"""The `Trainer` base class: the run loop, and the methods a recipe overrides to change it."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from types import TracebackType
from typing import Any, Self

import torch

from shipyard import record
from shipyard.config import Blueprint, Search, modules_dir
from shipyard.cost import Costs, billed_by, reported, sampled, sandbox_seconds
from shipyard.data import Generator
from shipyard.logs import Folder, Logger, Loggers, Wandb
from shipyard.modules import Candidate, keep, seed
from shipyard.progress import reporter
from shipyard.sampling.admit import judge
from shipyard.sampling.endpoint import NothingServed, endpoint_for
from shipyard.sampling.engine import HarborEngine
from shipyard.sampling.secrets import Pool, Secrets
from shipyard.training.client import TinkerClient
from shipyard.training.credit import advantages as relative
from shipyard.training.credit import anchor_logprobs, sampler_logprobs
from shipyard.training.pack import datum as packed
from shipyard.training.pack import groups
from shipyard.types import (
    Adam,
    Agent,
    Datum,
    Evaluation,
    Forward,
    Group,
    Job,
    Member,
    Policy,
    Sequence,
    Task,
    TrainingClient,
    TrainingStep,
    Update,
    harness,
)

logger = logging.getLogger(__name__)

#: Per-token values one training sequence is scored with, by name, aligned to its targets.
Signals = dict[str, list[float]]

SECONDS_PER_HOUR = 3600
#: Subdirectory of `<directory>/modules/` where the carried modules are kept for `seed` to read.
CARRIED = "carried"

#: Keys of a training step's log row, in order. A key with no value is left out.
ROW = (
    "step",
    "trained",
    "jobs",
    "groups",
    "rollouts",
    "graded",
    "masked",
    "degenerate",
    "sequences",
    "sequences_per_rollout",
    "train_tokens",
    "reward_mean",
    "reward_spread",
    "length_mean",
    "kl_v1",
    "kl_v2",
    "entropy",
    "anchor_kl",
    "overlong",
    "learning_rate",
    "substeps",
    "loss_fn",
    "seconds",
)


def warmed(learning_rate: float, warmup: int, done: int) -> float:
    """Return the learning rate after `done` updates, warmed up linearly over `warmup` updates."""
    if done >= warmup:
        return learning_rate
    return learning_rate * (done + 1) / warmup


def optimizer(adam: Adam, warmup: int, paper_warmup: int = 0) -> str:
    """Describe the optimizer settings for the `resolution` line."""
    said = f"adamw betas {adam.beta1} / {adam.beta2}, eps {adam.eps:g}"
    if adam.weight_decay > 0:
        said += f", weight decay {adam.weight_decay}"
    if adam.grad_clip_norm > 0:
        said += f", gradient norm clipped at {adam.grad_clip_norm}"
    if warmup > 0:
        steps = "step" if warmup == 1 else "steps"
        said += f", learning rate warmed up over {warmup} {steps}"
    elif paper_warmup > 0:
        said += f", no warm-up (the paper's is {paper_warmup} steps)"
    return said


def centred(rewards: Values[float | None], *, normalize: bool) -> list[float | None]:
    """Apply `advantages` to the measured rewards, leaving None entries in place."""
    found = [one for one in rewards if one is not None]
    scored = iter(relative(found, normalize=normalize) if found else [])
    return [next(scored) if one is not None else None for one in rewards]


@dataclass(frozen=True)
class Credit:
    """The datums a step trains on, by group, and the tokens spent computing them.

    A skipped group has an empty list. `reference_tokens` and `anchor_tokens` count the
    reference and anchor logprob passes; `anchor_kl` is the mean gap to the anchor, or None
    without a KL term."""

    datums: list[list[Datum]]
    reference_tokens: int
    anchor_tokens: int
    anchor_kl: float | None

    @property
    def sequences(self) -> int:
        return sum(len(one) for one in self.datums)


class Trainer:
    """Base trainer: a plain policy gradient and the run loop around it.

    `train` opens the agent, takes `[recipe] steps` steps, evaluates on the test set and saves
    a `final` checkpoint. Recipes subclass it and override the methods they change."""

    #: Tinker loss function; `bounds` supplies its config.
    loss_fn = "importance_sampling"
    #: Adam settings from the cookbook's `train_step` (no weight decay or gradient clipping),
    #: used where a paper does not state them. The SDK's default `eps` is 1e-12.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8)
    #: Warm-up steps used in the recipe's paper, shown next to the run's own `warmup`.
    PAPER_WARMUP = 0

    @classmethod
    def resolution(cls, recipe: Any) -> str:
        """Return the summary line `check` prints after the resolved config."""
        parts = [
            *cls.rules(recipe),
            "1 substep" if recipe.substeps == 1 else f"{recipe.substeps} substeps by prompt",
            optimizer(cls.ADAM, recipe.warmup, cls.PAPER_WARMUP),
            "degenerate groups dropped",
        ]
        if recipe.kl_coef > 0:
            parts.append(f"kl {recipe.kl_coef} to the starting weights")
        return f"# {recipe.kind}: " + "; ".join(parts)

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        """Return the clauses of `resolution` that describe the advantage and the loss.

        A recipe overrides this to describe its own rules."""
        return [
            "advantage = group mean, not divided by spread",
            f"loss = {cls.loss_fn}, {cls.aggregated()}",
        ]

    @classmethod
    def aggregated(cls) -> str:
        """Describe how the loss is aggregated over tokens, for `rules`.

        A recipe that aggregates differently overrides this."""
        return "summed over tokens"

    def __init__(self, config: Blueprint, directory: Path) -> None:
        self.config = config
        #: Run directory for log rows, checkpoints, modules and proxy records. A trainer on an
        #: existing directory continues from what an earlier one wrote there.
        self.directory = Path(directory)
        #: Run name, used for job names, the wandb run and the Tinker session.
        self.name = self.directory.name
        self.recipe = config.recipe
        # The blueprint's inputs are read first, so a missing dataset or a broken seed stops
        # the trainer before anything opens.
        #: Source of training batches and of the test set.
        self.generator = Generator.of(config)
        #: Modules the agent carries: the run directory's own, or those a checkpoint restores.
        self.modules = carried(config, self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        #: Terminal progress bars for running jobs.
        self.progress = reporter()
        #: A single engine, so all jobs in flight share one `[rollout] concurrency` limit.
        self.engine = HarborEngine(
            config.rollout,
            env=config.agent.env,
            kwargs=config.agent.kwargs,
            secrets=self.secrets(),
            progress=self.progress,
        )
        self.logs = Loggers(self.loggers())
        #: Accumulated costs, including those spent before a resume.
        self.costs = Costs.from_dict(record.read_json(self.directory / record.COSTS))
        self._next_job = next_job(self.name, self.directory, Path(config.rollout.jobs_dir))
        #: Checkpoint row the run resumed from; None on a fresh run.
        self.resumed: Mapping[str, Any] | None = None
        self.steps = 0
        self.updates = 0
        #: Number of batches drawn from the generator.
        self.drawn = 0
        #: Step after which the test set was last evaluated; None if it has not been.
        self.evaluated: int | None = None
        self._stream: Iterator[list[Task]] | None = None

    def loggers(self) -> list[Logger]:
        """Return the loggers that receive every row.

        The base writes to the run directory, and to wandb when `[logging.wandb]` names a
        project. A subclass may override this to add its own."""
        found: list[Logger] = [Folder(self.directory)]
        if self.config.logging.wandb is not None:
            found.append(Wandb(self.config, self.directory))
        return found

    def secrets(self) -> Secrets:
        """Return the secrets each trial acquires before it starts and releases after teardown.

        The default is a `Pool` of the identities `[rollout.secrets]` names, or no secrets.
        Called once, when the trainer is created. A subclass may override this to return its
        own `Secrets`."""
        table = self.config.rollout.secrets
        return Pool.of(table) if table else Secrets()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the progress bars, then the loggers, whether or not the run failed."""
        self.progress.close()
        self.logs.close(exc)

    def reserve_name(self) -> str:
        """Reserve and return the next job name, `<name>-NNNN`, by creating its directory.

        Raise `FileExistsError` if the directory already exists."""
        jobs_dir = Path(self.config.rollout.jobs_dir)
        name = f"{self.name}-{self._next_job:04d}"
        try:
            (jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {jobs_dir / name} exists") from None
        self._next_job += 1
        return name

    def agent(self) -> Agent:
        """Build the agent to sample, carrying `modules`.

        Its policy runs over the run's endpoint, with a training client when the recipe
        trains, and starts from the resumed checkpoint if there is one."""
        cfg, row = self.config, self.resumed or {}
        client = TinkerClient.of(cfg, name=self.name) if cfg.trains else None
        policy = Policy(
            endpoint_for(cfg, self.directory),
            client,
            start=row.get("sampler_path"),
            state=row.get("state_path"),
            anchored=cfg.trains and cfg.recipe.kl_coef > 0,
        )
        return Agent(policy, [harness(cfg.agent.harness)], self.modules)

    async def train(self) -> Agent:
        """Run training: open the agent, take `[recipe] steps` steps, then `finish`.

        A recipe that does not train (the evaluate kind) takes no steps. A subclass may
        override this to run a different loop around `advance` and `finish`."""
        steps = self.recipe.steps if self.config.trains else 0
        agent = self.agent()
        async with agent.policy:
            while self.steps < steps:
                await self.advance(agent)
            return await self.finish(agent)

    async def advance(self, agent: Agent) -> None:
        """Take one step of the run: `sample`, `step` and `log`.

        After an update, serve the new weights, and save a checkpoint every
        `[checkpoints] every` updates. Evaluate on the test set every `[recipe] evaluate_every`
        steps. Raise `ValueError` if `sample` returns groups that did not pass through
        `credited`."""
        found = await self.sample(agent)
        if any(len(group.shaped) != len(group.members) for group in found):
            raise ValueError(f"{type(self).__name__}.sample returned groups it never credited")
        done = await self.step(agent, found)
        self.log(done)
        self.steps += 1
        if done.update is not None:
            self.updates += 1
            await agent.policy.sync(f"sample-{self.steps}")
            every = self.config.checkpoints.every
            if every and self.updates % every == 0:
                await self.checkpoint(agent, f"step-{self.steps}")
        every = self.recipe.evaluate_every
        if every and self.steps % every == 0:
            await self.evaluate(agent, self.generator.test)

    async def finish(self, agent: Agent) -> Agent:
        """Evaluate the test set unless the last step just did, then save the `final` checkpoint."""
        if self.generator.test and self.evaluated != self.steps:
            await self.evaluate(agent, self.generator.test)
        await self.checkpoint(agent, "final", keep=True)
        return agent

    def draw(self) -> list[Task]:
        """Return the generator's next batch of tasks and count it in `drawn`.

        Raise `ValueError` if the run has no train tasks."""
        if self._stream is None:
            self._stream = self.generator.batches(self.drawn)
        batch = next(self._stream, None)
        if batch is None:
            raise ValueError("there is no batch to draw: the run names no train tasks")
        self.drawn += 1
        return batch

    async def sample(self, agent: Agent) -> list[Group]:
        """Return one step's credited groups, from one job over the next batch.

        A subclass may override this to sample differently, for example to keep sampling
        until enough groups carry signal. The groups it returns must have passed through
        `credited`."""
        return self.credited(await self.job(agent, self.draw()))

    async def job(
        self,
        agent: Agent,
        tasks: Values[Task],
        attempts: int | None = None,
        *,
        purpose: str = "training",
        **about: Any,
    ) -> list[Group]:
        """Sample the policy on `tasks` as one job and return its groups.

        This is the only place a policy is sampled: it runs the trials, takes the proxy's
        records, judges the trials, writes the job's row and groups the trials. `attempts`
        defaults to `[data] group_size`. The groups carry no advantages; `credited` sets them.
        Raise `NothingServed` if the endpoint served the job nothing."""
        cfg, endpoint = self.config, agent.policy.endpoint
        attempts = cfg.data.group_size if attempts is None else attempts
        job = Job(
            self.reserve_name(),
            purpose=purpose,
            step=self.steps,
            modules=agent.modules,
            about=about,
        )
        self.progress.started(
            job.name, trials=len(tasks) * len(agent.harness) * attempts, step=self.steps
        )
        job.trials = await self.engine.run(job.name, tasks, attempts, agent=agent)
        self.progress.done(job.name)
        job.records = await endpoint.record(job.name, job.trials)
        judge(job, check_turns=cfg.rollout.check_turns)
        masked = Counter(one.mask for one in job.verdicts if one.mask is not None)
        ended = Counter(one.ended for one in job.verdicts if one.ended is not None)
        self.record_job(
            job,
            party=endpoint.party or cfg.agent.provider,
            graded=sum(one.reward is not None for one in job.verdicts),
            **({"masked": dict(masked)} if masked else {}),
            **({"ended": dict(ended)} if ended else {}),
        )
        if job.records is not None:  # once the job is on record, stop if the endpoint is dead
            try:
                endpoint.check(job.name, job.records)
            except NothingServed as dead:
                dead.job = job
                raise
        return groups(job, agent)

    def record_job(self, job: Job, *, party: str, **facts: Any) -> None:
        """Write the job's row, including `facts`, and add the job's costs.

        A served job's tokens are counted from the proxy's records and billed to `party`.
        Otherwise the tokens are those the harness reported, billed to the provider the trials
        name, or else to `party`. Sandbox seconds are billed to the sandbox. A subclass that
        runs its own jobs calls this to record them."""
        served = job.records is not None
        if served:
            counted, noted = sampled(job)
        else:
            counted, party, noted = reported(job), billed_by(job) or party, {}
        sandbox = self.config.rollout.sandbox
        timed, seconds = sandbox_seconds(job)
        self.logs.job(
            {
                "job": job.name,
                "purpose": job.purpose,
                "trials": len(job.trials),
                **({"step": job.step} if job.step is not None else {}),
                "tasks": len({one.task for one in job.trials}),
                **job.about,
                **({"modules": job.modules.digest} if job.modules is not None else {}),
                **facts,
                "served": "tinker" if served else "provider",
                **{key: value for key, value in noted.items() if value},
                "party": party,
                "input_tokens": counted["input_tokens"],
                "cache_tokens": counted["cache_tokens"],
                "output_tokens": counted["output_tokens"],
                "sandbox": sandbox,
                "sandbox_seconds": seconds,
            }
        )
        self.costs.add_party(party, **counted)
        self.costs.add_sandbox(sandbox, timed, seconds)
        self.logs.costs(self.costs)

    def spent(self, party: str, **counts: int) -> None:
        """Add training token counts to `party`'s costs, skipping zero counts."""
        counted = {key: int(value) for key, value in counts.items() if value}
        if counted:
            self.costs.add_party(party, **counted)
            self.logs.costs(self.costs)

    async def evaluate(self, agent: Agent, tasks: Values[Task], **more: Any) -> Evaluation | None:
        """Evaluate the agent on `tasks` and log one `eval` row with the mean reward.

        The tasks run in jobs of `[data] batch_size` tasks, each with `[data] test_group_size`
        attempts (`group_size` when unset). Return None when `tasks` is empty."""
        if not tasks:
            return None
        data = self.config.data
        attempts = data.test_group_size or data.group_size
        found: list[Group] = []
        for start in range(0, len(tasks), data.batch_size):
            batch = tasks[start : start + data.batch_size]
            found += await self.job(agent, batch, attempts, purpose="evaluation")
        measured = [one for group in found for one in group.measured]
        rollouts = sum(len(group.members) for group in found)
        result = Evaluation(
            step=self.steps,
            tasks=len({group.task for group in found}),
            rollouts=rollouts,
            graded=len(measured),
            masked=rollouts - len(measured),
            reward_mean=fmean(float(one.verdict.reward or 0.0) for one in measured)
            if measured
            else None,
        )
        self.logs.log("eval", evaluation=True, **vars(result), **more)
        self.evaluated = self.steps
        return result

    async def measure(self, tasks: Values[Task]) -> Evaluation | None:
        """Evaluate the resumed checkpoint on `tasks` and log the result as an `eval` row.

        Raise `ValueError` if the run was not resumed from a checkpoint."""
        if self.resumed is None:
            raise ValueError("an evaluation needs a run reopened at a checkpoint")
        cfg = self.config
        policy = Policy(endpoint_for(cfg, self.directory), start=self.resumed["sampler_path"])
        agent = Agent(policy, [harness(cfg.agent.harness)], self.modules)
        async with agent.policy:
            return await self.evaluate(agent, list(tasks), checkpoint=self.resumed["tag"])

    def credited(self, found: list[Group]) -> list[Group]:
        """Set each group's shaped rewards and advantages, and return the groups.

        The values come from `shape` and `advantage`; override those to change how credit is
        assigned."""
        for group in found:
            group.shaped = self.shape(group)
            group.advantages = self.advantage(group)
        return found

    def shape(self, group: Group) -> list[float | None]:
        """Return each member's shaped reward, or None for an unmeasured member.

        The base returns the verifier's reward unchanged. A recipe overrides this to shape
        rewards, for example with a length penalty."""
        return [float(one.verdict.reward) if one.measured else None for one in group.members]

    def advantage(self, group: Group) -> list[float | None]:
        """Return each member's advantage, or None for an unmeasured member.

        The base subtracts the group's mean shaped reward. A recipe overrides this to
        normalize or otherwise change the advantage."""
        return centred(group.shaped, normalize=False)

    def weights(self, sizes: Values[int]) -> list[float]:
        """Return each credited group's factor on its tokens' advantages.

        `sizes` holds each group's target-token count. The base returns 1.0 for every group,
        which keeps Tinker's sum over tokens. A recipe overrides this to change how token
        losses are weighted across groups."""
        return [1.0] * len(sizes)

    def bounds(self) -> dict[str, float] | None:
        """Return the `loss_fn_config` sent to Tinker with `loss_fn`.

        The base loss takes none. A recipe overrides this to set its clipping bounds."""
        return None

    def trains_on(self, group: Group) -> bool:
        """Return whether `group` carries a gradient.

        The base requires reward variance: under a group-relative advantage, a group whose
        members all scored the same has none. A recipe with a per-token signal of its own,
        such as a teacher's, overrides this to keep such groups."""
        return group.reward_variance > 0

    async def credit(self, policy: Policy, found: Values[Group]) -> Credit:
        """Build the datums for each group, none for a group `trains_on` rejects.

        The layers run in order: `signals` scores every training sequence, `token_advantages`
        turns each rollout's advantage and its signals into per-token advantages, the group's
        factor from `weights` scales them, and `datum` packs each sequence for the loss."""
        recipe = self.recipe
        kept: list[list[tuple[Member, float]]] = [
            [
                (one, advantage)
                for one, advantage in zip(group.members, group.advantages, strict=True)
                if advantage is not None
            ]
            if self.trains_on(group)
            else []
            for group in found
        ]
        pairs = [
            (member, one) for members in kept for member, _ in members for one in member.sequences
        ]
        if not pairs:
            return Credit([[] for _ in found], 0, 0, None)
        scored = await self.signals(policy, pairs)
        sequences = [one for _, one in pairs]
        reference_tokens = (
            0 if recipe.reference == "sampler" else sum(len(one.tokens) - 1 for one in sequences)
        )
        anchor_tokens = sum(
            len(one.tokens) for one, row in zip(sequences, scored, strict=True) if "anchor" in row
        )
        sizes = [
            sum(len(one.targets) for member, _ in members for one in member.sequences)
            for members in kept
            if members
        ]
        scale = iter(self.weights(sizes))
        rows = iter(scored)
        datums: list[list[Datum]] = []
        for members in kept:
            built: list[Datum] = []
            factor = next(scale) if members else 1.0
            for member, advantage in members:
                for sequence in member.sequences:
                    row = next(rows)
                    per_token = self.token_advantages(member, sequence, row, advantage)
                    built.append(self.datum(sequence, row, [factor * one for one in per_token]))
            datums.append(built)
        gaps = [
            m - a
            for row in scored
            if "anchor" in row
            for m, a in zip(row["reference"], row["anchor"], strict=True)
        ]
        return Credit(datums, reference_tokens, anchor_tokens, fmean(gaps) if gaps else None)

    async def signals(
        self, policy: Policy, pairs: Values[tuple[Member, Sequence]]
    ) -> list[Signals]:
        """Return each training sequence's per-token signals, aligned to its targets.

        `reference`, the logprobs the loss's ratio is taken against, is always there;
        `anchor`, the starting weights' logprobs, is there when `kl_coef` is set. A recipe adds
        its own, such as a teacher's logprobs, for `token_advantages` or `datum` to read. Raise
        `ValueError` if `kl_coef` is set but there is no anchor."""
        sequences = [one for _, one in pairs]
        found: list[Signals] = [
            {"reference": list(one)}
            for one in await self.reference_logprobs(policy.client, sequences)
        ]
        kl_coef = self.recipe.kl_coef
        if kl_coef > 0:
            if policy.anchor is None:
                raise ValueError(f"kl_coef = {kl_coef} needs the starting weights to anchor to")
            anchored = await anchor_logprobs(policy.anchor, sequences)
            for row, base in zip(found, anchored, strict=True):
                row["anchor"] = base
        return found

    async def reference_logprobs(
        self, client: TrainingClient, sequences: Values[Sequence]
    ) -> list[list[float]]:
        """Return the logprobs the loss's ratio is taken against, at every target.

        They are read from the proxy's records when `[recipe] reference = "sampler"`, and
        otherwise recomputed by the training client."""
        if self.recipe.reference == "sampler":
            return [sampler_logprobs(one) for one in sequences]
        return await client.compute_logprobs(sequences)

    def token_advantages(
        self, member: Member, sequence: Sequence, signals: Signals, advantage: float
    ) -> list[float]:
        """Return the per-token advantages of one sequence of `member`'s rollout.

        The base puts the rollout's `advantage` on every target, less `kl_coef` times the gap
        between `reference` and `anchor` when there is one. A recipe overrides this to add, reshape
        or replace per-token terms; the group's factor from `weights` is applied after."""
        per_token = [advantage] * len(sequence.targets)
        if "anchor" not in signals:
            return per_token
        kl_coef = self.recipe.kl_coef
        return [
            a - kl_coef * (m - b)
            for a, m, b in zip(per_token, signals["reference"], signals["anchor"], strict=True)
        ]

    def datum(self, sequence: Sequence, signals: Signals, advantages: Values[float]) -> Datum:
        """Return the Tinker datum for one sequence, in the format the loss takes.

        The base carries `reference` and the per-token advantages, as `loss_fn` reads them. A recipe
        with another loss, such as `cross_entropy` over soft targets, overrides this together
        with `forward_backward`."""
        return packed(sequence, logprobs=signals["reference"], advantages=list(advantages))

    async def step(self, agent: Agent, found: Values[Group]) -> TrainingStep:
        """Take one training step on the credited groups.

        The datums are split into `substeps` parts of whole groups, each one forward-backward
        pass and one optimizer step. No update is taken without a training client or without
        datums."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0).
        recipe = self.recipe
        client = agent.policy.client
        if client is None:
            return TrainingStep(self.steps, list(found))
        began = time.monotonic()
        party = agent.policy.endpoint.party or self.config.agent.provider
        credit = await self.credit(agent.policy, found)
        self.spent(
            party, reference_tokens=credit.reference_tokens, anchor_tokens=credit.anchor_tokens
        )
        parts = [one for one in credit.datums if one]
        if not parts:
            return TrainingStep(self.steps, list(found))
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        count = min(recipe.substeps, len(parts))
        pieces = [[one for group in piece for one in group] for piece in split(parts, count)]
        datums = [one for piece in pieces for one in piece]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        # An earlier run's optimizer state continues its schedule; a resume stands at `updates`.
        rate = (
            recipe.learning_rate
            if self.config.agent.from_checkpoint and recipe.restore_optimizer
            else warmed(recipe.learning_rate, recipe.warmup, self.updates)
        )
        trained: list[torch.Tensor] = []
        forward, optim = await self._submit(client, pieces[0], rate)
        for index in range(count):
            following = (
                await self._submit(client, pieces[index + 1], rate) if index + 1 < count else None
            )
            result = await forward
            trained.extend(torch.tensor(row) for row in result.logprobs)
            await optim
            if following is not None:
                forward, optim = following
        self.spent(party, train_tokens=train_tokens)
        seen = observed(datums, trained)
        update = Update(
            train_tokens=train_tokens,
            substeps=count,
            learning_rate=rate,
            loss_fn=self.loss_fn,
            kl_v1=seen.get("kl_v1"),
            kl_v2=seen.get("kl_v2"),
            entropy=seen.get("entropy"),
            seconds=round(time.monotonic() - began, 3),
            sequences=credit.sequences,
            anchor_kl=credit.anchor_kl,
        )
        return TrainingStep(self.steps, list(found), update)

    async def forward_backward(
        self, client: TrainingClient, part: list[Datum]
    ) -> Awaitable[Forward]:
        """Submit one substep's forward-backward pass and return its result's awaitable.

        The base sends `loss_fn` with `bounds`. A recipe overrides this to send several losses,
        whose gradients add up before the optimizer step (return `together(...)` of their
        results), or a loss of its own. It must submit
        before it returns, so the next substep can be queued behind it, and it must not run
        concurrently with another training call."""
        return await client.forward_backward(part, self.loss_fn, self.bounds())

    async def _submit(
        self, client: TrainingClient, part: list[Datum], rate: float
    ) -> tuple[Awaitable[Forward], Awaitable[None]]:
        forward = await self.forward_backward(client, part)
        return forward, await client.optim_step(self.ADAM, rate)

    def log(self, done: TrainingStep, **more: Any) -> None:
        """Log the step's `train` row, with keys in `ROW` order followed by the rest of `more`.

        A subclass may override this to add its own fields through `more`."""
        found, update = done.groups, done.update
        measured = [one for group in found for one in group.measured]
        rollouts = sum(len(group.members) for group in found)
        spreads = [
            pstdev(rewards) if len(rewards) > 1 else 0.0
            for group in found
            if (rewards := [one for one in group.shaped if one is not None])
        ]
        credited = sum(len(group.measured) for group in found if self.trains_on(group))
        sequences = update.sequences if update is not None else 0
        jobs = list(dict.fromkeys(one.trial.job for group in found for one in group.members))
        row: dict[str, Any] = {
            "step": done.index,
            "trained": update is not None,
            "jobs": jobs,
            "groups": len(found),
            "rollouts": rollouts,
            "graded": len(measured),
            "masked": rollouts - len(measured),
            "degenerate": sum(1 for group in found if group.measured and not self.trains_on(group)),
            "sequences": sequences,
            "sequences_per_rollout": sequences / credited if credited else None,
            "reward_mean": fmean(float(one.verdict.reward or 0.0) for one in measured)
            if measured
            else None,
            "reward_spread": fmean(spreads) if spreads else None,
            "length_mean": fmean(one.sampled_tokens for one in measured) if measured else None,
        }
        if update is not None:
            row.update({key: getattr(update, key) for key in ROW if hasattr(update, key)})
        row.update(more)
        kept = {key: value for key in ROW if (value := row.get(key)) is not None}
        rest = {key: value for key, value in more.items() if key not in ROW}
        self.logs.log("train", **kept, **rest)

    async def checkpoint(self, agent: Agent, tag: str, *, keep: bool = False, **more: Any) -> None:
        """Save the training state and sampler weights under `tag`, and log a checkpoint row.

        The row records both paths, the step counters, and the agent's modules and harness.
        With `keep`, the checkpoint never expires; otherwise it expires after
        `[checkpoints] ttl_hours`. Do nothing without a training client. A subclass may
        override this to add fields to the row through `more`."""
        client = agent.policy.client
        if client is None:
            return
        hours = None if keep else self.config.checkpoints.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        state = await client.save_state(tag, ttl_seconds=ttl)
        sampler = await client.save_weights_for_sampler(tag, ttl_seconds=ttl)
        self.logs.checkpoint(
            {
                "tag": tag,
                "state_path": state,
                "sampler_path": sampler,
                "ttl_hours": hours,
                "step": self.steps,
                "updates": self.updates,
                "drawn": self.drawn,
                "modules": agent.modules.digest if agent.modules is not None else None,
                "harness": [one.name for one in agent.harness],
                **more,
            },
        )
        logger.info("checkpoint %s: %s", tag, sampler)

    def restore(self, row: Mapping[str, Any]) -> None:
        """Restore the step counters, the batches drawn and the weights from a checkpoint row.

        The weights are used when `agent` next builds the policy. A subclass that adds fields
        in `checkpoint` overrides this to restore them."""
        self.resumed = row
        self.steps, self.updates, self.drawn = row["step"], row["updates"], row["drawn"]


def carried(config: Blueprint, directory: Path) -> Candidate | None:
    """Return the modules every job carries.

    These are the copy kept in `directory`, else the ones `[agent] modules` seeds, which are
    then saved there so a resume carries the same text. Return None when the key is unset or
    the recipe (gepa, fst) searches over its own."""
    if isinstance(config.recipe, Search):
        return None
    kept = directory / record.MODULES / CARRIED
    if kept.is_dir():
        return seed(kept)
    found = seeded(config)
    if found is not None:
        keep(found, kept)
    return found


def seeded(config: Blueprint) -> Candidate | None:
    """Return the modules `[agent] modules` seeds, for any recipe, or None when it is unset."""
    named = modules_dir(config)
    return None if named is None else seed(named)


def next_job(name: str, directory: Path, jobs_dir: Path) -> int:
    """Return the next job number: one past the highest in the job log or on disk, else 0."""
    named = [row.get("job") for row in record.read(directory / record.JOBS)]
    if jobs_dir.is_dir():
        named += [one.name for one in jobs_dir.iterdir()]
    prefix = f"{name}-"
    numbers = [
        int(one[len(prefix) :])
        for one in named
        if isinstance(one, str) and one.startswith(prefix) and one[len(prefix) :].isdigit()
    ]
    return max(numbers, default=-1) + 1


async def together(*pending: Awaitable[Forward]) -> Forward:
    """Await several submitted forward-backward passes and return the last one's result.

    For a `forward_backward` that sends more than one loss: Tinker adds their gradients up
    before the optimizer step, and the last result's logprobs feed the step's metrics."""
    found = [await one for one in pending]
    return found[-1]


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """Split `items` into `parts` slices whose sizes differ by at most one.

    The slices are cut as the cookbook's `split_list` cuts them."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


def observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """Return kl_v1, kl_v2 and entropy over the target tokens, computed as the cookbook does.

    Return {} if the logprobs cannot be read."""
    if len(trained) != len(datums):
        return {}
    try:
        drifts: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        for one, scored in zip(datums, trained, strict=True):
            inputs = one.loss_fn_inputs
            if "logprobs" not in inputs or "mask" not in inputs or scored is None:
                return {}
            reference = inputs["logprobs"].to_torch()
            acted = inputs["mask"].to_torch() > 0
            if not bool(acted.any()):
                continue
            drifts.append(reference[acted] - scored[acted])
            chosen.append(reference[acted])
        if not drifts:
            return {}
        moved, sampled = torch.cat(drifts), torch.cat(chosen)
        return {
            "kl_v1": float(moved.mean()),
            "kl_v2": float(0.5 * (moved**2).mean()),
            "entropy": float(-sampled.mean()),
        }
    except Exception:  # noqa: BLE001 - these numbers are extras and must not fail the step
        logger.warning("could not read the step's own logprobs; the row goes without them")
        return {}
