"""The run: `train`, `sample`, `job`, `step`. A recipe is a subclass overriding what it changes."""

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
from shipyard.admit import judge
from shipyard.config import Blueprint, Search, modules_dir
from shipyard.cost import Costs, billed_by, reported, sampled, sandbox_seconds
from shipyard.data import Generator
from shipyard.logs import Folder, Logger, Loggers, Wandb
from shipyard.modules import Candidate, keep, seed
from shipyard.progress import reporter
from shipyard.sampling.endpoint import NothingServed, endpoint_for
from shipyard.sampling.engine import HarborEngine
from shipyard.training.client import TinkerClient
from shipyard.training.credit import advantages, anchor_logprobs, sampler_logprobs
from shipyard.training.pack import datum, groups
from shipyard.types import (
    Adam,
    Agent,
    Datum,
    Evaluation,
    Forward,
    Group,
    Job,
    Policy,
    Sequence,
    Task,
    TrainingClient,
    TrainingStep,
    Update,
    harness,
)

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600
#: Under `<directory>/modules/`: the candidate a trainer carries, as `seed` reads it back.
CARRIED = "carried"

#: The step's row keys, in this order. A measure with nothing behind it is left out.
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
    """The rate after `done` updates, rising linearly to `learning_rate` over `warmup`."""
    if done >= warmup:
        return learning_rate
    return learning_rate * (done + 1) / warmup


def optimizer(adam: Adam, warmup: int, paper_warmup: int = 0) -> str:
    """The optimizer clause of the resolution line."""
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
    """`advantages` over the measured positions, None kept where it was."""
    found = [one for one in rewards if one is not None]
    scored = iter(advantages(found, normalize=normalize) if found else [])
    return [next(scored) if one is not None else None for one in rewards]


@dataclass(frozen=True)
class Credit:
    """What a step trains on: the datums by group, empty where skipped, and the passes' cost."""

    datums: list[list[Datum]]
    reference_tokens: int
    anchor_tokens: int
    anchor_kl: float | None

    @property
    def sequences(self) -> int:
        return sum(len(one) for one in self.datums)


class Trainer:
    """The plain policy gradient, and the run around it: the agent opened, `[recipe] steps`
    steps taken, the test set measured and the weights checkpointed `final`."""

    #: Tinker's loss, with `bounds` as its config.
    loss_fn = "importance_sampling"
    #: Adam as the cookbook's `train_step` sets it (no weight decay or gradient clipping),
    #: for whatever a paper leaves unstated. The SDK's `eps` default is 1e-12.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8)
    #: The warm-up the recipe's paper trains with, shown beside the run's own `warmup`.
    PAPER_WARMUP = 0

    @classmethod
    def resolution(cls, recipe: Any) -> str:
        """The line `check` prints after the resolved config."""
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
        """The clauses of `resolution` that name this class's rules."""
        return [
            "advantage = group mean, not divided by spread",
            f"loss = {cls.loss_fn}, {cls.aggregated()}",
        ]

    @classmethod
    def aggregated(cls) -> str:
        return "summed over tokens"

    def __init__(self, config: Blueprint, directory: Path) -> None:
        self.config = config
        #: Where the trainer writes: its rows, its checkpoints, its modules, the proxy's
        #: records. A trainer goes on from what an earlier one wrote there.
        self.directory = Path(directory)
        #: What its jobs, its wandb run and its Tinker session are named after.
        self.name = self.directory.name
        self.recipe = config.recipe
        # The blueprint's inputs are read first, so a missing dataset or a broken seed stops
        # the trainer before anything opens.
        #: The train set as a stream of batches, and the test set.
        self.generator = Generator.of(config)
        #: What the agent carries: the directory's own modules, or the ones a checkpoint
        #: restores.
        self.modules = carried(config, self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        #: The jobs' bars on a terminal.
        self.progress = reporter()
        #: One engine, so every job in flight shares its `[rollout] concurrency`.
        self.engine = HarborEngine(
            config.rollout, env=config.agent.env, kwargs=config.agent.kwargs, progress=self.progress
        )
        #: Where every row goes.
        self.logs = Loggers(self.loggers())
        #: What has been spent, going on from what was spent before a resume.
        self.costs = Costs.from_dict(record.read_json(self.directory / record.COSTS))
        self._next_job = next_job(self.name, self.directory, Path(config.rollout.jobs_dir))
        #: The checkpoint row the run reopened at; None on a fresh run.
        self.resumed: Mapping[str, Any] | None = None
        self.steps = 0
        self.updates = 0
        #: Batches drawn from the generator.
        self.drawn = 0
        #: The step the test set was last measured after; None before it was.
        self.evaluated: int | None = None
        self._stream: Iterator[list[Task]] | None = None

    def loggers(self) -> list[Logger]:
        """Where this trainer writes: its directory, and wandb when `[logging.wandb]` names a
        project. A subclass adds its own."""
        found: list[Logger] = [Folder(self.directory)]
        if self.config.logging.wandb is not None:
            found.append(Wandb(self.config, self.directory))
        return found

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close what the trainer opened, however it ended: its bars, then its loggers."""
        self.progress.close()
        self.logs.close(exc)

    def reserve_name(self) -> str:
        """The next job name, `<name>-NNNN`, with its directory created; raises if it exists."""
        jobs_dir = Path(self.config.rollout.jobs_dir)
        name = f"{self.name}-{self._next_job:04d}"
        try:
            (jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {jobs_dir / name} exists") from None
        self._next_job += 1
        return name

    def agent(self) -> Agent:
        """The agent the trainer samples: the policy over its endpoint, with the training
        client when the recipe trains, standing at the checkpoint reopened at, and carrying
        `modules`."""
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
        """The run: open the agent, take `[recipe] steps` steps (the evaluate kind takes none),
        then `finish`."""
        steps = self.recipe.steps if self.config.trains else 0
        agent = self.agent()
        async with agent.policy:
            while self.steps < steps:
                await self.advance(agent)
            return await self.finish(agent)

    async def advance(self, agent: Agent) -> None:
        """One step of the run: `sample`, `step`, `log`; then the new weights served, and the
        checkpoint and the evaluation on their cadences."""
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
        """The run's end: the test set measured unless the last step just was, and the
        weights checkpointed `final`."""
        if self.generator.test and self.evaluated != self.steps:
            await self.evaluate(agent, self.generator.test)
        await self.checkpoint(agent, "final", keep=True)
        return agent

    def draw(self) -> list[Task]:
        """The generator's next batch, counted in `drawn`."""
        if self._stream is None:
            self._stream = self.generator.batches(self.drawn)
        batch = next(self._stream, None)
        if batch is None:
            raise ValueError("there is no batch to draw: the run names no train tasks")
        self.drawn += 1
        return batch

    async def sample(self, agent: Agent) -> list[Group]:
        """A step's groups, credited: one job over the generator's next batch."""
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
        """The one place a policy is sampled: run, judge, record, group. The groups carry no
        advantages; `credited` sets them where a step trains on them."""
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
        """The job's row with `facts`, and its bill: a served job's tokens counted off its
        records under `party`; else the tokens its harness reported, under the provider its
        trials name or else `party`. A job's sandbox seconds go to its sandbox."""
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
        """Training counts added to a party; zero counts are skipped."""
        counted = {key: int(value) for key, value in counts.items() if value}
        if counted:
            self.costs.add_party(party, **counted)
            self.logs.costs(self.costs)

    async def evaluate(self, agent: Agent, tasks: Values[Task], **more: Any) -> Evaluation | None:
        """One job per batch of `tasks` at `[data] test_group_size`, then one row with the mean."""
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
        """The checkpoint the trainer stands at, evaluated on `tasks` as one more row."""
        if self.resumed is None:
            raise ValueError("an evaluation needs a run reopened at a checkpoint")
        cfg = self.config
        policy = Policy(endpoint_for(cfg, self.directory), start=self.resumed["sampler_path"])
        agent = Agent(policy, [harness(cfg.agent.harness)], self.modules)
        async with agent.policy:
            return await self.evaluate(agent, list(tasks), checkpoint=self.resumed["tag"])

    def credited(self, found: list[Group]) -> list[Group]:
        """`found`, each group's shaped rewards and advantages set by this class's rules."""
        for group in found:
            group.shaped = self.shape(group)
            group.advantages = self.advantage(group)
        return found

    def shape(self, group: Group) -> list[float | None]:
        """The rewards as the verifier scored them; None on an unmeasured member."""
        return [float(one.verdict.reward) if one.measured else None for one in group.members]

    def advantage(self, group: Group) -> list[float | None]:
        """The shaped rewards less the group's mean; None on an unmeasured member."""
        return centred(group.shaped, normalize=False)

    def weights(self, sizes: Values[int]) -> list[float]:
        """Each credited group's factor on its tokens' advantages; the base leaves Tinker's sum."""
        return [1.0] * len(sizes)

    def bounds(self) -> dict[str, float] | None:
        """`loss_fn_config` for the step; the base loss takes none."""
        return None

    async def credit(self, policy: Policy, found: Values[Group]) -> Credit:
        """The datums by group, none for a flat one; the anchor gap times `kl_coef` subtracted."""
        recipe = self.recipe
        kept: list[list[tuple[Any, float]]] = []
        for group in found:
            if group.reward_variance <= 0:
                kept.append([])
                continue
            kept.append(
                [
                    (one, advantage)
                    for one, advantage in zip(group.members, group.advantages, strict=True)
                    if advantage is not None
                ]
            )
        sequences = [one for members in kept for member, _ in members for one in member.sequences]
        if not sequences:
            return Credit([[] for _ in found], 0, 0, None)
        mu = await self.mu(policy.client, sequences)
        reference_tokens = (
            0 if recipe.reference == "sampler" else sum(len(one.tokens) - 1 for one in sequences)
        )
        gaps = await self.gaps(sequences, mu, policy.anchor)
        anchor_tokens = 0 if gaps is None else sum(len(one.tokens) for one in sequences)
        sizes = [
            sum(len(one.targets) for member, _ in members for one in member.sequences)
            for members in kept
            if members
        ]
        scale = iter(self.weights(sizes))
        datums: list[list[Datum]] = []
        cursor = 0
        for members in kept:
            built: list[Datum] = []
            factor = next(scale) if members else 1.0
            for member, advantage in members:
                for sequence in member.sequences:
                    per_token = [advantage] * len(sequence.targets)
                    if gaps is not None:
                        per_token = [
                            a - recipe.kl_coef * g
                            for a, g in zip(per_token, gaps[cursor], strict=True)
                        ]
                    built.append(
                        datum(
                            sequence,
                            logprobs=mu[cursor],
                            advantages=[factor * one for one in per_token],
                        )
                    )
                    cursor += 1
            datums.append(built)
        gap_mean = fmean(g for row in gaps for g in row) if gaps and any(gaps) else None
        return Credit(datums, reference_tokens, anchor_tokens, gap_mean)

    async def mu(self, client: TrainingClient, sequences: Values[Sequence]) -> list[list[float]]:
        """The sampler's logprobs at every target, from the records or the training client."""
        if self.recipe.reference == "sampler":
            return [sampler_logprobs(one) for one in sequences]
        return await client.compute_logprobs(sequences)

    async def gaps(
        self, sequences: Values[Sequence], mu: Values[Values[float]], anchor: Any
    ) -> list[list[float]] | None:
        """mu less the anchor's logprobs at every target; None without a KL term."""
        kl_coef = self.recipe.kl_coef
        if kl_coef <= 0:
            return None
        if anchor is None:
            raise ValueError(f"kl_coef = {kl_coef} needs the starting weights to anchor to")
        anchored = await anchor_logprobs(anchor, sequences)
        return [
            [a - b for a, b in zip(scored, base, strict=True)]
            for scored, base in zip(mu, anchored, strict=True)
        ]

    async def step(self, agent: Agent, found: Values[Group]) -> TrainingStep:
        """The gradient in `substeps` parts of whole groups; no update without a client or
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

    async def _submit(
        self, client: TrainingClient, part: list[Datum], rate: float
    ) -> tuple[Awaitable[Forward], Awaitable[None]]:
        forward = await client.forward_backward(part, self.loss_fn, self.bounds())
        return forward, await client.optim_step(self.ADAM, rate)

    def log(self, done: TrainingStep, **more: Any) -> None:
        """The step's row in `ROW` order, naming its jobs, with `more` in it or after it."""
        found, update = done.groups, done.update
        measured = [one for group in found for one in group.measured]
        rollouts = sum(len(group.members) for group in found)
        spreads = [
            pstdev(rewards) if len(rewards) > 1 else 0.0
            for group in found
            if (rewards := [one for one in group.shaped if one is not None])
        ]
        credited = sum(len(group.measured) for group in found if group.reward_variance > 0)
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
            "degenerate": sum(
                1 for group in found if group.measured and group.reward_variance <= 0
            ),
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
        """Save state and sampler weights under `tag`, and the agent written down to the record."""
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
        """Stand where the checkpoint row stood: its counts, the batches it had drawn, and its
        weights for the agent to open at."""
        self.resumed = row
        self.steps, self.updates, self.drawn = row["step"], row["updates"], row["drawn"]


def carried(config: Blueprint, directory: Path) -> Candidate | None:
    """The candidate every job carries: the copy kept in `directory`, else the one `[agent]
    modules` seeds, kept there so a resume carries the same text. None when the key is unset
    or the recipe (gepa, fst) searches over its own."""
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
    """The candidate `[agent] modules` seeds, for any recipe. None when it is unset."""
    named = modules_dir(config)
    return None if named is None else seed(named)


def next_job(name: str, directory: Path, jobs_dir: Path) -> int:
    """One past the highest job number on the record or on disk; 0 for a new directory."""
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


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """`parts` slices, sizes within one of each other, cut as the cookbook's `split_list`."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


def observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """kl_v1, kl_v2 and entropy over the acted tokens, as the cookbook's; {} when unreadable."""
    if len(trained) != len(datums):
        return {}
    try:
        drifts: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        for one, scored in zip(datums, trained, strict=True):
            inputs = one.loss_fn_inputs
            if "logprobs" not in inputs or "mask" not in inputs or scored is None:
                return {}
            mu = inputs["logprobs"].to_torch()
            acted = inputs["mask"].to_torch() > 0
            if not bool(acted.any()):
                continue
            drifts.append(mu[acted] - scored[acted])
            chosen.append(mu[acted])
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
