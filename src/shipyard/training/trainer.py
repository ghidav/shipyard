"""The procedure: `train`, `job`, `step`. A recipe is a subclass overriding the rules it changes."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

import torch

from shipyard import record
from shipyard.credit import advantages, anchor_logprobs, sampler_logprobs
from shipyard.modules import Candidate
from shipyard.pack import datum, groups
from shipyard.sampling.endpoint import NothingServed
from shipyard.sampling.engine import HarborEngine
from shipyard.types import (
    Adam,
    Agent,
    Batches,
    Datum,
    Evaluation,
    Forward,
    Group,
    Job,
    Policy,
    Sequence,
    Task,
    TrainingClient,
    Update,
)

if TYPE_CHECKING:
    from shipyard.run import Run

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600

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
    """The plain policy gradient."""

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

    def __init__(self, run: Run) -> None:
        self.run = run
        self.recipe = run.config.recipe
        #: One engine, so every job in flight shares its `[rollout] concurrency`.
        self.engine = HarborEngine(run.config.rollout, progress=run.progress)
        self.steps = 0
        self.updates = 0
        self.epoch = 0

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        """One epoch: a step per yield of `sample`, the weights served after each update."""
        every = self.run.config.checkpoints.every
        async for found in self.sample(agent, tasks):
            if any(len(group.shaped) != len(group.members) for group in found):
                raise ValueError(f"{type(self).__name__}.sample yielded groups it never credited")
            update = await self.step(agent, found)
            self.log(found, update)
            self.steps += 1
            if update is None:
                continue
            self.updates += 1
            await agent.policy.sync(f"sample-{self.steps}")
            if every and self.updates % every == 0:
                await self.checkpoint(agent, f"step-{self.steps}")
        return agent

    async def sample(self, agent: Agent, tasks: Batches) -> AsyncIterator[list[Group]]:
        """How a step's groups are composed out of jobs: one job per batch."""
        for batch in tasks:
            yield self.credited(await self.job(agent, batch))

    async def job(
        self,
        agent: Agent,
        tasks: Values[Task],
        attempts: int | None = None,
        *,
        purpose: str = "training",
        **about: Any,
    ) -> list[Group]:
        """The one place a policy is sampled: run, record, group. The groups carry no
        advantages; `credited` sets them where a step trains on them."""
        run, endpoint = self.run, agent.policy.endpoint
        attempts = run.config.data.group_size if attempts is None else attempts
        job = Job.reserve(
            run,
            purpose=purpose,
            step=self.steps,
            epoch=self.epoch,
            modules=agent.modules,
            about=about,
        )
        run.progress.started(
            job.name, trials=len(tasks) * len(agent.harness) * attempts, step=self.steps
        )
        job.trials = await self.engine.run(job.name, tasks, attempts, agent=agent)
        run.progress.done(job.name)
        job.records = await endpoint.record(job.name, job.trials)
        run.recorded(job, party=endpoint.party)
        if job.records is not None:  # once the job is on record, stop if the endpoint is dead
            try:
                endpoint.check(job.name, job.records)
            except NothingServed as dead:
                dead.job = job
                raise
        return groups(job, agent)

    async def evaluate(self, agent: Agent, tasks: Values[Task], **more: Any) -> Evaluation | None:
        """One job per batch of `tasks` at `[data] test_group_size`, then one row with the mean."""
        if not tasks:
            return None
        data = self.run.config.data
        attempts = data.test_group_size or data.group_size
        found: list[Group] = []
        for start in range(0, len(tasks), data.batch_size):
            batch = tasks[start : start + data.batch_size]
            found += await self.job(agent, batch, attempts, purpose="evaluation")
        measured = [one for group in found for one in group.measured]
        rollouts = sum(len(group.members) for group in found)
        result = Evaluation(
            step=self.steps,
            epoch=self.epoch,
            tasks=len({group.task for group in found}),
            rollouts=rollouts,
            graded=len(measured),
            masked=rollouts - len(measured),
            reward_mean=fmean(float(one.verdict.reward or 0.0) for one in measured)
            if measured
            else None,
        )
        self.run.log(evaluation=True, **vars(result), **more)
        return result

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

    async def step(self, agent: Agent, found: Values[Group]) -> Update | None:
        """The gradient in `substeps` parts of whole groups; None without a client or datums."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0).
        recipe, run = self.recipe, self.run
        client = agent.policy.client
        if client is None:
            return None
        began = time.monotonic()
        party = agent.policy.endpoint.party or run.config.model.provider
        credit = await self.credit(agent.policy, found)
        run.spent(
            party, reference_tokens=credit.reference_tokens, anchor_tokens=credit.anchor_tokens
        )
        parts = [one for one in credit.datums if one]
        if not parts:
            return None
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        count = min(recipe.substeps, len(parts))
        pieces = [[one for group in piece for one in group] for piece in split(parts, count)]
        datums = [one for piece in pieces for one in piece]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        # An earlier run's optimizer state continues its schedule; a resume stands at `updates`.
        model = self.run.config.model
        rate = (
            recipe.learning_rate
            if model.from_checkpoint and model.restore_optimizer
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
        run.spent(party, train_tokens=train_tokens)
        seen = observed(datums, trained)
        return Update(
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

    async def _submit(
        self, client: TrainingClient, part: list[Datum], rate: float
    ) -> tuple[Awaitable[Forward], Awaitable[None]]:
        forward = await client.forward_backward(part, self.loss_fn, self.bounds())
        return forward, await client.optim_step(self.ADAM, rate)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> None:
        """The step's row in `ROW` order, naming its jobs, with `more` in it or after it."""
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
            "step": self.steps,
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
        self.run.log(**kept, **rest)

    async def checkpoint(self, agent: Agent, tag: str, *, keep: bool = False, **more: Any) -> None:
        """Save state and sampler weights under `tag`, and the agent written down to the record."""
        client = agent.policy.client
        if client is None:
            return
        hours = None if keep else self.run.config.checkpoints.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        state = await client.save_state(tag, ttl_seconds=ttl)
        sampler = await client.save_weights_for_sampler(tag, ttl_seconds=ttl)
        record.append(
            self.run.directory / record.CHECKPOINTS,
            {
                "tag": tag,
                "state_path": state,
                "sampler_path": sampler,
                "ttl_hours": hours,
                "step": self.steps,
                "updates": self.updates,
                "epoch": self.epoch,
                "modules": agent.modules.digest if agent.modules is not None else None,
                "harness": [one.name for one in agent.harness],
                **more,
            },
        )
        logger.info("checkpoint %s: %s", tag, sampler)

    def consumed(self, jobs: Values[Mapping[str, Any]], row: Mapping[str, Any]) -> int:
        """The batches of the checkpoint's epoch stepped before it: its training jobs. A run
        recorded before the keys were `step` and `training` wrote `batch` and `rollout`."""
        return sum(
            1
            for job in jobs
            if job.get("purpose") in ("training", "rollout")
            and job.get("epoch") == row["epoch"]
            and job.get("step", job.get("batch", -1)) < row["step"]
        )

    def restore(self, row: Mapping[str, Any], consumed: int) -> Candidate | None:
        """Stand where the checkpoint row stood; returns the modules the agent carries."""
        self.steps, self.updates, self.epoch = row["step"], row["updates"], row["epoch"]
        return self.run.carried


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
