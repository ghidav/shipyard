"""The procedure: `train` is one epoch, `job` the one place a policy is sampled, `step`
the gradient over the groups a step holds. The base is GRPO-shaped, and a recipe sets its
preset and replaces what it must."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable
from collections.abc import Sequence as Values
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

import torch

from shipyard import record
from shipyard.credit import advantages, anchor_logprobs, overlong, sampler_logprobs, shaped, weights
from shipyard.pack import datum, groups
from shipyard.types import (
    Agent,
    Batches,
    Checkpoint,
    Datum,
    Forward,
    Group,
    Sequence,
    Task,
    TrainingClient,
    Update,
)

if TYPE_CHECKING:
    from shipyard.gradient import Preset
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


@dataclass(frozen=True)
class Credit:
    """What a step trains on: the datums by group, empty for a group it skipped, and what
    the reference and anchor passes cost."""

    datums: list[list[Datum]]
    reference_tokens: int
    anchor_tokens: int
    anchor_kl: float | None

    @property
    def sequences(self) -> int:
        return sum(len(one) for one in self.datums)


class Trainer:
    """`anchor` is a Tinker sampling client on the agent's starting weights, needed when the
    preset has a KL term."""

    def __init__(self, run: Run, preset: Preset, *, anchor: Any = None) -> None:
        self.run = run
        self.preset = preset
        self.anchor = anchor
        self.steps = 0
        self.updates = 0
        self.epoch = 0

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        """One epoch: a step per yield of `sample`, the weights served after each update, a
        checkpoint every `[checkpoints] every` updates."""
        preset, every = self.preset, self.run.config.checkpoints.every
        if self.steps == 0 and preset.overlong_penalty > 0 and agent.policy.endpoint.budget is None:
            logger.warning(
                "%s's overlong term is off for this run: the proxy reports no token budget, "
                "so overlong_penalty = %s docks no rollout",
                preset.name,
                preset.overlong_penalty,
            )
        async for found in self.sample(agent, tasks):
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
            yield await self.job(agent, batch)

    async def job(
        self,
        agent: Agent,
        tasks: Values[Task],
        attempts: int | None = None,
        *,
        purpose: str = "rollout",
    ) -> list[Group]:
        """The one place a policy is sampled: run, record, group, and fill each group's
        shaped rewards and advantages. `attempts` defaults to `[data] group_size`."""
        run = self.run
        attempts = run.config.data.group_size if attempts is None else attempts
        run.noted()
        job = run.reserve_job()
        run.progress.started(
            job, trials=len(tasks) * len(agent.harness) * attempts, batch=self.steps
        )
        trials = await run.engine.run(job, tasks, attempts, agent=agent)
        run.progress.done(job)
        rolled = await run.recorded(
            job, trials, batch=self.steps, modules=agent.modules, purpose=purpose
        )
        found = groups(trials, rolled, agent)
        for group in found:
            group.shaped = self.shape(group)
            group.advantages = self.advantage(group)
        return found

    def shape(self, group: Group) -> list[float | None]:
        """The rewards after the length rule; None on an unmeasured member."""
        preset = self.preset
        measured = group.measured
        found = iter(
            shaped(
                [float(one.verdict.reward or 0.0) for one in measured],
                [one.sampled_tokens for one in measured],
                penalty=preset.length_penalty,
                floor=preset.length_floor,
                cap=preset.length_cap,
            )
        )
        return [next(found) if one.measured else None for one in group.members]

    def advantage(self, group: Group) -> list[float | None]:
        """The overlong term against the endpoint's token budget, then the rewards centred
        on the group's mean, divided by the spread under `normalize`; None on an unmeasured
        member."""
        preset = self.preset
        kept = [
            (one, reward)
            for one, reward in zip(group.members, group.shaped, strict=True)
            if reward is not None
        ]
        if not kept:
            return [None] * len(group.members)
        terms = overlong(
            [one.sampled_tokens for one, _ in kept],
            penalty=preset.overlong_penalty,
            buffer=preset.overlong_buffer,
            budget=group.agent.policy.endpoint.budget,
        )
        scored = iter(
            advantages(
                [reward + term for (_, reward), term in zip(kept, terms, strict=True)],
                normalize=preset.normalize,
            )
        )
        return [next(scored) if reward is not None else None for reward in group.shaped]

    async def credit(self, agent: Agent, found: Values[Group]) -> Credit:
        """The datums: a group with no reward spread has none. Each sequence takes its
        member's advantage at every target, less `kl_coef` times the gap to the anchor,
        scaled by its group's weight under the preset's aggregation."""
        preset = self.preset
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
        mu = await self.mu(agent, sequences)
        reference_tokens = (
            0 if preset.reference == "sampler" else sum(len(one.tokens) - 1 for one in sequences)
        )
        gaps = await self.gaps(sequences, mu)
        anchor_tokens = 0 if gaps is None else sum(len(one.tokens) for one in sequences)
        sizes = [
            sum(len(one.targets) for member, _ in members for one in member.sequences)
            for members in kept
            if members
        ]
        scale = iter(weights(sizes, preset.aggregation))
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
                            a - preset.kl_coef * g
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

    async def mu(self, agent: Agent, sequences: Values[Sequence]) -> list[list[float]]:
        """The sampler's logprobs at every target, from the records or from one batched
        forward on the training client, as `[recipe] reference` says."""
        if self.preset.reference == "sampler":
            return [sampler_logprobs(one) for one in sequences]
        client = agent.policy.client
        if client is None:
            raise ValueError(
                '`[recipe] reference = "trainer"` recomputes mu on the training client, '
                "and this agent has none"
            )
        return await client.compute_logprobs(sequences)

    async def gaps(
        self, sequences: Values[Sequence], mu: Values[Values[float]]
    ) -> list[list[float]] | None:
        """mu less the anchor's logprobs at every target; None without a KL term."""
        kl_coef = self.preset.kl_coef
        if kl_coef <= 0:
            return None
        if self.anchor is None:
            raise ValueError(f"kl_coef = {kl_coef} needs the starting weights to anchor to")
        anchored = await anchor_logprobs(self.anchor, sequences)
        return [
            [a - b for a, b in zip(scored, base, strict=True)]
            for scored, base in zip(mu, anchored, strict=True)
        ]

    async def step(self, agent: Agent, found: Values[Group]) -> Update | None:
        """Credit the groups and take the gradient in `substeps` parts of whole groups, each
        part's forward and optimizer call submitted before the part before is read. None
        when the agent has no client or nothing was credited."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0).
        preset, run = self.preset, self.run
        client = agent.policy.client
        if client is None:
            return None
        began = time.monotonic()
        party = agent.policy.endpoint.party or run.config.model.provider
        credit = await self.credit(agent, found)
        run.spent(
            party, reference_tokens=credit.reference_tokens, anchor_tokens=credit.anchor_tokens
        )
        parts = [one for one in credit.datums if one]
        if not parts:
            return None
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        count = min(preset.substeps, len(parts))
        pieces = [[one for group in piece for one in group] for piece in split(parts, count)]
        datums = [one for piece in pieces for one in piece]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        rate = (
            preset.learning_rate
            if client.restored
            else preset.adam.rate(preset.learning_rate, self.updates)
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
            loss_fn=preset.loss_fn,
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
        preset = self.preset
        forward = await client.forward_backward(part, preset.loss_fn, preset.loss_config)
        return forward, await client.optim_step(preset.adam, rate)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> dict[str, Any]:
        """The step's row in `ROW` order, naming its jobs, with `more` after it."""
        preset = self.preset
        measured = [one for group in found for one in group.measured]
        spreads = [
            pstdev(rewards) if len(rewards) > 1 else 0.0
            for group in found
            if (rewards := [one for one in group.shaped if one is not None])
        ]
        credited = sum(len(group.measured) for group in found if group.reward_variance > 0)
        budget = found[0].agent.policy.endpoint.budget if found else None
        docking = preset.overlong_penalty > 0 and bool(budget)
        docked = sum(
            term < 0
            for group in found
            for term in overlong(
                [one.sampled_tokens for one in group.measured],
                penalty=preset.overlong_penalty,
                buffer=preset.overlong_buffer,
                budget=budget,
            )
        )
        sequences = update.sequences if update is not None else 0
        jobs = list(dict.fromkeys(one.trial.job for group in found for one in group.members))
        row: dict[str, Any] = {
            "step": self.steps,
            "trained": update is not None,
            "jobs": jobs,
            "groups": len(found),
            "rollouts": sum(len(group.members) for group in found),
            "graded": len(measured),
            "masked": sum(len(group.members) for group in found) - len(measured),
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
            "overlong": docked if docking else None,
        }
        if update is not None:
            row.update({key: getattr(update, key) for key in ROW if hasattr(update, key)})
        kept = {key: value for key in ROW if (value := row.get(key)) is not None}
        self.run.log(**kept, **more)
        return kept

    async def checkpoint(self, agent: Agent, tag: str, *, keep: bool = False) -> Checkpoint | None:
        """Save the state and the sampler weights under `tag` with `[checkpoints] ttl_hours`,
        or with none when `keep`, and append the agent written down to the record. None
        for an agent with no client."""
        client = agent.policy.client
        if client is None:
            return None
        hours = None if keep else self.run.config.checkpoints.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        state = await client.save_state(tag, ttl_seconds=ttl)
        sampler = await client.save_weights_for_sampler(tag, ttl_seconds=ttl)
        saved = Checkpoint(
            tag,
            state,
            sampler,
            hours,
            self.steps,
            self.updates,
            self.epoch,
            agent.modules.digest if agent.modules is not None else None,
            tuple(one.name for one in agent.harness),
        )
        record.append(
            self.run.directory / record.CHECKPOINTS,
            {
                "tag": saved.tag,
                "state_path": saved.state_path,
                "sampler_path": saved.sampler_path,
                "ttl_hours": saved.ttl_hours,
                "step": saved.step,
                "updates": saved.updates,
                "epoch": saved.epoch,
                "modules": saved.modules,
                "harness": list(saved.harness),
            },
        )
        logger.info("checkpoint %s: %s", saved.tag, saved.sampler_path)
        return saved


class DAPOTrainer(Trainer):
    """DAPO's dynamic sampling as accumulation: jobs are held until as many groups have a
    reward spread as a batch yields, and the step takes them all at the weights they were
    sampled at. The epoch bounds it; the last step takes what it has."""

    async def sample(self, agent: Agent, tasks: Batches) -> AsyncIterator[list[Group]]:
        target = self.run.config.data.batch_size * len(agent.harness)
        held: list[Group] = []
        for batch in tasks:
            held += await self.job(agent, batch)
            if sum(group.reward_variance > 0 for group in held) >= target:
                yield held
                held = []
        if held:
            yield held


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """`parts` slices in order, with sizes differing by at most one. This is the cut of the
    cookbook's `split_list` (`np.linspace(0, len, parts + 1).astype(int)`), with its float
    arithmetic kept so a batch splits as it did under the cookbook."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


def observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """kl_v1 = mean(mu - pi), kl_v2 = half the mean square, entropy = mean(-mu) over the
    acted tokens (the cookbook's `compute_kl_sample_train`). Returns {} when a datum carries
    no mu or mask or the logprobs do not line up."""
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
