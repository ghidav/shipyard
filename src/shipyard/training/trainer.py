"""The procedure: `train` is one epoch, `job` the one place a policy is sampled, `step`
the gradient over the groups a step holds. The base is the plain policy gradient. A recipe
is a subclass that replaces the rules it changes, and the only values it reads are its own
`[recipe]` table."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable
from collections.abc import Sequence as Values
from dataclasses import dataclass, replace
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

import torch

from shipyard import record
from shipyard.credit import advantages, anchor_logprobs, overlong, sampler_logprobs, shaped, weights
from shipyard.pack import datum, groups
from shipyard.types import (
    Adam,
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
    from shipyard.run import Run

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600
#: Adam as the cookbook's `train_step` sets it (no weight decay, gradient clipping or
#: warm-up), for whatever a paper leaves unstated. The SDK's `eps` default is 1e-12.
COOKBOOK = Adam(beta1=0.9, beta2=0.95, eps=1e-8)

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


def clipped(low: float, high: float) -> dict[str, float]:
    """`loss_fn_config` for a ratio clipped to `[1 - low, 1 + high]`. Tinker's `ppo` and
    `cispo` take the bounds, while DAPO's `clip_low` / `clip_high` are epsilons."""
    # The route and the key names are the installed SDK's: tinker/types/
    # forward_backward_input.py:21 documents `loss_fn_config` as "Optional configuration
    # parameters for the loss function (e.g., PPO clip thresholds, DPO beta)", and
    # tinker/types/datum.py:32-33 names `clip_low_threshold` / `clip_high_threshold`.
    # That they are bounds is the losses page's `torch.clamp(prob_ratio,
    # clip_low_threshold, clip_high_threshold)` with the example `{"clip_low_threshold":
    # 0.9, "clip_high_threshold": 1.1}` (https://tinker-docs.thinkingmachines.ai/losses);
    # the installed SDK spells out neither that nor a default for `cispo`.
    return {"clip_low_threshold": round(1.0 - low, 6), "clip_high_threshold": round(1.0 + high, 6)}


def centred(rewards: Values[float | None], *, normalize: bool) -> list[float | None]:
    """`advantages` over the measured positions, None kept where it was."""
    found = [one for one in rewards if one is not None]
    scored = iter(advantages(found, normalize=normalize) if found else [])
    return [next(scored) if one is not None else None for one in rewards]


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
    """The plain policy gradient: the rewards as scored, the advantage is the reward less
    the group's mean, Tinker's importance-sampling loss summed over tokens, the cookbook's
    optimizer. `anchor` is a Tinker sampling client on the agent's starting weights, needed
    when `[recipe] kl_coef` is set."""

    #: Tinker's loss, with `bounds` as its config.
    loss_fn = "importance_sampling"
    #: The optimizer; `[recipe] warmup` is set on it at the step.
    ADAM = COOKBOOK

    def __init__(self, run: Run, *, anchor: Any = None) -> None:
        self.run = run
        self.recipe = run.config.recipe
        self.anchor = anchor
        self.steps = 0
        self.updates = 0
        self.epoch = 0

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        """One epoch: a step per yield of `sample`, the weights served after each update, a
        checkpoint every `[checkpoints] every` updates."""
        every = self.run.config.checkpoints.every
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
        """The rewards as the verifier scored them; None on an unmeasured member."""
        return [float(one.verdict.reward) if one.measured else None for one in group.members]

    def advantage(self, group: Group) -> list[float | None]:
        """The shaped rewards less the group's mean; None on an unmeasured member."""
        return centred(group.shaped, normalize=False)

    def weights(self, sizes: Values[int]) -> list[float]:
        """Each credited group's factor on its tokens' advantages, from the groups' target
        counts. Tinker sums token losses, and the base leaves that."""
        return [1.0] * len(sizes)

    def bounds(self) -> dict[str, float] | None:
        """`loss_fn_config` for the step; the base loss takes none."""
        return None

    async def credit(self, agent: Agent, found: Values[Group]) -> Credit:
        """The datums: a group with no reward spread has none. Each sequence takes its
        member's advantage at every target, less `kl_coef` times the gap to the anchor,
        scaled by its group's weight."""
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
        mu = await self.mu(agent, sequences)
        reference_tokens = (
            0 if recipe.reference == "sampler" else sum(len(one.tokens) - 1 for one in sequences)
        )
        gaps = await self.gaps(sequences, mu)
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

    async def mu(self, agent: Agent, sequences: Values[Sequence]) -> list[list[float]]:
        """The sampler's logprobs at every target, from the records or from one batched
        forward on the training client, as `[recipe] reference` says."""
        if self.recipe.reference == "sampler":
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
        kl_coef = self.recipe.kl_coef
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
        recipe, run = self.recipe, self.run
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
        count = min(recipe.substeps, len(parts))
        pieces = [[one for group in piece for one in group] for piece in split(parts, count)]
        datums = [one for piece in pieces for one in piece]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        adam = replace(self.ADAM, warmup=recipe.warmup)
        rate = (
            recipe.learning_rate
            if client.restored
            else adam.rate(recipe.learning_rate, self.updates)
        )
        trained: list[torch.Tensor] = []
        forward, optim = await self._submit(client, pieces[0], adam, rate)
        for index in range(count):
            following = (
                await self._submit(client, pieces[index + 1], adam, rate)
                if index + 1 < count
                else None
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
        self, client: TrainingClient, part: list[Datum], adam: Adam, rate: float
    ) -> tuple[Awaitable[Forward], Awaitable[None]]:
        forward = await client.forward_backward(part, self.loss_fn, self.bounds())
        return forward, await client.optim_step(adam, rate)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> dict[str, Any]:
        """The step's row in `ROW` order, naming its jobs, with `more` in it or after it."""
        measured = [one for group in found for one in group.measured]
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
        }
        if update is not None:
            row.update({key: getattr(update, key) for key in ROW if hasattr(update, key)})
        row.update(more)
        kept = {key: value for key in ROW if (value := row.get(key)) is not None}
        rest = {key: value for key, value in more.items() if key not in ROW}
        self.run.log(**kept, **rest)
        return {**kept, **rest}

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
    """DAPO (Yu et al., arXiv 2503.14476): the advantage divided by the group's spread, PPO
    with asymmetric clipping (`clip_low` / `clip_high`, 4.1, are epsilons; Tinker's `ppo`
    takes bounds), each prompt's token losses averaged (Eq. 8), the soft overlong
    punishment (Eq. 13), and dynamic sampling (Alg. 1) as accumulation: jobs are held until
    as many groups have a reward spread as a batch yields, and the step takes them all at
    the weights they were sampled at. The epoch bounds it; the last step takes what it
    has."""

    loss_fn = "ppo"
    #: 4.1 states no betas, eps, weight decay or gradient clipping. The authors' released
    #: recipe (verl recipe/dapo, run_dapo_qwen2.5_32b.sh) sets weight decay 0.1 and clips
    #: the gradient norm at 1.0. Betas and eps are the cookbook's.
    ADAM = replace(COOKBOOK, weight_decay=0.1, grad_clip_norm=1.0)

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        recipe = self.recipe
        if self.steps == 0 and recipe.overlong_penalty > 0 and agent.policy.endpoint.budget is None:
            logger.warning(
                "%s's overlong term is off for this run: the proxy reports no token budget, "
                "so overlong_penalty = %s docks no rollout",
                recipe.kind,
                recipe.overlong_penalty,
            )
        return await super().train(agent, tasks)

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

    def advantage(self, group: Group) -> list[float | None]:
        """The overlong term, then the rewards centred and divided by the spread."""
        terms = self.docked(group)
        return centred(
            [
                None if reward is None else reward + term
                for reward, term in zip(group.shaped, terms, strict=True)
            ],
            normalize=True,
        )

    def docked(self, group: Group) -> list[float | None]:
        """The overlong term per member, against the endpoint's token budget; None on an
        unmeasured one."""
        recipe = self.recipe
        measured = group.measured
        terms = iter(
            overlong(
                [one.sampled_tokens for one in measured],
                penalty=recipe.overlong_penalty,
                buffer=recipe.overlong_buffer,
                budget=group.agent.policy.endpoint.budget,
            )
        )
        return [next(terms) if one.measured else None for one in group.members]

    def weights(self, sizes: Values[int]) -> list[float]:
        """Each prompt weighs the same (Eq. 8): a group's tokens are scaled by the step's
        mean group size over its own."""
        return weights(sizes, "prompt")

    def bounds(self) -> dict[str, float] | None:
        return clipped(self.recipe.clip_low, self.recipe.clip_high)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> dict[str, Any]:
        """The row with `overlong`, the graded members the term docked, when the term is
        on and the budget known."""
        budget = found[0].agent.policy.endpoint.budget if found else None
        if self.recipe.overlong_penalty > 0 and budget:
            more["overlong"] = sum(
                term is not None and term < 0 for group in found for term in self.docked(group)
            )
        return super().log(found, update, **more)


class DrGRPOTrainer(Trainer):
    """Dr. GRPO (Liu et al., arXiv 2503.20783): the base advantage, with no division by the
    spread, PPO with one symmetric `clip` (0.2, App. G), token losses summed (Tinker's sum
    is the constant normalizer of 3.2), and AdamW as App. G sets it. The length rule among
    solved answers (`length_penalty`, `length_floor`, `length_cap`) is shipyard's addition
    and off by default; the paper has none."""

    loss_fn = "ppo"
    #: App. G (Table 6): betas 0.9 / 0.95, weight decay 0, the gradient norm clipped at
    #: 1.0, a constant learning rate. Eps is unstated, so the cookbook's applies.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=1.0)

    def shape(self, group: Group) -> list[float | None]:
        """The rewards after the length rule."""
        recipe = self.recipe
        measured = group.measured
        found = iter(
            shaped(
                [float(one.verdict.reward) for one in measured],
                [one.sampled_tokens for one in measured],
                penalty=recipe.length_penalty,
                floor=recipe.length_floor,
                cap=recipe.length_cap,
            )
        )
        return [next(found) if one.measured else None for one in group.members]

    def bounds(self) -> dict[str, float] | None:
        return clipped(self.recipe.clip, self.recipe.clip)


class CISPOTrainer(DAPOTrainer):
    """CISPO (MiniMax-M1, arXiv 2506.13585): DAPO's rules under Tinker's `cispo` loss, whose
    truncated importance weight is detached from the gradient, and AdamW at betas 0.9 /
    0.95 and eps 1e-15 (3.2). `clip_high` (3.0) is an epsilon on the weight's ceiling, with
    no lower bound."""

    loss_fn = "cispo"
    #: 3.2 sets the betas and eps for gradients mostly below 1e-14. It states no weight
    #: decay, gradient clipping or warm-up, so the cookbook's apply (none).
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-15)

    def bounds(self) -> dict[str, float] | None:
        # No lower bound: MiniMax-M1 sets epsilon_low large and tunes only epsilon_high,
        # without publishing it. The ceiling of 4.0 is ScaleRL's (A.17.2) and FST's
        # (appendix D).
        high = round(1.0 + float(self.recipe.clip_high), 6)
        return {"clip_low_threshold": 0.0, "clip_high_threshold": high}


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
