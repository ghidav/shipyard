"""Credit: which rollouts carry a gradient and how much, from the verdicts and the packed
sequences; the reference logprobs the ratio is formed against; the KL anchor; the datums
the step consumes, and what the batch had to leave out to get them."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence as Values
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

import tinker
import torch
from tinker import TensorData

from shipyard.pack import Group, Member, Sequence, datum, shifted

if TYPE_CHECKING:
    from shipyard.recipes.train import Preset
    from shipyard.trainer import Trainer

#: Guards the division when a group has no reward spread at all.
EPSILON = 1e-6
#: Harbor's reward for a task fully solved: only solved rollouts are compared on length.
SOLVED = 1.0
#: The most the length penalty takes from a solved rollout, so that the longest solved
#: answer still scores above every failure.
LENGTH_CAP = 0.5
#: Anchor passes in flight at once.
LOGPROB_CONCURRENCY = 16


@dataclass(frozen=True, kw_only=True)
class Batch:
    """What one step consumes and what it left out; a measure that had nothing to
    measure is None, never zero. `credited` counts the members carrying a gradient."""

    datums: tuple[tinker.Datum, ...]
    groups: int
    rollouts: int
    graded: int
    masked: int
    degenerate: int
    credited: int
    sequences: int
    reward_mean: float | None
    reward_spread: float | None
    length_mean: float | None
    anchor_kl: float | None
    reference_tokens: int
    anchor_tokens: int

    @property
    def empty(self) -> bool:
        return not self.datums

    @property
    def sequences_per_rollout(self) -> float | None:
        """The mean over the credited members; None when none was."""
        return self.sequences / self.credited if self.credited else None


def shaped(
    rewards: Values[float], lengths: Values[int], *, penalty: float, floor: int
) -> list[float]:
    """The rewards less the length penalty on the solved rollouts: among solved answers,
    at least two solved, docked `penalty * max(L - floor, 0) / mean solved L`, capped;
    the rewards themselves when the penalty is off or fewer than two solved."""
    if not penalty:
        return [float(one) for one in rewards]
    solved = [one >= SOLVED for one in rewards]
    if sum(solved) < 2:
        return [float(one) for one in rewards]
    mean = fmean(length for length, yes in zip(lengths, solved, strict=True) if yes)
    if mean <= 0.0:
        return [float(one) for one in rewards]
    out: list[float] = []
    for reward, length, yes in zip(rewards, lengths, solved, strict=True):
        excess = max(float(length) - float(floor), 0.0)
        docked = min(penalty * excess / mean, LENGTH_CAP)
        out.append(float(reward) - docked if yes else float(reward))
    return out


def measured(group: Group) -> list[Member]:
    """The members with a reward and no mask: the only ones a baseline is made of."""
    return [
        one for one in group.members if one.verdict.mask is None and one.verdict.reward is not None
    ]


def advantages(shaped_rewards: Values[float], *, normalize: bool) -> list[float]:
    """Centred on the group's mean; divided by the spread under `normalize`."""
    mean, spread = fmean(shaped_rewards), pstdev(shaped_rewards)
    scale = (spread + EPSILON) if normalize else 1.0
    return [(one - mean) / scale for one in shaped_rewards]


def pinned(groups: Values[Group], trainer: Trainer | None) -> None:
    """Refuse groups sampled before the trainer's last update: mu must come from the
    weights the rollouts were sampled at."""
    now = None if trainer is None else trainer.updates
    for group in groups:
        if group.updates is not None and now is not None and now != group.updates:
            raise ValueError(
                f"this run applied {now - group.updates} update(s) between sampling task "
                f"{group.task.name} and crediting it, so the trainer no longer holds the "
                "weights those rollouts came from; credit a batch before training on it"
            )


async def reference_logprobs(
    trainer: Trainer | None, sequences: Values[Sequence]
) -> list[list[float]]:
    """mu at every target, from one batched forward pass on the training client, one
    `shifted` datum per sequence: position `j` of the output scores `tokens[j + 1]`, so a
    target at `p` reads `scored[p - 1]`. An unscored position is 0.0."""
    if not sequences:
        return []
    if trainer is None:
        raise ValueError(
            '`[recipe] reference = "trainer"` recomputes mu on the training client, '
            "and none was given"
        )
    data: list[tinker.Datum] = []
    for one in sequences:
        model_input, target_tokens = shifted(one.tokens)
        weights = TensorData.from_torch(torch.ones(model_input.length))
        inputs = {"target_tokens": target_tokens, "weights": weights}
        data.append(tinker.Datum(model_input=model_input, loss_fn_inputs=inputs))
    future = await trainer.client.forward_async(data, loss_fn="cross_entropy")
    result = await future.result_async()
    out: list[list[float]] = []
    for one, row in zip(sequences, result.loss_fn_outputs, strict=True):
        scored = row["logprobs"]
        scored = scored.tolist() if hasattr(scored, "tolist") else list(scored)
        out.append([_number(scored[at - 1]) if at > 0 else 0.0 for at in one.targets])
    return out


def sampler_logprobs(sequence: Sequence) -> list[float]:
    """mu as the sampler reported it, refused when a record carried none: a short list
    would misalign every ratio after it."""
    if sequence.mu is None or len(sequence.mu) != len(sequence.targets):
        raise ValueError(
            '`[recipe] reference = "sampler"` reads the records\' logprobs, and a record '
            "of this batch carries none for its completion"
        )
    return list(sequence.mu)


async def anchor_logprobs(anchor: Any, sequences: Values[Sequence]) -> list[list[float]]:
    """The starting weights' logprobs at every target: one `compute_logprobs_async` over
    the whole sequence, where position `p` scores `tokens[p]` (the cookbook's
    `incorporate_kl_penalty` alignment, `[1:]` against the targets)."""
    gate = asyncio.Semaphore(LOGPROB_CONCURRENCY)

    async def one(sequence: Sequence) -> list[float]:
        async with gate:
            scored = await anchor.compute_logprobs_async(
                tinker.ModelInput.from_ints(list(sequence.tokens))
            )
        scored = list(scored)
        return [_number(scored[at]) for at in sequence.targets]

    return list(await asyncio.gather(*(one(sequence) for sequence in sequences)))


async def credit(
    groups: Values[Group],
    preset: Preset,
    trainer: Trainer | None = None,
    anchor: Any = None,  # a `tinker.SamplingClient` on the starting weights
) -> Batch:
    """The batch: measured members only; degenerate groups dropped before any reference
    pass; mu from the trainer's forward or the records; the KL to the anchor folded into
    the advantage per token when `kl_coef > 0`."""
    if preset.reference == "trainer":
        pinned(groups, trainer)
    rollouts = graded = dropped = 0
    rewards: list[float] = []
    spreads: list[float] = []
    lengths: list[int] = []
    kept: list[tuple[Member, float]] = []
    for group in groups:
        members = measured(group)
        rollouts += len(group.members)
        graded += len(members)
        if not members:
            continue
        found = [float(one.verdict.reward) for one in members]
        wrote = [one.sampled_tokens for one in members]
        shaped_rewards = shaped(
            found, wrote, penalty=preset.length_penalty, floor=preset.length_floor
        )
        spread = pstdev(shaped_rewards) if len(shaped_rewards) > 1 else 0.0
        rewards.extend(found)
        lengths.extend(wrote)
        spreads.append(spread)
        if len(shaped_rewards) < 2 or spread <= EPSILON:
            dropped += 1  # degenerate: a lone member or a flat group carries no gradient
            continue
        kept.extend(
            zip(members, advantages(shaped_rewards, normalize=preset.normalize), strict=True)
        )
    sequences = [one for member, _ in kept for one in member.sequences]
    # The token counters bill what is sent: the shifted input to the trainer's forward,
    # the whole sequence to the anchor.
    if preset.reference == "sampler":
        mu, reference_tokens = [sampler_logprobs(one) for one in sequences], 0
    else:
        mu = await reference_logprobs(trainer, sequences)
        reference_tokens = sum(len(one.tokens) - 1 for one in sequences)
    anchored: list[list[float]] | None = None
    anchor_tokens = 0
    if preset.kl_coef > 0 and sequences:
        if anchor is None:
            raise ValueError(f"kl_coef = {preset.kl_coef} needs the starting weights to anchor to")
        anchored = await anchor_logprobs(anchor, sequences)
        anchor_tokens = sum(len(one.tokens) for one in sequences)
    datums: list[tinker.Datum] = []
    gaps: list[float] = []
    cursor = 0
    for member, advantage in kept:
        for sequence in member.sequences:
            per_token = [advantage] * len(sequence.targets)
            if anchored is not None:
                gap = [a - b for a, b in zip(mu[cursor], anchored[cursor], strict=True)]
                per_token = [a - preset.kl_coef * g for a, g in zip(per_token, gap, strict=True)]
                gaps.extend(gap)
            datums.append(datum(sequence, logprobs=mu[cursor], advantages=per_token))
            cursor += 1
    return Batch(
        datums=tuple(datums),
        groups=len(groups),
        rollouts=rollouts,
        graded=graded,
        masked=rollouts - graded,
        degenerate=dropped,
        credited=len(kept),
        sequences=len(sequences),
        reward_mean=fmean(rewards) if rewards else None,
        reward_spread=fmean(spreads) if spreads else None,
        length_mean=fmean(lengths) if lengths else None,
        anchor_kl=fmean(gaps) if gaps else None,
        reference_tokens=reference_tokens,
        anchor_tokens=anchor_tokens,
    )


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0
