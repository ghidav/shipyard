"""Credit: which rollouts carry a gradient and how much, from the verdicts and the packed
sequences; the reference logprobs the ratio is formed against; the KL anchor; each prompt's
weight in the loss; the datums the step consumes, and what the batch had to leave out."""

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
    measure is None, never zero. `credited` counts the members carrying a gradient;
    `owners` holds, per datum, the index of the group it came from; `surplus` counts the
    groups past a full batch, None when the batch had no size to fill."""

    datums: tuple[tinker.Datum, ...]
    owners: tuple[int, ...]
    groups: int
    rollouts: int
    graded: int
    masked: int
    degenerate: int
    surplus: int | None = None
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


def scored(group: Group, preset: Preset) -> tuple[list[Member], list[float], list[float]]:
    """The measured members, their rewards, and the rewards as the preset shapes them."""
    members = measured(group)
    found = [float(one.verdict.reward) for one in members]
    wrote = [one.sampled_tokens for one in members]
    penalty, floor = preset.length_penalty, preset.length_floor
    return members, found, shaped(found, wrote, penalty=penalty, floor=floor)


def flat(shaped_rewards: Values[float]) -> bool:
    """Degenerate: a lone member, or rewards all equal; nothing to compare, no gradient."""
    return len(shaped_rewards) < 2 or pstdev(shaped_rewards) <= EPSILON


def carrying(groups: Values[Group], preset: Preset) -> int:
    """How many of the groups carry a gradient, read off the verdicts alone."""
    return sum(not flat(scored(group, preset)[2]) for group in groups)


def weights(tokens: Values[int], aggregation: str) -> list[float]:
    """Each group's factor on its tokens' advantages. Tinker sums token losses; under
    "prompt" a group's tokens are scaled by the step's mean group size over its own, so
    each prompt weighs the same and the sum keeps about its size. "sum" leaves them be."""
    if aggregation not in ("prompt", "sum"):
        raise ValueError(f"no token aggregation called {aggregation!r}; 'prompt' or 'sum'")
    sized = [one for one in tokens if one > 0]
    if aggregation == "sum" or not sized:
        return [1.0] * len(tokens)
    mean = fmean(sized)
    return [mean / one if one > 0 else 1.0 for one in tokens]


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
    *,
    limit: int | None = None,
) -> Batch:
    """The batch: measured members only; degenerate groups dropped before any reference
    pass, and with `limit` the groups carrying a gradient past the first `limit` left out
    as surplus; mu from the trainer's forward or the records; the KL to the anchor folded
    into the advantage per token when `kl_coef > 0`; each group's tokens weighted by the
    preset's aggregation."""
    if preset.reference == "trainer":
        pinned(groups, trainer)
    rollouts = graded = dropped = surplus = 0
    rewards: list[float] = []
    spreads: list[float] = []
    lengths: list[int] = []
    kept: list[tuple[int, Member, float]] = []
    owned: list[int] = []
    for group in groups:
        members, found, shaped_rewards = scored(group, preset)
        rollouts += len(group.members)
        graded += len(members)
        if not members:
            continue
        rewards.extend(found)
        lengths.extend(one.sampled_tokens for one in members)
        spreads.append(pstdev(shaped_rewards) if len(shaped_rewards) > 1 else 0.0)
        if flat(shaped_rewards):
            dropped += 1
            continue
        if limit is not None and len(owned) >= limit:
            surplus += 1  # the batch is full: DAPO keeps a constant number of prompts
            continue
        owner = len(owned)
        owned.append(owner)
        kept.extend(
            (owner, member, advantage)
            for member, advantage in zip(
                members, advantages(shaped_rewards, normalize=preset.normalize), strict=True
            )
        )
    sequences = [one for _, member, _ in kept for one in member.sequences]
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
    sizes = [0] * len(owned)
    for owner, member, _ in kept:
        sizes[owner] += sum(len(one.targets) for one in member.sequences)
    # The KL term is scaled with the advantage it is folded into, so kl_coef weighs it
    # against the reward the same way in every prompt, as a loss term aggregated alike.
    scale = weights(sizes, preset.aggregation)
    datums: list[tinker.Datum] = []
    owners: list[int] = []
    gaps: list[float] = []
    cursor = 0
    for owner, member, advantage in kept:
        for sequence in member.sequences:
            per_token = [advantage] * len(sequence.targets)
            if anchored is not None:
                gap = [a - b for a, b in zip(mu[cursor], anchored[cursor], strict=True)]
                per_token = [a - preset.kl_coef * g for a, g in zip(per_token, gap, strict=True)]
                gaps.extend(gap)
            per_token = [scale[owner] * one for one in per_token]
            datums.append(datum(sequence, logprobs=mu[cursor], advantages=per_token))
            owners.append(owner)
            cursor += 1
    return Batch(
        datums=tuple(datums),
        owners=tuple(owners),
        groups=len(groups),
        rollouts=rollouts,
        graded=graded,
        masked=rollouts - graded,
        degenerate=dropped,
        surplus=surplus if limit is not None else None,
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
