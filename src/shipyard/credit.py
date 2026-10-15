"""The arithmetic of credit: shaping, the advantage, the weights, the logprobs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence as Values
from statistics import fmean, pstdev
from typing import Any

from shipyard.pack import Sequence, model_input

#: Harbor's reward for a fully solved task. Only solved rollouts are compared on length.
SOLVED = 1.0
#: Anchor passes in flight at once.
LOGPROB_CONCURRENCY = 16

#: The normaliser's floor against a spread of zero.
EPSILON = 1e-6


def shaped(
    rewards: Values[float], lengths: Values[int], *, penalty: float, floor: int, cap: float
) -> list[float]:
    """The rewards less the length penalty. With the penalty on and at least two rollouts
    solved, each solved rollout is docked `penalty * max(L - floor, 0) / mean solved L`, at
    most `cap`. Otherwise the rewards are returned unchanged."""
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
        docked = min(penalty * excess / mean, cap)
        out.append(float(reward) - docked if yes else float(reward))
    return out


def overlong(
    lengths: Values[int], *, penalty: float, buffer: float, budget: int | None
) -> list[float]:
    """DAPO's soft overlong punishment (Eq. 13) per rollout, with the sampled tokens as the
    response length and the trial's token budget as L_max. The term is 0 up to the last
    `buffer` of the budget, falls linearly to `-penalty` at the budget, and stays there.
    All zeros when the term is off or the budget is unknown."""
    if not penalty or not budget:
        return [0.0] * len(lengths)
    cache = buffer * budget
    start = budget - cache
    return [-penalty * min(max(float(length) - start, 0.0) / cache, 1.0) for length in lengths]


def weights(tokens: Values[int]) -> list[float]:
    """Each group's factor on its tokens' advantages, so each prompt weighs the same: the
    step's mean group size over the group's own, since Tinker sums token losses."""
    sized = [one for one in tokens if one > 0]
    if not sized:
        return [1.0] * len(tokens)
    mean = fmean(sized)
    return [mean / one if one > 0 else 1.0 for one in tokens]


def advantages(shaped_rewards: Values[float], *, normalize: bool) -> list[float]:
    """The rewards centred on the group's mean, divided by the spread under `normalize`."""
    mean, spread = fmean(shaped_rewards), pstdev(shaped_rewards)
    scale = (spread + EPSILON) if normalize else 1.0
    return [(one - mean) / scale for one in shaped_rewards]


def sampler_logprobs(sequence: Sequence) -> list[float]:
    """mu as the sampler reported it. Raises when a record carries none, since a short list
    would misalign every ratio after it."""
    if sequence.mu is None or len(sequence.mu) != len(sequence.targets):
        raise ValueError(
            '`[recipe] reference = "sampler"` reads the records\' logprobs, and a record '
            "of this batch carries none for its completion"
        )
    return list(sequence.mu)


async def anchor_logprobs(anchor: Any, sequences: Values[Sequence]) -> list[list[float]]:
    """The starting weights' logprobs at every target, from one `compute_logprobs_async`
    over the whole sequence. Position `p` scores `tokens[p]` (the cookbook's
    `incorporate_kl_penalty` alignment, `[1:]` against the targets)."""
    gate = asyncio.Semaphore(LOGPROB_CONCURRENCY)

    async def one(sequence: Sequence) -> list[float]:
        async with gate:
            scored = await anchor.compute_logprobs_async(
                model_input(sequence.tokens, sequence.images)
            )
        scored = list(scored)
        return [_number(scored[at]) for at in sequence.targets]

    return list(await asyncio.gather(*(one(sequence) for sequence in sequences)))


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0
