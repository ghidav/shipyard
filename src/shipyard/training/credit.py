"""Credit assignment: reward shaping, advantages, group weights, logprobs to score against,
and what a step's credit holds."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence as Values
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any

from shipyard.proxy.wire import model_input
from shipyard.types import Datum, Sequence

#: Harbor's reward for a fully solved task. The length penalty compares solved rollouts only.
SOLVED = 1.0
#: Maximum number of concurrent logprob requests to one sampling client.
LOGPROB_CONCURRENCY = 16

#: Added to the standard deviation when normalizing, to avoid dividing by zero.
EPSILON = 1e-6

#: Per-token values one training sequence is scored with, by name, aligned to its targets.
Signals = dict[str, list[float]]


def shaped(
    rewards: Values[float], lengths: Values[int], *, penalty: float, floor: int, cap: float
) -> list[float]:
    """Return the rewards minus a length penalty.

    When `penalty` is nonzero and at least two rollouts are solved, each solved rollout loses
    `penalty * max(L - floor, 0) / mean solved L`, capped at `cap`. Otherwise the rewards are
    returned unchanged."""
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
    """Return DAPO's soft overlong punishment (Eq. 13) for each rollout.

    The response length is the number of sampled tokens, and L_max is the trial's token
    budget. The term is 0 up to the last `buffer` fraction of the budget, falls linearly to
    `-penalty` at the budget, and stays there. It is all zeros when `penalty` is 0 or the
    budget is unknown."""
    if not penalty or not budget:
        return [0.0] * len(lengths)
    cache = buffer * budget
    start = budget - cache
    return [-penalty * min(max(float(length) - start, 0.0) / cache, 1.0) for length in lengths]


def weights(tokens: Values[int]) -> list[float]:
    """Return each group's factor on its tokens' advantages, so every prompt weighs the same.

    `tokens` holds each group's target-token count. Because Tinker sums token losses, the
    factor is the step's mean count per group divided by the group's own. A group with no
    tokens gets 1.0."""
    sized = [one for one in tokens if one > 0]
    if not sized:
        return [1.0] * len(tokens)
    mean = fmean(sized)
    return [mean / one if one > 0 else 1.0 for one in tokens]


def advantages(shaped_rewards: Values[float], *, normalize: bool) -> list[float]:
    """Return the rewards minus their mean, divided by their standard deviation if `normalize`."""
    mean, spread = fmean(shaped_rewards), pstdev(shaped_rewards)
    scale = (spread + EPSILON) if normalize else 1.0
    return [(one - mean) / scale for one in shaped_rewards]


def centred(rewards: Values[float | None], *, normalize: bool) -> list[float | None]:
    """Apply `advantages` to the measured rewards, leaving None entries in place."""
    found = [one for one in rewards if one is not None]
    scored = iter(advantages(found, normalize=normalize) if found else [])
    return [next(scored) if one is not None else None for one in rewards]


def sampler_logprobs(sequence: Sequence) -> list[float]:
    """Return the sampler's logprobs at the sequence's targets, as the proxy recorded them.

    Raise `ValueError` when they are missing or incomplete, since a short list would
    misalign every ratio after it."""
    recorded = sequence.sampled_logprobs
    if recorded is None or len(recorded) != len(sequence.targets):
        raise ValueError(
            '`[recipe] reference = "sampler"` reads the records\' logprobs, and a record '
            "of this batch carries none for its completion"
        )
    return list(recorded)


async def scored_logprobs(sampler: Any, sequences: Values[Sequence]) -> list[list[float]]:
    """Return a sampling client's logprobs at every target of each sequence, such as the
    starting weights' or a teacher's.

    Each sequence is scored with one `compute_logprobs_async` call, at most
    `LOGPROB_CONCURRENCY` at once. Position `p` scores `tokens[p]` (the cookbook's
    `incorporate_kl_penalty` alignment, `[1:]` against the targets)."""
    gate = asyncio.Semaphore(LOGPROB_CONCURRENCY)

    async def one(sequence: Sequence) -> list[float]:
        async with gate:
            scored = await sampler.compute_logprobs_async(
                model_input(sequence.tokens, sequence.images)
            )
        scored = list(scored)
        return [_number(scored[at]) for at in sequence.targets]

    return list(await asyncio.gather(*(one(sequence) for sequence in sequences)))


def _number(value: Any) -> float:
    return float(value) if value is not None else 0.0


def scored_tokens(sequences: Values[Sequence], scored: Values[Signals], name: str) -> int:
    """Return the tokens sent to score the sequences that carry signal `name`."""
    return sum(len(one.tokens) for one, row in zip(sequences, scored, strict=True) if name in row)


def mean_gap(scored: Values[Signals], name: str) -> float | None:
    """Return the mean of `reference` minus signal `name` over every target that carries it,
    or None when none does."""
    gaps = [
        r - x
        for row in scored
        if name in row
        for r, x in zip(row["reference"], row[name], strict=True)
    ]
    return fmean(gaps) if gaps else None


@dataclass(frozen=True)
class Credit:
    """The datums a step trains on, by group, and the tokens spent computing them.

    A skipped group has an empty list. The token counts are those of the reference and anchor
    logprob passes; the teacher's are billed as it scores. `anchor_kl` and `teacher_kl` are
    the mean gaps from the reference to each, or None without one. `taught` and `untaught`
    are as in `Update`."""

    datums: list[list[Datum]]
    reference_tokens: int = 0
    anchor_tokens: int = 0
    anchor_kl: float | None = None
    teacher_kl: float | None = None
    taught: int | None = None
    untaught: int | None = None

    @property
    def sequences(self) -> int:
        return sum(len(one) for one in self.datums)
