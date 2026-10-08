"""The evaluate recipe: a measurement over the dataset, no gradient and no search."""

from __future__ import annotations

from statistics import fmean
from typing import TYPE_CHECKING

from shipyard.admit import scores
from shipyard.data import batches

if TYPE_CHECKING:
    from shipyard.run import Run


async def run(run: Run) -> None:
    """Run one job per batch and log the mean of the graded trials per batch and once over
    the run. A masked trial is left out of the mean."""
    cfg = run.config
    data = cfg.data
    rewards: list[float] = []
    count = rollouts = masked = 0
    planned = batches(cfg.datasets, size=data.batch_size, seed=data.seed, epochs=data.epochs)
    for index, batch in enumerate(planned):
        rolled = await run.sample(batch, rollouts=data.group_size, index=index)
        graded = scores(rolled)
        run.log(
            batch=index,
            tasks=len(batch),
            rollouts=len(rolled),
            graded=len(graded),
            masked=len(rolled) - len(graded),
            mean=mean(graded),
        )
        rewards.extend(graded)
        count += 1
        rollouts += len(rolled)
        masked += len(rolled) - len(graded)
    run.log(
        evaluation=True,
        batches=count,
        rollouts=rollouts,
        graded=len(rewards),
        masked=masked,
        mean=mean(rewards),
    )


def mean(values: list[float]) -> float | None:
    """The mean, or None for no values, since a 0 would read as a policy that failed."""
    return fmean(values) if values else None
