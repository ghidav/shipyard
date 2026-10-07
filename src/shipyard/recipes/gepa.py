"""The gepa recipe: the search over the text the harness reads, no gradient and no
checkpoint. The seed is the blueprint's `modules/`, kept under `runs/<id>/modules/seed/`;
the winner lands under `modules/best/` in the same layout, so it is the next seed."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import record
from shipyard.data import tasks as tasks_of
from shipyard.gepa.cycle import Evolution, evolve
from shipyard.gepa.outcomes import outcomes
from shipyard.gepa.reflect import reflect
from shipyard.modules import Candidate, keep, seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shipyard.gepa.fitness import Outcome
    from shipyard.run import Run

#: Under `runs/<id>/modules/`: the text the search started from, and the text it ends on.
SEED = "seed"
BEST = "best"
#: The budget when the blueprint names none, as a multiple of one pass over the tasks.
DEFAULT_PASSES = 2


async def run(run: Run) -> Evolution:
    """Seed, then evolve: every candidate scored by one job over the tasks it is asked
    about, every rewrite a reflection trial, one row per round, the winner kept, and a
    final row naming it. A served policy answers through the proxy as `evaluate`'s does."""
    cfg = run.config
    recipe, data = cfg.recipe, cfg.data
    seeded = seed(cfg.home / recipe.modules)
    keep(seeded, run.directory / record.MODULES / SEED)
    listed = [task for name in cfg.datasets for task in tasks_of(name)]
    budget = recipe.budget or DEFAULT_PASSES * len(listed) * data.group_size

    async def score(candidate: Candidate, over: Sequence[Path], round_index: int) -> list[Outcome]:
        """One job over `over`, its batch index the round it serves, read back."""
        rolled = await run.sample(
            list(over), rollouts=data.group_size, index=round_index, modules=candidate
        )
        return outcomes(rolled, over, data.group_size)

    write = reflect(
        run,
        harness=recipe.reflection_harness,
        model=recipe.reflection_model,
        image=recipe.reflection_image,
        incremental=recipe.edits == "incremental",
    )
    result = await evolve(
        seeded,
        listed,
        write=write,
        score=score,
        minibatch=recipe.minibatch,
        budget=budget,
        patience=recipe.patience,
        rollouts=data.group_size,
        log=run.log,
    )
    keep(result.best, run.directory / record.MODULES / BEST)
    run.log(
        evolution=True,
        rounds=result.rounds,
        spent=result.spent,
        best=result.best.digest,
        best_mean=result.fitness.aggregate(result.best.digest),
        frontier=len(result.fitness.frontier()),
        pool=len(result.pool),
        moved=result.best.digest != seeded.digest,
    )
    return result
