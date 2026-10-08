"""The gepa recipe: a search over the text the harness reads, with no gradient and no
checkpoint. The seed is the blueprint's `modules/`, kept under `runs/<id>/modules/seed/`.
The winner is kept under `modules/best/` in the same layout, so it can be the next seed."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import record
from shipyard.config import search_sets
from shipyard.gepa.cycle import Evolution, evolve
from shipyard.gepa.outcomes import outcomes
from shipyard.gepa.reflect import reflect
from shipyard.modules import Candidate, keep, seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shipyard.gepa.fitness import Outcome
    from shipyard.run import Run

#: Directory names under `runs/<id>/modules/` for the starting text and the winner.
SEED = "seed"
BEST = "best"
#: The budget when the blueprint names none, as a multiple of one pass over every task the
#: search may score.
DEFAULT_PASSES = 2


async def run(run: Run) -> Evolution:
    """Seed, then evolve. One job scores each candidate over the tasks it is asked about,
    and each rewrite is a reflection trial. The run logs one row per round and a final row
    naming the winner, which is kept. Minibatches come from the run's tasks less those
    `[recipe] pareto` holds out. The held-out tasks score what the search keeps.
    `[data] seed` draws both. A served policy answers through the proxy, as in `evaluate`."""
    cfg = run.config
    recipe, data = cfg.recipe, cfg.data
    seeded = seed(cfg.home / recipe.modules)
    keep(seeded, run.directory / record.MODULES / SEED)
    feedback, pareto = search_sets(cfg)
    budget = recipe.budget or DEFAULT_PASSES * len({*feedback, *pareto}) * data.group_size

    async def score(candidate: Candidate, over: Sequence[Path], round_index: int) -> list[Outcome]:
        """Sample `candidate` over `over` in one job, with the round as its batch index, and
        return the outcomes."""
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
        feedback,
        pareto=pareto,
        write=write,
        score=score,
        minibatch=recipe.minibatch,
        budget=budget,
        patience=recipe.patience,
        rollouts=data.group_size,
        rng=random.Random(data.seed),
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
