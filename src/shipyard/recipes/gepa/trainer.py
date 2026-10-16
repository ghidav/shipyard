"""GEPA (Agrawal et al., arXiv 2507.19457): a search over the modules, no gradient."""

from __future__ import annotations

import random
from collections.abc import Sequence as Values
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import record
from shipyard.data import search_sets
from shipyard.modules import Candidate, keep, seed
from shipyard.recipes.gepa.cycle import Evolution, Log, evolve
from shipyard.recipes.gepa.fitness import Outcome
from shipyard.recipes.gepa.outcomes import outcomes
from shipyard.recipes.gepa.reflect import reflect
from shipyard.training.trainer import Trainer
from shipyard.types import Agent, Batches

if TYPE_CHECKING:
    from shipyard.run import Run

#: Directory names under `runs/<id>/modules/` for the starting text and the winner.
SEED = "seed"
BEST = "best"
#: The budget when the blueprint names none, as a multiple of one pass over every task the
#: search may score.
DEFAULT_PASSES = 2


class GEPATrainer(Trainer):
    """The search over the modules: candidates scored through `job`, rewritten by a reflector."""

    #: Rollouts per candidate per task when the search scores it.
    score_group_size: int

    def __init__(self, run: Run) -> None:
        super().__init__(run)
        search = run.config.recipe
        self.score_group_size = run.config.data.group_size
        self.reflect = reflect(
            run,
            harness=search.reflection_harness,
            model=search.reflection_model,
            image=search.reflection_image,
            incremental=search.edits == "incremental",
        )

    async def evolve(
        self,
        agent: Agent,
        tasks: Values[Path],
        seed: Values[Candidate],
        *,
        pareto: Values[Path] | None = None,
        budget: int,
        minibatch: int,
        patience: int | None,
        rng: random.Random,
        log: Log | None = None,
    ) -> Evolution:
        """GEPA's rounds over `tasks`, each candidate scored on `pareto` when given."""
        return await evolve(
            seed,
            tasks,
            pareto=pareto,
            write=self.reflect,
            score=partial(self.score, agent),
            minibatch=minibatch,
            budget=budget,
            patience=patience,
            rollouts=self.score_group_size,
            rng=rng,
            log=self.run.log if log is None else log,
        )

    async def score(
        self, agent: Agent, candidate: Candidate, tasks: Values[Path], round: int
    ) -> list[Outcome]:
        """One job of `candidate` over `tasks`, `score_group_size` attempts each."""
        found = await self.job(
            replace(agent, modules=candidate),
            list(tasks),
            self.score_group_size,
            purpose="search",
            round=round,
        )
        return outcomes(found)

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        """One search over the pass's tasks, from the modules the agent carries; a first pass
        starts from `[recipe] modules`, kept under `seed`."""
        recipe, data = self.recipe, self.run.config.data
        kept = self.run.directory / record.MODULES
        start = agent.modules
        if start is None:
            start = seed(self.run.config.home / recipe.modules)
            keep(start, kept / SEED)
        feedback, pareto = search_sets(
            (task for batch in tasks for task in batch), recipe.pareto, seed=data.seed
        )
        budget = recipe.budget or DEFAULT_PASSES * len({*feedback, *pareto}) * data.group_size
        result = await self.evolve(
            agent,
            feedback,
            [start],
            pareto=pareto,
            budget=budget,
            minibatch=recipe.minibatch,
            patience=recipe.patience,
            rng=random.Random(data.seed + self.epoch),
        )
        keep(result.best, kept / BEST)
        self.run.log(
            evolution=True,
            epoch=self.epoch,
            rounds=result.rounds,
            spent=result.spent,
            best=result.best.digest,
            best_mean=result.fitness.aggregate(result.best.digest),
            frontier=len(result.fitness.frontier()),
            pool=len(result.pool),
            moved=result.best.digest != start.digest,
        )
        return replace(agent, modules=result.best)
