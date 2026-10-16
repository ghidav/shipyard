"""GEPA trainer (Agrawal et al., arXiv 2507.19457), which searches modules, not weights."""

from __future__ import annotations

import random
from collections.abc import Sequence as Values
from dataclasses import replace
from functools import partial
from pathlib import Path

from shipyard import record
from shipyard.config import Blueprint, modules_dir
from shipyard.data import search_sets
from shipyard.modules import Candidate, keep, seed
from shipyard.training.recipes.gepa.cycle import Evolution, Log, evolve
from shipyard.training.recipes.gepa.fitness import Outcome
from shipyard.training.recipes.gepa.outcomes import outcomes
from shipyard.training.recipes.gepa.reflect import reflect
from shipyard.training.trainer import Trainer
from shipyard.types import Agent

#: Directory names under `runs/<id>/modules/` for the seed candidate and the best one.
SEED = "seed"
BEST = "best"
#: Default search budget when the blueprint sets none, in passes over every task the search
#: may score.
DEFAULT_PASSES = 2


class GEPATrainer(Trainer):
    """Trainer for GEPA (Agrawal et al., arXiv 2507.19457): a search over the agent's modules.

    It takes no gradient steps. Candidates are scored through `job` and rewritten by a
    reflector."""

    #: Rollouts per task when the search scores a candidate.
    score_group_size: int

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        search = config.recipe
        self.score_group_size = config.data.group_size
        self.reflect = reflect(
            self,
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
        """Run the GEPA search from `seed` (see `cycle.evolve`), drawing minibatches from
        `tasks` and scoring candidates on `pareto` when given."""
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
            log=partial(self.logs.log, "search") if log is None else log,
        )

    async def score(
        self, agent: Agent, candidate: Candidate, tasks: Values[Path], round: int
    ) -> list[Outcome]:
        """Score `candidate` with one job over `tasks`, `score_group_size` attempts per task."""
        found = await self.job(
            replace(agent, modules=candidate),
            list(tasks),
            self.score_group_size,
            purpose="search",
            round=round,
        )
        return outcomes(found)

    async def train(self) -> Agent:
        """Run one search over the train tasks within the budget, then `finish`."""
        agent = self.agent()
        async with agent.policy:
            return await self.finish(await self.search(agent))

    async def search(self, agent: Agent) -> Agent:
        """Run GEPA over the train tasks, starting from `[agent] modules`, and return the agent
        carrying the best candidate. The seed and the best candidate are saved under
        `modules/seed/` and `modules/best/`."""
        recipe, data = self.recipe, self.config.data
        kept = self.directory / record.MODULES
        start = seed(modules_dir(self.config))
        keep(start, kept / SEED)
        feedback, pareto = search_sets(self.generator.train, recipe.pareto, seed=data.seed)
        budget = recipe.budget or DEFAULT_PASSES * len({*feedback, *pareto}) * data.group_size
        result = await self.evolve(
            agent,
            feedback,
            [start],
            pareto=pareto,
            budget=budget,
            minibatch=recipe.minibatch,
            patience=recipe.patience,
            rng=random.Random(data.seed),
        )
        keep(result.best, kept / BEST)
        self.logs.log(
            "search",
            evolution=True,
            rounds=result.rounds,
            spent=result.spent,
            best=result.best.digest,
            best_mean=result.fitness.aggregate(result.best.digest),
            frontier=len(result.fitness.frontier()),
            pool=len(result.pool),
            moved=result.best.digest != start.digest,
        )
        return replace(agent, modules=result.best)
