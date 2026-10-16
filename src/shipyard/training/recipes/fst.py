"""Fast-slow training recipe (Tiwari, Sareen, Agrawal et al., arXiv 2605.12484).

The weights and a population of K texts co-evolve in cycles: a fast phase evolves the texts
with GEPA, then a slow phase takes gradient steps with an RL recipe."""

from __future__ import annotations

import random
from collections.abc import Mapping
from collections.abc import Sequence as Values
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from shipyard import record
from shipyard.config import (
    SLOW_KNOBS,
    Blueprint,
    CispoRecipe,
    DapoRecipe,
    DrGrpoRecipe,
    modules_dir,
)
from shipyard.modules import Candidate, keep, seed
from shipyard.training.credit import weights
from shipyard.training.recipes.gepa import BEST, SEED, GEPATrainer
from shipyard.training.trainer import Trainer
from shipyard.types import Adam, Agent, Group, Task, TrainingStep

#: Config model for each `[recipe] slow` kind. The slow trainer is that kind's usual class.
SLOW: dict[str, type] = {"dapo": DapoRecipe, "dr-grpo": DrGrpoRecipe, "cispo": CispoRecipe}
#: GEPA budget per cycle, in passes over the anchor tasks, and rollouts per (task, text) cell.
#: Appendix D uses 960 metric calls over 192 examples, one rollout per metric call.
BUDGET_PASSES = 5
CELL_ROLLOUTS = 1
#: Appendix D: AdamW with PyTorch's betas 0.9 / 0.999 and weight decay 0. Eps is unstated;
#: 1e-8 is the PyTorch and cookbook default. No gradient clipping is stated, so none is used.
#: The paper's 10-step warm-up is off by default and is set with the `warmup` key.
ADAM = Adam(beta1=0.9, beta2=0.999, eps=1e-8)
PAPER_WARMUP = 10


def slow_recipe(recipe: Any) -> Any:
    """Build the slow recipe's config from an fst recipe.

    It takes the shared training settings and the slow kind's own knobs that are set, and
    turns the overlong penalty off."""
    model = SLOW[recipe.slow]
    knobs = {knob: getattr(recipe, knob) for knob in SLOW_KNOBS[recipe.slow]}
    knobs = {knob: value for knob, value in knobs.items() if value is not None}
    if model is not DrGrpoRecipe:
        knobs["overlong_penalty"] = 0.0
    shared = {
        name: getattr(recipe, name)
        for name in (
            "learning_rate",
            "steps",
            "evaluate_every",
            "substeps",
            "warmup",
            "reference",
            "lora_rank",
            "restore_optimizer",
        )
    }
    return model(kind=recipe.slow, kl_coef=recipe.kl_coef, **shared, **knobs)


def merged(parts: Values[list[Group]]) -> list[Group]:
    """Merge one step's per-text jobs into one group per task, with each text's members in
    turn. Raise `ValueError` if the jobs planned different numbers of groups."""
    first = parts[0]
    if any(len(part) != len(first) for part in parts):
        raise ValueError("the jobs of one step planned different numbers of groups")
    found: list[Group] = []
    for index, lead in enumerate(first):
        members = [one for part in parts for one in part[index].members]
        found.append(Group(lead.task, lead.agent, lead.harness, members))
    return found


class FSTTrainer(GEPATrainer):
    """Trainer for fast-slow training: a GEPA search over texts, then slow-recipe steps.

    `fst_trainer` mixes this class with the trainer class of the slow kind."""

    ADAM = ADAM
    PAPER_WARMUP = PAPER_WARMUP

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        return [
            f"cycles of {recipe.cycle} {recipe.slow} steps, each after gepa evolves "
            f"{recipe.population} texts on the next {recipe.cycle} batches",
            f"every group split group_size / {recipe.population} per text",
            *super().rules(slow_recipe(recipe)),
        ]

    @classmethod
    def aggregated(cls) -> str:
        return "averaged per prompt"

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.fst = config.recipe
        self.recipe = slow_recipe(self.fst)
        self.score_group_size = CELL_ROLLOUTS
        self.population: list[Candidate] = []
        self.cycles = 0
        #: Batches drawn for the current cycle that have not been stepped on yet.
        self.pending: list[list[Task]] = []

    async def train(self) -> Agent:
        """Run cycles until `[recipe] steps` steps are taken, then `finish`.

        Each cycle draws the next `cycle` batches, runs a fast phase over their tasks, then
        takes one slow step per batch."""
        kept = self.directory / record.MODULES
        if not self.population:
            seeded = seed(modules_dir(self.config))
            keep(seeded, kept / SEED)
            self.population = [seeded]
        self.modules = self.population[0]
        agent = self.agent()
        async with agent.policy:
            while self.steps < self.recipe.steps:
                if not self.pending:
                    count = min(self.fst.cycle, self.recipe.steps - self.steps)
                    self.pending = [self.draw() for _ in range(count)]
                    agent = await self.fast(agent)
                await self.advance(agent)
                if not self.pending:
                    self.cycles += 1
            keep(agent.modules, kept / BEST)
            return await self.finish(agent)

    async def fast(self, agent: Agent) -> Agent:
        """Run the fast phase: evolve the population with GEPA on the pending batches' tasks.

        The new population is saved under `modules/cycle-<n>/`, and the returned agent carries
        its top-ranked text."""
        fst, cycle = self.fst, self.cycles
        anchor = list(dict.fromkeys(task for batch in self.pending for task in batch))
        anchor = anchor[: fst.anchor] if fst.anchor else anchor
        # Scoring the carried-in population is added to the budget, except its first member.
        carried = len({one.digest for one in self.population}) - 1
        cells = len(anchor) * CELL_ROLLOUTS
        result = await self.evolve(
            agent,
            anchor,
            self.population,
            budget=(fst.budget or BUDGET_PASSES * cells) + carried * cells,
            minibatch=fst.minibatch,
            patience=fst.patience,
            rng=random.Random(self.config.data.seed + cycle),
            log=partial(self.logs.log, "search", cycle=cycle),
        )
        self.population = result.top(fst.population)
        for rank, text in enumerate(self.population):
            keep(text, self.directory / record.MODULES / f"cycle-{cycle}" / str(rank))
        self.logs.log(
            "search",
            cycle=cycle,
            evolution=True,
            rounds=result.rounds,
            spent=result.spent,
            population=[one.digest for one in self.population],
            frontier=len(result.fitness.frontier()),
            pool=len(result.pool),
        )
        return replace(agent, modules=self.population[0])

    async def sample(self, agent: Agent) -> list[Group]:
        """Sample the cycle's next batch with one job per text, each at its share of the group
        size, and credit the results merged into one group per task."""
        batch = self.pending.pop(0)
        share = self.config.data.group_size // self.fst.population
        parts = [
            await self.job(replace(agent, modules=text), batch, share) for text in self.population
        ]
        return self.credited(merged(parts))

    async def checkpoint(self, agent: Agent, tag: str, *, keep: bool = False, **more: Any) -> None:
        await super().checkpoint(
            agent, tag, keep=keep, cycle=self.cycles, pending=len(self.pending), **more
        )

    def restore(self, row: Mapping[str, Any]) -> None:
        """Restore the checkpoint's cycle: reload its population and redraw the batches it had
        not stepped on. After a finished cycle, the next one starts with its fast phase."""
        super().restore(row)
        kept = self.directory / record.MODULES / f"cycle-{row['cycle']}"
        self.population = [seed(kept / str(rank)) for rank in range(self.fst.population)]
        self.modules = self.population[0]
        self.drawn -= row["pending"]
        self.pending = [self.draw() for _ in range(row["pending"])]
        self.cycles = row["cycle"] if self.pending else row["cycle"] + 1

    def weights(self, sizes: Values[int]) -> list[float]:
        """Weight every prompt equally (Eq. 4), whatever the slow recipe does."""
        return weights(sizes)

    def log(self, done: TrainingStep, **more: Any) -> None:
        super().log(done, cycle=self.cycles, **more)


def fst_trainer(slow: type[Trainer]) -> type[Trainer]:
    """Return the fst trainer class: `FSTTrainer` mixed with `slow`, the `[recipe] slow` class."""
    return type(f"FST{slow.__name__}", (FSTTrainer, slow), {})
