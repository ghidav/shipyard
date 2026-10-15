"""Fast-slow training (Tiwari, Sareen, Agrawal et al., arXiv 2605.12484): the weights and a
population of K texts co-evolve in cycles."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from shipyard import record
from shipyard.config import SLOW_KNOBS, CispoRecipe, DapoRecipe, DrGrpoRecipe
from shipyard.credit import weights
from shipyard.modules import Candidate, keep, seed
from shipyard.recipes.gepa import BEST, SEED, Search
from shipyard.recipes.gepa.reflect import reflect
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer
from shipyard.types import Adam, Agent, Batches, Group, Policy, Update

if TYPE_CHECKING:
    from shipyard.run import Run

#: The config model and the trainer class of the slow recipe each `slow` value names.
SLOW: dict[str, tuple[type, type[Trainer]]] = {
    "dapo": (DapoRecipe, DAPOTrainer),
    "dr-grpo": (DrGrpoRecipe, DrGRPOTrainer),
    "cispo": (CispoRecipe, CISPOTrainer),
}
#: Gepa's budget per cycle, in passes over the anchor tasks with one rollout per (task, text)
#: cell. App. D uses 960 metric calls over 192 examples, a metric call being one rollout.
BUDGET_PASSES = 5
CELL_ROLLOUTS = 1
#: App. D: AdamW at PyTorch's betas 0.9 / 0.999, weight decay 0. Eps is unstated, and 1e-8 is
#: PyTorch's default and the cookbook's. No gradient clipping is stated, so none is used.
#: The `warmup` key sets the 10-step warm-up, which is off by default.
ADAM = Adam(beta1=0.9, beta2=0.999, eps=1e-8)
PAPER_WARMUP = 10


def slow_recipe(recipe: Any) -> Any:
    """The slow recipe's config: the shared knobs, its own that are set, no overlong term."""
    model, _ = SLOW[recipe.slow]
    knobs = {knob: getattr(recipe, knob) for knob in SLOW_KNOBS[recipe.slow]}
    knobs = {knob: value for knob, value in knobs.items() if value is not None}
    if model is not DrGrpoRecipe:
        knobs["overlong_penalty"] = 0.0
    shared = {
        name: getattr(recipe, name) for name in ("learning_rate", "substeps", "warmup", "reference")
    }
    return model(kind=recipe.slow, kl_coef=recipe.kl_coef, **shared, **knobs)


class FST(Search):
    """The fast-slow pass over the slow trainer it is mixed with."""

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

    def __init__(self, run: Run) -> None:
        super().__init__(run)
        self.fst = run.config.recipe
        self.recipe = slow_recipe(self.fst)
        self.score_group_size = CELL_ROLLOUTS
        self.reflect = reflect(
            run,
            harness=self.fst.reflection_harness,
            model=self.fst.reflection_model,
            image=self.fst.reflection_image,
            incremental=self.fst.edits == "incremental",
        )
        self.population: list[Agent] = []
        self.cycles = 0
        #: Batches of the cycle in progress already stepped, on a resume.
        self.offset = 0

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        """Cycles of `cycle` batches: a fast phase over the cycle's tasks, then its slow steps."""
        fst, data = self.fst, self.run.config.data
        kept = self.run.directory / record.MODULES
        if not self.population:
            seeded = seed(self.run.config.home / fst.modules)
            keep(seeded, kept / SEED)
            self.population = [replace(agent, modules=seeded)]
        else:
            self.population = [replace(agent, modules=one.modules) for one in self.population]
            agent = self.population[0]
        batches = list(tasks)
        start = 0
        while start < len(batches):
            chunk = batches[start : start + fst.cycle - self.offset]
            start += len(chunk)
            cycle = self.cycles
            if self.offset:
                self.offset = 0
                agent = await super().train(agent, iter(chunk))
                self.cycles += 1
                continue
            anchor = list(dict.fromkeys(task for batch in chunk for task in batch))
            anchor = anchor[: fst.anchor] if fst.anchor else anchor
            # Scoring the carried-in population is added to the budget, except its first member.
            carried = len({one.modules.digest for one in self.population}) - 1
            cells = len(anchor) * CELL_ROLLOUTS
            result = await self.evolve(
                agent,
                anchor,
                [one.modules for one in self.population],
                budget=(fst.budget or BUDGET_PASSES * cells) + carried * cells,
                minibatch=fst.minibatch,
                patience=fst.patience,
                rng=random.Random(data.seed + cycle),
                log=partial(self.run.log, cycle=cycle),
            )
            self.population = [replace(agent, modules=text) for text in result.top(fst.population)]
            agent = self.population[0]
            for rank, one in enumerate(self.population):
                keep(one.modules, kept / f"cycle-{cycle}" / str(rank))
            self.run.log(
                cycle=cycle,
                evolution=True,
                rounds=result.rounds,
                spent=result.spent,
                population=[one.modules.digest for one in self.population],
                frontier=len(result.fitness.frontier()),
                pool=len(result.pool),
            )
            agent = await super().train(agent, iter(chunk))
            self.cycles += 1
        keep(agent.modules, kept / BEST)
        return agent

    async def sample(self, agent: Agent, tasks: Batches) -> AsyncIterator[list[Group]]:
        """One job per text at its share of the group, merged into one group per task."""
        share = self.run.config.data.group_size // self.fst.population
        for batch in tasks:
            parts = [await self.job(one, batch, share) for one in self.population]
            yield self.merged(parts)

    def merged(self, parts: Values[list[Group]]) -> list[Group]:
        """The jobs of one step as one group per task, each text's share in turn."""
        first = parts[0]
        if any(len(part) != len(first) for part in parts):
            raise ValueError("the jobs of one step planned different numbers of groups")
        found: list[Group] = []
        for index, lead in enumerate(first):
            members = [one for part in parts for one in part[index].members]
            group = Group(lead.task, lead.agent, lead.harness, members)
            group.shaped = self.shape(group)
            group.advantages = self.advantage(group)
            found.append(group)
        return found

    async def checkpoint(self, agent: Agent, tag: str, *, keep: bool = False, **more: Any) -> None:
        await super().checkpoint(agent, tag, keep=keep, cycle=self.cycles, **more)

    def consumed(self, jobs: Values[Mapping[str, Any]], row: Mapping[str, Any]) -> int:
        return super().consumed(jobs, row) // self.fst.population

    def restore(self, row: Mapping[str, Any], consumed: int) -> Candidate | None:
        """The population of the checkpoint's cycle, and where in the cycle it stood."""
        super().restore(row, consumed)
        kept = self.run.directory / record.MODULES / f"cycle-{row['cycle']}"
        texts = [seed(kept / str(rank)) for rank in range(self.fst.population)]
        self.population = [Agent(Policy(self.run.endpoint), [], text) for text in texts]
        self.offset = consumed % self.fst.cycle
        self.cycles = row["cycle"] if self.offset else row["cycle"] + 1
        return texts[0]

    def weights(self, sizes: Values[int]) -> list[float]:
        """Each prompt weighs the same (Eq. 4), whatever the slow recipe does."""
        return weights(sizes)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> None:
        super().log(found, update, cycle=self.cycles, **more)


def fst_trainer(slow: type[Trainer]) -> type[Trainer]:
    """The fst kind over the slow class `[recipe] slow` names."""
    return type(f"FST{slow.__name__}", (FST, slow), {})
