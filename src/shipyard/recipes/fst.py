"""Fast-slow training (Tiwari, Sareen, Agrawal et al., arXiv 2605.12484): the weights and a
population of K texts co-evolve in cycles. Each cycle runs gepa against the current weights
on the next `cycle` batches, seeded with the last population, and keeps the top K of its
frontier; the next `cycle` steps sample every task's group as group_size / K rollouts per
text, normalised as one group, and update the weights with the slow recipe."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shipyard import record
from shipyard.config import SLOW_KNOBS, CispoRecipe, DapoRecipe, DrGrpoRecipe
from shipyard.data import batches
from shipyard.gepa.cycle import evolve
from shipyard.gepa.outcomes import outcomes
from shipyard.gepa.reflect import reflect
from shipyard.modules import Candidate, keep, seed
from shipyard.recipes import cispo, dapo, dr_grpo
from shipyard.recipes import train as loop
from shipyard.recipes.gepa import BEST, SEED
from shipyard.recipes.train import Preset
from shipyard.rollout import Rollouts
from shipyard.trainer import SERVE_TTL

if TYPE_CHECKING:
    from shipyard.run import Run

#: The slow recipe each `slow` names: its config model and its module.
SLOW = {
    "dapo": (DapoRecipe, dapo),
    "dr-grpo": (DrGrpoRecipe, dr_grpo),
    "cispo": (CispoRecipe, cispo),
}
#: Gepa's budget per cycle in passes over the anchor tasks: the paper's 960 metric calls
#: over 192 examples.
BUDGET_PASSES = 5


def slow_recipe(recipe: Any) -> Any:
    """The slow recipe's own config, from the knobs fst shares with it and the ones it set."""
    model, _ = SLOW[recipe.slow]
    knobs = {knob: getattr(recipe, knob) for knob in SLOW_KNOBS[recipe.slow]}
    knobs = {knob: value for knob, value in knobs.items() if value is not None}
    shared = {name: getattr(recipe, name) for name in ("learning_rate", "substeps", "reference")}
    return model(kind=recipe.slow, kl_coef=recipe.kl_coef, **shared, **knobs)


def preset(recipe: Any) -> Preset:
    """The slow recipe's preset, under fst's name, without refill and averaged per prompt
    whatever the slow loss: FST follows ScaleRL (section 2), whose zero-variance filtering
    drops flat groups without resampling, and aggregates at the prompt level (Eq. 4)."""
    _, module = SLOW[recipe.slow]
    return replace(module.preset(slow_recipe(recipe)), name="fst", refill=0, aggregation="prompt")


def resolution(recipe: Any) -> str:
    """The comment line `check` prints: the cycle, the population, then the slow recipe's."""
    slow = loop.resolution(preset(recipe)).removeprefix("# fst: ")
    return (
        f"# fst: cycles of {recipe.cycle} {recipe.slow} steps, each after gepa evolves "
        f"{recipe.population} texts on the next {recipe.cycle} batches; every group split "
        f"group_size / {recipe.population} per text; {slow}"
    )


async def run(run: Run) -> None:
    """Seed, then per cycle a fast phase and `cycle` slow steps until the batches run out;
    each cycle's population kept under `modules/cycle-<n>/<rank>`, the last one's first
    under `modules/best/`."""
    cfg = run.config
    recipe, data = cfg.recipe, cfg.data
    if data.group_size % recipe.population:
        raise ValueError(
            f"[recipe] population {recipe.population} does not divide [data] group_size "
            f"{data.group_size}"
        )
    share = data.group_size // recipe.population
    slow = preset(recipe)
    kept = run.directory / record.MODULES
    population = [seed(cfg.home / recipe.modules)]
    keep(population[0], kept / SEED)
    write = reflect(
        run,
        harness=recipe.reflection_harness,
        model=recipe.reflection_model,
        image=recipe.reflection_image,
        incremental=recipe.edits == "incremental",
    )
    planned = list(batches(cfg.datasets, size=data.batch_size, seed=data.seed, epochs=data.epochs))
    index = 0
    async with loop.training(run, slow) as (trainer, anchor):
        serving = loop.serving_of(run, slow)
        for cycle, start in enumerate(range(0, len(planned), recipe.cycle)):
            lookahead = planned[start : start + recipe.cycle]
            await serving.point(await trainer.publish(f"fast-{cycle}", ttl_seconds=SERVE_TTL))
            population = await fast(run, population, lookahead, cycle, index, write, share)
            for rank, member in enumerate(population):
                keep(member, kept / f"cycle-{cycle}" / str(rank))
            sample = partial(sampled, run, population, share)
            for tasks in lookahead:
                about = {"cycle": cycle}
                await loop.step(
                    run, trainer, anchor, slow, tasks, index, sample=sample, about=about
                )
                index += 1
    keep(population[0], kept / BEST)


async def fast(
    run: Run,
    population: Sequence[Candidate],
    lookahead: Sequence[Sequence[Path]],
    cycle: int,
    index: int,
    write: Any,
    share: int,
) -> list[Candidate]:
    """One fast phase under the weights the proxy now serves: gepa on the lookahead's tasks
    (the first `anchor` of them when set), seeded with the population, its top K back."""
    recipe = run.config.recipe
    tasks = list(dict.fromkeys(task for batch in lookahead for task in batch))
    tasks = tasks[: recipe.anchor] if recipe.anchor else tasks

    # Measuring the population carried in is on top of the budget, past the first member.
    carried = len({member.digest for member in population}) - 1

    async def score(candidate: Candidate, over: Sequence[Path], round_index: int) -> Any:
        rolled = await run.sample(list(over), rollouts=share, index=index, modules=candidate)
        return outcomes(rolled, over, share)

    result = await evolve(
        population,
        tasks,
        write=write,
        score=score,
        minibatch=recipe.minibatch,
        budget=(recipe.budget or BUDGET_PASSES * len(tasks) * share) + carried * len(tasks) * share,
        patience=recipe.patience,
        rollouts=share,
        log=partial(run.log, cycle=cycle),
    )
    chosen = result.top(recipe.population)
    run.log(
        cycle=cycle,
        evolution=True,
        rounds=result.rounds,
        spent=result.spent,
        population=[member.digest for member in chosen],
        frontier=len(result.fitness.frontier()),
        pool=len(result.pool),
    )
    return chosen


async def sampled(
    run: Run, population: Sequence[Candidate], share: int, tasks: Sequence[Path], index: int
) -> Rollouts:
    """A slow step's batch: one job per text at `share` rollouts a task, one after another
    (a job's name is reserved in order), merged into one group per task."""
    parts = [
        await run.sample(tasks, rollouts=share, index=index, modules=member)
        for member in population
    ]
    return merged(parts, share)


def merged(parts: Sequence[Rollouts], share: int) -> Rollouts:
    """The jobs of one step as one: task i's group is its `share` trials from each job in
    turn, so credit normalises prompt and sampling variation together, as the paper does."""
    first = parts[0]
    if any(len(part.trials) != len(first.trials) for part in parts):
        raise ValueError("the jobs of one step planned different numbers of trials")
    if len({part.updates for part in parts}) != 1:
        raise ValueError("the jobs of one step sampled different weights")
    trials: list[Path] = []
    plan: list[tuple[Path, int]] = []
    for start in range(0, len(first.trials), share):
        for part in parts:
            trials += part.trials[start : start + share]
            plan += part.plan[start : start + share]
    records = None
    if any(part.records is not None for part in parts):
        records = {name: found for part in parts for name, found in (part.records or {}).items()}
    asked = None
    if any(part.asked is not None for part in parts):
        asked = {name: count for part in parts for name, count in (part.asked or {}).items()}
    failed = None
    if any(part.failed is not None for part in parts):
        failed = {name for part in parts for name in (part.failed or set())}
    return Rollouts(
        job="+".join(part.job for part in parts),
        trials=trials,
        plan=plan,
        records=records,
        asked=asked,
        failed=failed,
        updates=first.updates,
    )
