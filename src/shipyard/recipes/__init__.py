"""The recipes: a trainer class per `[recipe] kind`, and the runner over them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard.config import Gradient
from shipyard.recipes.fst import SLOW, fst_trainer
from shipyard.recipes.gepa import GEPATrainer
from shipyard.runner import Runner
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer

if TYPE_CHECKING:
    from shipyard.run import Run

__all__ = ["TRAINERS", "resolution", "run_recipe", "trainer_for"]

#: The trainer class of each `[recipe] kind` the runner runs.
TRAINERS: dict[str, type[Trainer]] = {
    "dapo": DAPOTrainer,
    "dr-grpo": DrGRPOTrainer,
    "cispo": CISPOTrainer,
    "evaluate": Trainer,
    "gepa": GEPATrainer,
}


def trainer_for(recipe: Any) -> type[Trainer]:
    """The trainer class of the recipe: its kind's, or for fst the mix over its slow kind."""
    if recipe.kind == "fst":
        return fst_trainer(SLOW[recipe.slow][1])
    return TRAINERS[recipe.kind]


async def run_recipe(run: Run) -> None:
    await Runner.of(run, trainer_for(run.config.recipe)).run()


def resolution(recipe: Any) -> str | None:
    """The line `check` prints for a recipe that trains; None for the others."""
    return trainer_for(recipe).resolution(recipe) if isinstance(recipe, Gradient) else None
