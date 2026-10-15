"""The recipes: a trainer class per `[recipe] kind`."""

from __future__ import annotations

from typing import Any

from shipyard.config import Gradient
from shipyard.recipes.fst import fst_trainer
from shipyard.recipes.gepa import GEPATrainer
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer

__all__ = ["TRAINERS", "resolution", "trainer_for"]

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
        return fst_trainer(TRAINERS[recipe.slow])
    return TRAINERS[recipe.kind]


def resolution(recipe: Any, trainer: type[Trainer] | None = None) -> str | None:
    """The line `check` prints for a recipe that trains, under `trainer` or the kind's class;
    None for the others."""
    if not isinstance(recipe, Gradient):
        return None
    return (trainer or trainer_for(recipe)).resolution(recipe)
