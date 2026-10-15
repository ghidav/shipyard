"""The recipes: a trainer class per gradient kind; the other kinds are modules with `run`."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import TYPE_CHECKING, Any

from shipyard.config import KINDS
from shipyard.recipes.fst import SLOW, fst_trainer
from shipyard.recipes.gepa import GEPATrainer
from shipyard.runner import Runner
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer

if TYPE_CHECKING:
    from shipyard.run import Run

__all__ = ["TRAINERS", "module_for", "resolution", "run_recipe", "trainer_for"]

#: The trainer class of each `[recipe] kind` the runner runs.
TRAINERS: dict[str, type[Trainer]] = {
    "dapo": DAPOTrainer,
    "dr-grpo": DrGRPOTrainer,
    "cispo": CISPOTrainer,
    "evaluate": Trainer,
    "gepa": GEPATrainer,
}

#: The module of each kind. A hyphenated kind imports under its underscore name.
MODULES = {kind: kind.replace("-", "_") for kind in KINDS}


def module_for(kind: str) -> ModuleType:
    return importlib.import_module(f"shipyard.recipes.{MODULES[kind]}")


def trainer_for(recipe: Any) -> type[Trainer]:
    """The trainer class of the recipe: its kind's, or for fst the mix over its slow kind."""
    if recipe.kind == "fst":
        return fst_trainer(SLOW[recipe.slow][1])
    return TRAINERS[recipe.kind]


async def run_recipe(run: Run) -> None:
    await Runner.of(run, trainer_for(run.config.recipe)).run()


def resolution(recipe: Any) -> str | None:
    """The resolution line `check` prints; None for a kind without a preset."""
    if importlib.util.find_spec(f"shipyard.recipes.{MODULES[recipe.kind]}") is None:
        return None
    module = module_for(recipe.kind)
    own = getattr(module, "resolution", None)
    if own is not None:
        return own(recipe)
    found = getattr(module, "preset", None)
    if found is None:
        return None
    from shipyard import gradient

    return gradient.resolution(found(recipe))
