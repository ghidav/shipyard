"""The recipes: a trainer class per `[recipe] kind`, and the one a trainer file defines."""

from __future__ import annotations

import importlib.util
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from shipyard.config import Gradient
from shipyard.training.recipes.cispo import CISPOTrainer
from shipyard.training.recipes.dapo import DAPOTrainer
from shipyard.training.recipes.dr_grpo import DrGRPOTrainer
from shipyard.training.recipes.fst import fst_trainer
from shipyard.training.recipes.gepa import GEPATrainer
from shipyard.training.trainer import Trainer

__all__ = ["TRAINERS", "resolution", "trainer_for", "trainer_in"]

#: The trainer class of each `[recipe] kind`.
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


def trainer_in(path: Path) -> type[Trainer]:
    """The one `Trainer` subclass the Python file defines; a path is loaded once."""
    path = Path(path).resolve()
    name = f"shipyard_trainer_{sha256(str(path).encode()).hexdigest()[:12]}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{path} is not a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    found = [
        one
        for one in vars(module).values()
        if isinstance(one, type) and issubclass(one, Trainer) and one.__module__ == name
    ]
    if len(found) != 1:
        raise ValueError(
            f"{path} defines {len(found)} Trainer subclasses; a trainer file defines one"
        )
    return found[0]


def resolution(recipe: Any, trainer: type[Trainer] | None = None) -> str | None:
    """The line `check` prints for a recipe that trains, under `trainer` or the kind's class;
    None for the others."""
    if not isinstance(recipe, Gradient):
        return None
    return (trainer or trainer_for(recipe)).resolution(recipe)
