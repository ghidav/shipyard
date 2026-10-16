"""Trainer classes for each `[recipe] kind`, and loading a custom trainer from a file."""

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
from shipyard.training.recipes.opd import OPDTrainer
from shipyard.training.trainer import Trainer

__all__ = ["TRAINERS", "resolution", "trainer_for", "trainer_in"]

#: Trainer class for each `[recipe] kind` except fst, which `trainer_for` builds.
TRAINERS: dict[str, type[Trainer]] = {
    "dapo": DAPOTrainer,
    "dr-grpo": DrGRPOTrainer,
    "cispo": CISPOTrainer,
    "opd": OPDTrainer,
    "evaluate": Trainer,
    "gepa": GEPATrainer,
}


def trainer_for(recipe: Any) -> type[Trainer]:
    """Return the trainer class for `recipe`.

    For fst, this is `FSTTrainer` mixed with the trainer class of `recipe.slow`."""
    if recipe.kind == "fst":
        return fst_trainer(TRAINERS[recipe.slow])
    return TRAINERS[recipe.kind]


def trainer_in(path: Path) -> type[Trainer]:
    """Return the single `Trainer` subclass defined in a Python file.

    Each resolved path is imported once and cached in `sys.modules`. Raise `ValueError` if the
    file cannot be loaded as Python or does not define exactly one subclass."""
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
    """Return the summary line `check` prints for a recipe, using `trainer` or the kind's class.

    Return None when the recipe does not train."""
    if not isinstance(recipe, Gradient):
        return None
    return (trainer or trainer_for(recipe)).resolution(recipe)
