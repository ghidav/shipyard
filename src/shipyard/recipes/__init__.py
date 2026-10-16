"""The recipes: a trainer class per `[recipe] kind`, one a trainer file defines, and the
trainer a run reopens under."""

from __future__ import annotations

import importlib.util
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from shipyard import record
from shipyard.config import Gradient
from shipyard.recipes.cispo import CISPOTrainer
from shipyard.recipes.dapo import DAPOTrainer
from shipyard.recipes.dr_grpo import DrGRPOTrainer
from shipyard.recipes.fst import fst_trainer
from shipyard.recipes.gepa import GEPATrainer
from shipyard.run import Run
from shipyard.training.trainer import Trainer

__all__ = ["TRAINERS", "reopened", "resolution", "trainer_for", "trainer_in"]

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


def reopened(directory: Path, *, tag: str = "last") -> Trainer:
    """The trainer over a run reopened at one of its checkpoints, standing where it stood:
    the class of the trainer file the run was opened with, else the kind's."""
    directory = Path(directory)
    if not (directory / record.CONFIG).is_file():
        raise ValueError(f"no run at {directory}")
    rows = list(record.read(directory / record.CHECKPOINTS))
    if not rows:
        raise ValueError(f"{directory.name} has no checkpoint to go on from")
    row = rows[-1] if tag == "last" else next((r for r in rows if r["tag"] == tag), None)
    if row is None:
        tags = ", ".join(r["tag"] for r in rows)
        raise ValueError(f"{directory.name} has no checkpoint tagged {tag!r}; it has {tags}")
    kept = directory / record.TRAINER
    run = Run.reopen(directory, resumed=row["tag"])
    chosen = trainer_in(kept) if kept.is_file() else trainer_for(run.config.recipe)
    trainer = chosen(run)
    trainer.restore(row)
    return trainer


def resolution(recipe: Any, trainer: type[Trainer] | None = None) -> str | None:
    """The line `check` prints for a recipe that trains, under `trainer` or the kind's class;
    None for the others."""
    if not isinstance(recipe, Gradient):
        return None
    return (trainer or trainer_for(recipe)).resolution(recipe)
