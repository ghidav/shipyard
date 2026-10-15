"""The recipes: a trainer class per gradient kind; the other kinds are modules with `run`."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import TYPE_CHECKING, Any

from shipyard.config import KINDS
from shipyard.runner import TRAINERS, Runner

if TYPE_CHECKING:
    from shipyard.run import Run

__all__ = ["module_for", "resolution", "run_recipe"]

#: The module of each kind. A hyphenated kind imports under its underscore name.
MODULES = {kind: kind.replace("-", "_") for kind in KINDS}


def module_for(kind: str) -> ModuleType:
    return importlib.import_module(f"shipyard.recipes.{MODULES[kind]}")


async def run_recipe(run: Run) -> None:
    """The runner for a kind with a trainer class, else the module's `run(run)`."""
    kind = run.config.recipe.kind
    if kind in TRAINERS:
        await Runner.of(run).run()
    else:
        await module_for(kind).run(run)


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
