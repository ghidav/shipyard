"""The recipes: one module per `[recipe] kind`, each an `async def run(run: Run)`."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from shipyard.config import KINDS

if TYPE_CHECKING:
    from shipyard.run import Run

__all__ = ["module_for", "resolution", "run_recipe"]

#: The module of each kind. A hyphenated kind imports under its underscore name.
MODULES = {kind: kind.replace("-", "_") for kind in KINDS}


def module_for(kind: str) -> ModuleType:
    return importlib.import_module(f"shipyard.recipes.{MODULES[kind]}")


async def run_recipe(run: Run) -> None:
    """Import the module named by `run.config.recipe.kind` and await its `run(run)`."""
    await module_for(run.config.recipe.kind).run(run)


def resolution(recipe: Any) -> str | None:
    """The one-line description of what a gradient recipe's knobs resolve to, printed by
    `check`. None when the recipe's module has no `preset`."""
    module = module_for(recipe.kind)
    own = getattr(module, "resolution", None)
    if own is not None:
        return own(recipe)
    found = getattr(module, "preset", None)
    if found is None:
        return None
    from shipyard.recipes import train as loop

    return loop.resolution(found(recipe))
