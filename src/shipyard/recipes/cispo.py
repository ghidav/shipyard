"""CISPO: the advantage as DAPO forms it and Tinker's `cispo` loss, whose truncated importance
weight scales the gradient without passing it. `clip_high` (0.2) is an epsilon on the weight's
ceiling: `loss_fn_config` gets `clip_high_threshold = 1 + clip_high`, `clip_low_threshold = 0`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset

if TYPE_CHECKING:
    from shipyard.run import Run


def preset(recipe: Any) -> Preset:
    # No lower bound: the MiniMax-M1 paper sets epsilon_low large and tunes epsilon_high
    # alone, without publishing its value.
    high = round(1.0 + float(recipe.clip_high), 6)
    return Preset(
        name="cispo",
        normalize=True,
        loss_fn="cispo",
        loss_config={"clip_low_threshold": 0.0, "clip_high_threshold": high},
        clipping=f"weight truncated above {high}, no lower bound",
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
