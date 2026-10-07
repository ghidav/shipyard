"""DAPO: advantage = (r - mean) / spread, PPO with asymmetric clipping, degenerate groups
dropped. `clip_low` / `clip_high` (0.2 / 0.28) are epsilons; Tinker's `ppo` takes bounds, so
`loss_fn_config` gets `1 - clip_low` and `1 + clip_high` (`recipes.train.clipped`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset

if TYPE_CHECKING:
    from shipyard.run import Run


def preset(recipe: Any) -> Preset:
    return Preset(
        name="dapo",
        normalize=True,
        loss_fn="ppo",
        loss_config=loop.clipped(float(recipe.clip_low), float(recipe.clip_high)),
        clipping=f"clip {recipe.clip_low} / {recipe.clip_high}",
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
