"""Dr. GRPO: advantage = r - mean (no division by spread), PPO with one symmetric `clip`
(0.2; the bounds `1 - clip` / `1 + clip` via `recipes.train.clipped`), the length rule among
solved answers (`length_penalty`, `length_floor`; `credit.shaped`) here only; degenerate dropped."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset

if TYPE_CHECKING:
    from shipyard.run import Run


def preset(recipe: Any) -> Preset:
    return Preset(
        name="dr-grpo",
        normalize=False,
        loss_fn="ppo",
        loss_config=loop.clipped(float(recipe.clip), float(recipe.clip)),
        clipping=f"clip {recipe.clip} / {recipe.clip}",
        length_penalty=float(recipe.length_penalty),
        length_floor=int(recipe.length_floor),
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
