"""DAPO (Yu et al., arXiv 2503.14476): advantage = (r - mean) / spread, PPO with asymmetric
clipping, each prompt's token losses averaged (Eq. 8), 16 substeps by prompt (4.1), degenerate
groups dropped and refilled from the plan (dynamic sampling, Alg. 1). `clip_low` / `clip_high`
(0.2 / 0.28, 4.1) are epsilons; Tinker's `ppo` takes bounds, so `loss_fn_config` gets
`1 - clip_low` and `1 + clip_high` (`recipes.train.clipped`)."""

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
        aggregation="prompt",
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
