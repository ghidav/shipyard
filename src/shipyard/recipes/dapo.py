"""DAPO (Yu et al., arXiv 2503.14476): advantage = (r - mean) / spread, PPO with asymmetric
clipping, each prompt's token losses averaged (Eq. 8), 16 substeps by prompt (4.1), degenerate
groups dropped and refilled from the plan (dynamic sampling, Alg. 1), the soft overlong
punishment (Eq. 13), and AdamW with a 20-step linear warm-up (4.1). `clip_low` / `clip_high`
(0.2 / 0.28, 4.1) are epsilons; Tinker's `ppo` takes bounds, so `loss_fn_config` gets
`1 - clip_low` and `1 + clip_high` (`recipes.train.clipped`)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset
from shipyard.trainer import COOKBOOK

if TYPE_CHECKING:
    from shipyard.run import Run

#: 4.1 states the warm-up and no betas, eps, weight decay or gradient clipping: those are
#: the cookbook's, which clips no gradient.
ADAM = replace(COOKBOOK, warmup=20)


def preset(recipe: Any) -> Preset:
    return Preset(
        name="dapo",
        normalize=True,
        loss_fn="ppo",
        loss_config=loop.clipped(float(recipe.clip_low), float(recipe.clip_high)),
        clipping=f"clip {recipe.clip_low} / {recipe.clip_high}",
        aggregation="prompt",
        overlong_penalty=float(recipe.overlong_penalty),
        overlong_buffer=float(recipe.overlong_buffer),
        adam=ADAM,
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
