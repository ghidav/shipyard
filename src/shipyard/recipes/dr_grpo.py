"""Dr. GRPO (Liu et al., arXiv 2503.20783): advantage = r - mean (no division by spread), PPO
with one symmetric `clip` (0.2, App. G; the bounds `1 - clip` / `1 + clip` via
`recipes.train.clipped`), token losses summed, the constant normalizer of 3.2 that Tinker's sum
already is, AdamW as App. G sets it; degenerate groups dropped, never refilled unless asked.
The length rule among solved answers (`length_penalty`, `length_floor`, `length_cap`;
`credit.shaped`) is shipyard's own, off by default, as the paper has none."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset
from shipyard.trainer import Adam

if TYPE_CHECKING:
    from shipyard.run import Run

#: App. G (Table 6): betas 0.9 / 0.95, weight decay 0, the gradient norm clipped at 1.0, a
#: constant learning rate; eps is unstated, so the cookbook's.
ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=1.0)


def preset(recipe: Any) -> Preset:
    return Preset(
        name="dr-grpo",
        normalize=False,
        loss_fn="ppo",
        loss_config=loop.clipped(float(recipe.clip), float(recipe.clip)),
        clipping=f"clip {recipe.clip} / {recipe.clip}",
        aggregation="sum",
        length_penalty=float(recipe.length_penalty),
        length_floor=int(recipe.length_floor),
        length_cap=float(recipe.length_cap),
        adam=ADAM,
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
