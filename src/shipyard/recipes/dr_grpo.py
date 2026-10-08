"""Dr. GRPO (Liu et al., arXiv 2503.20783): advantage = r - mean (no division by spread), PPO
with one symmetric `clip` (0.2, App. G; the bounds `1 - clip` / `1 + clip` via
`recipes.train.clipped`), token losses summed (Tinker's sum is the constant normalizer of
3.2), and AdamW as App. G sets it. Degenerate groups are dropped, and refilled only when
`refill` is set. The length rule among solved answers (`length_penalty`, `length_floor`,
`length_cap`; `credit.shaped`) is shipyard's addition and off by default. The paper has
none."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset
from shipyard.trainer import Adam

if TYPE_CHECKING:
    from shipyard.run import Run

#: App. G (Table 6): betas 0.9 / 0.95, weight decay 0, the gradient norm clipped at 1.0, a
#: constant learning rate. Eps is unstated, so the cookbook's applies.
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
        adam=replace(ADAM, warmup=int(recipe.warmup)),
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
