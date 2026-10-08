"""CISPO (MiniMax-M1, arXiv 2506.13585): the advantage as DAPO forms it and Tinker's `cispo`
loss, whose truncated importance weight scales the gradient without passing it; each prompt's
token losses averaged (Eq. 4), 16 substeps by prompt, DAPO's dynamic sampling and length
penalty (3.1), and AdamW at betas 0.9 / 0.95 and eps 1e-15 (3.2). `clip_high` (3.0) is an
epsilon on the weight's ceiling: `loss_fn_config` gets `clip_high_threshold = 1 + clip_high`,
`clip_low_threshold = 0`."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from shipyard.recipes import train as loop
from shipyard.recipes.train import Preset
from shipyard.trainer import Adam

if TYPE_CHECKING:
    from shipyard.run import Run

#: 3.2 sets the betas and eps for gradients mostly below 1e-14; it states no weight decay,
#: gradient clipping or warm-up, so those are the cookbook's: none.
ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-15)


def preset(recipe: Any) -> Preset:
    # No lower bound: MiniMax-M1 sets epsilon_low large and tunes epsilon_high alone without
    # publishing it; the ceiling of 4.0 is ScaleRL's (A.17.2) and FST's (appendix D).
    high = round(1.0 + float(recipe.clip_high), 6)
    return Preset(
        name="cispo",
        normalize=True,
        loss_fn="cispo",
        loss_config={"clip_low_threshold": 0.0, "clip_high_threshold": high},
        clipping=f"weight truncated above {high}, no lower bound",
        aggregation="prompt",
        overlong_penalty=float(recipe.overlong_penalty),
        overlong_buffer=float(recipe.overlong_buffer),
        adam=replace(ADAM, warmup=int(recipe.warmup)),
        **loop.shared(recipe),
    )


async def run(run: Run) -> None:
    await loop.train(run, preset(run.config.recipe))
