"""CISPO recipe (MiniMax-M1, arXiv 2506.13585)."""

from __future__ import annotations

from typing import Any

from shipyard.training.recipes.dapo import DAPOTrainer, overlong_clause
from shipyard.types import Adam


class CISPOTrainer(DAPOTrainer):
    """Trainer for CISPO (MiniMax-M1, arXiv 2506.13585): DAPO rules with Tinker's `cispo` loss."""

    loss_fn = "cispo"
    #: Section 3.2 sets the betas and eps for gradient magnitudes mostly below 1e-14. It states
    #: no weight decay, gradient clipping or warm-up, so the cookbook defaults (none) apply.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-15)
    PAPER_WARMUP = 0

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        high = round(1.0 + float(recipe.clip_high), 6)
        return [
            "advantage = group mean, divided by spread",
            f"loss = cispo, weight truncated above {high}, no lower bound, {cls.aggregated()}",
            *overlong_clause(recipe),
        ]

    def bounds(self) -> dict[str, float] | None:
        # No lower bound: MiniMax-M1 sets epsilon_low large and tunes only epsilon_high,
        # without publishing it. The ceiling of 4.0 is ScaleRL's (A.17.2) and FST's
        # (appendix D).
        high = round(1.0 + float(self.recipe.clip_high), 6)
        return {"clip_low_threshold": 0.0, "clip_high_threshold": high}
