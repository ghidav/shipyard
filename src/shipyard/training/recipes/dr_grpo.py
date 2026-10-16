"""Dr. GRPO recipe (Liu et al., arXiv 2503.20783), with an optional length penalty."""

from __future__ import annotations

from typing import Any

from shipyard.training.credit import shaped
from shipyard.training.recipes.dapo import clipped
from shipyard.training.trainer import Trainer
from shipyard.types import Adam, Group


class DrGRPOTrainer(Trainer):
    """Trainer for Dr. GRPO (Liu et al., arXiv 2503.20783).

    The length penalty in `shape` is shipyard's addition, not the paper's, and is off by
    default."""

    loss_fn = "ppo"
    #: Appendix G (Table 6): betas 0.9 / 0.95, weight decay 0, gradient norm clipped at 1.0,
    #: constant learning rate. Eps is unstated, so the cookbook default applies.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=1.0)

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        found = [
            "advantage = group mean, not divided by spread",
            f"loss = ppo, clip {recipe.clip} / {recipe.clip}, {cls.aggregated()}",
        ]
        if recipe.length_penalty > 0:
            cap = recipe.length_cap
            capped = "uncapped" if cap == float("inf") else f"capped at {cap}"
            found.append(
                f"length penalty {recipe.length_penalty} over {recipe.length_floor} tokens "
                f"among solved answers, {capped}"
            )
        return found

    def shape(self, group: Group) -> list[float | None]:
        """Return the rewards after the length penalty (see `credit.shaped`), or None for an
        unmeasured member."""
        recipe = self.recipe
        measured = group.measured
        found = iter(
            shaped(
                [float(one.verdict.reward) for one in measured],
                [one.sampled_tokens for one in measured],
                penalty=recipe.length_penalty,
                floor=recipe.length_floor,
                cap=recipe.length_cap,
            )
        )
        return [next(found) if one.measured else None for one in group.members]

    def bounds(self) -> dict[str, float] | None:
        return clipped(self.recipe.clip, self.recipe.clip)
