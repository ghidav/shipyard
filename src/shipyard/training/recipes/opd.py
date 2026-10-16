"""On-policy distillation (Agarwal et al., arXiv 2306.13649): a teacher scores the policy's
own tokens."""

from __future__ import annotations

from typing import Any

from shipyard.training.trainer import Trainer
from shipyard.types import Group


class OPDTrainer(Trainer):
    """Trainer for on-policy distillation, as tinker-cookbook's
    `distillation/train_on_policy.py` runs it: each token's advantage is `teacher_coef` times the
    teacher's logprob minus the reference's, with no reward term and no discount.

    The verifier still runs, so the step rows show whether the policy gets better at the
    tasks."""

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        return [
            "advantage = teacher - reference per token, no reward term",
            f"loss = {cls.loss_fn}, {cls.aggregated()}",
        ]

    def advantage(self, group: Group) -> list[float | None]:
        """Return 0 for every measured member: the reward is logged, not trained on."""
        return [0.0 if one.measured else None for one in group.members]
