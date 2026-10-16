"""DAPO recipe (Yu et al., arXiv 2503.14476)."""

from __future__ import annotations

import logging
from collections.abc import Sequence as Values
from typing import Any

from shipyard.training.credit import centred, overlong, weights
from shipyard.training.trainer import Trainer
from shipyard.types import Adam, Agent, Group, TrainingStep

logger = logging.getLogger(__name__)


def clipped(low: float, high: float) -> dict[str, float]:
    """Return the `loss_fn_config` that clips the importance ratio to `[1 - low, 1 + high]`.

    `low` and `high` are epsilons, as in DAPO; Tinker takes the bounds themselves."""
    # The route and the key names are the installed SDK's: tinker/types/
    # forward_backward_input.py:21 documents `loss_fn_config` as "Optional configuration
    # parameters for the loss function (e.g., PPO clip thresholds, DPO beta)", and
    # tinker/types/datum.py:32-33 names `clip_low_threshold` / `clip_high_threshold`.
    # That they are bounds is the losses page's `torch.clamp(prob_ratio,
    # clip_low_threshold, clip_high_threshold)` with the example `{"clip_low_threshold":
    # 0.9, "clip_high_threshold": 1.1}` (https://tinker-docs.thinkingmachines.ai/losses);
    # the installed SDK spells out neither that nor a default for `cispo`.
    return {"clip_low_threshold": round(1.0 - low, 6), "clip_high_threshold": round(1.0 + high, 6)}


def overlong_clause(recipe: Any) -> list[str]:
    """Return the resolution-line clause for the overlong penalty; empty when it is off."""
    if recipe.overlong_penalty <= 0:
        return []
    return [
        f"overlong penalty up to {recipe.overlong_penalty} over the last "
        f"{recipe.overlong_buffer * 100:g}% of the token budget"
    ]


class DAPOTrainer(Trainer):
    """Trainer for DAPO (Yu et al., arXiv 2503.14476).

    Dynamic sampling is implemented by accumulating jobs until enough groups have nonzero
    reward variance."""

    loss_fn = "ppo"
    #: Section 4.1 states no betas, eps, weight decay or gradient clipping. Weight decay 0.1
    #: and gradient-norm clipping at 1.0 come from the authors' released recipe (verl
    #: recipe/dapo, run_dapo_qwen2.5_32b.sh). Betas and eps are the cookbook defaults.
    ADAM = Adam(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0)
    PAPER_WARMUP = 20

    @classmethod
    def rules(cls, recipe: Any) -> list[str]:
        return [
            "advantage = group mean, divided by spread",
            f"loss = ppo, clip {recipe.clip_low} / {recipe.clip_high}, {cls.aggregated()}",
            *overlong_clause(recipe),
        ]

    @classmethod
    def aggregated(cls) -> str:
        return "averaged per prompt"

    async def sample(self, agent: Agent) -> list[Group]:
        """Run jobs until a batch's worth of groups have nonzero reward variance (DAPO's
        dynamic sampling), drawing at most one pass over the train tasks."""
        recipe = self.recipe
        if self.steps == 0 and recipe.overlong_penalty > 0 and agent.policy.endpoint.budget is None:
            logger.warning(
                "%s's overlong term is off for this run: the proxy reports no token budget, "
                "so overlong_penalty = %s docks no rollout",
                recipe.kind,
                recipe.overlong_penalty,
            )
        target = self.config.data.batch_size * len(agent.harness)
        held: list[Group] = []
        for _ in range(self.generator.per_pass):
            held += self.credited(await self.job(agent, self.draw()))
            if sum(self.trains_on(group) for group in held) >= target:
                break
        return held

    def advantage(self, group: Group) -> list[float | None]:
        """Add the overlong penalty to the shaped rewards, then center them and divide by the
        group's standard deviation."""
        terms = self.docked(group)
        return centred(
            [
                None if reward is None else reward + term
                for reward, term in zip(group.shaped, terms, strict=True)
            ],
            normalize=True,
        )

    def docked(self, group: Group) -> list[float | None]:
        """Return each member's overlong penalty against the token budget, or None for an
        unmeasured member."""
        recipe = self.recipe
        measured = group.measured
        terms = iter(
            overlong(
                [one.sampled_tokens for one in measured],
                penalty=recipe.overlong_penalty,
                buffer=recipe.overlong_buffer,
                budget=group.agent.policy.endpoint.budget,
            )
        )
        return [next(terms) if one.measured else None for one in group.members]

    def weights(self, sizes: Values[int]) -> list[float]:
        """Weight every prompt equally (Eq. 8): scale each group by the step's mean group size,
        in tokens, over its own."""
        return weights(sizes)

    def bounds(self) -> dict[str, float] | None:
        return clipped(self.recipe.clip_low, self.recipe.clip_high)

    def log(self, done: TrainingStep, **more: Any) -> None:
        """Log the step's row, adding `overlong` (the number of penalized members) when the
        penalty is on and the token budget is known."""
        found = done.groups
        budget = found[0].agent.policy.endpoint.budget if found else None
        if self.recipe.overlong_penalty > 0 and budget:
            more["overlong"] = sum(
                term is not None and term < 0 for group in found for term in self.docked(group)
            )
        super().log(done, **more)
