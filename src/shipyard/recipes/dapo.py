"""DAPO (Yu et al., arXiv 2503.14476), with dynamic sampling as accumulation across jobs."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from collections.abc import Sequence as Values
from dataclasses import replace
from typing import Any

from shipyard.credit import overlong, weights
from shipyard.training.trainer import COOKBOOK, Trainer, centred
from shipyard.types import Agent, Batches, Group, Update

logger = logging.getLogger(__name__)


def clipped(low: float, high: float) -> dict[str, float]:
    """`loss_fn_config` for a ratio clipped to `[1 - low, 1 + high]`; DAPO's are epsilons."""
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
    """The overlong term's clause of the resolution line; none when the term is off."""
    if recipe.overlong_penalty <= 0:
        return []
    return [
        f"overlong penalty up to {recipe.overlong_penalty} over the last "
        f"{recipe.overlong_buffer * 100:g}% of the token budget"
    ]


class DAPOTrainer(Trainer):
    """DAPO (Yu et al., arXiv 2503.14476), with dynamic sampling as accumulation across jobs."""

    loss_fn = "ppo"
    #: 4.1 states no betas, eps, weight decay or gradient clipping. The authors' released
    #: recipe (verl recipe/dapo, run_dapo_qwen2.5_32b.sh) sets weight decay 0.1 and clips
    #: the gradient norm at 1.0. Betas and eps are the cookbook's.
    ADAM = replace(COOKBOOK, weight_decay=0.1, grad_clip_norm=1.0)
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

    async def train(self, agent: Agent, tasks: Batches) -> Agent:
        recipe = self.recipe
        if self.steps == 0 and recipe.overlong_penalty > 0 and agent.policy.endpoint.budget is None:
            logger.warning(
                "%s's overlong term is off for this run: the proxy reports no token budget, "
                "so overlong_penalty = %s docks no rollout",
                recipe.kind,
                recipe.overlong_penalty,
            )
        return await super().train(agent, tasks)

    async def sample(self, agent: Agent, tasks: Batches) -> AsyncIterator[list[Group]]:
        """Jobs are held until a batch's worth of groups have spread; the epoch bounds it."""
        target = self.run.config.data.batch_size * len(agent.harness)
        held: list[Group] = []
        for batch in tasks:
            held += await self.job(agent, batch)
            if sum(group.reward_variance > 0 for group in held) >= target:
                yield held
                held = []
        if held:
            yield held

    def advantage(self, group: Group) -> list[float | None]:
        """The overlong term, then the rewards centred and divided by the spread."""
        terms = self.docked(group)
        return centred(
            [
                None if reward is None else reward + term
                for reward, term in zip(group.shaped, terms, strict=True)
            ],
            normalize=True,
        )

    def docked(self, group: Group) -> list[float | None]:
        """The overlong term per member against the token budget; None on an unmeasured one."""
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
        """Each prompt weighs the same (Eq. 8): the mean group size over the group's own."""
        return weights(sizes)

    def bounds(self) -> dict[str, float] | None:
        return clipped(self.recipe.clip_low, self.recipe.clip_high)

    def log(self, found: Values[Group], update: Update | None, **more: Any) -> None:
        """The row with `overlong`, the docked members, when the term is on and the budget known."""
        budget = found[0].agent.policy.endpoint.budget if found else None
        if self.recipe.overlong_penalty > 0 and budget:
            more["overlong"] = sum(
                term is not None and term < 0 for group in found for term in self.docked(group)
            )
        super().log(found, update, **more)
