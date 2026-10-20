"""Dr. GRPO (Liu et al., arXiv 2503.20783), GRPO without its two biases. Each rollout's
advantage is its reward minus its group's mean, not divided by the group's spread, and the loss
is `ppo`, clipped the same on both sides and summed over the tokens without dividing by each
answer's length. Tinker sums the token losses, so that sum is what `update` already does.

It also takes a length penalty that is shipyard's addition, not the paper's: a solved rollout
loses `length_penalty` times its tokens past `length_floor` over the mean length of its group's
solved rollouts, at most `length_cap`; it needs two solved rollouts in the group to compare."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pvariance

import tinker

from shipyard import Group, Trainer


class DrGRPOTrainer(Trainer):
    #: Appendix G (Table 6): betas 0.9 / 0.95, no weight decay, the gradient norm clipped at
    #: 1.0. Eps is unstated, so the cookbook's applies.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=1.0)

    class Recipe(Trainer.Recipe):
        #: How far from 1 a token's ratio may go, on either side, before `ppo` clips it.
        clip: float
        #: Shipyard's length penalty on solved rollouts; 0 turns it off.
        length_penalty: float
        length_floor: int
        length_cap: float

    async def step(self) -> None:
        groups = await self.rollout(self.draw())
        await self.score(groups)
        clip = self.recipe.clip
        clipped = {"clip_low_threshold": 1 - clip, "clip_high_threshold": 1 + clip}
        await self.update(groups, loss="ppo", loss_config=clipped)

    async def score(self, groups: Sequence[Group]) -> None:
        for group in groups:
            shaped = self.shaped(group)
            # A group whose rewards, penalty included, are all equal has nothing to teach.
            if len(shaped) < 2 or pvariance(shaped) == 0:
                continue
            mean = fmean(shaped)
            for t, r in zip(group.measured, shaped, strict=True):
                t.set_advantage(r - mean)

    def shaped(self, group: Group) -> list[float]:
        """Return the group's rewards, each solved rollout's less its length penalty."""
        rewards, penalty = group.rewards, self.recipe.length_penalty
        solved = [r >= 1.0 for r in rewards]
        if not penalty or sum(solved) < 2:
            return rewards
        lengths = [t.sampled_tokens for t in group.measured]
        mean = fmean(n for n, yes in zip(lengths, solved, strict=True) if yes)
        if mean <= 0:
            return rewards
        floor, cap = self.recipe.length_floor, self.recipe.length_cap
        return [
            r - min(penalty * max(n - floor, 0) / mean, cap) if yes else r
            for r, n, yes in zip(rewards, lengths, solved, strict=True)
        ]
