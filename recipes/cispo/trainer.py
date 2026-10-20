"""CISPO (MiniMax-M1, arXiv 2506.13585): DAPO's rules with Tinker's `cispo` loss. Instead of
clipping each token's update, CISPO truncates its importance weight and stops the gradient
through it, so every token keeps contributing. Each rollout's reward, less DAPO's soft overlong
penalty (Eq. 13), is normalized within its group; every prompt weighs the same; and a step keeps
drawing batches until a batch's worth of groups have something to teach (dynamic sampling).

The DAPO part is ../dapo/trainer.py's, copied, so this file runs on its own."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pstdev

import tinker

from shipyard import Group, Trainer


class CISPOTrainer(Trainer):
    #: Section 3.2 sets the betas and an eps for gradients mostly below 1e-14. No weight decay,
    #: gradient clipping or warm-up is stated, so none is used.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-15)

    class Recipe(Trainer.Recipe):
        #: Epsilon above 1 at which a token's importance weight is truncated. MiniMax-M1 tunes
        #: only this bound and leaves the lower one off.
        clip_high: float
        #: DAPO's overlong penalty at the token budget (Eq. 13), and the last share of the
        #: budget over which it grows to it.
        overlong_penalty: float
        overlong_buffer: float
        #: Tokens a rollout should stay under, its model calls together. Unset, the proxy's
        #: per-rollout budget, which is the model's context length.
        overlong_budget: int | None = None

    async def step(self) -> None:
        active: list[Group] = []
        for _ in range(self.generator.per_pass):  # at most one pass over the tasks
            groups = await self.rollout(self.draw())
            await self.score(groups)
            active += [g for g in groups if g.is_scored]
            if len(active) >= self.config.data.batch_size:
                break
        # Every prompt's tokens weigh the same in all (DAPO's Eq. 8), where Tinker sums them.
        sizes = [sum(len(s.targets) for s in g.scored) for g in active]
        for group, size in zip(active, sizes, strict=True):
            for s in group.scored:
                s.advantages *= fmean(sizes) / size
        # No lower bound: MiniMax-M1 sets epsilon_low large and tunes only epsilon_high.
        high = 1 + self.recipe.clip_high
        truncated = {"clip_low_threshold": 0.0, "clip_high_threshold": high}
        await self.update(active, loss="cispo", loss_config=truncated)

    async def score(self, groups: Sequence[Group]) -> None:
        for group in groups:
            # DAPO keeps a prompt by its accuracy (Eq. 11): the penalty alone would train on
            # length.
            if group.reward_variance == 0:
                continue
            shaped = [
                r + self.overlong(t.sampled_tokens)
                for t, r in zip(group.measured, group.rewards, strict=True)
            ]
            mean, spread = fmean(shaped), pstdev(shaped)
            for t, r in zip(group.measured, shaped, strict=True):
                t.set_advantage((r - mean) / (spread + 1e-6))

    def overlong(self, tokens: int) -> float:
        """Return the soft overlong punishment (DAPO's Eq. 13) of a rollout that sampled
        `tokens`: nothing until the last `overlong_buffer` of the token budget, `overlong_budget`,
        then down linearly to -`overlong_penalty` at the budget."""
        penalty = self.recipe.overlong_penalty
        budget = self.recipe.overlong_budget or self.policy.budget
        if not penalty or not budget:
            return 0.0
        cache = self.recipe.overlong_buffer * budget
        return -penalty * min(max(tokens - (budget - cache), 0) / cache, 1.0)
