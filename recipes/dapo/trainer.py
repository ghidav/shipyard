"""DAPO (Yu et al., arXiv 2503.14476). Each trial's reward, less a soft penalty for running
into the end of its token budget (Eq. 13), is normalized within its group; the loss is `ppo`
clipped asymmetrically and averaged per prompt; and a step keeps drawing batches until a
batch's worth of groups have something to teach (dynamic sampling)."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pstdev

import tinker

from shipyard import Group, Trainer


class DAPOTrainer(Trainer):
    #: Weight decay and clipping from the authors' released recipe (verl recipe/dapo).
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0)

    class Recipe(Trainer.Recipe):
        #: How far below and above 1 a token's ratio may go before `ppo` clips it.
        clip_low: float
        clip_high: float
        #: The overlong penalty at the token budget (Eq. 13), and the last share of the budget
        #: over which it grows to it.
        overlong_penalty: float
        overlong_buffer: float
        #: Tokens a trial should stay under, its model calls together. Unset, the proxy's
        #: per-trial budget, which is the model's context length.
        overlong_budget: int | None = None

    async def step(self) -> None:
        active: list[Group] = []
        for _ in range(self.generator.per_pass):  # at most one pass over the tasks
            groups = await self.rollout(self.draw())
            await self.score(groups)
            active += [g for g in groups if g.is_scored]
            if len(active) >= self.config.data.batch_size:
                break
        # Every prompt's tokens weigh the same in all (Eq. 8), where Tinker sums them.
        sizes = [sum(len(s.targets) for s in g.scored) for g in active]
        for group, size in zip(active, sizes, strict=True):
            for s in group.scored:
                s.advantages *= fmean(sizes) / size
        low, high = self.recipe.clip_low, self.recipe.clip_high
        clipped = {"clip_low_threshold": 1 - low, "clip_high_threshold": 1 + high}
        await self.update(active, loss="ppo", loss_config=clipped)

    async def score(self, groups: Sequence[Group]) -> None:
        for group in groups:
            # DAPO keeps a prompt by its accuracy (Eq. 11): the penalty alone would train on
            # length.
            if group.reward_variance == 0:
                continue
            shaped = [
                r + self.overlong(t.sampled_tokens)
                for t, r in zip(group.graded, group.rewards, strict=True)
            ]
            mean, spread = fmean(shaped), pstdev(shaped)
            for t, r in zip(group.graded, shaped, strict=True):
                t.set_advantage((r - mean) / (spread + 1e-6))

    def overlong(self, tokens: int) -> float:
        """Return the soft overlong punishment (Eq. 13) of a trial that sampled `tokens`:
        nothing until the last `overlong_buffer` of the token budget, `overlong_budget`, then
        down linearly to -`overlong_penalty` at the budget."""
        penalty = self.recipe.overlong_penalty
        budget = self.recipe.overlong_budget or self.agent.budget
        if not penalty or not budget:
            return 0.0
        cache = self.recipe.overlong_buffer * budget
        return -penalty * min(max(tokens - (budget - cache), 0) / cache, 1.0)
