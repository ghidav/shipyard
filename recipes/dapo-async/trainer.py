"""DAPO (Yu et al., arXiv 2503.14476) with the rollouts overlapping the updates: dapo's scoring,
in a loop of its own. Workers keep `in_flight` groups rolling out, `group_size` trials of a task
each as a job of its own, and hand each group over once its last trial ends. Each step takes the
groups in the order they finished until a batch's worth have rewards that differ (dynamic
sampling), at most a pass over the tasks, and updates on them while the next ones sample. Each
call takes the weights published last when it starts, so a group may straddle an update: one
more than `max_lag` updates behind is left out, and its task goes out again."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev

import tinker
from pydantic import Field

from shipyard import Config, Group, Harness, Trainer


class DAPOAsyncTrainer(Trainer):
    #: Weight decay and clipping from the authors' released recipe (verl recipe/dapo).
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0)

    class Recipe(Trainer.Recipe):
        #: The harness the agent runs through, and the trials per task in a group.
        harness: Harness
        group_size: int
        #: Optimizer steps each update takes, on a share of the groups each.
        substeps: int
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
        #: Groups rolling out at once; unset, half again the groups `[rollout] sandboxes` holds,
        #: so the slots a group's finished trials free never wait on its last.
        in_flight: int | None = Field(default=None, ge=1)
        #: The most updates a group may lag the weights served and be trained on; unset, any.
        max_lag: int | None = Field(default=None, ge=0)

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]
        self.agent.training = True
        #: The groups the workers finished, in the order they finished, until a step takes them.
        self.finished: asyncio.Queue[Group] | None = None

    async def step(self) -> None:
        if self.finished is None:
            # At most a batch waits, so a worker holds back while one does: that bounds the lag.
            self.finished = asyncio.Queue(maxsize=self.config.data.batch_size)
            for _ in range(self.in_flight):
                self.spawn(self.work(self.finished))
        taken, batch, stale = await self.take(self.finished)
        lags = [self.agent.lag(group) for group in batch]
        await self.score(batch)
        active = [g for g in batch if g.is_scored]
        # Every prompt's tokens weigh the same in all (Eq. 8), where Tinker sums them.
        sizes = [sum(len(s.targets) for s in g.scored) for g in active]
        for group, size in zip(active, sizes, strict=True):
            for s in group.scored:
                s.advantages *= fmean(sizes) / size
        self.metrics.update(
            lag_mean=fmean(lags) if lags else 0.0, lag_max=max(lags, default=0), stale=stale
        )
        low, high = self.recipe.clip_low, self.recipe.clip_high
        clipped = {"clip_low_threshold": 1 - low, "clip_high_threshold": 1 + high}
        # Every group taken goes to update, which trains the scored ones: the step's row counts
        # the groups dropped for equal rewards or lag too.
        await self.update(
            self.agent,
            taken,
            substeps=self.recipe.substeps,
            loss="ppo",
            loss_config=clipped,
        )

    @property
    def in_flight(self) -> int:
        """Groups rolling out at once: `in_flight`, else half again the groups the sandboxes
        hold."""
        held = self.config.rollout.alive // self.recipe.group_size
        return self.recipe.in_flight or max(1, held * 3 // 2)

    async def work(self, finished: asyncio.Queue[Group]) -> None:
        """Roll the next task out `group_size` times, as a job of its own, and hand its group
        over once its last trial ends; again."""
        while True:
            (group,) = await self.rollout(
                self.draw(1),
                model=self.agent,
                harness=self.recipe.harness,
                attempts=self.recipe.group_size,
                modules=self.state.modules,
            )
            await finished.put(group)

    async def take(self, finished: asyncio.Queue[Group]) -> tuple[list[Group], list[Group], int]:
        """Take finished groups, in the order they finished, until `[data] batch_size` of them
        have rewards that differ, or a pass over the tasks' worth has been taken. Return every
        group taken, the ones to train on, and how many lagged more than `max_lag`, whose tasks
        go out again."""
        taken: list[Group] = []
        batch: list[Group] = []
        stale = 0
        while len(batch) < self.config.data.batch_size and len(taken) < len(self.lender.train):
            group = await finished.get()
            taken.append(group)
            if self.recipe.max_lag is not None and self.agent.lag(group) > self.recipe.max_lag:
                self.lender.back(group.task)
                stale += 1
            elif group.reward_variance > 0:
                batch.append(group)
        return taken, batch, stale

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

    async def evaluate(self, tasks: list[Path]) -> list[Group]:
        return await self.rollout(
            tasks,
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )
