"""On-policy distillation (Agarwal et al., arXiv 2306.13649): each token's advantage is how much
more likely the teacher finds it than the agent does. The rewards are logged, not trained on."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from shipyard import Config, Group, Harness, Trainer


class OPDTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The harness the agent runs through, and the trials per task in a rollout.
        harness: Harness
        group_size: int

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]
        self.agent.training = True
        #: A model on Tinker that reads the agent's token ids: it shares its tokenizer.
        self.teacher = self.models["teacher"]

    async def step(self) -> None:
        groups = await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )
        await self.score(groups)
        await self.update(self.agent, groups)

    async def score(self, groups: Sequence[Group]) -> None:
        seqs = [s for g in groups for t in g.graded for s in t.sequences]
        if not seqs:
            return
        ours, theirs = await asyncio.gather(self.agent.logprobs(seqs), self.teacher.logprobs(seqs))
        for seq, ref, taught in zip(seqs, ours, theirs, strict=True):
            seq.advantages = taught - ref
        self.metrics["teacher_kl"] = fmean(
            float((r - t).mean()) for r, t in zip(ours, theirs, strict=True)
        )

    async def evaluate(self, tasks: list[Path]) -> list[Group]:
        return await self.rollout(
            tasks,
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )
