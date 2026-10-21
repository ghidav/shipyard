"""On-policy distillation (Agarwal et al., arXiv 2306.13649): each token's advantage is how much
more likely the teacher finds it than the agent does. The rewards are logged, not trained on."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from shipyard import Blueprint, Group, Model, Trainer


class OPDTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The teacher: a model on Tinker that shares the agent's tokenizer, or a checkpoint.
        teacher: str

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.teacher = Model(self.recipe.teacher)

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
