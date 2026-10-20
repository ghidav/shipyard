"""On-policy distillation (Agarwal et al., arXiv 2306.13649): each token's advantage is how much
more likely the teacher finds it than the policy does, times `teacher_coef`. The rewards are
logged, not trained on."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from shipyard import Blueprint, Group, Policy, Trainer


class OPDTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The teacher: a model on Tinker that shares the agent's tokenizer, or a checkpoint.
        teacher: str
        #: What the teacher's gap on each token is scaled by.
        teacher_coef: float

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.teacher = Policy(self.recipe.teacher)

    async def score(self, groups: Sequence[Group]) -> None:
        seqs = [s for g in groups for t in g.measured for s in t.sequences]
        if not seqs:
            return
        ours, theirs = await asyncio.gather(self.policy.logprobs(seqs), self.teacher.logprobs(seqs))
        coef = self.recipe.teacher_coef
        for seq, ref, taught in zip(seqs, ours, theirs, strict=True):
            seq.advantages = coef * (taught - ref)
        self.metrics["teacher_kl"] = fmean(
            float((r - t).mean()) for r, t in zip(ours, theirs, strict=True)
        )
