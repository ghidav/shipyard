"""Rejection-sampling fine-tuning (RFT; Yuan et al., arXiv 2308.01825) with the sampling
overlapping the updates: the agent is trained by cross-entropy on correct rollouts, and on nothing
else. Workers keep `in_flight` groups rolling out, `group_size` trials of a task each as a job of
its own, from `[models.teacher]` when run.toml names one and else from the agent itself. Each step
takes the groups in the order they finished until `[data] batch_size` of them hold a trial whose
reward reaches `threshold`, at most a pass over the tasks, and trains the agent on those trials,
up to `keep_per_group` of a group's, every kept token weighing the same, while the next ones
sample. A teacher must be served through the proxy and share the agent's tokenizer and renderer:
the agent trains on its token ids as the proxy recorded them."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from shipyard import Config, Group, Harness, Trainer, Trial


class RFTAsyncTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The harness the sampler runs through, and the trials per task in a group.
        harness: Harness
        group_size: int
        #: The reward a trial must reach to be trained on.
        threshold: float = 1.0
        #: The most correct trials a group gives the update; unset, every one.
        keep_per_group: int | None = Field(default=None, ge=1)
        #: Groups rolling out at once; unset, half again the groups `[rollout] sandboxes` holds,
        #: so the slots a group's finished trials free never wait on its last.
        in_flight: int | None = Field(default=None, ge=1)

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]
        self.agent.training = True
        #: The model whose correct rollouts the agent learns from: the teacher when run.toml
        #: names one, else the agent itself.
        self.sampler = self.models["teacher"] if "teacher" in self.models else self.agent
        #: The groups the workers finished, in the order they finished, until a step takes them.
        self.finished: asyncio.Queue[Group] | None = None

    async def step(self) -> None:
        if self.finished is None:
            # At most a batch waits, so a worker holds back while one does.
            self.finished = asyncio.Queue(maxsize=self.config.data.batch_size)
            for _ in range(self.in_flight):
                self.spawn(self.work(self.finished))
        taken = await self.take(self.finished)
        kept = self.score(taken)
        self.metrics.update(kept=len(kept), kept_tokens=_tokens(kept))
        # Every group taken goes to update, which trains the kept trials: the step's row counts
        # the others too.
        await self.update(self.agent, taken, loss="cross_entropy")

    @property
    def in_flight(self) -> int:
        """Groups rolling out at once: `in_flight`, else half again the groups the sandboxes
        hold."""
        held = self.config.rollout.alive // self.recipe.group_size
        return self.recipe.in_flight or max(1, held * 3 // 2)

    async def work(self, finished: asyncio.Queue[Group]) -> None:
        """Roll the next task out `group_size` times from the sampler, as a job of its own, and
        hand its group over once its last trial ends; again."""
        while True:
            (group,) = await self.rollout(
                self.draw(1),
                model=self.sampler,
                harness=self.recipe.harness,
                attempts=self.recipe.group_size,
                modules=self.state.modules,
            )
            await finished.put(group)

    def correct(self, group: Group) -> list[Trial]:
        """Return the group's graded trials whose reward reaches `threshold`, `keep_per_group`
        of them at most."""
        found = [
            trial
            for trial, reward in zip(group.graded, group.rewards, strict=True)
            if reward >= self.recipe.threshold
        ]
        return found[: self.recipe.keep_per_group]

    async def take(self, finished: asyncio.Queue[Group]) -> list[Group]:
        """Take finished groups, in the order they finished, until `[data] batch_size` of them
        hold a correct trial, or a pass over the tasks' worth has been taken; return them all."""
        taken: list[Group] = []
        holding = 0
        while holding < self.config.data.batch_size and len(taken) < len(self.lender.train):
            group = await finished.get()
            taken.append(group)
            holding += bool(self.correct(group))
        return taken

    def score(self, groups: Sequence[Group]) -> list[Trial]:
        """Give every target token of the groups' correct trials the same weight, one over their
        tokens together, so the update's loss is their mean negative log-likelihood, and return
        the trials kept. Raise `RuntimeError` for correct trials that hold no tokens: their
        model's provider served them, so the proxy recorded nothing to train on."""
        kept = [trial for group in groups for trial in self.correct(group)]
        tokens = _tokens(kept)
        if kept and not tokens:
            raise RuntimeError(
                f"[models.{self.sampler.role}]'s correct trials hold no tokens to train on: its "
                "provider served it, out of the proxy's sight. Serve it on Tinker or Together"
            )
        for trial in kept:
            trial.set_advantage(1 / tokens)
        return kept

    async def evaluate(self, tasks: list[Path]) -> list[Group]:
        return await self.rollout(
            tasks,
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )


def _tokens(trials: Sequence[Trial]) -> int:
    """Return the target tokens of the trials' sequences, together."""
    return sum(len(one.targets) for trial in trials for one in trial.sequences)
