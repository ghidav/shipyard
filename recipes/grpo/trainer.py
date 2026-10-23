"""GRPO (Shao et al., arXiv 2402.03300) on verifiable rewards: each trial's advantage is its
reward minus its group's mean, a group of equal rewards teaches nothing, and the loss is
importance sampling against the sampler's logprobs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from shipyard import Config, Group, Harness, Trainer


class GRPOTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The harness the agent runs through.
        harness: Harness
        #: Trials per task in a step's rollout, whose rewards are compared with each other.
        group_size: int
        #: Trials per test task; unset, `group_size`.
        test_group_size: int | None = None

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]
        self.agent.training = True

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
        """Give each trial its reward minus its group's mean, skipping groups of equal rewards."""
        for group in groups:
            if group.reward_variance == 0:
                continue
            for trial, advantage in zip(group.graded, group.centered_rewards, strict=True):
                trial.set_advantage(advantage)

    async def evaluate(self, tasks: list[Path]) -> list[Group]:
        return await self.rollout(
            tasks,
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.test_group_size or self.recipe.group_size,
            modules=self.state.modules,
        )
