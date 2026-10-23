"""Tests of an update's substeps: the optimizer steps it takes on the groups it is given, as many
as the recipe says, and the rate it says, as DAPO does from its own `substeps`."""

from __future__ import annotations

from pathlib import Path

from shipyard.run.data import Generator, tasks
from tests.fakes import START, Alternating, FakeProxy, Harbor, Spread
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService
from tests.test_trainer import DAPO, RECIPES

#: The rate the recipe below trains at, over its table's 2e-5.
RATE = 1e-4


class Split(GRPO):
    """GRPO that splits each update into three substeps, at its own rate."""

    async def step(self) -> None:
        groups = await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=None,
        )
        await self.score(groups)
        await self.update(self.agent, groups, substeps=3, learning_rate=RATE)


async def test_an_update_takes_a_substep_for_each_share_of_its_scored_groups_at_the_rate_given(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    home = run_toml(tmp_path, **ONE_STEP, **{"batch_size = 2": "batch_size = 3"})
    plan = Generator(tasks("four", root=tmp_path / "tasks"), 3, 0)
    first = [one.name for one in next(plan.batches())]
    harbor(Spread(set(first[:2])))
    directory = await train(home, tmp_path / "runs", Split)
    calls = service.client.calls
    steps = [c[0] for c in calls if c[0] in ("forward_backward", "optim_step")]
    assert steps == ["forward_backward", "optim_step"] * 2, (
        "three substeps asked, two groups scored: the uniform one is left out before the split"
    )
    assert [len(c[1]) for c in calls if c[0] == "forward_backward"] == [2, 2], "a group each"
    assert [c[1].learning_rate for c in calls if c[0] == "optim_step"] == [RATE, RATE]
    assert set(proxy.weights) == {START}, "the step's trials sampled the starting weights"
    published = [c[1] for c in calls if c[0] == "save_weights_for_sampler" and c[1] != "final"]
    assert published == ["start", "sample-1"], "served once"
    (row,) = rows(directory)
    assert (row["substeps"], row["learning_rate"], row["uniform"]) == (2, RATE, 1)


async def test_dapo_takes_as_many_substeps_an_update_as_its_config_says(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(Alternating())
    four = {"substeps = 16": "substeps = 2", "batch_size = 2": "batch_size = 4"}
    home = run_toml(tmp_path, *DAPO, **ONE_STEP, **four)
    directory = await train(home, tmp_path / "runs", RECIPES / "dapo/trainer.py")
    calls = service.client.calls
    assert [len(c[1]) for c in calls if c[0] == "forward_backward"] == [4, 4], (
        "four groups of two, in two substeps of two groups"
    )
    assert sum(c[0] == "optim_step" for c in calls) == 2
    (row,) = rows(directory)
    assert (row["groups"], row["substeps"]) == (4, 2)
