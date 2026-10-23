"""Tests for `Trainer.spawn`: a task the recipe starts beside its steps, which the run owns. One
that fails fails the run at once, wherever the step under way waits; the run's end stops the
ones still running, closing their trials and leaving the tasks they drew out, for a resume."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import Config, Group
from shipyard.run.checkpoint import found
from shipyard.run.data import Lender, tasks
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService


class Stalled(Alternating):
    """Fake trial runner that rewards as `Alternating` does, but runs a trial of `stalled` until
    it is cancelled."""

    def __init__(self, stalled: str) -> None:
        super().__init__()
        self.stalled = stalled

    async def __call__(self, config: TrialConfig) -> None:
        if Path(str(config.task.path)).name == self.stalled:
            await asyncio.Event().wait()
        await super().__call__(config)


async def test_a_spawned_task_that_raises_fails_the_run_at_once_wherever_its_step_waits(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(Alternating())

    class Waiting(GRPO):
        async def step(self) -> None:
            ready: asyncio.Queue[list[Group]] = asyncio.Queue()

            async def worker() -> None:
                await self.rollout(
                    self.draw(),
                    model=self.agent,
                    harness=self.recipe.harness,
                    attempts=1,
                    modules=None,
                )
                raise RuntimeError("the worker broke")

            self.spawn(worker())
            await ready.get()  # the worker fails before it hands anything over

    with pytest.raises(RuntimeError, match="the worker broke"):
        await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Waiting)
    assert service.closed == [("errored", "RuntimeError: the worker broke")]
    (directory,) = (tmp_path / "runs").iterdir()
    with pytest.raises(ValueError, match="has no checkpoint"):
        found(directory)


async def test_the_runs_end_stops_a_spawned_rollout_closing_its_trial_and_leaving_its_task_out(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    home = run_toml(tmp_path, **ONE_STEP)
    stalled = Lender(tasks("four", root=tmp_path / "tasks"), 2, 0).draw(1)[0].name
    harbor(Stalled(stalled))

    class Lingering(GRPO):
        async def step(self) -> None:
            self.spawn(
                self.rollout(
                    self.draw(1),
                    model=self.agent,
                    harness=self.recipe.harness,
                    attempts=1,
                    modules=None,
                )
            )
            await super().step()

    directory = await train(home, tmp_path / "runs", Lingering)
    assert any(name.startswith(f"{stalled}__") for name in proxy.closed), "its trial closed"
    (row,) = rows(directory)
    assert row["trained"] and row["groups"] == 2, "the step trained its own batch"
    final = found(directory)
    assert final.tag == "final"
    out = final.saved_state(directory)["lender"]["out"]
    assert [Path(one).name for one in out] == [stalled], "out, for a resume to give first"


async def test_a_step_awaits_what_it_spawned_and_trains_on_it(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())

    class Awaiting(GRPO):
        async def step(self) -> None:
            groups = await self.spawn(
                self.rollout(
                    self.draw(),
                    model=self.agent,
                    harness=self.recipe.harness,
                    attempts=2,
                    modules=None,
                )
            )
            await self.score(groups)
            await self.update(self.agent, groups)

    directory = await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Awaiting)
    (row,) = rows(directory)
    assert row["trained"] and (row["groups"], row["trials"]) == (2, 4)


async def test_a_spawned_task_that_fails_as_the_run_stops_it_is_logged_and_the_run_ends_well(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harbor(Alternating())

    class Untidy(GRPO):
        async def step(self) -> None:
            async def worker() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise ValueError("its clean-up broke") from None

            self.spawn(worker())
            await super().step()

    directory = await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Untidy)
    assert "its clean-up broke" in caplog.text
    assert found(directory).tag == "final", "the run ended as it would have"


async def test_spawning_before_the_run_starts_is_refused(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())

    class Early(GRPO):
        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.spawn(asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="spawn from `step`"):
        await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Early)
