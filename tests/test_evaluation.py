"""Tests for evaluation: `evaluate` is a hook the recipe writes, which rolls the test set out as
the recipe measures it. The run calls it on schedule from the start, beside the next step's
update, which serves its new weights once the evaluation has finished, and after the last step;
each evaluation's row holds what its groups came to as `eval_` fields, at the step whose weights
it measured, apart from the training rows."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from shipyard import Config, Group, Trainer
from shipyard.models.backends import Session
from shipyard.run import files
from shipyard.trainer import EVALUATING
from tests.fakes import Alternating, FakeProxy, FakeTrials, Harbor
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService

#: `[data]` with the four tasks as the test set too.
TESTED = {'train = "four"': 'train = "four"\ntest = "four"'}


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor's own check of each harness's config, which is not what these tests are about."""
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)


async def test_evaluate_rolls_out_for_evaluation_and_its_groups_stay_out_of_the_training_row(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(FakeTrials())
    home = run_toml(tmp_path, "test_group_size = 1", **ONE_STEP, **TESTED)
    directory = await train(home, tmp_path / "runs")
    jobs = list(files.read(directory / files.JOBS))
    assert [job["purpose"] for job in jobs] == ["training", "evaluation"], "by default"
    training, evaluation = jobs
    assert (evaluation["tasks"], evaluation["trials"], evaluation["step"]) == (4, 4, 1)
    row, evaluated = rows(directory)
    assert (row["groups"], row["trials"]) == (2, 4), "the step's two tasks, twice each"
    assert row["jobs"] == [training["job"]], "the evaluation's job is not the step's"
    assert not any(key.startswith("eval_") for key in row), "the evaluation's row is its own"
    assert {key: value for key, value in evaluated.items() if key != "at"} == {
        "step": 1,
        "eval_tasks": 4,
        "eval_trials": 4,
        "eval_graded": 4,
        "eval_masked": 0,
        "eval_reward_mean": 1.0,
    }


async def test_the_run_evaluates_from_the_start_every_evaluate_every_steps_and_at_its_last(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(FakeTrials())
    three = {"steps = 2": "steps = 3"}
    home = run_toml(tmp_path, "evaluate_every = 2", **three, **TESTED)
    directory = await train(home, tmp_path / "runs")
    tested = [
        job["step"] for job in files.read(directory / files.JOBS) if job["purpose"] == "evaluation"
    ]
    assert tested == [0, 2, 3], "the model the run starts from, every second step, and the last"
    assert [(row["step"], "eval_trials" in row) for row in rows(directory)] == [
        (0, True),
        (1, False),
        (2, False),
        (2, True),
        (3, False),
        (3, True),
    ], "each evaluation's row after the row of the step whose weights it measured"
    home = run_toml(tmp_path, **TESTED)
    once = await train(home, tmp_path / "runs")
    assert [(row["step"], "eval_trials" in row) for row in rows(once)] == [
        (1, False),
        (2, False),
        (2, True),
    ], "with no evaluate_every, the last step alone"


async def test_an_evaluation_runs_beside_the_next_update_which_serves_once_it_has_finished(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The samplers serve a step's weights until the next update serves its own: an evaluation
    of them runs while that update trains, and the new weights wait for it to finish."""
    harbor(Alternating())
    events: list[str] = []
    started = asyncio.Event()
    fit, serve = Session.fit, Session.serve

    async def fitting(self: Session, *args: Any, **kwargs: Any) -> Any:
        await asyncio.wait_for(started.wait(), 5)  # the evaluation is under way as it trains
        events.append("trained")
        return await fit(self, *args, **kwargs)

    async def serving(self: Session, tag: str) -> str | None:
        events.append(f"served {tag}")
        return await serve(self, tag)

    class Watched(GRPO):
        async def evaluate(self, tasks: list[Path]) -> list[Group]:
            step = EVALUATING.get()
            started.set()
            found = await super().evaluate(tasks)
            await asyncio.sleep(0.2)  # longer than the update beside it
            events.append(f"evaluated {step}")
            return found

    monkeypatch.setattr(Session, "fit", fitting)
    monkeypatch.setattr(Session, "serve", serving)
    home = run_toml(tmp_path, "evaluate_every = 1", "test_group_size = 1", **ONE_STEP, **TESTED)
    directory = await train(home, tmp_path / "runs", Watched)
    assert events == ["served start", "trained", "evaluated 0", "served sample-1", "evaluated 1"]
    jobs = [(job["purpose"], job["step"]) for job in files.read(directory / files.JOBS)]
    assert jobs == [("training", 1), ("evaluation", 0), ("evaluation", 1)]
    assert [(row["step"], "eval_trials" in row) for row in rows(directory)] == [
        (0, True),
        (1, False),
        (1, True),
    ]
    (updated,) = [row for row in rows(directory) if "update_seconds" in row]
    assert updated["update_seconds"] < 0.2, "training and serving, not the wait between them"


async def test_a_test_set_with_no_evaluate_is_refused_before_anything_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(FakeTrials())

    class Unmeasured(Trainer):
        Recipe = GRPO.Recipe

        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.agent = self.models["agent"]

        async def step(self) -> None:
            raise AssertionError("no step is taken")

    with pytest.raises(ValueError, match=r"\[data\] test: Unmeasured writes no `evaluate`"):
        await train(run_toml(tmp_path, **TESTED), tmp_path / "runs", Unmeasured)
    assert proxy.placed == [] and trials.configs == []


async def test_evaluate_rolls_out_as_the_recipe_measures_and_every_group_it_returns_counts(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """A recipe that measures each of two harnesses gathers a rollout of each; the row tallies
    every group `evaluate` returns, with no breakdown by harness."""
    harbor(FakeTrials())

    class Both(GRPO):
        async def evaluate(self, tasks: list[Path]) -> list[Group]:
            both = await asyncio.gather(
                *(
                    self.rollout(tasks, model=self.agent, harness=one, attempts=1, modules=None)
                    for one in ("pi@0.85.1", "claude-code@2.1.0")
                )
            )
            return [group for groups in both for group in groups]

    directory = await train(run_toml(tmp_path, **ONE_STEP, **TESTED), tmp_path / "runs", Both)
    tested = [
        job["harness"]
        for job in files.read(directory / files.JOBS)
        if job["purpose"] == "evaluation"
    ]
    assert sorted(tested) == ["claude-code@2.1.0", "pi@0.85.1"], "both for evaluation"
    _, row = rows(directory)
    assert (row["eval_tasks"], row["eval_trials"], row["eval_graded"]) == (4, 8, 8)
    assert not any("pi" in key or "claude" in key for key in row), "no breakdown by harness"
