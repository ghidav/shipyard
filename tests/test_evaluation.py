"""Tests for how the test set is rolled out: every harness on its own, even in a run that mixes
them, through the recipe's own `rollout`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from shipyard import Group, Trainer
from shipyard.jobs.engine import EVALUATION
from shipyard.run import files
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import ONE_STEP, blueprint, rows, train
from tests.service import FakeService
from tests.test_renderers import EFFORTS, HIGH, LOW
from tests.test_rollout import TWO, Answered, _engine, _tasks


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor's own check of each harness's config, which is not what these tests are about."""
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)


async def test_an_evaluation_runs_each_harness_on_its_own_where_training_mixes_them(
    tmp_path: Path,
) -> None:
    beta, alpha = _tasks(tmp_path, "beta", "alpha")
    engine, _, _ = _engine(tmp_path, mix=True)

    async def rolled(purpose: str) -> list[tuple[str, str | None, int]]:
        groups = await engine.rollout(
            [beta, alpha],
            attempts=2,
            harnesses=TWO,
            renderers=[None],
            endpoint=Answered(),
            purpose=purpose,
        )
        return [(group.task.name, group.harness, len(group.trials)) for group in groups]

    assert await rolled(EVALUATION) == [
        ("beta", "pi", 2),
        ("beta", "claude-code", 2),
        ("alpha", "pi", 2),
        ("alpha", "claude-code", 2),
    ], "each harness runs every attempt, in a group of its own"
    assert await rolled("training") == [("beta", None, 2), ("alpha", None, 2)], "shared"


async def test_the_test_set_runs_at_every_harness_and_renderer_through_a_recipes_own_rollout(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """A recipe's `rollout` that takes neither `harness` nor `renderer` evaluates them all."""
    harbor(Alternating())

    class Plain(Trainer):
        async def rollout(  # type: ignore[override]
            self, tasks: Sequence[Path], *, attempts: int | None = None, purpose: str = "training"
        ) -> list[Group]:
            return await super().rollout(tasks, attempts=attempts, purpose=purpose)

    two = 'harness = ["pi@0.85.1", "claude-code@2.1.0"]\nmix_harnesses = true'
    with_test = {"group_size = 2": 'group_size = 2\ntest = "four"\ntest_group_size = 1'}
    home = blueprint(tmp_path, **EFFORTS, **ONE_STEP, **with_test, **{'harness = "pi@0.85.1"': two})
    directory = await train(home, tmp_path / "runs", Plain)
    tested = [
        (job["harness"], job["renderer"])
        for job in files.read(directory / files.JOBS)
        if job["purpose"] == EVALUATION
    ]
    pairs = [
        (harness, renderer)
        for harness in ("pi@0.85.1", "claude-code@2.1.0")
        for renderer in (LOW, HIGH)
    ]
    assert sorted(tested) == sorted(pairs * 2), "each test batch at each harness and renderer"
    (row,) = rows(directory)
    assert row["eval_trials"] == 4 * 2 * 2, "four tasks, once at each harness and renderer"
    for name in ("pi@0.85.1", "claude-code@2.1.0", LOW, HIGH):
        assert f"eval_reward_mean_{name}" in row
