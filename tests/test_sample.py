"""`Run.sample`: the job reserved by name and directory before anything runs, the row it
appends for the job, and the counts it adds to `costs.json` after each one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shipyard import record
from shipyard.rollout import Rollouts
from shipyard.run import Run
from tests.trials import FakeTrials, fixture_tasks

EVALUATE = Path(__file__).parent / "blueprints" / "evaluate"


@pytest.fixture
def opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """An open evaluate run in a working directory holding the fixture tasks."""
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    return Run.open(EVALUATE, root=tmp_path / "runs")


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    return [tmp_path / "tasks" / "fixture" / name for name in names]


async def test_jobs_are_named_from_the_rows_and_reserved_by_directory_first(
    opened: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool, dict[str, Any]]] = []

    async def fake_rollout(batch: list[Path], job: str, **options: Any) -> Rollouts:
        seen.append((job, (tmp_path / "jobs" / job).is_dir(), options))
        plan = [(task, n) for task in batch for n in range(options["rollouts"])]
        trials = [tmp_path / "jobs" / job / f"{task.name}__{n}" for task, n in plan]
        return Rollouts(job=job, trials=trials, plan=plan)

    monkeypatch.setattr("shipyard.run.rollout", fake_rollout)
    first = await opened.sample(_batch(tmp_path, "alpha", "beta"), rollouts=2, index=0)
    second = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=1)
    assert first.job == f"{opened.id}-0000" and second.job == f"{opened.id}-0001"
    assert [(job, reserved) for job, reserved, _ in seen] == [(first.job, True), (second.job, True)]
    options = seen[0][2]
    assert options["harness"] == "pi@0.85.1" and options["model"] == "some-hosted-model"
    assert options["sandbox"] == "modal" and options["concurrency"] == 4
    assert options["jobs_dir"] == Path("jobs") and options["rollouts"] == 2
    assert options["env"] == {} and options["kwargs"] == {}
    assert options["setup_timeout"] is None and options["timeout"] is None
    assert options["run_trial"] is None and options["progress"] is opened.progress
    rows = list(record.read(opened.directory / record.JOBS))
    assert [row["job"] for row in rows] == [first.job, second.job]
    assert [row["batch"] for row in rows] == [0, 1]
    # A row already in `jobs.jsonl`, whoever wrote it, is counted; the batch is the caller's.
    record.append(opened.directory / record.JOBS, {"job": "elsewhere"})
    third = await opened.sample(_batch(tmp_path, "beta"), rollouts=1, index=1)
    assert third.job == f"{opened.id}-0003" and (tmp_path / "jobs" / third.job).is_dir()
    assert list(record.read(opened.directory / record.JOBS))[-1]["batch"] == 1


async def test_a_job_name_already_taken_on_disk_is_refused(opened: Run, tmp_path: Path) -> None:
    taken = tmp_path / "jobs" / f"{opened.id}-0000"
    taken.mkdir(parents=True)
    opened.run_trial = FakeTrials()
    with pytest.raises(FileExistsError, match=f"{opened.id}-0000 is taken"):
        await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=0)
    assert not (opened.directory / record.JOBS).exists() and not any(taken.iterdir())


async def test_the_job_row_counts_verdicts_endings_tokens_and_seconds(
    opened: Run, tmp_path: Path
) -> None:
    opened.run_trial = FakeTrials(
        outcomes={
            "alpha": {
                "reward": 1.0,
                "tokens": (100, 20, 10),
                "provider": "OpenRouter",
                "seconds": 30.0,
            },
            "beta": {"reward": 0.0, "exception": "AgentTimeoutError", "seconds": 10.0},
        }
    )
    rolled = await opened.sample(_batch(tmp_path, "alpha", "beta"), rollouts=2, index=0)
    assert len(rolled) == 4
    [row] = list(record.read(opened.directory / record.JOBS))
    assert row == {
        "at": row["at"],
        "job": f"{opened.id}-0000",
        "purpose": "rollout",
        "batch": 0,
        "tasks": 2,
        "trials": 4,
        "graded": 2,
        "masked": {"timeout": 2},
        "ended": {"AgentTimeoutError": 2},
        "served": "provider",
        "party": "openrouter",
        "input_tokens": 200,
        "cache_tokens": 40,
        "output_tokens": 20,
        "sandbox": "modal",
        "sandbox_seconds": 80.0,
    }
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs == {
        "parties": {
            "openrouter": {
                "trials": 4,
                "input_tokens": 200,
                "cache_tokens": 40,
                "output_tokens": 20,
            }
        },
        "sandbox": {"modal": {"trials": 4, "seconds": 80.0}},
    }
    assert opened.costs.to_dict() == costs


async def test_empty_maps_are_left_off_and_costs_add_up_across_jobs(
    opened: Run, tmp_path: Path
) -> None:
    opened.run_trial = FakeTrials(outcomes={"alpha": {"reward": 0.5}, "beta": {"reward": 1.0}})
    await opened.sample(_batch(tmp_path, "alpha", "beta"), rollouts=1, index=0)
    opened.run_trial = FakeTrials(breaks=("alpha",), outcomes={"beta": {"reward": 0.0}})
    await opened.sample(_batch(tmp_path, "alpha", "beta"), rollouts=1, index=1)
    first, second = list(record.read(opened.directory / record.JOBS))
    assert "masked" not in first and "ended" not in first and first["graded"] == 2
    assert second["masked"] == {"env_error": 1} and "ended" not in second
    assert second["graded"] == 1 and second["trials"] == 2 and second["batch"] == 1
    # No provider recorded, so the blueprint's names the party; nothing reported counts 0
    # tokens, and the party's trials are the ones with a result, as the sandbox's are.
    assert first["party"] == second["party"] == "openrouter"
    assert first["input_tokens"] == 0 and first["sandbox_seconds"] == 0.0
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs["parties"] == {
        "openrouter": {"trials": 3, "input_tokens": 0, "cache_tokens": 0, "output_tokens": 0}
    }
    assert costs["sandbox"] == {"modal": {"trials": 3, "seconds": 0.0}}


async def test_jobs_dir_comes_from_the_blueprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "bp"
    home.mkdir()
    text = (
        (EVALUATE / "run.toml")
        .read_text()
        .replace("[rollout]", '[rollout]\njobs_dir = "elsewhere/jobs"')
    )
    (home / "run.toml").write_text(text)
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = FakeTrials()
    rolled = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=0)
    assert rolled.trials[0].parent == Path("elsewhere/jobs") / rolled.job
    assert (tmp_path / "elsewhere" / "jobs" / rolled.job).is_dir()
    assert not (tmp_path / "jobs").exists()
    # Nothing of shipyard's is written inside the job directory.
    assert sorted(p.name for p in (tmp_path / "elsewhere" / "jobs" / rolled.job).iterdir()) == [
        rolled.trials[0].name
    ]
