"""The trainer's `job` reserves the job name and directory before anything runs, appends a
row for each job, and adds the job's counts to `costs.json`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shipyard import record
from shipyard.run import Run
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.sampling.engine import HarborEngine
from shipyard.training.trainer import Trainer
from shipyard.types import Agent, Settings, Trial
from tests.trials import BLUEPRINTS, FakeTrials, Harbor, fixture_tasks, job, job_of, trials_of

EVALUATE = BLUEPRINTS / "evaluate"


@pytest.fixture
def opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """An open evaluate run in a working directory holding the fixture tasks."""
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    return Run.open(EVALUATE, root=tmp_path / "runs")


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    return [tmp_path / "tasks" / "fixture" / name for name in names]


async def test_jobs_are_named_in_turn_and_reserved_by_directory_first(
    opened: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool, int, Agent]] = []

    async def fake_run(
        engine: HarborEngine,
        job: str,
        tasks: list[Path],
        attempts: int,
        *,
        agent: Agent,
        settings: Any = None,
    ) -> list[Trial]:
        seen.append((job, (tmp_path / "jobs" / job).is_dir(), attempts, agent))
        return [
            Trial(job, task, n, agent.harness[0], tmp_path / "jobs" / job / f"{task.name}__{n}")
            for task in tasks
            for n in range(attempts)
        ]

    monkeypatch.setattr(HarborEngine, "run", fake_run)
    first = await job(opened, _batch(tmp_path, "alpha", "beta"), rollouts=2, step=0)
    second = await job(opened, _batch(tmp_path, "alpha"), rollouts=1, step=1)
    assert job_of(first) == f"{opened.id}-0000" and job_of(second) == f"{opened.id}-0001"
    assert [(name, reserved) for name, reserved, *_ in seen] == [
        (job_of(first), True),
        (job_of(second), True),
    ]
    _, _, attempts, agent = seen[0]
    assert attempts == 2 and agent.modules is None
    assert [one.agent for one in agent.harness] == ["pi@0.85.1"]
    assert agent.policy.endpoint == ProviderEndpoint("some-hosted-model")
    engine = Trainer(opened).engine
    assert engine.sandbox == "modal" and engine.jobs_dir == Path("jobs")
    assert engine.settings == Settings() and engine.progress is opened.progress
    assert engine.run_trial is None
    rows = list(record.read(opened.directory / record.JOBS))
    assert [row["job"] for row in rows] == [job_of(first), job_of(second)]
    assert [row["step"] for row in rows] == [0, 1]
    # The counter was seeded when the run opened: a row that lands later does not move it.
    record.append(opened.directory / record.JOBS, {"job": "elsewhere"})
    third = await job(opened, _batch(tmp_path, "beta"), rollouts=1, step=1)
    assert job_of(third) == f"{opened.id}-0002" and (tmp_path / "jobs" / job_of(third)).is_dir()
    assert list(record.read(opened.directory / record.JOBS))[-1]["step"] == 1


def test_the_job_counter_is_seeded_from_the_record_and_the_disk(
    opened: Run, tmp_path: Path
) -> None:
    record.append(opened.directory / record.JOBS, {"job": f"{opened.id}-0003"})
    record.append(opened.directory / record.JOBS, {"job": "elsewhere"})
    (tmp_path / "jobs" / f"{opened.id}-0005").mkdir(parents=True)
    reopened = Run(opened.directory, opened.config)
    assert reopened.reserve_job() == f"{opened.id}-0006"
    assert Run(opened.directory, opened.config).reserve_job() == f"{opened.id}-0007"


async def test_a_job_name_already_taken_on_disk_is_refused(
    opened: Run, tmp_path: Path, harbor: Harbor
) -> None:
    taken = tmp_path / "jobs" / f"{opened.id}-0000"
    taken.mkdir(parents=True)
    harbor(FakeTrials())
    with pytest.raises(FileExistsError, match=f"{opened.id}-0000 is taken"):
        await job(opened, _batch(tmp_path, "alpha"), rollouts=1, step=0)
    assert not (opened.directory / record.JOBS).exists() and not any(taken.iterdir())


async def test_the_job_row_counts_verdicts_endings_tokens_and_seconds(
    opened: Run, tmp_path: Path, harbor: Harbor
) -> None:
    harbor(
        FakeTrials(
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
    )
    rolled = await job(opened, _batch(tmp_path, "alpha", "beta"), rollouts=2, step=0)
    assert len(trials_of(rolled)) == 4
    [row] = list(record.read(opened.directory / record.JOBS))
    assert row == {
        "at": row["at"],
        "job": f"{opened.id}-0000",
        "purpose": "training",
        "step": 0,
        "tasks": 2,
        "epoch": 0,
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
        "sandboxes": {"modal": {"trials": 4, "seconds": 80.0}},
    }
    assert opened.costs.to_dict() == costs


async def test_empty_maps_are_left_off_and_costs_add_up_across_jobs(
    opened: Run,
    tmp_path: Path,
    harbor: Harbor,
) -> None:
    harbor(FakeTrials(outcomes={"alpha": {"reward": 0.5}, "beta": {"reward": 1.0}}))
    await job(opened, _batch(tmp_path, "alpha", "beta"), rollouts=1, step=0)
    harbor(FakeTrials(breaks=("alpha",), outcomes={"beta": {"reward": 0.0}}))
    await job(opened, _batch(tmp_path, "alpha", "beta"), rollouts=1, step=1)
    first, second = list(record.read(opened.directory / record.JOBS))
    assert "masked" not in first and "ended" not in first and first["graded"] == 2
    assert second["masked"] == {"env_error": 1} and "ended" not in second
    assert second["graded"] == 1 and second["trials"] == 2 and second["step"] == 1
    # With no provider recorded, the blueprint names the party. Unreported tokens count as 0.
    # The party's trials and the sandbox's trials are those with a result.
    assert first["party"] == second["party"] == "openrouter"
    assert first["input_tokens"] == 0 and first["sandbox_seconds"] == 0.0
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs["parties"] == {
        "openrouter": {"trials": 3, "input_tokens": 0, "cache_tokens": 0, "output_tokens": 0}
    }
    assert costs["sandboxes"] == {"modal": {"trials": 3, "seconds": 0.0}}


async def test_jobs_dir_comes_from_the_blueprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
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
    harbor(FakeTrials())
    rolled = await job(opened, _batch(tmp_path, "alpha"), rollouts=1, step=0)
    assert trials_of(rolled)[0].parent == Path("elsewhere/jobs") / job_of(rolled)
    assert (tmp_path / "elsewhere" / "jobs" / job_of(rolled)).is_dir()
    assert not (tmp_path / "jobs").exists()
    # Shipyard writes nothing inside the job directory.
    assert sorted(p.name for p in (tmp_path / "elsewhere" / "jobs" / job_of(rolled)).iterdir()) == [
        trials_of(rolled)[0].name
    ]
