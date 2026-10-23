"""Tests for `shipyard.jobs.progress`: per-job trial bars on a terminal, silence otherwise,
and the events the engine reports to them."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from shipyard.config import Rollout
from shipyard.jobs.engine import HarborEngine, Job
from shipyard.jobs.harnesses import Harness
from shipyard.jobs.progress import QUIET, Bars
from shipyard.models.endpoints import ProviderEndpoint
from tests.fakes import FakeTrials


def test_the_bars_are_silent_off_a_terminal_and_under_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(QUIET, raising=False)
    piped = Console(file=io.StringIO(), force_terminal=False)
    assert Bars(piped).bars.disable
    terminal = Console(file=io.StringIO(), force_terminal=True)
    assert not Bars(terminal).bars.disable
    monkeypatch.setenv(QUIET, "1")
    assert Bars(terminal).bars.disable


def test_bars_count_finished_trials_per_job_and_show_the_masked_once_it_is_done() -> None:
    out = io.StringIO()
    bars = Bars(Console(file=out, force_terminal=True, width=120))
    bars.started("run-0000", trials=3)
    bars.finished("run-0000")
    bars.finished("run-0000")
    [task] = bars.bars.tasks
    assert task.completed == 2 and task.total == 3 and task.fields["masked"] == ""
    assert task.description == "run-0000"
    bars.finished("elsewhere")
    assert task.completed == 2, "a trial of a job with no bar advances none"
    bars.started("run-0001", trials=1)
    assert [t.description for t in bars.bars.tasks] == ["run-0000", "run-0001"]
    bars.done("run-0000", masked=1)
    assert task.fields["masked"] == "1 masked"
    bars.done("never-started", masked=0)
    bars.close()
    bars.close()
    assert "run-0000" in out.getvalue() and "1 masked" in out.getvalue()


class Heard:
    """Stand-in for `Bars` that keeps every event it is told, with the job each one is about."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int | None]] = []

    def started(self, job: str, *, trials: int) -> None:
        self.events.append(("started", job, trials))

    def finished(self, job: str) -> None:
        self.events.append(("finished", job, None))

    def done(self, job: str, *, masked: int) -> None:
        self.events.append(("done", job, masked))


async def test_the_engine_reports_each_trial_and_the_jobs_masked_count_once_judged(
    tmp_path: Path,
) -> None:
    tasks = [tmp_path / "tasks" / name for name in ("alpha", "beta", "gamma")]
    for task in tasks:
        task.mkdir(parents=True)
    rollout = Rollout(jobs_dir=str(tmp_path / "jobs"))
    engine = HarborEngine(rollout, run_trial=FakeTrials(breaks=("beta",)))
    engine.progress = heard = Heard()
    endpoint = ProviderEndpoint("m", "openrouter")
    await engine.run(Job("job-0000", harness=Harness(name="pi")), tasks, 1, endpoint=endpoint)
    assert heard.events == [
        ("started", "job-0000", 3),
        *[("finished", "job-0000", None)] * 3,
        ("done", "job-0000", 1),
    ]
