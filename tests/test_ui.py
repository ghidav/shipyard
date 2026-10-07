"""Bars on a terminal counting each job's trials; nothing drawn anywhere else."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from shipyard.ui import QUIET, Bars, Reporter, reporter


def test_the_reporter_is_silent_off_a_terminal_and_under_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(QUIET, raising=False)
    piped = Console(file=io.StringIO(), force_terminal=False)
    assert type(reporter(piped)) is Reporter
    terminal = Console(file=io.StringIO(), force_terminal=True)
    assert isinstance(reporter(terminal), Bars)
    monkeypatch.setenv(QUIET, "1")
    assert type(reporter(terminal)) is Reporter


def test_bars_count_finished_and_masked_per_job(tmp_path: Path) -> None:
    out = io.StringIO()
    bars = Bars(Console(file=out, force_terminal=True, width=120))
    bars.started("run-0000", trials=3, batch=0)
    bars.finished(tmp_path / "jobs" / "run-0000" / "alpha__1", masked=False)
    bars.finished(tmp_path / "jobs" / "run-0000" / "beta__1", masked=True)
    [task] = bars.bars.tasks
    assert task.completed == 2 and task.total == 3 and task.fields["masked"] == 1
    assert "batch 0" in task.description and "run-0000" in task.description
    bars.finished(tmp_path / "elsewhere" / "gamma__1", masked=True)  # of no job this drew
    assert task.completed == 2
    bars.started("run-0001", trials=1)
    assert [t.description for t in bars.bars.tasks] == ["batch 0  run-0000", "run-0001"]
    bars.done("run-0000")
    bars.done("never-started")
    bars.close()
    bars.close()
    assert "run-0000" in out.getvalue() and "1 masked" in out.getvalue()
