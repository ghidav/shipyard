"""Live progress on a terminal: one bar per job, counting its trials; silent anywhere else."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

#: Set to `1` to keep a terminal run from drawing.
QUIET = "SHIPYARD_QUIET"


def reporter(console: Console | None = None) -> Reporter:
    """Bars on a terminal, a silent reporter elsewhere or under `SHIPYARD_QUIET=1`, so logs
    and pipes get no redrawn frames."""
    console = console if console is not None else Console()
    if os.environ.get(QUIET) == "1" or not console.is_terminal:
        return Reporter()
    return Bars(console)


class Reporter:
    """Receives a run's job events. This base class ignores them; `Bars` draws them."""

    def started(self, job: str, *, trials: int, batch: int | None = None) -> None:
        """A job of `trials` planned trials has begun, for batch `batch`."""

    def finished(self, trial: Path, *, masked: bool) -> None:
        """One trial of a started job is back; `masked` says admission left it out."""

    def done(self, job: str) -> None:
        """Every trial of the job is back."""

    def close(self) -> None:
        """Stop drawing. Idempotent."""


class Bars(Reporter):
    """One bar per job: trials finished out of planned, trials masked, the batch index and
    the elapsed time. Redrawn in place, as Harbor draws a job."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.bars = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[yellow]{task.fields[masked]} masked"),
            TimeElapsedColumn(),
            console=console,
        )
        self._live: Live | None = None
        self._tasks: dict[str, Any] = {}
        self._masked: dict[str, int] = {}

    def started(self, job: str, *, trials: int, batch: int | None = None) -> None:
        if self._live is None:
            self._live = Live(self.bars, console=self.console, refresh_per_second=8)
            self._live.start()
        label = f"batch {batch}  {job}" if batch is not None else job
        self._masked[job] = 0
        self._tasks[job] = self.bars.add_task(label, total=trials, masked=0)

    def finished(self, trial: Path, *, masked: bool) -> None:
        """The job is the trial directory's parent in `jobs/<job>/`."""
        job = Path(trial).parent.name
        task = self._tasks.get(job)
        if task is None:
            return
        if masked:
            self._masked[job] += 1
        self.bars.update(task, advance=1, masked=self._masked[job])

    def done(self, job: str) -> None:
        task = self._tasks.get(job)
        if task is not None:
            self.bars.stop_task(task)

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
