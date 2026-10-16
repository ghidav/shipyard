"""Live progress bars on a terminal, one per job, counting its trials; silent elsewhere."""

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

#: Environment variable; set it to `1` to turn off progress bars on a terminal.
QUIET = "SHIPYARD_QUIET"


def reporter(console: Console | None = None) -> Reporter:
    """Return progress bars on a terminal, or a silent reporter elsewhere or under
    `SHIPYARD_QUIET=1`, so that logs and pipes get no redrawn frames."""
    console = console if console is not None else Console()
    if os.environ.get(QUIET) == "1" or not console.is_terminal:
        return Reporter()
    return Bars(console)


class Reporter:
    """Receiver of a run's job events. This base class ignores them; `Bars` draws them."""

    def started(self, job: str, *, trials: int) -> None:
        """Report that a job with `trials` planned trials has started."""

    def finished(self, trial: Path, *, masked: bool) -> None:
        """Report that one trial of a started job has finished. `masked` is true when
        admission left it out."""

    def done(self, job: str) -> None:
        """Report that every trial of the job has finished."""

    def close(self) -> None:
        """Stop drawing. Idempotent."""


class Bars(Reporter):
    """Progress bars, one per job, redrawn in place as Harbor draws a job. Each shows the
    job, trials finished out of planned, trials masked and the elapsed time."""

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

    def started(self, job: str, *, trials: int) -> None:
        if self._live is None:
            self._live = Live(self.bars, console=self.console, refresh_per_second=8)
            self._live.start()
        self._masked[job] = 0
        self._tasks[job] = self.bars.add_task(job, total=trials, masked=0)

    def finished(self, trial: Path, *, masked: bool) -> None:
        """Advance the bar of the trial's job, which is the trial directory's parent in
        `jobs/<job>/`."""
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
