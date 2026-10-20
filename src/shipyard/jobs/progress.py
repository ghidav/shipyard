from __future__ import annotations

import os

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

#: Environment variable; set it to `1` to turn off progress bars on a terminal.
QUIET = "SHIPYARD_QUIET"


class Bars:
    def __init__(self, console: Console | None = None) -> None:
        console = console if console is not None else Console()
        self.bars = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[yellow]{task.fields[masked]}"),
            TimeElapsedColumn(),
            console=console,
            disable=os.environ.get(QUIET) == "1" or not console.is_terminal,
        )
        self._tasks: dict[str, TaskID] = {}

    def started(self, job: str, *, trials: int) -> None:
        """Add a bar for a job of `trials` planned trials."""
        self.bars.start()
        self._tasks[job] = self.bars.add_task(job, total=trials, masked="")

    def finished(self, job: str) -> None:
        """Advance the job's bar by one finished trial."""
        if job in self._tasks:
            self.bars.update(self._tasks[job], advance=1)

    def done(self, job: str, *, masked: int) -> None:
        """Show how many of the judged job's trials were masked, and stop its clock."""
        if job in self._tasks:
            self.bars.update(self._tasks[job], masked=f"{masked} masked")
            self.bars.stop_task(self._tasks[job])

    def close(self) -> None:
        """Stop drawing. Idempotent."""
        self.bars.stop()
