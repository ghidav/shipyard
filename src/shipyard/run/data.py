from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task.task import Task

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: Directory holding the datasets, relative to the working directory, as Harbor uses `tasks/`.
TASKS_DIR = "tasks"


def home_of(dataset: str, *, root: Path | None = None) -> Path:
    """Return `<root>/<dataset>`, `root` being `tasks/` in the working directory by default."""
    return (root if root is not None else Path(TASKS_DIR)) / dataset


def tasks(dataset: str, *, root: Path | None = None) -> list[Path]:
    """List the dataset's task directories that Harbor accepts, sorted by name."""
    home = home_of(dataset, root=root)
    listed = _task_dirs(home)
    if listed:
        return listed
    raise FileNotFoundError(
        f"no dataset at {home}: no valid Harbor task directory under it; "
        "`harbor datasets download` fills it"
    )


@dataclass(frozen=True)
class Generator:
    """The trainer's source of tasks: batches of task paths, and nothing else."""

    train: list[Path]
    batch_size: int
    seed: int
    test: list[Path] = field(default_factory=list)

    @classmethod
    def of(cls, config: Blueprint) -> Generator:
        """Build a generator from the blueprint's `[data]` table."""
        data = config.data

        def listed(names: Sequence[str]) -> list[Path]:
            return [task for name in names for task in tasks(name)]

        return cls(listed(config.datasets), data.batch_size, data.seed, listed(config.test_sets))

    @property
    def per_pass(self) -> int:
        """Number of batches in one pass over the training tasks, counting a short last one."""
        return -(-len(self.train) // self.batch_size)

    def batches(self, start: int = 0) -> Iterator[list[Path]]:
        """Yield batches without end, from batch index `start`."""
        if not self.train:
            return
        turn, at = divmod(start, self.per_pass)
        while True:
            order = list(self.train)
            random.Random(self.seed + turn).shuffle(order)
            for begin in range(at * self.batch_size, len(order), self.batch_size):
                yield order[begin : begin + self.batch_size]
            turn, at = turn + 1, 0


def _task_dirs(home: Path) -> list[Path]:
    if not home.is_dir():
        return []
    return sorted(
        path
        for path in home.iterdir()
        if path.is_dir() and not path.name.startswith(".") and Task.is_valid_dir(path)
    )
