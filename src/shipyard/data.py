"""Datasets as directories of Harbor tasks under `tasks/`, and how a run cuts them up."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task.task import Task

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: Where datasets live, relative to the working directory, the way Harbor takes `tasks/`.
TASKS_DIR = "tasks"


class NoSuchDataset(FileNotFoundError):
    """A dataset named in a blueprint is absent on disk. Raised so a run cannot finish over
    zero batches and report success."""


def home_of(dataset: str, *, root: Path | None = None) -> Path:
    """`tasks/<dataset>`, under `root` or the working directory."""
    return (root if root is not None else Path(TASKS_DIR)) / dataset


def tasks(dataset: str, *, root: Path | None = None) -> list[Path]:
    """Every task directory under `tasks/<dataset>/` that Harbor accepts, by name.
    Raises `NoSuchDataset` naming the path when there is none."""
    home = home_of(dataset, root=root)
    listed = _task_dirs(home)
    if listed:
        return listed
    raise NoSuchDataset(
        f"no dataset at {home}: no valid Harbor task directory under it; "
        "`harbor datasets download` fills it"
    )


def held_out(listed: Sequence[Path], count: int, *, seed: int) -> tuple[list[Path], list[Path]]:
    """Split `listed` in two, each part in list order: the tasks to learn from, and
    `count` tasks drawn with `seed` to hold out and measure on. The same seed gives the
    same split."""
    if not 0 <= count <= len(listed):
        raise ValueError(f"cannot hold out {count} of {len(listed)} task(s)")
    drawn = set(random.Random(seed).sample(range(len(listed)), count))
    kept = [task for at, task in enumerate(listed) if at not in drawn]
    return kept, [task for at, task in enumerate(listed) if at in drawn]


@dataclass(frozen=True)
class Generator:
    """How tasks are fed to the trainer: batches of task references and nothing else."""

    train: list[Path]
    batch_size: int
    seed: int
    test: list[Path] = field(default_factory=list)

    @classmethod
    def of(cls, config: Blueprint) -> Generator:
        """Over the blueprint's `[data]` table: each set's tasks in list order."""
        data = config.data

        def listed(names: Sequence[str]) -> list[Path]:
            return [task for name in names for task in tasks(name)]

        return cls(listed(config.datasets), data.batch_size, data.seed, listed(config.test_sets))

    def batches(self, epoch: int) -> Iterator[list[Path]]:
        """One epoch: the train tasks shuffled under `seed + epoch`, `batch_size` at a time,
        the short last batch kept."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        order = list(self.train)
        random.Random(self.seed + epoch).shuffle(order)
        for start in range(0, len(order), self.batch_size):
            yield order[start : start + self.batch_size]


def _task_dirs(home: Path) -> list[Path]:
    """The valid task directories under a dataset directory, sorted by name."""
    if not home.is_dir():
        return []
    return sorted(
        path
        for path in home.iterdir()
        if path.is_dir() and not path.name.startswith(".") and Task.is_valid_dir(path)
    )
