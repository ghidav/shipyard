"""Datasets of Harbor tasks under `tasks/`: listing, batching and held-out splits."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task.task import Task

if TYPE_CHECKING:
    from shipyard.config import Blueprint

#: Directory holding the datasets, relative to the working directory, as Harbor uses `tasks/`.
TASKS_DIR = "tasks"


class NoSuchDataset(FileNotFoundError):
    """Raised when a dataset named in a blueprint is missing on disk, so that a run cannot
    report success after zero batches."""


def home_of(dataset: str, *, root: Path | None = None) -> Path:
    """Return `<root>/<dataset>`, where `root` defaults to `tasks/` in the working
    directory."""
    return (root if root is not None else Path(TASKS_DIR)) / dataset


def tasks(dataset: str, *, root: Path | None = None) -> list[Path]:
    """List the dataset's task directories that Harbor accepts, sorted by name.

    Raises `NoSuchDataset` when there are none."""
    home = home_of(dataset, root=root)
    listed = _task_dirs(home)
    if listed:
        return listed
    raise NoSuchDataset(
        f"no dataset at {home}: no valid Harbor task directory under it; "
        "`harbor datasets download` fills it"
    )


def held_out(listed: Sequence[Path], count: int, *, seed: int) -> tuple[list[Path], list[Path]]:
    """Split `listed` into kept and held-out tasks.

    `count` tasks are drawn at random with `seed` and held out. Both parts keep list order,
    and the same seed gives the same split."""
    if not 0 <= count <= len(listed):
        raise ValueError(f"cannot hold out {count} of {len(listed)} task(s)")
    drawn = set(random.Random(seed).sample(range(len(listed)), count))
    kept = [task for at, task in enumerate(listed) if at not in drawn]
    return kept, [task for at, task in enumerate(listed) if at in drawn]


@dataclass(frozen=True)
class Generator:
    """The trainer's source of tasks: batches of task paths, and nothing else."""

    train: list[Path]
    batch_size: int
    seed: int
    test: list[Path] = field(default_factory=list)

    @classmethod
    def of(cls, config: Blueprint) -> Generator:
        """Build a generator from the blueprint's `[data]` table, with each dataset's tasks
        in list order."""
        data = config.data

        def listed(names: Sequence[str]) -> list[Path]:
            return [task for name in names for task in tasks(name)]

        return cls(listed(config.datasets), data.batch_size, data.seed, listed(config.test_sets))

    @property
    def per_pass(self) -> int:
        """Number of batches in one pass over the training tasks, counting a short last one."""
        return -(-len(self.train) // self.batch_size)

    def batches(self, start: int = 0) -> Iterator[list[Path]]:
        """Yield batches without end, starting from batch index `start`.

        Pass `n` over the training tasks shuffles them with seed `seed + n` and cuts them
        into batches of `batch_size`, keeping a short last batch. Yields nothing when there
        are no training tasks."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
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


#: Fraction of the run's tasks `gepa` holds out for selection when `[recipe] pareto` is
#: unset, rounded down, as in three of GEPA's four benchmarks: GEPA §4.1 (split sizes) and
#: §4.3 (the validation set is D_pareto). HotpotQA, IFBench and HoVer have 300 validation
#: tasks beside 150 training ones (only PUPA splits 111 and 111), and IFBench's, the one
#: split the paper draws itself, is among them.
PAPER_PARETO = (2, 3)


def search_sets(
    found: Iterable[Path], pareto: int | str | None, *, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split the tasks `found` into GEPA's D_feedback and D_pareto (Alg. 1 line 1).

    Minibatches are drawn from D_feedback, and every candidate the search keeps is scored
    on D_pareto. Each list holds each task once, in path order. `pareto` selects the split:
    a dataset name is used as D_pareto, with all of `found` as D_feedback; a count holds
    that many tasks out, drawn with `seed`; 0 puts every task in both (GEPA 6); None holds
    out two thirds, as in three of GEPA's four benchmarks. Raises `ValueError` when either
    list would be empty."""
    listed = sorted(set(found))
    if isinstance(pareto, str):
        return listed, tasks(pareto)
    if pareto == 0:
        return listed, list(listed)
    share, whole = PAPER_PARETO
    count = len(listed) * share // whole if pareto is None else pareto
    if count < 1:
        raise ValueError(
            "one task cannot be split; 0 selects on the task the search reflects on, and a "
            "dataset name selects on that dataset"
        )
    if count >= len(listed):
        raise ValueError(f"holding out {count} of {_tasks(len(listed))} leaves none to reflect on")
    return held_out(listed, count, seed=seed)


def _tasks(count: int) -> str:
    return f"{count} task" if count == 1 else f"{count} tasks"
