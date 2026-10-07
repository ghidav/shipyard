"""Datasets as directories of Harbor tasks under `tasks/`, and how a run cuts them up."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from harbor.models.task.task import Task

#: Where datasets live, relative to the working directory, the way Harbor takes `tasks/`.
TASKS_DIR = "tasks"


class NoSuchDataset(FileNotFoundError):
    """Named in a blueprint, absent on disk: raised rather than yielding nothing, so a run
    cannot finish over zero batches and report success."""


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


def batches(
    datasets: Sequence[str],
    *,
    size: int,
    seed: int,
    epochs: int,
    root: Path | None = None,
) -> Iterator[list[Path]]:
    """The datasets' tasks, in list order, shuffled once per epoch with `seed + epoch`
    and cut into batches of `size`; the final short batch is kept, since dropping it
    would leave a run covering less than its dataset. A fresh generator per call."""
    if size < 1:
        raise ValueError(f"size must be at least 1; got {size}")
    listed = [task for dataset in datasets for task in tasks(dataset, root=root)]
    return _cut(listed, size=size, seed=seed, epochs=epochs)


def _cut(listed: list[Path], *, size: int, seed: int, epochs: int) -> Iterator[list[Path]]:
    """The generator half of `batches`, kept apart so a missing dataset raises at the call
    rather than at the first batch."""
    for epoch in range(epochs):
        order = list(listed)
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), size):
            yield order[start : start + size]


def _task_dirs(home: Path) -> list[Path]:
    """The valid task directories under a dataset directory, sorted by name."""
    if not home.is_dir():
        return []
    return sorted(
        path
        for path in home.iterdir()
        if path.is_dir() and not path.name.startswith(".") and Task.is_valid_dir(path)
    )
