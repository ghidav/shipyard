"""Datasets as directories of Harbor tasks under `tasks/`, and how a run cuts them up."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from harbor.models.task.task import Task

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


def batches(
    datasets: Sequence[str],
    *,
    size: int,
    seed: int,
    epochs: int,
    root: Path | None = None,
) -> Iterator[list[Path]]:
    """The datasets' tasks, in list order, shuffled once per epoch with `seed + epoch`
    and cut into batches of `size`. The final short batch is kept so the run covers the
    whole dataset. Returns a fresh generator per call."""
    if size < 1:
        raise ValueError(f"size must be at least 1; got {size}")
    listed = [task for dataset in datasets for task in tasks(dataset, root=root)]
    return _cut(listed, size=size, seed=seed, epochs=epochs)


def held_out(listed: Sequence[Path], count: int, *, seed: int) -> tuple[list[Path], list[Path]]:
    """Split `listed` in two, each part in list order: the tasks to learn from, and
    `count` tasks drawn with `seed` to hold out and measure on. The same seed gives the
    same split."""
    if not 0 <= count <= len(listed):
        raise ValueError(f"cannot hold out {count} of {len(listed)} task(s)")
    drawn = set(random.Random(seed).sample(range(len(listed)), count))
    kept = [task for at, task in enumerate(listed) if at not in drawn]
    return kept, [task for at, task in enumerate(listed) if at in drawn]


def _cut(listed: list[Path], *, size: int, seed: int, epochs: int) -> Iterator[list[Path]]:
    """The generator half of `batches`. It is separate so a missing dataset raises when
    `batches` is called, not at the first batch."""
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
