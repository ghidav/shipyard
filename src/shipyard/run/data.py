from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task.task import Task

from shipyard.state import Lending

if TYPE_CHECKING:
    from shipyard.config import Config

logger = logging.getLogger(__name__)

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


@dataclass
class Lender:
    """The run's source of training tasks. It gives them out in a seeded order, each pass over
    them shuffled with `seed + pass`; gives the ones put back out before new ones; gives retired
    ones out no more; and keeps which are out until their groups complete. What it has given
    out is `lending`, the run's state's, so a checkpoint saves it. It lists the test set's tasks
    too, and never gives them out."""

    train: list[Path]
    batch_size: int
    seed: int
    test: list[Path] = field(default_factory=list)
    lending: Lending = field(default_factory=Lending)

    @classmethod
    def of(cls, config: Config, lending: Lending | None = None) -> Lender:
        """Build the lender of the config's `[data]` table, going on from `lending`."""
        data = config.data

        def listed(names: Sequence[str]) -> list[Path]:
            return [task for name in names for task in tasks(name)]

        return cls(
            listed(config.datasets),
            data.batch_size,
            data.seed,
            listed(config.test_sets),
            lending if lending is not None else Lending(),
        )

    @property
    def per_pass(self) -> int:
        """Number of batches in one pass over the training tasks, counting a short last one."""
        return -(-len(self.train) // self.batch_size)

    def draw(self, n: int | None = None) -> list[Path]:
        """Give out the next `n` training tasks, `batch_size` unless given: the ones put back
        first, then the seeded order, which stops at the end of a pass, so a batch never holds
        a task twice. Each is out until its group completes. Raise `RuntimeError` when there is
        no task to give: none to train on, or every one retired."""
        count = self.batch_size if n is None else n
        if count < 1:
            raise ValueError(f"draw({count}): draw at least one task")
        held = self.lending
        found = [Path(one) for one in held.back[:count]]
        del held.back[:count]
        if len(found) < count:
            found += self._fresh(count - len(found), have=bool(found))
        held.out += [str(one) for one in found]
        return found

    def back(self, task: Path) -> None:
        """Put `task` back: give it out again before new ones. It is out no more, nor retired."""
        name = self._known(task, "back")
        held = self.lending
        if name in held.out:
            held.out.remove(name)
        if name in held.retired:
            held.retired.remove(name)
        held.back.append(name)

    def retire(self, task: Path) -> None:
        """Give `task` out no more in the run. A group of it that is running completes as any."""
        name = self._known(task, "retire")
        held = self.lending
        if name not in held.retired:
            held.retired.append(name)
        held.back[:] = [one for one in held.back if one != name]

    def complete(self, task: Path) -> None:
        """Note that a group of `task` completed, so that it is out once fewer; the run calls it
        as each rollout returns. A task that was not out, such as a test task, is let be."""
        name = str(task)
        if name in self.lending.out:
            self.lending.out.remove(name)

    def resume(self, lending: Lending) -> None:
        """Go on from a checkpoint's `lending`. The tasks out then are given out first, as their
        trials ended with the run that drew them; a task the training set no longer holds is
        dropped, with a warning."""
        known = {str(one) for one in self.train}
        gone = sorted({*lending.out, *lending.back, *lending.retired} - known)
        if gone:
            logger.warning("the lender drops tasks the training set no longer holds: %s", gone)
        lending.back = [one for one in (*lending.out, *lending.back) if one in known]
        lending.out = []
        lending.retired = [one for one in lending.retired if one in known]
        self.lending = lending

    def place_after(self, batches: int) -> int:
        """Return the place in the seeded order after `batches` batches of `batch_size`, each
        pass ending in a short one: what a checkpoint saved before the lender counted."""
        if not self.train:
            return 0
        turn, at = divmod(batches, self.per_pass)
        return turn * len(self.train) + at * self.batch_size

    def _fresh(self, count: int, *, have: bool) -> list[Path]:
        """Take up to `count` tasks from the seeded order, skipping retired ones, and stop at the
        end of a pass once any was taken; `have` says some were taken from those put back."""
        if not self.train:
            raise RuntimeError("the lender has no training task: [data] train names none")
        held = self.lending
        retired = set(held.retired)
        if all(str(one) in retired for one in self.train):
            if have:
                return []
            raise RuntimeError("the lender has no task to give: every training task is retired")
        size = len(self.train)
        found: list[Path] = []
        while len(found) < count:
            turn, at = divmod(held.place, size)
            task = self._order(turn)[at]
            held.place += 1
            if str(task) not in retired:
                found.append(task)
            if held.place % size == 0 and (found or have):
                break
        return found

    def _order(self, turn: int) -> list[Path]:
        """Return pass `turn`'s order: the training tasks shuffled with `seed + turn`."""
        if getattr(self, "_turn", None) != turn:
            order = list(self.train)
            random.Random(self.seed + turn).shuffle(order)
            self._turn, self._cached = turn, order
        return self._cached

    def _known(self, task: Path, verb: str) -> str:
        """Return `task` as the lender names it; raise `ValueError` for one it does not lend."""
        name = str(task)
        if name not in {str(one) for one in self.train}:
            raise ValueError(f"{verb}({task}): the lender gives out training tasks, not this one")
        return name


def _task_dirs(home: Path) -> list[Path]:
    if not home.is_dir():
        return []
    return sorted(
        path
        for path in home.iterdir()
        if path.is_dir() and not path.name.startswith(".") and Task.is_valid_dir(path)
    )
