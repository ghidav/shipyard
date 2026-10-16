"""A run: a directory holding the blueprint it consumed, opened, reopened and closed. What
happens in it is the trainer's to write."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shortuuid import ShortUUID

from shipyard import record
from shipyard.config import Blueprint, Search, config_file, load, modules_dir
from shipyard.modules import Candidate, keep, seed
from shipyard.sampling.endpoint import tinker_base_url

logger = logging.getLogger(__name__)

#: Under `runs/<id>/modules/`: the candidate a run carries, as `seed` reads it back.
CARRIED = "carried"


class RunRefused(RuntimeError):
    """A run that could not be opened; nothing was written."""


class Run:
    """An open run: its directory, the blueprint it consumed, the names of its jobs, and how
    it opened and closed."""

    def __init__(
        self, directory: Path, config: Blueprint, carried: Candidate | None = None
    ) -> None:
        self.directory = directory
        self.id = directory.name
        self.config = config
        #: The candidate every job carries unless handed another; None when `[recipe]
        #: modules` names none, or the recipe searches over them itself.
        self.carried = carried
        self.jobs_dir = Path(config.rollout.jobs_dir)
        self._next_job = next_job(self.id, directory, self.jobs_dir)
        #: Called as the run closes, last added first: what the trainer opened for it.
        self._closing: list[Callable[[str | None], object]] = []
        #: Set by `reopen`: the process note is written on entry, keeping the earlier ones.
        self._reopened = False
        self._resumed: str | None = None

    @classmethod
    def open(
        cls,
        blueprint: Path,
        *,
        root: Path | None = None,
        smoke: bool = False,
        trainer: Path | None = None,
    ) -> Self:
        """Create `<root>/<id>/`, copy `run.toml` byte for byte and the trainer file as
        `trainer.py` when one is given, keep the recipe's modules under `modules/carried/`, and
        write the process note (marked `smoke` for a smoke run). Raises `RunRefused` if the
        directory exists."""
        config = load(blueprint)
        source = config_file(blueprint)
        home = source.parent
        carried = carried_modules(config)
        directory = runs_root(root) / _run_id(home.name)
        if directory.exists():
            raise RunRefused(f"{directory} already exists")
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / record.CONFIG)
        if trainer is not None:
            shutil.copyfile(trainer, directory / record.TRAINER)
        if carried is not None:
            keep(carried, directory / record.MODULES / CARRIED)
        record.write_json(
            directory / record.PROCESS,
            _process(
                config,
                str(home.resolve()),
                trainer=None if trainer is None else str(Path(trainer).resolve()),
                smoke=smoke,
            ),
        )
        return cls(directory, config, carried)

    @classmethod
    def reopen(cls, directory: Path, *, resumed: str | None = None) -> Self:
        """The run as it stands, to go on; its process note is written on entry, so a refusal
        before the work leaves the earlier one as it was."""
        directory = Path(directory)
        config = load(directory)
        carried = directory / record.MODULES / CARRIED
        run = cls(directory, config, seed(carried) if carried.is_dir() else None)
        run._reopened, run._resumed = True, resumed
        return run

    @staticmethod
    def read(directory: Path) -> dict[str, Any]:
        """The process note of a run directory, open or closed; `{}` when it has none."""
        return record.read_json(Path(directory) / record.PROCESS)

    def note(self, **about: Any) -> None:
        """Merge facts about this invocation into `process.json`."""
        current = record.read_json(self.directory / record.PROCESS)
        current.update(about)
        record.write_json(self.directory / record.PROCESS, current)

    def reserve_name(self) -> str:
        """The next job name, `<run id>-NNNN`, with its directory created; raises if it exists."""
        name = f"{self.id}-{self._next_job:04d}"
        try:
            (self.jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {self.jobs_dir / name} exists") from None
        self._next_job += 1
        return name

    def closing(self, close: Callable[[str | None], object]) -> None:
        """Have `close` called as the run closes, with why it stopped (None when it finished)."""
        self._closing.append(close)

    def __enter__(self) -> Self:
        if self._reopened:
            earlier = record.read_json(self.directory / record.PROCESS)
            before = earlier.pop("earlier", [])
            note = _process(self.config, earlier.get("blueprint"), resumed=self._resumed)
            note["earlier"] = [*before, earlier] if earlier else before
            record.write_json(self.directory / record.PROCESS, note)
            self._reopened = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close what was opened for the run, then note how it ended, whether or not the loop
        raised. A `KeyboardInterrupt` from a signal is noted as `stopped: <signal>`."""
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        while self._closing:
            close = self._closing.pop()
            try:
                close(error)
            except Exception:  # one that fails to close must not keep the end off the note
                logger.exception("closing run %s", self.id)
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def _process(config: Blueprint, blueprint: str | None, **extra: Any) -> dict[str, Any]:
    """A process note: this invocation, the blueprint and the kind, then `extra` where set."""
    return {
        "pid": os.getpid(),
        "started_at": record.now(),
        "version": version(),
        "blueprint": blueprint,
        "kind": config.recipe.kind,
        **({"tinker_base_url": tinker_base_url()} if config.model.served else {}),
        **{key: value for key, value in extra.items() if value},
    }


def carried_modules(config: Blueprint) -> Candidate | None:
    """The candidate a run carries into every job, seeded from `[recipe] modules`. None
    when the key is unset or the recipe (gepa, fst) seeds its own search."""
    return None if isinstance(config.recipe, Search) else seeded(config)


def seeded(config: Blueprint) -> Candidate | None:
    """The candidate `[recipe] modules` seeds, for any recipe. None when it is unset."""
    directory = modules_dir(config)
    return None if directory is None else seed(directory)


def next_job(run_id: str, directory: Path, jobs_dir: Path) -> int:
    """One past the highest job number on the record or on disk; 0 for a new run."""
    named = [row.get("job") for row in record.read(directory / record.JOBS)]
    if jobs_dir.is_dir():
        named += [one.name for one in jobs_dir.iterdir()]
    prefix = f"{run_id}-"
    numbers = [
        int(name[len(prefix) :])
        for name in named
        if isinstance(name, str) and name.startswith(prefix) and name[len(prefix) :].isdigit()
    ]
    return max(numbers, default=-1) + 1


def runs_root(flag: Path | None = None) -> Path:
    """Where runs live: `--root`, else `SHIPYARD_RUNS`, else `./runs`. Every command uses it."""
    if flag is not None:
        return Path(flag)
    named = os.environ.get("SHIPYARD_RUNS")
    return Path(named) if named else Path.cwd() / "runs"


def version() -> str:
    """The installed distribution's version, or "source" from a tree never installed."""
    try:
        return installed("shipyard")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "source"


def _run_id(name: str) -> str:
    """`<name>__<7>`, named as Harbor names a trial. Each run gets a new id."""
    stem = name[:32].rstrip("_-") or "run"
    return f"{stem}__{ShortUUID().random(length=7)}"
