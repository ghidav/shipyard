"""Run directories: opening, reopening and closing a run, and choosing its trainer.

A run directory holds a copy of its blueprint (and trainer file, if one was given) and a process
note; the trainer writes everything else in it.
"""

from __future__ import annotations

import os
import shutil
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shortuuid import ShortUUID

from shipyard import record
from shipyard.config import Blueprint, config_file, load
from shipyard.sampling.endpoint import tinker_base_url
from shipyard.training.recipes import trainer_for, trainer_in
from shipyard.training.trainer import Trainer


class RunRefused(RuntimeError):
    """Raised when a run cannot be opened. Nothing has been written."""


class Run:
    """An open run directory, used as a context manager around the work done in it.

    Entering a reopened run writes a new process note; exiting records how the run ended.
    """

    def __init__(self, directory: Path, config: Blueprint) -> None:
        self.directory = directory
        self.id = directory.name
        self.config = config
        #: Set by `reopen`. On entry, a new process note is written and the earlier ones kept.
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
        """Create the run directory `<root>/<id>/` for a blueprint.

        Copies `run.toml` byte for byte, copies the trainer file as `trainer.py` when one is
        given, and writes the process note, marked `smoke` for a smoke run. Raises `RunRefused`
        if the directory already exists.
        """
        config = load(blueprint)
        source = config_file(blueprint)
        home = source.parent
        directory = runs_root(root) / _run_id(home.name)
        if directory.exists():
            raise RunRefused(f"{directory} already exists")
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / record.CONFIG)
        if trainer is not None:
            shutil.copyfile(trainer, directory / record.TRAINER)
        record.write_json(
            directory / record.PROCESS,
            _process(
                config,
                str(home.resolve()),
                trainer=None if trainer is None else str(Path(trainer).resolve()),
                smoke=smoke,
            ),
        )
        return cls(directory, config)

    @classmethod
    def reopen(cls, directory: Path, *, resumed: str | None = None) -> Self:
        """Reopen an existing run to continue it.

        The new process note is written only on entering the run, so a refusal before the work
        starts leaves the previous note unchanged.
        """
        directory = Path(directory)
        run = cls(directory, load(directory))
        run._reopened, run._resumed = True, resumed
        return run

    @staticmethod
    def read(directory: Path) -> dict[str, Any]:
        """Return the process note of a run directory, open or closed, or `{}` if it has none."""
        return record.read_json(Path(directory) / record.PROCESS)

    def note(self, **about: Any) -> None:
        """Merge the given fields about this invocation into `process.json`."""
        current = record.read_json(self.directory / record.PROCESS)
        current.update(about)
        record.write_json(self.directory / record.PROCESS, current)

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
        """Record in the process note how the run ended, whether or not the work raised.

        A `KeyboardInterrupt` is recorded as `stopped: <signal name>`, with `SIGINT` for Ctrl-C.
        """
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def trainer_of(run: Run) -> type[Trainer]:
    """Return the trainer class for a run.

    This is the class defined in the run's copy of the trainer file, else the default trainer
    for its recipe kind.
    """
    kept = run.directory / record.TRAINER
    return trainer_in(kept) if kept.is_file() else trainer_for(run.config.recipe)


def reopened(directory: Path, *, tag: str = "last") -> tuple[Run, Trainer]:
    """Reopen a run at a checkpoint and return it with its trainer restored to that checkpoint.

    `tag` names the checkpoint; "last" means the most recent one. Raises `ValueError` if there
    is no run at `directory` or no matching checkpoint.
    """
    directory = Path(directory)
    if not (directory / record.CONFIG).is_file():
        raise ValueError(f"no run at {directory}")
    rows = list(record.read(directory / record.CHECKPOINTS))
    if not rows:
        raise ValueError(f"{directory.name} has no checkpoint to go on from")
    row = rows[-1] if tag == "last" else next((r for r in rows if r["tag"] == tag), None)
    if row is None:
        tags = ", ".join(r["tag"] for r in rows)
        raise ValueError(f"{directory.name} has no checkpoint tagged {tag!r}; it has {tags}")
    run = Run.reopen(directory, resumed=row["tag"])
    trainer = trainer_of(run)(run.config, run.directory)
    trainer.restore(row)
    return run, trainer


def _process(config: Blueprint, blueprint: str | None, **extra: Any) -> dict[str, Any]:
    """Build the process note for this invocation, adding the entries of `extra` that are set."""
    return {
        "pid": os.getpid(),
        "started_at": record.now(),
        "version": version(),
        "blueprint": blueprint,
        "kind": config.recipe.kind,
        **({"tinker_base_url": tinker_base_url()} if config.agent.served else {}),
        **{key: value for key, value in extra.items() if value},
    }


def runs_root(flag: Path | None = None) -> Path:
    """Return the directory runs live in: `--root`, else `SHIPYARD_RUNS`, else `./runs`."""
    if flag is not None:
        return Path(flag)
    named = os.environ.get("SHIPYARD_RUNS")
    return Path(named) if named else Path.cwd() / "runs"


def version() -> str:
    """Return the installed shipyard version, or "source" when running from an uninstalled tree."""
    try:
        return installed("shipyard")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "source"


def _run_id(name: str) -> str:
    """Return a new run id, `<name>__<7 random chars>`, following Harbor's trial naming."""
    stem = name[:32].rstrip("_-") or "run"
    return f"{stem}__{ShortUUID().random(length=7)}"
