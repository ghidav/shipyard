"""A run: what `shipyard run` wraps around a trainer. A directory holding the blueprint it
consumed, opened, reopened and closed; what happens in it is the trainer's to write. Also the
trainer a run goes under, and a run reopened at a checkpoint with its trainer."""

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
    """A run that could not be opened; nothing was written."""


class Run:
    """An open run: its directory, the blueprint it consumed, and how it opened and closed."""

    def __init__(self, directory: Path, config: Blueprint) -> None:
        self.directory = directory
        self.id = directory.name
        self.config = config
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
        `trainer.py` when one is given, and write the process note (marked `smoke` for a smoke
        run). Raises `RunRefused` if the directory exists."""
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
        """The run as it stands, to go on; its process note is written on entry, so a refusal
        before the work leaves the earlier one as it was."""
        directory = Path(directory)
        run = cls(directory, load(directory))
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
        """Note how the run ended, whether or not the work raised. A `KeyboardInterrupt` from
        a signal is noted as `stopped: <signal>`."""
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def trainer_of(run: Run) -> type[Trainer]:
    """The trainer class a run goes under: the one its kept trainer file defines, else its
    kind's."""
    kept = run.directory / record.TRAINER
    return trainer_in(kept) if kept.is_file() else trainer_for(run.config.recipe)


def reopened(directory: Path, *, tag: str = "last") -> tuple[Run, Trainer]:
    """A run reopened at one of its checkpoints, and its trainer standing where it stood,
    under `trainer_of` the run."""
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
    """A process note: this invocation, the blueprint and the kind, then `extra` where set."""
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
