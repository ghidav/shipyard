from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from shortuuid import ShortUUID

from shipyard.config import Blueprint, config_file, load
from shipyard.run import record
from shipyard.trainer import Trainer

logger = logging.getLogger(__name__)


def runs_root(flag: Path | None = None) -> Path:
    """Return the directory runs live in: `--root`, else `SHIPYARD_RUNS`, else `./runs`."""
    if flag is not None:
        return Path(flag)
    named = os.environ.get("SHIPYARD_RUNS")
    return Path(named) if named else Path.cwd() / "runs"


def open_run(
    blueprint: Path, *, root: Path | None = None, trainer: Path | None = None
) -> tuple[Path, Blueprint]:
    """Create a run's directory for a blueprint, noting its recipe's place."""
    source = config_file(blueprint)
    if trainer is None and (source.parent / record.TRAINER).is_file():
        trainer = source.parent / record.TRAINER
    made = trainer_in(trainer) if trainer is not None else Trainer
    config = load(blueprint, made.Recipe)
    stem = source.parent.name[:32].rstrip("_-") or "run"
    directory = runs_root(root) / f"{stem}__{ShortUUID().random(length=7)}"
    directory.mkdir(parents=True)
    shutil.copyfile(source, directory / record.CONFIG)
    if trainer is not None:
        trainer = Path(trainer).resolve()
        noted = {"trainer": str(trainer), "fingerprint": fingerprint(trainer.parent)}
        record.write_json(directory / record.RECIPE, noted)
    return directory, config


def reopened(directory: Path, *, tag: str = "last") -> tuple[Blueprint, Mapping[str, Any]]:
    """Return a run's blueprint and its checkpoint row `tag`, the last one by default."""
    directory = Path(directory)
    if not (directory / record.CONFIG).is_file():
        raise ValueError(f"no run at {directory}")
    noted = record.read_json(directory / record.RECIPE)
    if noted and Path(noted["trainer"]).is_file():
        now = fingerprint(Path(noted["trainer"]).parent)
        if now != noted.get("fingerprint"):
            logger.warning(
                "%s: the recipe's code at %s changed since the run started; it goes on with "
                "the code as it is now",
                directory.name,
                Path(noted["trainer"]).parent,
            )
    rows = list(record.read(directory / record.CHECKPOINTS))
    if tag == "last":
        row = rows[-1] if rows else None
    else:
        row = next((r for r in reversed(rows) if r["tag"] == tag), None)
    if row is None:
        tags = ", ".join(r["tag"] for r in rows) or "none"
        raise ValueError(f"{directory.name} has no checkpoint {tag!r}; it has {tags}")
    return load(directory, trainer_class(directory).Recipe), row


def trainer_class(directory: Path) -> type[Trainer]:
    """Return the run's trainer: its recipe's, else its own copy's, else the base `Trainer`."""
    directory = Path(directory)
    noted = record.read_json(directory / record.RECIPE)
    if noted:
        path = Path(noted["trainer"])
        if not path.is_file():
            raise ValueError(
                f"{directory.name} is trained by {path}, which is gone: put the recipe back "
                "where it was"
            )
        return trainer_in(path)
    kept = directory / record.TRAINER
    return trainer_in(kept) if kept.is_file() else Trainer


def fingerprint(folder: Path) -> str:
    """Return a digest of the names and contents of the Python files in `folder`."""
    digest = sha256()
    for one in sorted(Path(folder).glob("*.py")):
        digest.update(one.name.encode() + b"\0" + one.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def trainer_in(path: Path) -> type[Trainer]:
    """Return the single `Trainer` subclass a Python file defines, loaded with its folder."""
    path = Path(path).resolve()
    package = f"shipyard_recipe_{sha256(str(path.parent).encode()).hexdigest()[:12]}"
    if package not in sys.modules:
        folder = importlib.util.spec_from_loader(package, loader=None, is_package=True)
        assert folder is not None
        held = importlib.util.module_from_spec(folder)
        held.__path__ = [str(path.parent)]
        sys.modules[package] = held
    name = f"{package}.{path.stem}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{path} is not a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    found = [
        one
        for one in vars(module).values()
        if isinstance(one, type) and issubclass(one, Trainer) and one.__module__ == name
    ]
    if len(found) != 1:
        raise ValueError(
            f"{path} defines {len(found)} Trainer subclasses; a trainer file defines one"
        )
    return found[0]


@contextmanager
def running(directory: Path, **about: Any) -> Iterator[None]:
    """Write this invocation's process note, and record how it ended."""
    path = Path(directory) / record.PROCESS
    earlier = record.read_json(path)
    before = earlier.pop("earlier", []) if earlier else []
    note: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": record.now(),
        "tinker_base_url": tinker_base_url(),
        **{key: value for key, value in about.items() if value},
    }
    if earlier:
        note["earlier"] = [*before, earlier]
    record.write_json(path, note)
    error: str | None = None
    try:
        yield
    except KeyboardInterrupt as stopped:
        error = f"stopped: {str(stopped) or 'SIGINT'}"
        raise
    except BaseException as failed:
        error = f"{type(failed).__name__}: {failed}"
        raise
    finally:
        note.update(finished_at=record.now(), failed=error is not None, error=error)
        record.write_json(path, note)


def tinker_base_url() -> str:
    """Return the Tinker server this process talks to."""
    from tinker.lib.base_url import DEFAULT_BASE_URL

    return (os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
