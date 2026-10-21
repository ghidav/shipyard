from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import shutil
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import tomli_w
from shortuuid import ShortUUID

from shipyard.config import Blueprint, config_file, load, parsed
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint, found
from shipyard.trainer import Trainer

logger = logging.getLogger(__name__)


def runs_root(flag: Path | None = None) -> Path:
    """Return the directory runs live in: `--root`, else `SHIPYARD_RUNS`, else `./runs`."""
    if flag is not None:
        return Path(flag)
    named = os.environ.get("SHIPYARD_RUNS")
    return Path(named) if named else Path.cwd() / "runs"


def open_run(
    blueprint: Path,
    *,
    root: Path | None = None,
    trainer: Path | None = None,
    origin: Mapping[str, Any] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> tuple[Path, Blueprint]:
    """Create a blueprint's run directory, noting its recipe and any `origin` in `run.json`."""
    source = config_file(blueprint)
    if trainer is None and (source.parent / files.TRAINER).is_file():
        trainer = source.parent / files.TRAINER
    made = trainer_in(trainer) if trainer is not None else Trainer
    config = load(blueprint, made.Recipe, overrides)
    # A run that goes on from another is named as that one is, without its id.
    named = source.parent.name.rsplit("__", 1)[0] if origin else source.parent.name
    stem = named[:32].rstrip("_-") or "run"
    directory = runs_root(root) / f"{stem}__{ShortUUID().random(length=7)}"
    directory.mkdir(parents=True)
    if overrides:
        # The blueprint as the run has it, which a resume reads again.
        given = " ".join(shlex.join([f"--{name}", text]) for name, text in overrides.items())
        header = " ".join(f"# {source} with {given}".splitlines())
        written = tomli_w.dumps(parsed(source, overrides))
        (directory / files.CONFIG).write_text(f"{header}\n{written}", encoding="utf-8")
    else:
        shutil.copyfile(source, directory / files.CONFIG)
    noted: dict[str, Any] = {}
    if trainer is not None:
        trainer = Path(trainer).resolve()
        noted["recipe"] = {"trainer": str(trainer), "fingerprint": fingerprint(trainer.parent)}
    if origin is not None:
        noted["origin"] = dict(origin)
    files.write_json(directory / files.RUN, noted)
    return directory, config


def resumed(origin: Path, *, tag: str = "last") -> tuple[Path, Blueprint, Checkpoint] | None:
    """Open a run beside `origin` that goes on from its checkpoint `tag`, the last by default,
    and return its directory, its blueprint and that checkpoint; None when the checkpoint is
    at the run's last step, so nothing is left to run."""
    config, checkpoint = reopened(origin, tag=tag)
    if checkpoint.step >= config.recipe.steps:
        return None
    directory, config = continued(origin, checkpoint)
    return directory, config, checkpoint


def continued(origin: Path, checkpoint: Checkpoint) -> tuple[Path, Blueprint]:
    """Open a run beside `origin` that goes on from its `checkpoint`, noting the chain."""
    origin = Path(origin).resolve()
    noted = files.read_json(origin / files.RUN)
    trainer = (noted.get("recipe") or {}).get("trainer")
    first = (noted.get("origin") or {}).get("first", origin.name)
    directory, config = open_run(
        origin / files.CONFIG,
        root=origin.parent,
        trainer=Path(trainer) if trainer else None,
        origin={"run": str(origin), "tag": checkpoint.tag, "first": first},
    )
    carried = origin / files.MODULES / files.CARRIED
    if carried.is_dir():
        shutil.copytree(carried, directory / files.MODULES / files.CARRIED)
    return directory, config


def reopened(directory: Path, *, tag: str = "last") -> tuple[Blueprint, Checkpoint]:
    """Return a run's blueprint and its checkpoint `tag`, the last one by default."""
    directory = Path(directory)
    if not (directory / files.CONFIG).is_file():
        raise ValueError(f"no run at {directory}")
    recipe = files.read_json(directory / files.RUN).get("recipe")
    if recipe and Path(recipe["trainer"]).is_file():
        now = fingerprint(Path(recipe["trainer"]).parent)
        if now != recipe.get("fingerprint"):
            logger.warning(
                "%s: the recipe's code at %s changed since the run started; the run that goes "
                "on from it runs the code as it is now",
                directory.name,
                Path(recipe["trainer"]).parent,
            )
    checkpoint = found(directory, tag)
    return load(directory, trainer_class(directory).Recipe), checkpoint


def trainer_class(directory: Path) -> type[Trainer]:
    """Return the run's trainer: its recipe's, else its own copy's, else the base `Trainer`."""
    directory = Path(directory)
    recipe = files.read_json(directory / files.RUN).get("recipe")
    if recipe:
        path = Path(recipe["trainer"])
        if not path.is_file():
            raise ValueError(
                f"{directory.name} is trained by {path}, which is gone: put the recipe back "
                "where it was"
            )
        return trainer_in(path)
    kept = directory / files.TRAINER
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
    """Note the process that runs the run in its `run.json`, and how it ended."""
    path = Path(directory) / files.RUN
    noted = files.read_json(path)
    note: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": files.now(),
        "tinker_base_url": tinker_base_url(),
        **{key: value for key, value in about.items() if value},
    }
    files.write_json(path, {**noted, "process": note})
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
        note.update(finished_at=files.now(), failed=error is not None, error=error)
        files.write_json(path, {**noted, "process": note})


def tinker_base_url() -> str:
    """Return the Tinker server this process talks to."""
    from tinker.lib.base_url import DEFAULT_BASE_URL

    return (os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
