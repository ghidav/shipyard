from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import shlex
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import tomli_w
from shortuuid import ShortUUID

from shipyard.config import Config, config_file, load, parsed, runs
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
    path: Path,
    *,
    root: Path | None = None,
    trainer: Path | type[Trainer] | None = None,
    origin: Mapping[str, Any] | None = None,
    overrides: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> tuple[Path, Config]:
    """Create a config's run directory, noting its recipe and any `origin` in `run.json`. The
    recipe is `trainer`: its trainer file, the config folder's `trainer.py` by default, or a
    `Trainer` subclass, which a run made from Python passes and a resume cannot find again.
    `home` is the folder the config was written in, when `path` is a run's copy of it."""
    source = config_file(path)
    if trainer is None:
        trainer = recipe_of(source)
    made = trainer if isinstance(trainer, type) else trainer_in(trainer)
    home = Path(home) if home is not None else source.parent.resolve()
    config = load(path, made.Recipe, overrides, origin=home)
    # A run that goes on from another is named as that one is, without its id.
    named = source.parent.name.rsplit("__", 1)[0] if origin else source.parent.name
    stem = named[:32].rstrip("_-") or "run"
    directory = runs_root(root) / f"{stem}__{ShortUUID().random(length=7)}"
    directory.mkdir(parents=True)
    if overrides:
        # The config as the run has it, which a resume reads again.
        given = " ".join(shlex.join([f"--{name}", text]) for name, text in overrides.items())
        header = " ".join(f"# {source} with {given}".splitlines())
        written = tomli_w.dumps(parsed(source, overrides))
        (directory / files.CONFIG).write_text(f"{header}\n{written}", encoding="utf-8")
    else:
        shutil.copyfile(source, directory / files.CONFIG)
    # Where the config was written: a harness it gives as a file is found from there again.
    noted: dict[str, Any] = {"home": str(home)}
    if not isinstance(trainer, type):
        trainer = Path(trainer).resolve()
        noted["recipe"] = {"trainer": str(trainer), "fingerprint": fingerprint(trainer.parent)}
    if origin is not None:
        noted["origin"] = dict(origin)
    files.write_json(directory / files.RUN, noted)
    return directory, config


def open_runs(
    path: Path,
    *,
    root: Path | None = None,
    trainer: Path | type[Trainer] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> list[tuple[Path, Config]]:
    """Create the run directory of each run a config starts, as `open_run` does: one, or one
    for each seed when a model's `seed` is a list, each run's copy of the config holding its
    own seed. Runs that start together note each other in `run.json`, so that a resume of one
    brings the others back with it. Raise `ConfigError` for a config no run could take, or
    one whose runs could not start together, before any directory is made."""
    source = config_file(path)
    made = trainer if isinstance(trainer, type) else trainer_in(recipe_of(source, trainer))
    opened = [
        open_run(path, root=root, trainer=trainer, overrides=one)
        for one in runs(path, made.Recipe, overrides)
    ]
    together([directory for directory, _ in opened])
    return opened


def together(directories: Sequence[Path]) -> None:
    """Note in the `run.json` of each of several runs the runs it started with, itself among
    them, in the order they started."""
    if len(directories) < 2:
        return
    names = [one.name for one in directories]
    for one in directories:
        noted = files.read_json(one / files.RUN)
        files.write_json(one / files.RUN, {**noted, "together": names})


def recipe_of(path: Path, trainer: Path | None = None) -> Path:
    """Return the trainer file a config runs with: `trainer`, else the `trainer.py` beside the
    config's run.toml. Raise `ValueError` when there is none."""
    if trainer is not None:
        return Path(trainer)
    found = config_file(path).parent / files.TRAINER
    if not found.is_file():
        raise ValueError(
            f"{found.parent} holds no {files.TRAINER}: a recipe is its run.toml and the "
            "trainer.py that writes its step"
        )
    return found


def resumed(origin: Path, *, tag: str = "last") -> tuple[Path, Config, Checkpoint] | None:
    """Open a run beside `origin` that goes on from its checkpoint `tag`, the last by default,
    and return its directory, its config and that checkpoint; None when the checkpoint is
    at the run's last step, so nothing is left to run."""
    config, checkpoint = reopened(origin, tag=tag)
    if checkpoint.step >= config.recipe.steps:
        return None
    directory, config = continued(origin, checkpoint)
    return directory, config, checkpoint


def resumed_together(
    origin: Path, *, tag: str = "last"
) -> list[tuple[Path, Config, Checkpoint, Path]]:
    """Open, for `origin` and each run it started with, a run beside it that goes on from its
    checkpoint `tag`, its own last by default; return each new run's directory, config and
    checkpoint, and the run it goes on from, in the order those started. The new runs note
    each other in turn. A run with nothing left to run is left out; so is one started with
    `origin` that is gone, whose process still runs, or that has no checkpoint yet. Raise
    `ValueError` for a `tag` one of them lacks, before any run is opened."""
    origin = Path(origin)
    own = origin.resolve().name
    names = files.read_json(origin / files.RUN).get("together") or []
    names = names if own in names else [own, *names]
    going: list[tuple[Path, Checkpoint]] = []
    for name in names:
        run = origin if name == own else origin.resolve().parent / name
        if run is not origin:
            why = _left(run, tag)
            if why:
                logger.warning(
                    "%s, started with %s, %s: it is left as it is", name, origin.name, why
                )
                continue
        config, checkpoint = reopened(run, tag=tag)
        if checkpoint.step < config.recipe.steps:
            going.append((run, checkpoint))
    opened = [(*continued(run, checkpoint), checkpoint, run) for run, checkpoint in going]
    together([directory for directory, *_ in opened])
    return opened


def _left(run: Path, tag: str) -> str | None:
    """Why a resume leaves alone a run started with the one it was asked for; None to take it."""
    if not (run / files.CONFIG).is_file():
        return "is gone"
    if alive(run):
        return "still runs"
    if tag == "last" and not list(files.read(run / files.CHECKPOINTS)):
        return "has no checkpoint to go on from"
    return None


def alive(directory: Path) -> bool:
    """Whether the process that runs the run in `directory` is still up, as its `run.json`
    notes it: one that noted no end, under a process id that still answers."""
    note = files.read_json(Path(directory) / files.RUN).get("process") or {}
    pid = note.get("pid")
    if not pid or note.get("finished_at"):
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process answers
    return True


def continued(origin: Path, checkpoint: Checkpoint) -> tuple[Path, Config]:
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
        home=_home(noted),
    )
    carried = origin / files.MODULES / files.CARRIED
    if carried.is_dir():
        shutil.copytree(carried, directory / files.MODULES / files.CARRIED)
    return directory, config


def reopened(directory: Path, *, tag: str = "last") -> tuple[Config, Checkpoint]:
    """Return a run's config and its checkpoint `tag`, the last one by default."""
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
    home = _home(files.read_json(directory / files.RUN))
    return load(directory, trainer_class(directory).Recipe, origin=home), checkpoint


def _home(noted: Mapping[str, Any]) -> Path | None:
    """The folder a run's config was written in, as its `run.json` notes it; None for a run
    that noted none."""
    return Path(noted["home"]) if noted.get("home") else None


def trainer_class(directory: Path) -> type[Trainer]:
    """Return the run's trainer: its recipe's, else its own copy's."""
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
    if not kept.is_file():
        raise ValueError(f"{directory.name} names no recipe, and holds no {files.TRAINER}")
    return trainer_in(kept)


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
    made = found[0]
    if made.step is Trainer.step:
        raise ValueError(
            f"{path}: {made.__name__} writes no step: a recipe writes `async def step`"
        )
    return made


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
    except (KeyboardInterrupt, asyncio.CancelledError) as stopped:
        # `shipyard run` cancels the task a run trains in, where a run made from Python is
        # interrupted; the cancellation carries the signal's name.
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
