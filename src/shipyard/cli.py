from __future__ import annotations

import asyncio
import logging
import shutil
import signal
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from typer.core import TyperCommand

from shipyard.config import Blueprint, ConfigError
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.directory import open_run, resumed, running, trainer_class

#: Where `init` copies the recipes, and the env template it writes, in the directory it runs in.
RECIPES = "recipes"
ENV_EXAMPLE = ".env.example"
#: What a first run needs once `init` has run.
NEXT = """
next:
  cp .env.example .env                              # your keys
  uvx harbor datasets download <dataset> -o tasks   # the dataset a recipe's run.toml names
  shipyard run recipes/dapo                         # each recipe's README.md says what it needs"""

#: Where `run` keeps the blueprint keys given as `--table.key VALUE`, in Click's context.
OVERRIDES = "shipyard.overrides"
#: What `run --help` says of them.
SETTABLE = (
    "Any key of the blueprint can be set for this run as --table.key VALUE, such as "
    "--data.train swe-bench."
)

app = typer.Typer(
    help="Train with a recipe written from four primitives: draw, rollout, score, update.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)  # the shell wins over the file
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def init() -> None:
    """Copy the built-in recipes into ./recipes, and write ./.env.example, keeping what is there."""
    try:
        copied, kept, wrote = initialized(Path.cwd())
    except FileNotFoundError as missing:
        _refuse(missing)
    if copied:
        typer.echo(f"copied recipes/: {', '.join(copied)}")
    if kept:
        typer.echo(f"kept your recipes/: {', '.join(kept)}")
    if wrote:
        typer.echo(f"wrote {ENV_EXAMPLE}")
    typer.echo(NEXT)


class Overridable(TyperCommand):
    """A command that takes blueprint keys as `--table.key VALUE`, before Typer parses the rest."""

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        try:
            ctx.meta[OVERRIDES], rest = overrides_in(args)
        except ValueError as missing:
            ctx.fail(str(missing))
        return super().parse_args(ctx, rest)


@app.command(cls=Overridable, epilog=SETTABLE)
def run(
    ctx: typer.Context,
    blueprint: Annotated[Path, typer.Argument(help="A run.toml, or the directory holding one.")],
    trainer: Annotated[
        Path | None,
        typer.Argument(
            help="A Python file defining one Trainer subclass; by default the trainer.py "
            "beside the run.toml."
        ),
    ] = None,
    root: Annotated[Path | None, typer.Option(help="Where runs live; default ./runs.")] = None,
) -> None:
    """Run BLUEPRINT, trained by TRAINER, the trainer.py beside it, or the base trainer."""
    try:
        overrides = ctx.meta.get(OVERRIDES)
        directory, config = open_run(blueprint, root=root, trainer=trainer, overrides=overrides)
    except (ConfigError, ValueError) as refused:
        _refuse(refused)
    made = trainer_class(directory).__name__
    typer.echo(f"run {directory.name} in {directory.parent}, trained by {made}")
    _train(directory, config)


@app.command()
def resume(
    directory: Annotated[Path, typer.Argument(help="The run directory.")],
    tag: Annotated[str, typer.Option(help="The checkpoint to go on from.")] = "last",
) -> None:
    """Go on from checkpoint TAG of the run in DIRECTORY, the last by default, in a new run."""
    origin = Path(directory)
    try:
        going = resumed(origin, tag=tag)
    except (ConfigError, ValueError) as refused:
        _refuse(refused)
    if going is None:
        typer.echo(f"run {origin.name} finished at checkpoint {tag}: nothing is left to run")
        return
    directory, config, checkpoint = going
    typer.echo(f"run {directory.name} goes on from {origin.name} at {checkpoint.tag}")
    _train(directory, config, origin, checkpoint)


def _train(
    directory: Path,
    config: Blueprint,
    origin: Path | None = None,
    checkpoint: Checkpoint | None = None,
) -> None:
    """Train the run, from `checkpoint` of `origin` if given; exit 130 if a signal stops it."""
    made = trainer_class(directory)
    try:
        # The trainer is made inside the process note, so a recipe's `__init__` that fails,
        # such as on a missing [recipe] key, is recorded as the run's error.
        with running(directory, trainer=made.__name__), made(config, directory) as trainer:
            if origin is not None and checkpoint is not None:
                trainer.restore(origin, checkpoint)
            _until_stopped(trainer.run())
    except KeyboardInterrupt as stopped:
        typer.echo(f"run {directory.name} stopped: {str(stopped) or 'SIGINT'}", err=True)
        raise typer.Exit(130) from None


def initialized(into: Path) -> tuple[list[str], list[str], bool]:
    """Copy the shipped recipes and `.env.example` that `into` lacks: (copied, kept, wrote)."""
    recipes, env = shipped()
    target = into / RECIPES
    target.mkdir(parents=True, exist_ok=True)
    copied, kept = [], []
    for one in sorted(path for path in recipes.iterdir() if path.is_dir()):
        if one.name.startswith((".", "_")):
            continue
        if (target / one.name).exists():
            kept.append(one.name)
            continue
        skipped = shutil.ignore_patterns("__pycache__", "*.pyc", ".*")
        shutil.copytree(one, target / one.name, ignore=skipped)
        copied.append(one.name)
    if (recipes / "README.md").is_file() and not (target / "README.md").exists():
        shutil.copyfile(recipes / "README.md", target / "README.md")
    wrote = not (into / ENV_EXAMPLE).exists()
    if wrote:
        shutil.copyfile(env, into / ENV_EXAMPLE)
    return copied, kept, wrote


def shipped() -> tuple[Path, Path]:
    """Return the recipes and env template `init` copies: the package's, else the checkout's."""
    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    for recipes, env in (
        (package / RECIPES, package / "env.example"),
        (checkout / RECIPES, checkout / ENV_EXAMPLE),
    ):
        if recipes.is_dir() and env.is_file():
            return recipes, env
    raise FileNotFoundError(
        "this install of shipyard ships no recipes: reinstall it, or clone the repository"
    )


def _until_stopped(work: Coroutine[Any, Any, None]) -> None:
    """Run `work`, which SIGTERM and SIGHUP stop as Ctrl-C does."""
    caught: list[str] = []

    async def main() -> None:
        loop, task = asyncio.get_running_loop(), asyncio.current_task()
        previous: dict[int, Any] = {}
        for name in ("SIGTERM", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is None or task is None:
                continue
            handler = signal.getsignal(number)
            try:
                loop.add_signal_handler(number, _stop, task, caught, name)
            except (NotImplementedError, RuntimeError, ValueError):
                continue  # not the main thread, or a loop that cannot take signals
            previous[number] = handler
        try:
            await work
        finally:
            for number, handler in previous.items():
                loop.remove_signal_handler(number)
                signal.signal(number, handler)

    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        if caught:
            raise KeyboardInterrupt(caught[0]) from None
        raise


def _stop(task: asyncio.Task[Any], caught: list[str], name: str) -> None:
    caught.append(name)
    task.cancel()


def overrides_in(args: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Return the `--table.key VALUE` and `--table.key=VALUE` flags in `args`, and the rest."""
    found: dict[str, str] = {}
    rest: list[str] = []
    tokens = iter(args)
    for one in tokens:
        if one == "--":
            rest += [one, *tokens]
            break
        name, given, value = one.partition("=")
        if not (name.startswith("--") and "." in name):
            rest.append(one)
            continue
        if not given:
            value = next(tokens, "")
            if not value or value.startswith("--"):
                raise ValueError(f"{name} needs a value: {name} VALUE, or {name}=VALUE")
        found[name.removeprefix("--")] = value
    return found, rest


def _refuse(refused: Exception) -> None:
    problems = refused.problems if isinstance(refused, ConfigError) else [str(refused)]
    for one in problems:
        typer.echo(one, err=True)
    raise typer.Exit(1)
