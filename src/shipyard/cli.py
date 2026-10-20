from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from shipyard.config import Blueprint, ConfigError
from shipyard.run.directory import open_run, reopened, running, trainer_class

app = typer.Typer(
    help="Train with a recipe written from four primitives: draw, rollout, logprobs, update.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)  # the shell wins over the file
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def run(
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
    """Open a run for BLUEPRINT and train it with its recipe: TRAINER, else the trainer.py
    beside the run.toml, else the base trainer."""
    try:
        directory, config = open_run(blueprint, root=root, trainer=trainer)
    except (ConfigError, ValueError) as refused:
        _refuse(refused)
    made = trainer_class(directory).__name__
    typer.echo(f"run {directory.name} in {directory.parent}, trained by {made}")
    _train(directory, config, None)


@app.command()
def resume(
    directory: Annotated[Path, typer.Argument(help="The run directory.")],
    tag: Annotated[str, typer.Option(help="The checkpoint to go on from.")] = "last",
) -> None:
    """Go on with the run in DIRECTORY from its checkpoint TAG, the last one by default."""
    try:
        config, row = reopened(directory, tag=tag)
    except (ConfigError, ValueError) as refused:
        _refuse(refused)
    typer.echo(f"run {Path(directory).name} from {row['tag']} at step {row['step']}")
    _train(Path(directory), config, row)


def _train(directory: Path, config: Blueprint, row: Mapping[str, Any] | None) -> None:
    """Train the run, and exit 130 if a signal stopped it: Ctrl-C, SIGTERM or SIGHUP."""
    made = trainer_class(directory)
    resumed = row["tag"] if row else None
    try:
        # The trainer is made inside the process note, so a recipe's `__init__` that fails,
        # such as on a missing [recipe] key, is recorded as the run's error.
        with (
            running(directory, trainer=made.__name__, resumed=resumed),
            made(config, directory) as trainer,
        ):
            if row is not None:
                trainer.restore(row)
            _until_stopped(trainer.train())
    except KeyboardInterrupt as stopped:
        typer.echo(f"run {directory.name} stopped: {str(stopped) or 'SIGINT'}", err=True)
        raise typer.Exit(130) from None


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


def _refuse(refused: Exception) -> None:
    problems = refused.problems if isinstance(refused, ConfigError) else [str(refused)]
    for one in problems:
        typer.echo(one, err=True)
    raise typer.Exit(1)
