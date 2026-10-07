"""The command line: check a blueprint, run it, list the runs, show one, serve a model."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table

from shipyard import config, record, resolved
from shipyard.config import Finding
from shipyard.proxy.client import Unreachable
from shipyard.recipes import run_recipe
from shipyard.run import Run, RunRefused, runs_root
from shipyard.serving import NothingServed, ProxyGone
from shipyard.show import print_run, sections, state_of, styled
from shipyard.smoke import SmokeFailed
from shipyard.smoke import smoke as smoke_job

app = typer.Typer(
    name="shipyard",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

BlueprintArgument = Annotated[
    Path, typer.Argument(help="a directory holding run.toml, or the file")
]
RunArgument = Annotated[str, typer.Argument(help="a run id, `<blueprint>__<7 chars>`")]
RootOption = Annotated[
    Path | None, typer.Option("--root", help="where runs live; SHIPYARD_RUNS or ./runs by default")
]
JsonOption = Annotated[bool, typer.Option("--json", help="as JSON")]
VerboseOption = Annotated[bool, typer.Option("--verbose", help="show the ok lines too")]
SectionsOption = Annotated[
    bool, typer.Option("--json", help="one object with the five sections, every row included")
]
FullOption = Annotated[bool, typer.Option("--full", help="every job row, not the last few")]
SmokeOption = Annotated[
    bool,
    typer.Option(
        "--smoke",
        help="one task, one rollout, no gradient, checkpoint or reflection: the sampling "
        "path end to end, then a report; exits 0 only if a trial came back",
    ),
]

MARKS = {
    "ok": "[green]ok[/green]",
    "warning": "[yellow]warning[/yellow]",
    "blocked": "[red]blocked[/red]",
}


@app.callback()
def main() -> None:
    """Post-training on Harbor jobs: a blueprint is a config, a run is its record."""
    load_dotenv(Path.cwd() / ".env", override=False)  # the shell wins over the file
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_path=False)],
    )


@app.command()
def check(
    blueprint: BlueprintArgument, as_json: JsonOption = False, verbose: VerboseOption = False
) -> None:
    """Whether a blueprint could run: the config as resolved, then every problem at once,
    before anything is spent; the `ok` lines only under --verbose."""
    found = resolved.report(blueprint)
    if as_json:
        report = {"config": found.config, "findings": [asdict(f) for f in found.findings]}
        print(json.dumps(report, indent=2))
    else:
        if found.config is not None:
            console.print(found.text(), markup=False, highlight=False, soft_wrap=True)
        _print_findings([f for f in found.findings if verbose or f.level != "ok"])
    raise typer.Exit(1 if found.blocked else 0)


@app.command()
def run(blueprint: BlueprintArgument, root: RootOption = None, smoke: SmokeOption = False) -> None:
    """Check the blueprint, open a run, hand it to its recipe (or, with --smoke, sample one
    trial and report it), and record how it ended."""
    findings = config.check(blueprint)
    _print_findings([found for found in findings if found.level != "ok"])
    if _blocked(findings):
        console.print("[red]blocked; nothing was started[/red]")
        raise typer.Exit(1)
    try:
        opened = Run.open(blueprint, root=root, smoke=smoke)
    except RunRefused as refused:
        console.print(f"[red]{escape(str(refused))}[/red]")
        raise typer.Exit(1) from None
    console.print(
        f"[bold cyan]{opened.id}[/bold cyan]  [dim]{escape(str(opened.directory))}[/dim]",
        highlight=False,
        soft_wrap=True,
    )
    restore = _stop_on_signals()
    code = 0
    try:
        with opened:
            if smoke:
                _smoke(opened)
            else:
                asyncio.run(run_recipe(opened))
    except KeyboardInterrupt:
        code = 130
    except (SmokeFailed, NothingServed, ProxyGone, Unreachable):
        code = 1  # a stop that says why in one line: the run's error, printed below
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        restore()
    note = Run.read(opened.directory)
    if note.get("error"):  # the one text `__exit__` recorded, not a second composition
        style = "yellow" if code == 130 else "red"
        console.print(f"[{style}]{escape(str(note['error']))}[/]", highlight=False, soft_wrap=True)
    console.print(f"{opened.id}  {styled(state_of(note))}", highlight=False)
    raise typer.Exit(code)


def _smoke(opened: Run) -> None:
    """The smoke job and its report; a job that brought nothing back raises after it."""
    smoked = asyncio.run(smoke_job(opened))
    for line in smoked.lines():
        console.print(line, markup=False, highlight=False, soft_wrap=True)
    smoked.verify()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def serve(ctx: typer.Context) -> None:
    """Serve a Tinker model to a harness in a sandbox; `shipyard serve --help` lists the flags.
    They are argparse's, in `shipyard.serve`, so a container entrypoint and this verb parse
    one list and not two."""
    from shipyard.serve import main as serve_main

    raise typer.Exit(serve_main(list(ctx.args)))


@app.command()
def runs(root: RootOption = None) -> None:
    """Every run under the root, newest first."""
    found = [(directory.name, Run.read(directory)) for directory in _run_dirs(runs_root(root))]
    if not found:
        console.print("[dim](no runs)[/dim]")
        return
    found.sort(key=lambda pair: str(pair[1].get("started_at") or ""), reverse=True)
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("state")
    table.add_column("started", style="dim")
    table.add_column("finished", style="dim")
    for run_id, note in found:
        table.add_row(
            run_id,
            str(note.get("kind") or ""),
            styled(state_of(note)),
            str(note.get("started_at") or "")[:19],
            str(note.get("finished_at") or "")[:19],
        )
    console.print(table)


@app.command()
def show(
    run_id: RunArgument,
    root: RootOption = None,
    full: FullOption = False,
    as_json: SectionsOption = False,
) -> None:
    """One run, section by section: process, metrics, jobs, checkpoints, costs."""
    directory = runs_root(root) / run_id
    if not (directory / record.CONFIG).is_file():
        console.print(
            f"[red]no run named[/red] {escape(run_id)} [dim]under {runs_root(root)}[/dim]"
        )
        raise typer.Exit(1)
    found = sections(directory)
    if as_json:
        print(json.dumps(found, indent=2))
        return
    print_run(run_id, found, full=full)


def _blocked(findings: list[Finding]) -> bool:
    return any(found.level == "blocked" for found in findings)


def _print_findings(findings: list[Finding]) -> None:
    for found in findings:
        console.print(
            f"{MARKS[found.level]}  {escape(found.text)}", highlight=False, soft_wrap=True
        )


def _run_dirs(root: Path) -> list[Path]:
    """Every directory under the root that holds a `run.toml`."""
    if not root.is_dir():
        return []
    return sorted(found for found in root.iterdir() if (found / record.CONFIG).is_file())


def _stop_on_signals() -> Callable[[], None]:
    """Make SIGTERM and SIGHUP raise `KeyboardInterrupt(<name>)` in the main thread, as
    Ctrl-C does, so the recipe unwinds and `Run.__exit__` records the stop; returns the
    restorer of the previous handlers."""

    def stop(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    previous: dict[int, Any] = {}
    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.signal(number, stop)
        except ValueError:
            pass  # not the main thread: a caller embedding the CLI keeps its own handlers

    def restore() -> None:
        for number, handler in previous.items():
            signal.signal(number, handler)

    return restore
