"""What `runs` and `show` read off a run directory and how `show` prints it: a run's
state, its whole record as plain data, and each section at the console's width."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from shipyard import record

console = Console()

STYLES = {"running": "cyan", "finished": "green", "failed": "red", "died": "yellow"}
#: What `show` prints of the process note, as `label: key`.
FACTS = {
    "pid": "pid",
    "started": "started_at",
    "finished": "finished_at",
    "version": "version",
    "kind": "kind",
    "blueprint": "blueprint",
    "directory": "directory",
}
#: What `show` prints of the process note only when the run noted it.
NOTED = {"smoke": "smoke", "proxy": "proxy", "backend": "tinker_base_url"}
#: How many job rows `show` prints without `--full`.
LAST_JOBS = 10
#: The columns a table of rows leads with, never wrapped.
PINNED = ("name", "at")


def alive(pid: int | None) -> bool:
    """Whether that process exists: signal 0 delivers nothing, and a `PermissionError`
    means it is there under another user."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def state_of(note: dict[str, Any]) -> str:
    """`failed`, `finished`, `running` or `died`: a note with no finish and no live
    process was killed, which must not look like one still working."""
    if note.get("failed"):
        return "failed"
    if note.get("finished_at"):
        return "finished"
    return "running" if alive(note.get("pid")) else "died"


def sections(directory: Path) -> dict[str, Any]:
    """The whole record as plain data, one entry per section `show` prints: the process
    note with its state, the metrics count and last row, every job and checkpoint row,
    and the costs document. A section whose file is absent is None, not empty."""
    directory = Path(directory)
    note = record.read_json(directory / record.PROCESS)
    costs = directory / record.COSTS
    return {
        "process": {"state": state_of(note), **note, "directory": str(directory)},
        "metrics": _tail(directory / record.METRICS),
        "jobs": _listed(directory / record.JOBS),
        "checkpoints": _listed(directory / record.CHECKPOINTS),
        "costs": record.read_json(costs) if costs.is_file() else None,
    }


def _listed(path: Path) -> dict[str, Any] | None:
    """A JSONL file's rows with their count, or None when there is no such file."""
    if not path.is_file():
        return None
    rows = list(record.read(path))
    return {"count": len(rows), "rows": rows}


def _tail(path: Path) -> dict[str, Any] | None:
    """A JSONL file's row count and last row, read without holding every row; None if absent."""
    if not path.is_file():
        return None
    count, last = 0, None
    for row in record.read(path):
        count, last = count + 1, row
    return {"count": count, "last": last}


def styled(state: str) -> str:
    return f"[{STYLES.get(state, 'white')}]{state}[/]"


def print_run(run_id: str, found: dict[str, Any], *, full: bool) -> None:
    """`sections` as `show` prints them: the process note, the metrics' last row, the job
    rows (the last few unless `full`), every checkpoint row, and the costs."""
    console.print(f"[bold cyan]{escape(run_id)}[/bold cyan]", highlight=False)
    _print_process(found["process"])
    if found["metrics"] is not None:
        _print_metrics(found["metrics"])
    if found["jobs"] is not None:
        print_rows("jobs", found["jobs"], None if full else LAST_JOBS)
    if found["checkpoints"] is not None:
        print_rows("checkpoints", found["checkpoints"], None)
    if found["costs"] is not None:
        console.print("[bold]costs[/bold]")
        for line in json.dumps(found["costs"], indent=2).splitlines():
            console.print(f"  {line}", markup=False, highlight=False, soft_wrap=True)


def _print_process(process: dict[str, Any]) -> None:
    console.print("[bold]process[/bold]")
    width = max(len(label) for label in [*FACTS, *NOTED]) + 2
    _pair("state", styled(process["state"]), width)
    if process.get("error"):
        _pair("error", f"[red]{escape(str(process['error']))}[/red]", width)
    for label, key in FACTS.items():
        _pair(label, escape(_text(process.get(key))), width)
    for label, key in NOTED.items():
        if key in process:
            _pair(label, escape(_text(process[key])), width)


def _print_metrics(metrics: dict[str, Any]) -> None:
    """The row count and the last row's fields, one per line."""
    last = metrics["last"] or {}
    console.print(f"[bold]metrics[/bold]  {_rows(metrics['count'])}", highlight=False)
    width = max((len(key) for key in last), default=0) + 2
    for key, value in last.items():
        _pair(key, escape(_text(value)), width)


def print_rows(name: str, section: dict[str, Any], limit: int | None) -> None:
    """The row count and a table of the rows, the last `limit` of them when that cuts; a
    table wider than the console would fold every cell to "…", so each row is a block."""
    count, rows = section["count"], section["rows"]
    shown = rows if limit is None or count <= limit else rows[-limit:]
    cut = f" (last {len(shown)}; --full for all)" if len(shown) < count else ""
    console.print(f"[bold]{name}[/bold]  {_rows(count)}{cut}", highlight=False)
    if not shown:
        return
    table = _table(shown)
    if console.measure(table).maximum + 2 <= console.width:
        console.print(Padding(table, (0, 0, 0, 2), expand=False))
        return
    columns = _columns(shown)
    for index, row in enumerate(shown):
        if index:
            console.print()
        width = max(len(key) for key in columns if key in row) + 2
        for key in columns:
            if key in row:
                _pair(key, escape(_cell(key, row)), width)


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    """Every key the rows carry: `name`, `at`, then the rest in row order, so a reader
    sees whatever later blocks record without the CLI knowing the keys."""
    seen = list(dict.fromkeys(key for row in rows for key in row))
    pinned = [key for key in PINNED if key in seen]
    return pinned + [key for key in seen if key not in pinned]


def _table(rows: list[dict[str, Any]]) -> Table:
    """One column per key the rows carry, in `_columns`' order."""
    columns = _columns(rows)
    table = Table(show_header=True, header_style="bold")
    for key in columns:
        table.add_column(key, no_wrap=key in PINNED)
    for row in rows:
        table.add_row(*(_cell(key, row) for key in columns))
    return table


def _cell(key: str, row: dict[str, Any]) -> str:
    """A table cell: blank for a key the row lacks; `at` cut to the second, as `runs` does."""
    if key not in row:
        return ""
    return _text(row[key])[:19] if key == "at" else _text(row[key])


def _text(value: Any) -> str:
    """Strings verbatim, nothing as a dash, anything else as compact JSON."""
    if value is None:
        return "—"
    return value if isinstance(value, str) else json.dumps(value)


def _rows(count: int) -> str:
    return f"{count} row{'' if count == 1 else 's'}"


def _pair(key: str, value: str, width: int) -> None:
    """One `  key  value` line; the value is already marked up or escaped."""
    console.print(f"  {key:<{width}}{value}", highlight=False, soft_wrap=True)
