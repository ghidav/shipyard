"""Run-directory file names, and readers and writers for append-only JSONL and atomic JSON."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: File names in a run directory, defined in one place for every reader and writer.
CONFIG = "run.toml"
PROCESS = "process.json"
JOBS = "jobs.jsonl"
METRICS = "metrics.jsonl"
CHECKPOINTS = "checkpoints.jsonl"
REQUESTS = "requests.jsonl"
COSTS = "costs.json"
#: Copy of the trainer file the run was started with, if one was given.
TRAINER = "trainer.py"
#: Directory, not a file: the modules the run carried (`carried/`), and a search's `seed/`
#: and `best/`.
MODULES = "modules"


def now() -> str:
    return datetime.now(UTC).isoformat()


def append(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Append one row as a JSON line stamped with `at`, and return the stamped row.

    The file is opened per call and flushed before returning, without fsync. If the last
    line was cut short, a newline is written first so the new row stays on its own line."""
    stamped = {"at": now(), **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if handle.tell() and not _ends_cleanly(path):
            handle.write("\n")
        handle.write(json.dumps(stamped) + "\n")
        handle.flush()
    return stamped


def _ends_cleanly(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            return handle.read(1) == b"\n"
    except OSError:  # pragma: no cover - an empty or unreadable file appends cleanly
        return True


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each row of a JSONL file in order. Blank lines and lines that do not parse as a
    JSON object, such as a truncated last line, are skipped."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document atomically: write it to a file beside `path`, then rename it
    over `path`, so a reader sees either the old document or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, or return `{}` for a file that is missing, unreadable or not an
    object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
