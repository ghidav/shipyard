"""The record's writers and readers: append-only JSONL, atomic JSON, and the file names."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The record's files, so a reader has one place to learn their names.
CONFIG = "run.toml"
PROCESS = "process.json"
JOBS = "jobs.jsonl"
METRICS = "metrics.jsonl"
CHECKPOINTS = "checkpoints.jsonl"
REQUESTS = "requests.jsonl"
COSTS = "costs.json"
#: A directory, not a file: the modules the run carried (`carried/`), and a search's
#: `seed/` and `best/`.
MODULES = "modules"


def now() -> str:
    return datetime.now(UTC).isoformat()


def append(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """One line, stamped `at`, opened per call and flushed before returning, no fsync.
    A cut last line gets its newline first, so the new row is not welded onto it."""
    stamped = {"at": now(), **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if handle.tell() and not _ends_cleanly(path):
            handle.write("\n")
        handle.write(json.dumps(stamped) + "\n")
        handle.flush()
    return stamped


def _ends_cleanly(path: Path) -> bool:
    """Whether the file ends in a newline, i.e. whether its last line is whole."""
    try:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            return handle.read(1) == b"\n"
    except OSError:  # pragma: no cover - an empty or unreadable file appends cleanly
        return True


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Every line that parses, in order; blank lines and a truncated last line are skipped."""
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
    """A whole document, written beside the file and renamed over it, so a reader sees
    the old document or the new one and never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """The document, or `{}` for a file that is missing, unreadable or not an object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
