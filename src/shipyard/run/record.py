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
#: The trainer file a recipe folder holds. A run started before runs kept their recipe's
#: place, rather than a copy of its code, holds its own copy under this name.
TRAINER = "trainer.py"
#: Where the run's trainer file is, and a fingerprint of its recipe's code then.
RECIPE = "recipe.json"
#: Directory, not a file: the modules the run carried (`carried/`), and a search's `seed/`
#: and `best/`.
MODULES = "modules"


def now() -> str:
    return datetime.now(UTC).isoformat()


def append(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Append one row as a JSON line stamped with `at`, and return the stamped row."""
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
    """Yield each row of a JSONL file in order, skipping any line that does not parse."""
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
    """Write a JSON document atomically, through a file beside `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, or `{}` for a file missing, unreadable or not an object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def next_job(name: str, directory: Path, jobs_dir: Path) -> int:
    """Return the next job number: one past the highest in the job log or on disk, else 0."""
    named = [row.get("job") for row in read(directory / JOBS)]
    if jobs_dir.is_dir():
        named += [one.name for one in jobs_dir.iterdir()]
    prefix = f"{name}-"
    numbers = [
        int(one[len(prefix) :])
        for one in named
        if isinstance(one, str) and one.startswith(prefix) and one[len(prefix) :].isdigit()
    ]
    return max(numbers, default=-1) + 1
