from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shipyard.run import files


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint of a run: its tag and step, where the trainer's state was saved, and each
    trained model's saves by role. It is a row of the run's `checkpoints.jsonl`."""

    tag: str
    step: int
    #: The trainer's state, a path relative to the run directory; None for a row that has none.
    state: str | None = None
    #: Each trained model's training state, by role; none until an update moved its weights.
    training_state: dict[str, str] = field(default_factory=dict)
    #: Each trained model's sampler weights, by role, when the checkpoint kept them.
    sampler_weights: dict[str, str] = field(default_factory=dict)
    #: Hours its saves live: None for good, or when it saved no weights.
    ttl_hours: float | None = None

    @classmethod
    def save(
        cls,
        directory: Path,
        tag: str,
        step: int,
        state: Mapping[str, Any],
        *,
        training_state: Mapping[str, str] | None = None,
        sampler_weights: Mapping[str, str] | None = None,
        ttl_hours: float | None = None,
    ) -> Checkpoint:
        """Write the trainer's `state` into the run in `directory` and add the checkpoint's row;
        return the checkpoint."""
        made = cls(
            tag,
            step,
            f"{files.STATES}/step-{step}.json",
            dict(training_state or {}),
            dict(sampler_weights or {}),
            ttl_hours,
        )
        files.write_json(Path(directory) / made.state, dict(state))  # type: ignore[operator]
        files.append(Path(directory) / files.CHECKPOINTS, made.row())
        return made

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> Checkpoint:
        """Read a `checkpoints.jsonl` row."""
        return cls(
            row["tag"],
            row["step"],
            row.get("state"),
            dict(row.get("training_state") or {}),
            dict(row.get("sampler_weights") or {}),
            row.get("ttl_hours"),
        )

    def row(self) -> dict[str, Any]:
        """Return the checkpoint as its `checkpoints.jsonl` row."""
        return {
            "tag": self.tag,
            "step": self.step,
            "state": self.state,
            "training_state": dict(self.training_state),
            "sampler_weights": dict(self.sampler_weights),
            **({"ttl_hours": self.ttl_hours} if self.training_state else {}),
        }

    def saved_state(self, run: Path) -> dict[str, Any]:
        """Return the trainer's state as the run in `run` saved it; empty when it is missing or
        unreadable."""
        return files.read_json(Path(run) / self.state) if self.state else {}


def found(directory: Path, tag: str = "last") -> Checkpoint:
    """Return the run's checkpoint `tag`, its last by default. Raise `ValueError` for none."""
    directory = Path(directory)
    rows = list(files.read(directory / files.CHECKPOINTS))
    if tag == "last":
        row = rows[-1] if rows else None
    else:
        row = next((one for one in reversed(rows) if one["tag"] == tag), None)
    if row is None:
        tags = ", ".join(one["tag"] for one in rows) or "none"
        raise ValueError(f"{directory.name} has no checkpoint {tag!r}; it has {tags}")
    return Checkpoint.of(row)
