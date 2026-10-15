"""A finished job as the loop still carries it, until the trainer takes groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shipyard.types import Records, Trial


@dataclass(frozen=True)
class Rollouts:
    """A finished job: its trials in plan order, and once recorded, its records and harness logs."""

    job: str
    trials: list[Path]
    plan: list[tuple[Path, int]]
    records: Records | None = None
    asked: dict[str, int] | None = None
    #: The trials whose harness log says their last model call failed.
    failed: set[str] | None = None
    #: The trainer's update count when this job sampled, for credit to check against.

    def __len__(self) -> int:
        return len(self.trials)

    @classmethod
    def of(cls, job: str, trials: Sequence[Trial]) -> Rollouts:
        return cls(job, [one.dir for one in trials], [(one.task, one.attempt) for one in trials])
