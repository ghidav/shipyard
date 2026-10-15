"""A finished job as the record side carries it to the groups, the bill and admission."""

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
    records: Records | None = None
    asked: dict[str, int] | None = None
    #: The trials whose harness log says their last model call failed.
    failed: set[str] | None = None

    def __len__(self) -> int:
        return len(self.trials)

    @classmethod
    def of(cls, job: str, trials: Sequence[Trial]) -> Rollouts:
        return cls(job, [one.dir for one in trials])
