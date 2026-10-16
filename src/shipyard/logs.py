"""Trainer loggers: the run directory, and wandb when `[logging.wandb]` is set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shipyard import record

if TYPE_CHECKING:
    from shipyard.config import Blueprint
    from shipyard.cost import Costs


class Logger:
    """Destination for a trainer's rows. This base class discards them; subclasses write
    them somewhere."""

    def log(self, section: str, /, **row: Any) -> None:
        """Record a row of metrics under `section`: `train` for a step, `eval` for an
        evaluation, or `search` for a search round or its end."""

    def job(self, row: Mapping[str, Any]) -> None:
        """Record a job's row: what it sampled, its token counts and the party billed."""

    def costs(self, costs: Costs) -> None:
        """Record the run's total costs so far. Called each time they grow."""

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        """Record a checkpoint's row: where its weights are saved and the trainer's counters
        at that point."""

    def close(self, error: BaseException | None) -> None:
        """Finish logging. `error` is the exception that stopped the trainer, or None if it
        completed."""


class Folder(Logger):
    """Logger that writes to the run directory: `metrics.jsonl`, `jobs.jsonl`,
    `checkpoints.jsonl` and `costs.json`. A resumed run reads them back.

    Metric rows are numbered continuing from the rows already in `metrics.jsonl`."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._seq = sum(1 for _ in record.read(directory / record.METRICS))

    def log(self, section: str, /, **row: Any) -> None:
        """Append the row with a sequence number, since timestamps can collide."""
        self._seq += 1
        record.append(self.directory / record.METRICS, {"seq": self._seq, **row})

    def job(self, row: Mapping[str, Any]) -> None:
        record.append(self.directory / record.JOBS, dict(row))

    def costs(self, costs: Costs) -> None:
        record.write_json(self.directory / record.COSTS, costs.to_dict())

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        record.append(self.directory / record.CHECKPOINTS, dict(row))


class Wandb(Logger):
    """Logger that writes to a wandb run.

    The wandb run id is the run directory's name, so a resumed run continues the same wandb
    run. A row's numeric values are logged under `<section>/` and plotted against that
    section's axis; costs go to the run summary. wandb keeps its files under `wandb/` in
    the run directory."""

    #: The metric each section's rows are plotted against.
    AXES = {"train": "step", "eval": "step", "search": "round"}

    def __init__(self, config: Blueprint, directory: Path) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("[logging.wandb] needs wandb: install shipyard[wandb]") from None
        table = config.logging.wandb
        if table is None:
            raise ValueError("the blueprint has no [logging.wandb] table")
        self.run = wandb.init(
            project=table.project,
            entity=table.entity,
            id=directory.name,
            name=directory.name,
            resume="allow",
            reinit="create_new",
            dir=directory,
            config=config.model_dump(mode="json"),
        )
        self._sections: set[str] = set()

    def log(self, section: str, /, **row: Any) -> None:
        if section not in self._sections:
            self._sections.add(section)
            if (axis := self.AXES.get(section)) is not None:
                self.run.define_metric(f"{section}/{axis}")
                self.run.define_metric(f"{section}/*", step_metric=f"{section}/{axis}")
        numbers = {
            f"{section}/{key}": value
            for key, value in row.items()
            if isinstance(value, int | float)
        }
        if numbers:
            self.run.log(numbers)

    def costs(self, costs: Costs) -> None:
        self.run.summary["costs"] = costs.to_dict()

    def close(self, error: BaseException | None) -> None:
        self.run.finish(exit_code=0 if error is None else 1)


class Loggers(Logger):
    """Several loggers as one: each call is forwarded to every logger, in order."""

    def __init__(self, each: Sequence[Logger]) -> None:
        self.each = list(each)

    def log(self, section: str, /, **row: Any) -> None:
        for one in self.each:
            one.log(section, **row)

    def job(self, row: Mapping[str, Any]) -> None:
        for one in self.each:
            one.job(row)

    def costs(self, costs: Costs) -> None:
        for one in self.each:
            one.costs(costs)

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        for one in self.each:
            one.checkpoint(row)

    def close(self, error: BaseException | None) -> None:
        for one in self.each:
            one.close(error)
