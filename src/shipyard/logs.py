"""Where a trainer writes what happened: the run's own folder, and wandb when `[wandb]`
names a project. A trainer writes every row to each of its loggers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shipyard import record

if TYPE_CHECKING:
    from shipyard.config import Blueprint
    from shipyard.cost import Costs


class Logger:
    """Receives a trainer's rows. This base drops them; a subclass writes them somewhere."""

    def log(self, section: str, /, **row: Any) -> None:
        """A row of metrics: a step's (`train`), an evaluation's (`eval`), or a search's
        round or end (`search`)."""

    def job(self, row: Mapping[str, Any]) -> None:
        """A job's row: what it sampled, the tokens it took and the party billed for them."""

    def costs(self, costs: Costs) -> None:
        """Everything the run has spent so far, after it spent more."""

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        """A checkpoint's row: where its weights are, and where the run stood."""

    def close(self, error: BaseException | None) -> None:
        """The trainer is done: `error` is what stopped it, None when it finished."""


class Folder(Logger):
    """The run's directory: `metrics.jsonl` numbered from one past the rows already there,
    `jobs.jsonl`, `checkpoints.jsonl` and `costs.json`. A resume reads them back."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._seq = sum(1 for _ in record.read(directory / record.METRICS))

    def log(self, section: str, /, **row: Any) -> None:
        """Append the row with a sequence number from 1: timestamps collide."""
        self._seq += 1
        record.append(self.directory / record.METRICS, {"seq": self._seq, **row})

    def job(self, row: Mapping[str, Any]) -> None:
        record.append(self.directory / record.JOBS, dict(row))

    def costs(self, costs: Costs) -> None:
        record.write_json(self.directory / record.COSTS, costs.to_dict())

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        record.append(self.directory / record.CHECKPOINTS, dict(row))


class Wandb(Logger):
    """A wandb run with the directory's name as its id, so a resume goes on with it. A row's
    numbers go under its section, plotted against the section's axis; the costs go to the
    summary. Its own files go under `wandb/` in the directory."""

    #: What each section's rows are plotted against.
    AXES = {"train": "step", "eval": "step", "search": "round"}

    def __init__(self, config: Blueprint, directory: Path) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("[wandb] needs wandb: install shipyard[wandb]") from None
        table = config.wandb
        if table is None:
            raise ValueError("the blueprint has no [wandb] table")
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
    """Several loggers as one: each row to every one of them, in order."""

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
