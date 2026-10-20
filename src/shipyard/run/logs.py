from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any

from shipyard.groups import Group, by_task
from shipyard.run import record
from shipyard.run.costs import (
    UNKNOWN,
    Costs,
    billed_by,
    cached,
    reported,
    sampled,
    sandbox_seconds,
)

if TYPE_CHECKING:
    from harness_proxy.wire import Record
    from shipyard.config import Blueprint
    from shipyard.groups import Trial
    from shipyard.jobs.engine import Job

#: Keys of a step's `train` row, in order, before the recipe's own `metrics`. A key with no
#: value is left out.
ROW = (
    "step",
    "trained",
    "jobs",
    "groups",
    "rollouts",
    "graded",
    "masked",
    "degenerate",
    "sequences",
    "train_tokens",
    "reward_mean",
    "reward_spread",
    "length_mean",
    "kl_v1",
    "kl_v2",
    "entropy",
    "learning_rate",
    "substeps",
    "loss_fn",
    "seconds",
)


class Wandb:
    """The wandb run the rows and costs are mirrored to, named after the run directory."""

    #: The metric each section's rows are plotted against.
    AXES = {"train": "step", "eval": "step"}

    def __init__(self, config: Blueprint, directory: Path) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("[logging.wandb] needs wandb: install shipyard[wandb]") from None
        table = config.logging.wandb
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


class Logs:
    """The run's record: the rows of its jobs, steps and checkpoints, and its costs."""

    def __init__(self, config: Blueprint, directory: Path) -> None:
        self.directory = Path(directory)
        self.wandb = Wandb(config, self.directory) if config.logging.wandb is not None else None
        self._seq = sum(1 for _ in record.read(self.directory / record.METRICS))
        self.sandbox = config.rollout.sandbox
        self.jobs_dir = Path(config.rollout.jobs_dir)
        #: Tokens by provider and model, and sandbox seconds, including a resumed run's.
        self.costs = Costs.from_dict(record.read_json(self.directory / record.COSTS))
        self._next = record.next_job(self.directory.name, self.directory, self.jobs_dir)

    @cached_property
    def attempt(self) -> int:
        """Which invocation of the run this is: 0, then one more for each resume."""
        return len(record.read_json(self.directory / record.PROCESS).get("earlier") or [])

    def reserve(self) -> str:
        """Reserve the next job name, `<run>-NNNN`, by creating its directory."""
        name = f"{self.directory.name}-{self._next:04d}"
        try:
            (self.jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {self.jobs_dir / name} exists") from None
        self._next += 1
        return name

    def job(self, job: Job, endpoint: Any) -> None:
        """Write a judged job's row and its model calls' rows, and add its tokens to the costs."""
        trials = job.trials
        served = any(one.records is not None for one in trials)
        provider, model = endpoint.provider or UNKNOWN, endpoint.model or UNKNOWN
        if served:
            tokens, noted = sampled(trials, provider, model)
        else:
            tokens, noted = reported(trials, billed_by(trials) or provider, model), {}
        timed, seconds = sandbox_seconds(trials)
        verdicts = [one.verdict for one in trials if one.verdict is not None]
        masked = Counter(one.mask for one in verdicts if one.mask is not None)
        ended = Counter(one.ended for one in verdicts if one.ended is not None)
        row = {
            "job": job.name,
            "purpose": job.purpose,
            **({"harness": job.harness.agent} if job.harness is not None else {}),
            "trials": len(job.trials),
            "step": job.step,
            "attempt": self.attempt,
            "tasks": len({one.task for one in job.trials}),
            **({"modules": job.modules.digest} if job.modules is not None else {}),
            "graded": sum(one.reward is not None for one in verdicts),
            **({"masked": dict(masked)} if masked else {}),
            **({"ended": dict(ended)} if ended else {}),
            "served": "tinker" if served else "provider",
            **{key: value for key, value in noted.items() if value},
            "tokens": tokens,
            "sandbox": self.sandbox,
            "sandbox_seconds": seconds,
        }
        record.append(self.directory / record.JOBS, row)
        for trial in trials:
            for item in trial.records or ():
                record.append(self.directory / record.REQUESTS, request(job.name, trial, item))
        self.costs.merge(tokens)
        self.costs.add_sandbox(self.sandbox, timed, seconds)
        self._write_costs()

    def step(
        self, index: int, groups: Sequence[Group], update: Any, metrics: Mapping[str, Any]
    ) -> None:
        """Log a step's `train` row: its groups, its update and the recipe's own `metrics`."""
        groups = by_task(groups)
        measured = [one for group in groups for one in group.measured]
        rollouts = sum(len(group.trials) for group in groups)
        spreads = [pstdev(group.rewards) for group in groups if len(group.rewards) > 1]
        row: dict[str, Any] = {
            "step": index,
            "trained": update is not None,
            "jobs": list(dict.fromkeys(one.job for group in groups for one in group.trials)),
            "groups": len(groups),
            "rollouts": rollouts,
            "graded": len(measured),
            "masked": rollouts - len(measured),
            "degenerate": sum(1 for group in groups if group.measured and not group.is_scored),
            "reward_mean": fmean(one.reward or 0.0 for one in measured) if measured else None,
            "reward_spread": fmean(spreads) if spreads else None,
            "length_mean": fmean(one.sampled_tokens for one in measured) if measured else None,
            **(vars(update) if update is not None else {}),
        }
        kept = {key: value for key in ROW if (value := row.get(key)) is not None}
        taken = {*kept, "attempt", "seq"}
        own = {(f"recipe_{key}" if key in taken else key): value for key, value in metrics.items()}
        self.log("train", **kept, **own)
        self._write_costs()

    def log(self, section: str, /, **row: Any) -> None:
        """Write a row of metrics under `section`, `train` or `eval`."""
        row = {**row, "attempt": self.attempt}
        self._seq += 1
        numbered = {"seq": self._seq, "section": section, **row}
        record.append(self.directory / record.METRICS, numbered)
        if self.wandb is not None:
            self.wandb.log(section, **row)

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        """Write a checkpoint's row: where its weights are saved and the trainer's counters."""
        record.append(self.directory / record.CHECKPOINTS, {**row, "attempt": self.attempt})

    def close(self, error: BaseException | None) -> None:
        """Finish the wandb run, if any. `error` is what stopped the trainer, or None."""
        if self.wandb is not None:
            self.wandb.close(error)

    def _write_costs(self) -> None:
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())
        if self.wandb is not None:
            self.wandb.costs(self.costs)


def request(job: str, trial: Trial, item: Record) -> dict[str, Any]:
    """Return the `requests.jsonl` row of one model call the proxy served the trial."""
    return {
        "job": job,
        "trial": trial.name,
        "seq": item.seq,
        "sample": item.sample,
        "samples": item.samples,
        "prompt_tokens": len(item.prompt_token_ids),
        "cached_tokens": cached(item),
        "completion_tokens": len(item.completion_token_ids),
        "stop_reason": item.stop_reason,
        "sample_ms": round(float(item.sample_ms), 1),
        "served": item.served,
        "request_id": item.request_id,
        "bridged": item.bridged,
        "error": item.error,
    }
