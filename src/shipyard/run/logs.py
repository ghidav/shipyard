from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
    counted,
    filed,
    reported,
    sampled,
    sandbox_seconds,
)

if TYPE_CHECKING:
    from harness_proxy.wire import Record
    from shipyard.config import Blueprint
    from shipyard.groups import Trial
    from shipyard.jobs.engine import Job

#: The role the agent's own spend is filed under: its rollouts, and its passes on Tinker.
#: Another policy's is the name the trainer holds it by, such as `teacher`.
AGENT = "agent"
#: Keys of a step's row, in order, before its evaluation's and the recipe's own `metrics`. A
#: key with no value is left out.
ROW = (
    "step",
    "trained",
    "jobs",
    "groups",
    "trials",
    "graded",
    "masked",
    "uniform",
    "sequences",
    "train_tokens",
    "reward_mean",
    "reward_spread",
    "sampled_mean",
    "kl_v1",
    "kl_v2",
    "entropy",
    "learning_rate",
    "substeps",
    "loss",
    "update_seconds",
)
#: What an evaluation's fields are prefixed with in the row of the step it belongs to.
EVAL = "eval_"


class Wandb:
    """The wandb run the step rows and costs are mirrored to, named after the run directory, and
    grouped with the runs of its chain: the run it went on from, and that one's, and so on."""

    def __init__(self, config: Blueprint, directory: Path) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("[logging.wandb] needs wandb: install shipyard[wandb]") from None
        table = config.logging.wandb
        origin = record.read_json(directory / record.RUN).get("origin") or {}
        self.run = wandb.init(
            project=table.project,
            entity=table.entity,
            id=directory.name,
            name=directory.name,
            group=origin.get("first", directory.name),
            reinit="create_new",
            dir=directory,
            config=config.model_dump(mode="json"),
        )

    def log(self, row: Mapping[str, Any]) -> None:
        """Mirror a step row's numbers at its step, its evaluation's under `eval/`."""
        numbers = {
            (f"eval/{key.removeprefix(EVAL)}" if key.startswith(EVAL) else key): value
            for key, value in row.items()
            if isinstance(value, int | float) and key != "step"
        }
        if numbers:
            self.run.log(numbers, step=row["step"])

    def costs(self, costs: Costs) -> None:
        self.run.summary["costs"] = costs.to_dict()

    def close(self, error: BaseException | None) -> None:
        self.run.finish(exit_code=0 if error is None else 1)


class Logs:
    """The run's record: the rows of its jobs, steps and checkpoints, and its costs."""

    def __init__(
        self,
        config: Blueprint,
        directory: Path,
        *,
        step: Callable[[], int | None] = lambda: None,
    ) -> None:
        self.directory = Path(directory)
        #: The step a spend happens in, as the trainer counts it.
        self._step = step
        self.wandb = Wandb(config, self.directory) if config.logging.wandb is not None else None
        self.sandbox = config.rollout.sandbox
        self.jobs_dir = Path(config.rollout.jobs_dir)
        #: Tokens by provider and model, and sandbox seconds: the run's own.
        self.costs = Costs()
        self._next = 0

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
        exceptions = Counter(one.ended for one in verdicts if one.ended is not None)
        # What the job spent is in its `usage.jsonl` rows, filed under its name.
        row = {
            "job": job.name,
            "dir": str((self.jobs_dir / job.name).resolve()),
            "purpose": job.purpose,
            **({"harness": job.harness.agent} if job.harness is not None else {}),
            **({"renderer": job.renderer} if job.renderer is not None else {}),
            "trials": len(job.trials),
            "step": job.step,
            "tasks": len({one.task for one in job.trials}),
            **({"modules": job.modules.digest} if job.modules is not None else {}),
            "graded": sum(one.reward is not None for one in verdicts),
            "masked": sum(masked.values()),
            **({"masked_by": dict(masked)} if masked else {}),
            **({"exceptions": dict(exceptions)} if exceptions else {}),
            "backend": provider.split("@", 1)[0] if served else "provider",
            **{key: value for key, value in noted.items() if value},
        }
        record.append(self.directory / record.JOBS, row)
        for trial in trials:
            for item in trial.records or ():
                record.append(self.directory / record.REQUESTS, request(job.name, trial, item))
        about = {"step": job.step, "role": AGENT, "job": job.name}
        for billed, models in tokens.items():
            for name, counts in models.items():
                self._spent(about, billed, name, counts)
        ran = {"sandbox": self.sandbox, "trials": timed, "seconds": seconds}
        record.append(self.directory / record.USAGE, {**_known(about), **ran})
        self.costs.add_sandbox(self.sandbox, timed, seconds)
        self._write_costs()

    def spend(self, role: str, provider: str, model: str, /, **counts: int) -> None:
        """File tokens `role` spent on a provider's model, by kind: a row of the usage ledger,
        added to the run's costs at once."""
        if self._spent({"step": self._step(), "role": role}, provider, model, counts):
            self._write_costs()

    def _spent(
        self, about: Mapping[str, Any], provider: str, model: str, counts: Mapping[str, int]
    ) -> bool:
        """Append a row of tokens to the usage ledger and add them to the costs; False when
        nothing was spent."""
        found = counted(counts)
        if not found:
            return False
        model = filed(provider, model)
        row = {**_known(about), "provider": provider, "model": model, **found}
        record.append(self.directory / record.USAGE, row)
        self.costs.add(provider, model, **found)
        return True

    def step(
        self,
        index: int,
        groups: Sequence[Group],
        update: Any,
        evaluation: Any,
        metrics: Mapping[str, Any],
    ) -> None:
        """Write a step's row: its groups, its update, the evaluation that belongs to it, as
        `eval_` fields, and the recipe's own `metrics`."""
        groups = by_task(groups)
        measured = [one for group in groups for one in group.measured]
        trials = sum(len(group.trials) for group in groups)
        spreads = [pstdev(group.rewards) for group in groups if len(group.rewards) > 1]
        row: dict[str, Any] = {
            "step": index,
            "trained": update is not None,
            "jobs": list(dict.fromkeys(one.job for group in groups for one in group.trials)),
            "groups": len(groups),
            "trials": trials,
            "graded": len(measured),
            "masked": trials - len(measured),
            "uniform": sum(1 for group in groups if group.measured and not group.is_scored),
            "reward_mean": fmean(one.reward or 0.0 for one in measured) if measured else None,
            "reward_spread": fmean(spreads) if spreads else None,
            "sampled_mean": fmean(one.sampled_tokens for one in measured) if measured else None,
            **(vars(update) if update is not None else {}),
        }
        kept = {key: value for key in ROW if (value := row.get(key)) is not None}
        kept |= by_renderer(groups)
        if evaluation is not None:
            evaluated = {key: value for key, value in vars(evaluation).items() if key != "step"}
            renderers = evaluated.pop("by_renderer", None) or {}
            kept |= {
                f"{EVAL}{key}": value
                for key, value in {**evaluated, **renderers}.items()
                if value is not None
            }
        own = {(f"recipe_{key}" if key in kept else key): value for key, value in metrics.items()}
        row = {**kept, **own}
        record.append(self.directory / record.METRICS, row)
        if self.wandb is not None:
            self.wandb.log(row)

    def checkpoint(self, row: Mapping[str, Any]) -> None:
        """Write a checkpoint's row: where its weights are saved and the trainer's counters."""
        record.append(self.directory / record.CHECKPOINTS, dict(row))

    def close(self, error: BaseException | None) -> None:
        """Finish the wandb run, if any. `error` is what stopped the trainer, or None."""
        if self.wandb is not None:
            self.wandb.close(error)

    def _write_costs(self) -> None:
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())
        if self.wandb is not None:
            self.wandb.costs(self.costs)


def _known(about: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in about.items() if value is not None}


def by_renderer(groups: Sequence[Group]) -> dict[str, float]:
    """Return each renderer's mean reward and sampled tokens over its graded trials, as
    `reward_mean_<renderer>` and `sampled_mean_<renderer>`, when the groups' trials were rendered
    with more than one renderer; nothing otherwise."""
    graded: dict[str, list[Trial]] = {}
    for one in (trial for group in groups for trial in group.trials):
        if one.renderer is not None:
            graded.setdefault(one.renderer, []).extend([one] if one.measured else [])
    if len(graded) < 2:
        return {}
    found: dict[str, float] = {}
    for name, trials in graded.items():
        if trials:
            found[f"reward_mean_{name}"] = fmean(one.reward or 0.0 for one in trials)
            found[f"sampled_mean_{name}"] = fmean(one.sampled_tokens for one in trials)
    return found


def request(job: str, trial: Trial, item: Record) -> dict[str, Any]:
    """Return the `requests.jsonl` row of one model call the proxy served the trial."""
    return {
        "job": job,
        "trial": trial.name,
        "call": item.seq,
        "sample": item.sample,
        "samples": item.samples,
        "prompt_tokens": len(item.prompt_token_ids),
        "cached_tokens": cached(item),
        "completion_tokens": len(item.completion_token_ids),
        "stop_reason": item.stop_reason,
        "sample_ms": round(float(item.sample_ms), 1),
        "weights": item.served,
        "request_id": item.request_id,
        "bridged": item.bridged,
        "error": item.error,
    }
