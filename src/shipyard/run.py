"""A run: a directory holding the blueprint it consumed and the record of what happened."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shortuuid import ShortUUID

from shipyard import record, types
from shipyard.admit import read_logs, verdicts
from shipyard.config import Blueprint, Search, config_file, load, modules_dir
from shipyard.cost import Costs, billed_by, reported, sampled, sandbox_seconds
from shipyard.modules import Candidate, keep, seed
from shipyard.progress import Reporter, reporter
from shipyard.sampling.endpoint import tinker_base_url

logger = logging.getLogger(__name__)

#: Under `runs/<id>/modules/`: the candidate a run carries, as `seed` reads it back.
CARRIED = "carried"


class RunRefused(RuntimeError):
    """A run that could not be opened; nothing was written."""


class Run:
    """An open run: its directory, the blueprint it consumed, and the record it appends to."""

    def __init__(
        self, directory: Path, config: Blueprint, carried: Candidate | None = None
    ) -> None:
        self.directory = directory
        self.id = directory.name
        self.config = config
        #: The candidate every job carries unless handed another; None when `[recipe]
        #: modules` names none, or the recipe searches over them itself.
        self.carried = carried
        self.costs = Costs.from_dict(record.read_json(directory / record.COSTS))
        self.jobs_dir = Path(config.rollout.jobs_dir)
        self.progress: Reporter = reporter()
        self._next_job = next_job(self.id, directory, self.jobs_dir)
        self._seq = 0
        #: Set by `reopen`: the process note is written on entry, keeping the earlier ones.
        self._reopened = False
        self._resumed: str | None = None

    @classmethod
    def open(
        cls,
        blueprint: Path,
        *,
        root: Path | None = None,
        smoke: bool = False,
        runner: Path | None = None,
    ) -> Self:
        """Create `<root>/<id>/`, copy `run.toml` byte for byte and the runner file as
        `runner.py` when one is given, keep the recipe's modules under `modules/carried/`, and
        write the process note (marked `smoke` for a smoke run). Raises `RunRefused` if the
        directory exists."""
        config = load(blueprint)
        source = config_file(blueprint)
        home = source.parent
        carried = carried_modules(config)
        directory = runs_root(root) / _run_id(home.name)
        if directory.exists():
            raise RunRefused(f"{directory} already exists")
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / record.CONFIG)
        if runner is not None:
            shutil.copyfile(runner, directory / record.RUNNER)
        if carried is not None:
            keep(carried, directory / record.MODULES / CARRIED)
        record.write_json(
            directory / record.PROCESS,
            _process(
                config,
                str(home.resolve()),
                runner=None if runner is None else str(Path(runner).resolve()),
                smoke=smoke,
            ),
        )
        return cls(directory, config, carried)

    @classmethod
    def reopen(cls, directory: Path, *, resumed: str | None = None) -> Self:
        """The run as it stands, to go on; its process note is written on entry, so a refusal
        before the work leaves the earlier one as it was."""
        directory = Path(directory)
        config = load(directory)
        carried = directory / record.MODULES / CARRIED
        run = cls(directory, config, seed(carried) if carried.is_dir() else None)
        run._seq = sum(1 for _ in record.read(directory / record.METRICS))
        run._reopened, run._resumed = True, resumed
        return run

    @staticmethod
    def read(directory: Path) -> dict[str, Any]:
        """The process note of a run directory, open or closed; `{}` when it has none."""
        return record.read_json(Path(directory) / record.PROCESS)

    def log(self, **metrics: Any) -> dict[str, Any]:
        """Append to `metrics.jsonl` with a sequence number from 1: timestamps collide."""
        self._seq += 1
        return record.append(self.directory / record.METRICS, {"seq": self._seq, **metrics})

    def note(self, **about: Any) -> None:
        """Merge facts about this invocation into `process.json`."""
        current = record.read_json(self.directory / record.PROCESS)
        current.update(about)
        record.write_json(self.directory / record.PROCESS, current)

    def reserve_name(self) -> str:
        """The next job name, `<run id>-NNNN`, with its directory created; raises if it exists."""
        name = f"{self.id}-{self._next_job:04d}"
        try:
            (self.jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {self.jobs_dir / name} exists") from None
        self._next_job += 1
        return name

    def recorded(self, job: types.Job, *, party: str | None) -> None:
        """The job's record side: logs read against the records the endpoint served (None for
        a provider), the trials judged once, the row written with its modules digest, graded
        count, masks, endings and the party billed: `party` for served records, else the one
        the trials name."""
        if job.records is not None:
            job.asked, job.failed = read_logs(
                job.trials, job.records, check_turns=self.config.rollout.check_turns
            )
        job.verdicts = verdicts(job)
        served = job.records is not None
        if served:
            counted, noted = sampled(job)
        else:
            counted, party, noted = reported(job), billed_by(job), {}
        masked = _tally(each.mask for each in job.verdicts)
        ended = _tally(each.ended for each in job.verdicts)
        self.record_job(
            job,
            party=party or self.config.model.provider,
            counted=counted,
            tasks=len({one.task for one in job.trials}),
            **({"epoch": job.epoch} if job.epoch is not None else {}),
            **job.about,
            **({"modules": job.modules.digest} if job.modules is not None else {}),
            graded=sum(each.reward is not None for each in job.verdicts),
            **({"masked": masked} if masked else {}),
            **({"ended": ended} if ended else {}),
            served="tinker" if served else "provider",
            **{key: value for key, value in noted.items() if value},
        )

    def spent(self, party: str, **counts: int) -> None:
        """Add training counts to a party on `costs.json`. Zero counts are skipped."""
        counted = {key: int(value) for key, value in counts.items() if value}
        if not counted:
            return
        self.costs.add_party(party, **counted)
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())

    def record_job(
        self,
        job: types.Job,
        *,
        party: str,
        counted: Mapping[str, int] | None = None,
        **about: Any,
    ) -> dict[str, Any]:
        """Append a job row and bill it: the job's name, purpose, trial count and step,
        `about`, the tokens (`counted`, else what the harness reported) under `party`, and
        the sandbox's seconds. A reflection records no verdicts because nothing was graded."""
        counted = reported(job) if counted is None else counted
        timed, seconds = sandbox_seconds(job)
        row = record.append(
            self.directory / record.JOBS,
            {
                "job": job.name,
                "purpose": job.purpose,
                "trials": len(job.trials),
                **({"step": job.step} if job.step is not None else {}),
                **about,
                "party": party,
                "input_tokens": counted["input_tokens"],
                "cache_tokens": counted["cache_tokens"],
                "output_tokens": counted["output_tokens"],
                "sandbox": self.config.rollout.sandbox,
                "sandbox_seconds": seconds,
            },
        )
        self.costs.add_party(party, **counted)
        self.costs.add_sandbox(self.config.rollout.sandbox, timed, seconds)
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())
        return row

    def __enter__(self) -> Self:
        if self._reopened:
            earlier = record.read_json(self.directory / record.PROCESS)
            before = earlier.pop("earlier", [])
            note = _process(self.config, earlier.get("blueprint"), resumed=self._resumed)
            note["earlier"] = [*before, earlier] if earlier else before
            record.write_json(self.directory / record.PROCESS, note)
            self._reopened = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the record, whether or not the loop raised. A `KeyboardInterrupt` from a
        signal is recorded as `stopped: <signal>`."""
        self.progress.close()
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def _process(config: Blueprint, blueprint: str | None, **extra: Any) -> dict[str, Any]:
    """A process note: this invocation, the blueprint and the kind, then `extra` where set."""
    return {
        "pid": os.getpid(),
        "started_at": record.now(),
        "version": version(),
        "blueprint": blueprint,
        "kind": config.recipe.kind,
        **({"tinker_base_url": tinker_base_url()} if config.model.served else {}),
        **{key: value for key, value in extra.items() if value},
    }


def carried_modules(config: Blueprint) -> Candidate | None:
    """The candidate a run carries into every job, seeded from `[recipe] modules`. None
    when the key is unset or the recipe (gepa, fst) seeds its own search."""
    return None if isinstance(config.recipe, Search) else seeded(config)


def seeded(config: Blueprint) -> Candidate | None:
    """The candidate `[recipe] modules` seeds, for any recipe. None when it is unset."""
    directory = modules_dir(config)
    return None if directory is None else seed(directory)


def next_job(run_id: str, directory: Path, jobs_dir: Path) -> int:
    """One past the highest job number on the record or on disk; 0 for a new run."""
    named = [row.get("job") for row in record.read(directory / record.JOBS)]
    if jobs_dir.is_dir():
        named += [one.name for one in jobs_dir.iterdir()]
    prefix = f"{run_id}-"
    numbers = [
        int(name[len(prefix) :])
        for name in named
        if isinstance(name, str) and name.startswith(prefix) and name[len(prefix) :].isdigit()
    ]
    return max(numbers, default=-1) + 1


def runs_root(flag: Path | None = None) -> Path:
    """Where runs live: `--root`, else `SHIPYARD_RUNS`, else `./runs`. Every command uses it."""
    if flag is not None:
        return Path(flag)
    named = os.environ.get("SHIPYARD_RUNS")
    return Path(named) if named else Path.cwd() / "runs"


def version() -> str:
    """The installed distribution's version, or "source" from a tree never installed."""
    try:
        return installed("shipyard")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "source"


def _run_id(name: str) -> str:
    """`<name>__<7>`, named as Harbor names a trial. Each run gets a new id."""
    stem = name[:32].rstrip("_-") or "run"
    return f"{stem}__{ShortUUID().random(length=7)}"


def _tally(values: Iterable[str | None]) -> dict[str, int]:
    """Count each value, in order of first appearance. None is skipped."""
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return counts
