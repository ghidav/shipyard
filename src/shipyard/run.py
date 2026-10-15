"""A run: a directory holding the blueprint it consumed and the record of what happened."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from shortuuid import ShortUUID

from shipyard import record, types
from shipyard.admit import read_logs, verdicts
from shipyard.config import Blueprint, FstRecipe, GepaRecipe, config_file, load, modules_dir
from shipyard.cost import Costs, billed_by, reported, sampled, sandbox_seconds
from shipyard.modules import Candidate, keep, seed
from shipyard.progress import Reporter, reporter
from shipyard.rollout import Rollouts
from shipyard.sampling.endpoint import (
    NothingServed,
    ProviderEndpoint,
    ProxyEndpoint,
    tinker_base_url,
)
from shipyard.sampling.engine import HarborEngine, TrialRunner
from shipyard.types import Agent, Endpoint, Policy

if TYPE_CHECKING:
    from shipyard.trainer import Checkpoint

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
        self.engine = HarborEngine(config.rollout, progress=self.progress)
        #: The proxy over this run's weights, or the provider the harness dials itself.
        self.endpoint: Endpoint = (
            ProxyEndpoint(config, directory)
            if config.model.served
            else ProviderEndpoint(config.model.name)
        )
        self._noted = False
        self._next_job = next_job(self.id, directory, self.jobs_dir)
        #: The trainer whose weights the proxy serves, for the update count every batch
        #: is stamped with; None for a run that trains nothing.
        self.trainer: Any | None = None
        self._seq = 0

    @classmethod
    def open(cls, blueprint: Path, *, root: Path | None = None, smoke: bool = False) -> Self:
        """Create `<root>/<id>/`, copy `run.toml` byte for byte, keep the recipe's modules
        under `modules/carried/`, and write the process note (marked `smoke` for a smoke
        run). Raises `RunRefused` if the directory exists."""
        config = load(blueprint)
        source = config_file(blueprint)
        home = source.parent
        carried = carried_modules(config)
        directory = runs_root(root) / _run_id(home.name)
        if directory.exists():
            raise RunRefused(f"{directory} already exists")
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / record.CONFIG)
        if carried is not None:
            keep(carried, directory / record.MODULES / CARRIED)
        record.write_json(
            directory / record.PROCESS,
            {
                "pid": os.getpid(),
                "started_at": record.now(),
                "version": version(),
                "blueprint": str(home.resolve()),
                "kind": config.recipe.kind,
                **({"tinker_base_url": tinker_base_url()} if config.model.served else {}),
                **({"smoke": True} if smoke else {}),
            },
        )
        return cls(directory, config, carried)

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

    @property
    def run_trial(self) -> TrialRunner | None:
        """How one trial is run; None is Harbor's `Trial`. A test hands in its own."""
        return self.engine.run_trial

    @run_trial.setter
    def run_trial(self, runner: TrialRunner | None) -> None:
        self.engine.run_trial = runner

    def reserve_job(self) -> str:
        """The next job name, `<run id>-NNNN`, from a counter seeded by the record when
        the run opened, so jobs in flight together never share one. Creates its directory
        and raises `FileExistsError` if it exists."""
        name = f"{self.id}-{self._next_job:04d}"
        try:
            (self.jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {self.jobs_dir / name} exists") from None
        self._next_job += 1
        return name

    async def sample(
        self,
        batch: Sequence[Path],
        *,
        rollouts: int,
        index: int,
        modules: Candidate | None = None,
        purpose: str = "rollout",
    ) -> Rollouts:
        """One Harbor job over `batch`: the verdicts on its row, then the endpoint's check."""
        cfg, endpoint = self.config, self.endpoint
        await endpoint.open()
        self.noted()
        candidate = self.carried if modules is None else modules
        agent = Agent(Policy(endpoint), [types.harness(cfg.rollout.harness)], candidate)
        job = self.reserve_job()
        self.progress.started(job, trials=len(batch) * rollouts, batch=index)
        trials = await self.engine.run(job, batch, rollouts, agent=agent)
        self.progress.done(job)
        return await self.recorded(job, trials, batch=index, modules=candidate, purpose=purpose)

    async def recorded(
        self,
        job: str,
        trials: Sequence[types.Trial],
        *,
        batch: int,
        modules: Candidate | None = None,
        purpose: str = "rollout",
    ) -> Rollouts:
        """The job's record side: its model calls drained, the harness logs read, the job
        row written, and the endpoint's check after the row."""
        endpoint = self.endpoint
        rolled = Rollouts.of(job, trials)
        records = await endpoint.record(job, trials)
        if records is not None:
            asked, failed = read_logs(trials, records, check_turns=self.config.rollout.check_turns)
            rolled = replace(rolled, records=records, asked=asked, failed=failed)
        rolled = replace(rolled, updates=getattr(self.trainer, "updates", None))
        tasks = len({one.task for one in trials})
        self._record_job(rolled, batch=batch, tasks=tasks, modules=modules, purpose=purpose)
        if records is not None:
            try:
                endpoint.check(job, records)
            except NothingServed as dead:
                dead.rolled = rolled
                raise
        return rolled

    def noted(self) -> None:
        """Note the proxy's placement and origin on `process.json`, once it is open."""
        endpoint = self.endpoint
        if self._noted or not isinstance(endpoint, ProxyEndpoint) or endpoint.proxy is None:
            return
        self.note(proxy={"placement": endpoint.placement, "origin": endpoint.origin})
        self._noted = True

    async def checkpoint(self, trainer: Any, tag: str, *, keep: bool = False) -> Checkpoint:
        """Save state and sampler weights under `tag` with `[checkpoints] ttl_hours`, or with
        no TTL when `keep` (the final weights). Append both paths to `checkpoints.jsonl`; a
        record with only the sampler path cannot be continued."""
        ttl = None if keep else self.config.checkpoints.ttl_hours
        saved = await trainer.save(tag, ttl_hours=ttl)
        named = ("tag", "state_path", "sampler_path", "ttl_hours")
        record.append(
            self.directory / record.CHECKPOINTS, {key: getattr(saved, key) for key in named}
        )
        logger.info("checkpoint %s: %s", saved.tag, saved.sampler_path)
        return saved

    def spent(self, party: str, **counts: int) -> None:
        """Add training counts to a party on `costs.json`. Zero counts are skipped."""
        counted = {key: int(value) for key, value in counts.items() if value}
        if not counted:
            return
        self.costs.add_party(party, **counted)
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())

    def _record_job(
        self,
        rolled: Rollouts,
        *,
        batch: int,
        tasks: int,
        modules: Candidate | None = None,
        purpose: str = "rollout",
    ) -> dict[str, Any]:
        """The job row: modules digest, graded count, masks, endings, and the party billed."""
        judged = verdicts(rolled)
        served = rolled.records is not None
        if served:
            counted, noted = sampled(rolled)
            party = self.endpoint.party
        else:
            counted, party, noted = reported(rolled), billed_by(rolled), {}
        masked = _tally(each.mask for each in judged)
        ended = _tally(each.ended for each in judged)
        return self.record_job(
            rolled,
            purpose=purpose,
            party=party or self.config.model.provider,
            counted=counted,
            batch=batch,
            tasks=tasks,
            **({"modules": modules.digest} if modules is not None else {}),
            graded=sum(each.reward is not None for each in judged),
            **({"masked": masked} if masked else {}),
            **({"ended": ended} if ended else {}),
            served="tinker" if served else "provider",
            **{key: value for key, value in noted.items() if value},
        )

    def record_job(
        self,
        rolled: Rollouts,
        *,
        purpose: str,
        party: str,
        counted: Mapping[str, int] | None = None,
        **about: Any,
    ) -> dict[str, Any]:
        """Append a job row and bill it: the purpose, the trial count, `about`, the tokens
        (`counted`, else what the harness reported) under `party`, and the sandbox's
        seconds. A reflection records no verdicts because nothing was graded."""
        counted = reported(rolled) if counted is None else counted
        timed, seconds = sandbox_seconds(rolled)
        row = record.append(
            self.directory / record.JOBS,
            {
                "job": rolled.job,
                "purpose": purpose,
                "trials": len(rolled),
                **about,
                "party": party,
                "input_tokens": counted["input_tokens"],
                "cache_tokens": counted["cache_tokens"],
                "output_tokens": counted["output_tokens"],
                "sandbox": self.config.rollout.sandbox,
                "sandbox_seconds": seconds,
            },
        )
        self._bill(party, counted, timed, seconds)
        return row

    def _bill(self, party: str, counted: Mapping[str, int], timed: int, seconds: float) -> None:
        """Add the party's counts and the sandbox's seconds to `costs.json`."""
        self.costs.add_party(party, **counted)
        self.costs.add_sandbox(self.config.rollout.sandbox, timed, seconds)
        record.write_json(self.directory / record.COSTS, self.costs.to_dict())

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the proxy if this run started one, then close the record, whether or not
        the loop raised. A `KeyboardInterrupt` from a signal is recorded as
        `stopped: <signal>`."""
        self.progress.close()
        self.endpoint.close()
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def carried_modules(config: Blueprint) -> Candidate | None:
    """The candidate a run carries into every job, seeded from `[recipe] modules`. None
    when the key is unset or the recipe (gepa, fst) seeds its own search."""
    return None if isinstance(config.recipe, GepaRecipe | FstRecipe) else seeded(config)


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
