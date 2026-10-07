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

from shipyard import record, ui
from shipyard.admit import verdicts
from shipyard.config import Blueprint, GepaRecipe, config_file, load, modules_dir
from shipyard.cost import Costs, billed_by, reported, sandbox_seconds
from shipyard.modules import Candidate, keep, seed
from shipyard.rollout import Rollouts, Runner, rollout
from shipyard.serving import Harvest, NothingServed, Serving, served_nothing, tinker_base_url

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
        self.progress: ui.Reporter = ui.reporter()
        #: How one trial is run; None is Harbor's `Trial`. A test hands in its own.
        self.run_trial: Runner | None = None
        #: The proxy and its records, for a run serving its own weights; else None.
        self.serving: Serving | None = Serving(config, directory) if config.model.served else None
        #: The trainer whose weights the proxy serves, for the update count every batch
        #: is stamped with; None for a run that trains nothing.
        self.trainer: Any | None = None
        self._seq = 0

    @classmethod
    def open(cls, blueprint: Path, *, root: Path | None = None, smoke: bool = False) -> Self:
        """Create `<root>/<id>/`, copy `run.toml` byte for byte, keep the modules the
        recipe carries under `modules/carried/`, and note the process (a smoke run says
        so). Refuses a directory that exists rather than writing into another run's record."""
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
        """Merge into `process.json`: a fact about this invocation, not a metric."""
        current = record.read_json(self.directory / record.PROCESS)
        current.update(about)
        record.write_json(self.directory / record.PROCESS, current)

    def reserve_job(self) -> str:
        """The next job name, `<run id>-NNNN` counted from the rows in `jobs.jsonl`, with
        its directory made now; one that exists already is refused, never written into."""
        count = sum(1 for _ in record.read(self.directory / record.JOBS))
        name = f"{self.id}-{count:04d}"
        try:
            (self.jobs_dir / name).mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f"job {name} is taken: {self.jobs_dir / name} exists") from None
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
        """One Harbor job over `batch`, the recipe's batch `index`, carrying `modules` (the
        run's own when None), stamped with the trainer's update count, every verdict on the
        job row; a served job the proxy served nothing for then raises `NothingServed`."""
        cfg = self.config
        if self.serving is not None:
            first = self.serving.proxy is None
            await self.serving.start()
            self.serving.alive()
            if first:
                placed = {"placement": self.serving.placement, "origin": self.serving.origin}
                self.note(proxy=placed)
        updates = getattr(self.trainer, "updates", None)
        candidate = self.carried if modules is None else modules
        job = self.reserve_job()
        self.progress.started(job, trials=len(batch) * rollouts, batch=index)
        rolled = await rollout(
            batch,
            job,
            harness=cfg.rollout.harness,
            model=self.serving.model_name() if self.serving else cfg.model.name,
            sandbox=cfg.rollout.sandbox,
            concurrency=cfg.rollout.concurrency,
            rollouts=rollouts,
            jobs_dir=self.jobs_dir,
            env=cfg.rollout.env,
            kwargs=cfg.rollout.kwargs,
            setup_timeout=cfg.rollout.setup_timeout,
            timeout=cfg.rollout.timeout,
            run_trial=self.run_trial,
            progress=self.progress,
            served=self.serving.served() if self.serving else None,
            modules=candidate,
        )
        self.progress.done(job)
        harvested = await self.serving.harvest(rolled) if self.serving else None
        if harvested is not None:
            rolled = harvested.rollouts
        rolled = replace(rolled, updates=updates)
        self._record_job(
            rolled,
            batch=index,
            tasks=len(batch),
            harvested=harvested,
            modules=candidate,
            purpose=purpose,
        )
        if self.serving is not None:
            self.serving.alive()  # a proxy that died under the batch: its masks are not data
        why = served_nothing(rolled) if harvested is not None else None
        if why is not None:
            raise NothingServed(f"job {rolled.job}: {why}", rolled)
        return rolled

    async def checkpoint(self, trainer: Any, tag: str) -> Checkpoint:
        """Save state and sampler weights under `tag` at `[checkpoints] ttl_hours`, and
        append both paths to `checkpoints.jsonl`: a record naming only the sampler path
        can be measured again but never continued."""
        saved = await trainer.save(tag, ttl_hours=self.config.checkpoints.ttl_hours)
        named = ("tag", "state_path", "sampler_path", "ttl_hours")
        record.append(
            self.directory / record.CHECKPOINTS, {key: getattr(saved, key) for key in named}
        )
        logger.info("checkpoint %s: %s", saved.tag, saved.sampler_path)
        return saved

    def spent(self, party: str, **counts: int) -> None:
        """Add training counts to a party on `costs.json`; a count of nothing adds no key."""
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
        harvested: Harvest | None = None,
        modules: Candidate | None = None,
        purpose: str = "rollout",
    ) -> dict[str, Any]:
        """A rollout of the policy on the record: the modules' digest when it carried
        any, what was graded, what was masked and why, how each trial ended, and who is
        billed for which counts (a served job from its records, the proxy's notes with it)."""
        judged = verdicts(rolled)
        if harvested is not None and self.serving is not None:
            counted, party, noted = harvested.counted, self.serving.party, harvested.noted
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
            served="tinker" if harvested is not None else "provider",
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
        """A job on the record and its costs: its purpose, its trials, what `about` says
        of it, the tokens (`counted`, else what the harness reported) under `party`, and
        the sandbox's seconds. A reflection passes no verdicts: nothing was graded."""
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
        """The party's counts and the sandbox's seconds onto `costs.json`."""
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
        """Close the record whether or not the loop raised: a run that crashed still happened.
        A signal's `KeyboardInterrupt` is recorded as `stopped: <signal>`; the proxy, if
        this run started one, is stopped first."""
        self.progress.close()
        if self.serving is not None:
            self.serving.close()
        if exc is None:
            error = None
        elif isinstance(exc, KeyboardInterrupt):
            error = f"stopped: {str(exc) or 'SIGINT'}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        self.note(finished_at=record.now(), failed=exc is not None, error=error)


def carried_modules(config: Blueprint) -> Candidate | None:
    """The candidate a run carries into every job, seeded from `[recipe] modules`; None
    when the key is unset, or for the one recipe that seeds its own search, gepa."""
    return None if isinstance(config.recipe, GepaRecipe) else seeded(config)


def seeded(config: Blueprint) -> Candidate | None:
    """The candidate `[recipe] modules` seeds, whichever recipe; None when it is unset."""
    directory = modules_dir(config)
    return None if directory is None else seed(directory)


def runs_root(flag: Path | None = None) -> Path:
    """Where runs live: `--root`, then `SHIPYARD_RUNS`, then `./runs`; every command uses it."""
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
    """`<name>__<7>`, the way Harbor names a trial: an id per run, not per blueprint."""
    stem = name[:32].rstrip("_-") or "run"
    return f"{stem}__{ShortUUID().random(length=7)}"


def _tally(values: Iterable[str | None]) -> dict[str, int]:
    """How many of each named value, in order of first appearance; None is not a value."""
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return counts
