"""`run --smoke`: one task (the first of the first batch), one rollout, one job, down the
whole sampling path (serving, tunnel, records, admission) with no gradient, no checkpoint
and no reflection, then a report of what came back. A smoke run is a normal run directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import record
from shipyard.admit import verdicts
from shipyard.config import Blueprint, FstRecipe, GepaRecipe
from shipyard.data import batches
from shipyard.modules import Candidate
from shipyard.proxy.profiles import PROFILES, bare_name, profile_for
from shipyard.run import seeded
from shipyard.serving import NothingServed

if TYPE_CHECKING:
    from shipyard.rollout import Rollouts
    from shipyard.run import Run

#: What a smoke job is on the record, in its row's `purpose`.
PURPOSE = "smoke"
LABEL = 12


class SmokeFailed(RuntimeError):
    """A smoke job that came back with nothing to show: no trial with a record on a served
    run, or no trial with a verdict on a provider-served one."""


def planned(cfg: Blueprint) -> list[Path]:
    """The one task a smoke run samples: the first of the first batch the recipe would."""
    data = cfg.data
    first = next(batches(cfg.datasets, size=data.batch_size, seed=data.seed, epochs=1))
    return first[:1]


def carried(cfg: Blueprint) -> Candidate | None:
    """The modules the job carries: gepa's seed, since a search's first job scores it;
    None for the other recipes, whose run carries its own."""
    return seeded(cfg) if isinstance(cfg.recipe, GepaRecipe | FstRecipe) else None


@dataclass(frozen=True)
class Trial:
    name: str
    records: int | None
    served: str
    verdict: str
    graded: bool


@dataclass(frozen=True)
class Smoked:
    """What the smoke job showed, enough for the report and for its exit code."""

    run: str
    sandbox: str
    proxy: str
    harness: str
    profile: str
    proxied: bool
    trials: list[Trial] = field(default_factory=list)
    seconds: float = 0.0
    tokens: tuple[int, int, int] = (0, 0, 0)
    #: The stop the job raised once it was recorded, which `verify` raises in turn.
    dead: NothingServed | None = field(default=None, compare=False)

    @property
    def passed(self) -> bool:
        """A trial with a record on a served run, a trial with a verdict otherwise."""
        if self.proxied:
            return any(trial.records for trial in self.trials)
        return any(trial.graded for trial in self.trials)

    def lines(self) -> list[str]:
        prompt, cached, sampled = self.tokens
        made = [
            f"smoke  {self.run}",
            f"  {'sandbox':<{LABEL}}{self.sandbox:<{LABEL}} proxy  {self.proxy}",
            f"  {'harness':<{LABEL}}{self.harness:<{LABEL}} profile {self.profile}",
        ]
        for trial in self.trials:
            records = "—" if trial.records is None else str(trial.records)
            made.append(
                f"  {'trial':<{LABEL}}{trial.name}   records {records}   served "
                f"{trial.served}   verdict {trial.verdict}"
            )
        made.append(
            f"  sandbox seconds {self.seconds:.1f}   tokens prompt {prompt} cached {cached} "
            f"sampled {sampled}"
        )
        return made

    def verify(self) -> None:
        """Raise the job's own stop when it had one, else `SmokeFailed` unless it passed."""
        if self.dead is not None:
            raise self.dead
        if not self.passed:
            wanted = "a record" if self.proxied else "a verdict"
            raise SmokeFailed(f"no trial of the smoke job came back with {wanted}")


async def smoke(run: Run) -> Smoked:
    """The one job, rolled out and recorded with `purpose = "smoke"`; nothing else. A job
    the proxy served nothing for is still reported, its stop kept for `verify`."""
    try:
        rolled = await run.sample(
            planned(run.config), rollouts=1, index=0, modules=carried(run.config), purpose=PURPOSE
        )
    except NothingServed as dead:
        if dead.rolled is None:
            raise
        return replace(report(run, dead.rolled), dead=dead)
    return report(run, rolled)


def report(run: Run, rolled: Rollouts) -> Smoked:
    """The smoke job as the report says it: where the proxy stood, how the harness was
    wired, each trial's records, weights and verdict, and the job row's seconds and tokens."""
    cfg, serving = run.config, run.serving
    judged = verdicts(rolled)
    trials = []
    for trial, verdict in zip(rolled.trials, judged, strict=True):
        name = Path(trial).name
        taken = None if rolled.records is None else rolled.records.get(name) or []
        if taken is None:
            served = cfg.model.provider
        else:
            paths = sorted({item.served or cfg.model.name for item in taken})
            served = ", ".join(paths) or "nothing"
        said = f"masked {verdict.mask}" if verdict.reward is None else str(verdict.reward)
        trials.append(
            Trial(name, None if taken is None else len(taken), served, said, verdict.mask is None)
        )
    *_, row = record.read(run.directory / record.JOBS)
    cached, prompt = int(row["cache_tokens"]), int(row["input_tokens"])
    if serving is not None:  # the proxy's `input_tokens` leave the cached ones out
        proxy, prompt = f"{serving.origin}  ({serving.placement})", prompt + cached
    else:
        proxy = f"none (the harness calls {cfg.model.provider} itself)"
    return Smoked(
        run=run.id,
        sandbox=cfg.rollout.sandbox,
        proxy=proxy,
        harness=cfg.rollout.harness,
        profile=wiring(cfg) if serving is not None else "none (provider-served)",
        proxied=serving is not None,
        trials=trials,
        seconds=float(row["sandbox_seconds"]),
        tokens=(prompt, cached, int(row["output_tokens"])),
    )


def wiring(cfg: Blueprint) -> str:
    """The harness's profile by name with what it adds: `pi (model_api=openai-completions)`;
    a table kwarg by its name alone, `opencode (provider=shipyard, opencode_config)`."""
    name = bare_name(cfg.rollout.harness)
    profile = profile_for(cfg.rollout.harness)
    said = [
        key if isinstance(value, Mapping) else f"{key}={value}"
        for key, value in profile.kwargs.items()
    ]
    if profile.provider:
        said.insert(0, f"provider={profile.provider}")
    if profile.dialect != "openai":
        said.insert(0, f"dialect={profile.dialect}")
    named = name if name in PROFILES else "generic"
    return f"{named} ({', '.join(said)})" if said else named
