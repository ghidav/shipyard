"""`run --smoke`: one task (the first of the first batch), one rollout, one job, through
the whole sampling path (serving, tunnel, records, admission) with no gradient, checkpoint
or reflection, then a report of what came back. A smoke run writes a normal run
directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard import record
from shipyard.admit import verdicts
from shipyard.config import Blueprint
from shipyard.data import Generator
from shipyard.harnesses import PROFILES, bare_name, profile_for
from shipyard.run import seeded
from shipyard.sampling.endpoint import NothingServed, ProxyEndpoint, endpoint_for
from shipyard.training.trainer import Trainer
from shipyard.types import Agent, Endpoint, Policy, Verdict, harness

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shipyard.run import Run

#: The `purpose` of a smoke job's row.
PURPOSE = "smoke"
LABEL = 12


class SmokeFailed(RuntimeError):
    """The smoke job returned nothing usable: no trial with a record on a served run, or no
    trial with a verdict on a provider-served one."""


def planned(cfg: Blueprint) -> list[Path]:
    """The task a smoke run samples: the first of the first batch, of the test set when there is
    no train set."""
    made = Generator.of(cfg)
    if not made.train:
        made = replace(made, train=made.test)
    return next(made.batches(0))[:1]


@dataclass(frozen=True)
class Trial:
    name: str
    records: int | None
    served: str
    verdict: str
    graded: bool


@dataclass(frozen=True)
class Smoked:
    """The smoke job's result, used for the report and the exit code."""

    run: str
    sandbox: str
    proxy: str
    harness: str
    profile: str
    proxied: bool
    trials: list[Trial] = field(default_factory=list)
    seconds: float = 0.0
    tokens: tuple[int, int, int] = (0, 0, 0)
    #: The stop the job raised after it was recorded. `verify` raises it again.
    dead: NothingServed | None = field(default=None, compare=False)

    @property
    def passed(self) -> bool:
        """True if a trial has a record on a served run, or a verdict otherwise."""
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
        """Raise the job's stop if it had one, else `SmokeFailed` unless the job passed."""
        if self.dead is not None:
            raise self.dead
        if not self.passed:
            wanted = "a record" if self.proxied else "a verdict"
            raise SmokeFailed(f"no trial of the smoke job came back with {wanted}")


async def smoke(run: Run) -> Smoked:
    """Run the smoke job through the trainer's `job`, carrying the modules the blueprint seeds,
    recorded with `purpose = "smoke"`. A job the proxy served nothing for is still reported,
    and its stop is kept for `verify`."""
    cfg = run.config
    endpoint = endpoint_for(cfg, run.directory)
    agent = Agent(Policy(endpoint), [harness(cfg.rollout.harness)], seeded(cfg))
    async with agent.policy:
        try:
            found = await Trainer(run).job(agent, planned(cfg), 1, purpose=PURPOSE)
        except NothingServed as dead:
            if dead.rolled is None:
                raise
            judged = verdicts(dead.rolled)
            return replace(report(run, endpoint, dead.rolled.trials, judged), dead=dead)
    members = [one for group in found for one in group.members]
    trials = [one.trial.dir for one in members]
    return report(run, endpoint, trials, [one.verdict for one in members])


def report(
    run: Run, endpoint: Endpoint, trials: Sequence[Path], judged: Sequence[Verdict]
) -> Smoked:
    """The smoke report: where the proxy stood, how the harness was wired, each trial's
    records, weights and verdict off the record, and the job row's seconds and tokens."""
    cfg = run.config
    serving = endpoint if isinstance(endpoint, ProxyEndpoint) else None
    *_, row = record.read(run.directory / record.JOBS)
    requests = [
        one for one in record.read(run.directory / record.REQUESTS) if one["job"] == row["job"]
    ]
    found = []
    for trial, verdict in zip(trials, judged, strict=True):
        name = Path(trial).name
        if serving is None:
            served, records = cfg.model.provider, None
        else:
            taken = [one for one in requests if one["trial"] == name]
            served = (
                ", ".join(sorted({one["served"] or cfg.model.name for one in taken})) or "nothing"
            )
            records = len(taken)
        said = f"masked {verdict.mask}" if verdict.reward is None else str(verdict.reward)
        found.append(Trial(name, records, served, said, verdict.mask is None))
    cached, prompt = int(row["cache_tokens"]), int(row["input_tokens"])
    if serving is not None:
        proxy = f"{serving.origin}  ({serving.placement})"
    else:
        proxy = f"none (the harness calls {cfg.model.provider} directly)"
    return Smoked(
        run=run.id,
        sandbox=cfg.rollout.sandbox,
        proxy=proxy,
        harness=cfg.rollout.harness,
        profile=wiring(cfg) if serving is not None else "none (provider-served)",
        proxied=serving is not None,
        trials=found,
        seconds=float(row["sandbox_seconds"]),
        tokens=(prompt, cached, int(row["output_tokens"])),
    )


def wiring(cfg: Blueprint) -> str:
    """The harness's profile name with what it adds, as in `pi (model_api=openai-completions)`.
    A table kwarg shows by name alone: `opencode (provider=shipyard, opencode_config)`."""
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
