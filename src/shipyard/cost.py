"""Token and sandbox-time accounting for a run."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from harbor.models.agent.context import AgentContext

from shipyard.proxy.wire import BUDGET, CONTEXT
from shipyard.sampling.admit import results

if TYPE_CHECKING:
    from shipyard.proxy.wire import Record
    from shipyard.types import Job

logger = logging.getLogger(__name__)

#: Counts kept for each party. Only counts are stored, never prices.
COUNTS = ("trials", "input_tokens", "cache_tokens", "output_tokens")
#: Counts training adds to the Tinker party. Each appears only once the run has spent some.
TRAINING_COUNTS = ("train_tokens", "reference_tokens", "anchor_tokens", "teacher_tokens")
#: The phases Harbor times on a trial, in the order they run.
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


@dataclass
class Costs:
    """Running totals per party (trials and tokens) and per sandbox (trials and seconds).
    `to_dict` gives the contents of `costs.json`."""

    parties: dict[str, dict[str, int]] = field(default_factory=dict)
    sandboxes: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_party(self, name: str, **counts: int) -> None:
        """Add counts to a party's totals.

        Raises `ValueError` for a key outside `COUNTS` and `TRAINING_COUNTS`, so that no
        price can be recorded."""
        unknown = set(counts) - set(COUNTS) - set(TRAINING_COUNTS)
        if unknown:
            named = ", ".join(COUNTS + TRAINING_COUNTS)
            raise ValueError(f"a party counts {named}; got {sorted(unknown)}")
        held = self.parties.setdefault(name, dict.fromkeys(COUNTS, 0))
        for key, value in counts.items():
            held[key] = held.get(key, 0) + int(value)

    def add_sandbox(self, name: str, trials: int, seconds: float) -> None:
        held = self.sandboxes.setdefault(name, {"trials": 0, "seconds": 0.0})
        held["trials"] += int(trials)
        held["seconds"] += float(seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parties": {name: dict(held) for name, held in self.parties.items()},
            "sandboxes": {name: dict(held) for name, held in self.sandboxes.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Costs:
        """Rebuild costs from the output of `to_dict`. A missing table is read as empty."""
        costs = cls()
        for name, held in (payload.get("parties") or {}).items():
            base = {key: held.get(key, 0) for key in COUNTS}
            trained = {key: held[key] for key in TRAINING_COUNTS if key in held}
            costs.add_party(name, **base, **trained)
        # A costs.json written before the key was `sandboxes` says `sandbox`.
        for name, held in (payload.get("sandboxes") or payload.get("sandbox") or {}).items():
            costs.add_sandbox(name, held.get("trials", 0), held.get("seconds", 0.0))
        return costs


def reported(job: Job) -> dict[str, int]:
    """Sum the token counts the agents reported in a job's trial results.

    Counts the trials that have a result, and totals their `n_input_tokens` (which include
    cached tokens), `n_cache_tokens` and `n_output_tokens`. A trial that reported nothing
    adds no tokens."""
    totals = dict.fromkeys(COUNTS, 0)
    for trial, payload in results(job):
        totals["trials"] += 1
        stated = [payload.get("agent_result")] + [
            step.get("agent_result")
            for step in payload.get("step_results") or []
            if isinstance(step, dict)
        ]
        try:
            contexts = [AgentContext.model_validate(one) for one in stated if isinstance(one, dict)]
        except Exception:  # noqa: BLE001 - a result this cannot read is a count it cannot know
            logger.warning("could not read what the trial at %s reported", trial.name)
            continue
        totals["input_tokens"] += sum(one.n_input_tokens or 0 for one in contexts)
        totals["cache_tokens"] += sum(one.n_cache_tokens or 0 for one in contexts)
        totals["output_tokens"] += sum(one.n_output_tokens or 0 for one in contexts)
    return totals


def sampled(job: Job) -> tuple[dict[str, int], dict[str, int]]:
    """Count a served job's tokens and trials from the proxy's records.

    Also returns how many calls the proxy refused for budget or context (`turned_away`) and
    how many prompts the bridge built (`bridged`)."""
    calls = [item for found in (job.records or {}).values() for item in found]
    counted = {
        "trials": sum(1 for _ in results(job)),
        "input_tokens": sum(len(item.prompt_token_ids) for item in calls),
        "cache_tokens": sum(cached(item) for item in calls),
        "output_tokens": sum(len(item.completion_token_ids) for item in calls),
    }
    noted = {
        "turned_away": sum(item.error in (BUDGET, CONTEXT) for item in calls),
        "bridged": sum(item.bridged for item in calls),
    }
    return counted, noted


def cached(item: Record) -> int:
    """Return the record's cached prompt tokens, capped at the prompt length."""
    return min(int(item.cached_tokens or 0), len(item.prompt_token_ids))


def billed_by(job: Job) -> str | None:
    """Return the provider the job's trials report in `agent_info.model_info.provider`,
    lower-cased.

    Returns None when no trial reports one or they disagree, in which case the blueprint's
    provider is billed."""
    providers: set[str] = set()
    for _, payload in results(job):
        info = payload.get("agent_info")
        model = info.get("model_info") if isinstance(info, dict) else None
        named = model.get("provider") if isinstance(model, dict) else None
        if isinstance(named, str) and named.strip():
            providers.add(named.strip().lower())
    return providers.pop() if len(providers) == 1 else None


def sandbox_seconds(job: Job) -> tuple[int, float]:
    """Return the number of trials with a result and their total container seconds.
    Trials without a result are not counted."""
    trials, seconds = 0, 0.0
    for _, payload in results(job):
        trials += 1
        seconds += _container_seconds(payload)
    return trials, seconds


def _container_seconds(payload: Mapping[str, Any]) -> float:
    """Return the seconds from the first phase start to the last phase end, or 0.0 if no
    phase was timed."""
    phases = [payload[phase] for phase in PHASES if isinstance(payload.get(phase), dict)]
    started = [when for phase in phases if (when := _at(phase.get("started_at"))) is not None]
    finished = [when for phase in phases if (when := _at(phase.get("finished_at"))) is not None]
    if not started or not finished:
        return 0.0
    return max((max(finished) - min(started)).total_seconds(), 0.0)


def _at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
