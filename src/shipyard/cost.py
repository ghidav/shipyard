"""What a run spent, as counts: tokens per party and seconds per sandbox, never a total."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from harbor.models.agent.context import AgentContext

from shipyard.admit import results

if TYPE_CHECKING:
    from shipyard.rollout import Rollouts

logger = logging.getLogger(__name__)

#: What a party's entry counts, and nothing else: no price can be added to it.
COUNTS = ("trials", "input_tokens", "cache_tokens", "output_tokens")
#: What training adds to the Tinker party, present only once a run has spent them.
TRAINING_COUNTS = ("train_tokens", "reference_tokens", "anchor_tokens")
#: The phases Harbor times on a trial, in the order they run.
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


@dataclass
class Costs:
    """One entry per party and per sandbox, each a sum of counts; `to_dict` is `costs.json`."""

    parties: dict[str, dict[str, int]] = field(default_factory=dict)
    sandbox: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_party(self, name: str, **counts: int) -> None:
        """Add to a party's counts; a key outside `COUNTS` and `TRAINING_COUNTS` is
        refused, so no price gets in."""
        unknown = set(counts) - set(COUNTS) - set(TRAINING_COUNTS)
        if unknown:
            named = ", ".join(COUNTS + TRAINING_COUNTS)
            raise ValueError(f"a party counts {named}; not {sorted(unknown)}")
        held = self.parties.setdefault(name, dict.fromkeys(COUNTS, 0))
        for key, value in counts.items():
            held[key] = held.get(key, 0) + int(value)

    def add_sandbox(self, name: str, trials: int, seconds: float) -> None:
        held = self.sandbox.setdefault(name, {"trials": 0, "seconds": 0.0})
        held["trials"] += int(trials)
        held["seconds"] += float(seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parties": {name: dict(held) for name, held in self.parties.items()},
            "sandbox": {name: dict(held) for name, held in self.sandbox.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Costs:
        """Read back what `to_dict` wrote; a missing table is empty."""
        costs = cls()
        for name, held in (payload.get("parties") or {}).items():
            base = {key: held.get(key, 0) for key in COUNTS}
            trained = {key: held[key] for key in TRAINING_COUNTS if key in held}
            costs.add_party(name, **base, **trained)
        for name, held in (payload.get("sandbox") or {}).items():
            costs.add_sandbox(name, held.get("trials", 0), held.get("seconds", 0.0))
        return costs


def reported(rollouts: Rollouts) -> dict[str, int]:
    """Over the trials with a result: how many (`trials`), and the `n_input_tokens` (cache
    inside), `n_cache_tokens` and `n_output_tokens` their agents reported, summed. A trial
    that reported nothing adds no tokens, since a zero would claim it spent none."""
    totals = dict.fromkeys(COUNTS, 0)
    for trial, payload in results(rollouts):
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


def billed_by(rollouts: Rollouts) -> str | None:
    """`agent_info.model_info.provider`, lower-cased, when every trial that recorded one
    agrees; None when none did or they differ, and the blueprint's provider stands."""
    providers: set[str] = set()
    for _, payload in results(rollouts):
        info = payload.get("agent_info")
        model = info.get("model_info") if isinstance(info, dict) else None
        named = model.get("provider") if isinstance(model, dict) else None
        if isinstance(named, str) and named.strip():
            providers.add(named.strip().lower())
    return providers.pop() if len(providers) == 1 else None


def sandbox_seconds(rollouts: Rollouts) -> tuple[int, float]:
    """How many trials left a result, and their summed container seconds. A trial with
    no result has no time on it: writing a zero would say its container never ran."""
    trials, seconds = 0, 0.0
    for _, payload in results(rollouts):
        trials += 1
        seconds += _container_seconds(payload)
    return trials, seconds


def _container_seconds(payload: Mapping[str, Any]) -> float:
    """From the first timed phase starting to the last one finishing; zero if none was."""
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
