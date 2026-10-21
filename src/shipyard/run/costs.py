from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from harbor.models.agent.context import ModelUsage

if TYPE_CHECKING:
    from harbor.models.trial.result import TimingInfo, TrialResult

    from shipyard.groups import Trial

#: The kinds of token, each billed at its own price: prompt tokens read fresh, prompt tokens
#: read from the provider's cache, tokens sampled, and tokens through a training client's pass.
KINDS = ("prefill", "cache", "sample", "train")
#: The model counts are filed under when nothing says which model spent them.
UNKNOWN = "unknown"

#: Token counts by provider, then model, then kind.
Tokens = dict[str, dict[str, dict[str, int]]]


class Spend(Protocol):
    """Files a policy's spend, `spend(provider, model, **counts)`, under its role in the costs."""

    def __call__(self, provider: str, model: str, /, **counts: int) -> None: ...


def counted(counts: Mapping[str, int]) -> dict[str, int]:
    """Return token counts by kind, in `KINDS` order, leaving out zeros."""
    unknown = set(counts) - set(KINDS)
    if unknown:
        raise ValueError(f"tokens are counted as {', '.join(KINDS)}; got {sorted(unknown)}")
    return {kind: int(counts[kind]) for kind in KINDS if counts.get(kind)}


def filed(provider: str, model: str) -> str:
    """Return the name a provider's model is counted under: without the provider's prefix."""
    return model.removeprefix(f"{provider}/")


@dataclass
class Costs:
    """Tokens by provider, model and kind, and trials and seconds per sandbox."""

    tokens: Tokens = field(default_factory=dict)
    sandboxes: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, provider: str, model: str, **counts: int) -> None:
        """Add token counts by kind to a provider's model, leaving out zeros."""
        found = counted(counts)
        if not found:
            return
        held = self.tokens.setdefault(provider, {}).setdefault(filed(provider, model), {})
        for kind, value in found.items():
            held[kind] = held.get(kind, 0) + value

    def merge(self, tokens: Tokens) -> None:
        """Add counts in the shape of `tokens`."""
        for provider, models in tokens.items():
            for model, counts in models.items():
                self.add(provider, model, **counts)

    def add_sandbox(self, name: str, trials: int, seconds: float) -> None:
        held = self.sandboxes.setdefault(name, {"trials": 0, "seconds": 0.0})
        held["trials"] += int(trials)
        held["seconds"] += float(seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": {
                provider: {model: dict(counts) for model, counts in models.items()}
                for provider, models in self.tokens.items()
            },
            "sandboxes": {name: dict(held) for name, held in self.sandboxes.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Costs:
        """Rebuild costs from the output of `to_dict`. A missing table is read as empty."""
        costs = cls()
        costs.merge(payload.get("tokens") or {})
        for name, held in (payload.get("sandboxes") or {}).items():
            costs.add_sandbox(name, held.get("trials", 0), held.get("seconds", 0.0))
        return costs

    @classmethod
    def of(cls, rows: Iterable[Mapping[str, Any]]) -> Costs:
        """Return the totals of usage-ledger rows: each a model's tokens, or a sandbox's time."""
        costs = cls()
        for row in rows:
            if "sandbox" in row:
                costs.add_sandbox(row["sandbox"], row.get("trials", 0), row.get("seconds", 0.0))
            else:
                costs.add(row["provider"], row["model"], **{k: row[k] for k in KINDS if k in row})
        return costs


def sampled(trials: Sequence[Trial], provider: str, model: str) -> tuple[Tokens, dict[str, int]]:
    """Count a served job's tokens from its records, and its refusals and bridged prompts."""
    costs = Costs()
    for one in trials:
        costs.add(provider, model, **one.records.billed)
    noted = {
        "turned_away": sum(one.records.turned_away for one in trials),
        "bridged": sum(one.records.bridged for one in trials),
    }
    return costs.tokens, noted


def reported(trials: Sequence[Trial], provider: str, model: str) -> Tokens:
    """Count the tokens the agents reported in their trials' results, billed to `provider`."""
    costs = Costs()
    for trial in trials:
        result = trial.result
        if result is None:
            continue
        stated = [result.agent_result, *(step.agent_result for step in result.step_results or ())]
        for one in (context for context in stated if context is not None):
            used = one.model_usage or {
                model: ModelUsage(
                    n_input_tokens=one.n_input_tokens or 0,
                    n_cache_tokens=one.n_cache_tokens or 0,
                    n_output_tokens=one.n_output_tokens or 0,
                )
            }
            for name, usage in used.items():
                costs.add(
                    provider,
                    name,
                    prefill=usage.n_input_tokens - usage.n_cache_tokens,
                    cache=usage.n_cache_tokens,
                    sample=usage.n_output_tokens,
                )
    return costs.tokens


def billed_by(trials: Sequence[Trial]) -> str | None:
    """Return the provider every trial that names one reports, or None."""
    providers: set[str] = set()
    for trial in trials:
        model = trial.result.agent_info.model_info if trial.result is not None else None
        named = model.provider if model is not None else None
        if named and named.strip():
            providers.add(named.strip().lower())
    return providers.pop() if len(providers) == 1 else None


def sandbox_seconds(trials: Sequence[Trial]) -> tuple[int, float]:
    """Return how many trials have a result, and their containers' seconds in all."""
    count, seconds = 0, 0.0
    for trial in trials:
        if trial.result is None:
            continue
        count += 1
        seconds += _container_seconds(trial.result)
    return count, seconds


def _container_seconds(result: TrialResult) -> float:
    """Return the seconds from the first timed phase's start to the last one's end."""
    phases: list[TimingInfo | None] = [
        result.environment_setup,
        result.agent_setup,
        result.agent_execution,
        result.verifier,
    ]
    for step in result.step_results or ():
        phases += [step.agent_execution, step.verifier]
    timed = [one for one in phases if one is not None]
    started = [one.started_at for one in timed if one.started_at is not None]
    finished = [one.finished_at for one in timed if one.finished_at is not None]
    if not started or not finished:
        return 0.0
    return max((max(finished) - min(started)).total_seconds(), 0.0)
