"""Admission: which trials are measured, which are masked, and why, off `result.json`
and, for a served run, off the proxy's records of what the policy was asked and said."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shipyard.proxy.wire import IMAGE_TOKEN, Record

if TYPE_CHECKING:
    from shipyard.rollout import Rollouts

#: The masks: a trial left out of every mean and every gradient, by reason.
ENV_ERROR = "env_error"
GRADING_ERROR = "grading_error"
TIMEOUT = "timeout"
API_ERROR = "api_error"
CONTEXT_OVERFLOW = "context_overflow"
MULTIMODAL = "multimodal"

#: What Harbor names on a trial its clock cut.
TIMED_OUT = "AgentTimeoutError"
#: The proxy's two refusals, each the policy's own doing: a spent budget, a full context.
BUDGET = "budget"
CONTEXT = "context"

#: Harbor's `ApiError` subclasses that are the endpoint's doing, from
#: `harbor.agents.installed.base`. The three left out are the policy's own
#: (`OutputTokenExceededError`, `ContextWindowExceededError`, `AgentSafetyRefusalError`),
#: and none of them ends in "ApiError", so `endpoint_failed` also takes that suffix.
ENDPOINT_FAILED = frozenset(
    {
        "ApiError",
        "UnknownApiError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiConnectionClosedError",
        "ApiResponseStalledError",
        "ApiRateLimitError",
        "ApiUsageLimitError",
        "ApiProviderResourceNotFoundError",
    }
)
#: Harbor's verifier-side endings: the agent finished and the grader did not. A filled
#: context does not read them as the harness giving up.
GRADER_FAILED = frozenset(
    {
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierTimeoutError",
        "VerifierOutputParseError",
    }
)


@dataclass(frozen=True)
class Verdict:
    """A trial's reward or its mask (exactly one is set), and how Harbor said it ended."""

    reward: float | None
    mask: str | None
    ended: str | None


def _ended(payload: dict[str, Any] | None) -> str | None:
    """Harbor's `exception_info.exception_type` from a result payload, or None."""
    found = payload.get("exception_info") if payload else None
    named = found.get("exception_type") if isinstance(found, dict) else None
    return str(named) if named else None


def endpoint_failed(why: str | None) -> bool:
    """Whether the ending Harbor named is an endpoint failure."""
    return why is not None and (why in ENDPOINT_FAILED or why.endswith("ApiError"))


def verdict(
    trial: Path,
    records: Sequence[Record] | None = None,
    *,
    asked: int | None = None,
    failed: bool = False,
) -> Verdict:
    """With records, the served endings come first (`served_verdict`). Then: no result is
    `env_error`, the clock's cut is `timeout`, an endpoint failure is `api_error`, a
    numeric `rewards.reward` is the reward, and anything else is `grading_error`."""
    payload = result_of(trial)
    why = _ended(payload)
    if records is not None:
        found = served_verdict(records, why, asked=asked, failed=failed)
        if found is not None:
            return found
    if payload is None:
        return Verdict(None, ENV_ERROR, None)
    if why == TIMED_OUT:
        return Verdict(None, TIMEOUT, why)
    if endpoint_failed(why):
        return Verdict(None, API_ERROR, why)
    verifier = payload.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if isinstance(reward, int | float) and not isinstance(reward, bool):
        return Verdict(float(reward), None, why)
    return Verdict(None, GRADING_ERROR, why)


def served_verdict(
    records: Sequence[Record],
    why: str | None,
    *,
    asked: int | None = None,
    failed: bool = False,
) -> Verdict | None:
    """The served endings, decided before the verifier's number is read, or None. No
    records, or too few, is `env_error`. A first turn that never fit is
    `context_overflow`. A budget cut or a filled context scores 0. A quit on a failed call
    is `api_error`. An image is `multimodal`. `failed` is the harness's log saying its last
    call failed. With no failure on the last record, the proxy never saw that call and
    the trial measured the connection."""
    if not records:
        return Verdict(None, ENV_ERROR, why)
    if asked is not None and asked > len(records):
        return Verdict(None, ENV_ERROR, why)
    spoke = any(item.error is None for item in records)
    if records[0].error == CONTEXT and not spoke:
        return Verdict(None, CONTEXT_OVERFLOW, why)
    if any(item.error == BUDGET for item in records):
        return Verdict(0.0, None, why)
    filled = spoke and any(item.error == CONTEXT for item in records)
    if filled and why is not None and why != TIMED_OUT and why not in GRADER_FAILED:
        return Verdict(0.0, None, why)
    last = records[-1].error
    if last is not None and last not in (BUDGET, CONTEXT):
        return Verdict(None, API_ERROR, why)
    if failed and last is None:
        return Verdict(None, API_ERROR, why)
    if any(IMAGE_TOKEN in item.prompt_token_ids for item in records):
        return Verdict(None, MULTIMODAL, why)
    return None


def verdicts(rollouts: Rollouts) -> list[Verdict]:
    """One verdict per trial, in plan order. When the run attached records to the
    rollouts, each trial is judged with its own records and its harness's turn count."""
    if rollouts.records is None:
        return [verdict(Path(trial)) for trial in rollouts.trials]
    asked, failed = rollouts.asked or {}, rollouts.failed or set()
    return [
        verdict(
            Path(trial),
            list(rollouts.records.get(Path(trial).name) or []),
            asked=asked.get(Path(trial).name),
            failed=Path(trial).name in failed,
        )
        for trial in rollouts.trials
    ]


def scores(rollouts: Rollouts) -> list[float]:
    """The rewards of the graded trials. A masked trial is left out, not scored zero."""
    return [found.reward for found in verdicts(rollouts) if found.reward is not None]


def results(rollouts: Rollouts) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Each trial that left a readable `result.json`, with its payload, in plan order.
    Every reader of a job's results uses this walk, so none skips a trial another counts."""
    for trial in rollouts.trials:
        payload = result_of(trial)
        if payload is not None:
            yield Path(trial), payload


def result_of(trial: Path) -> dict[str, Any] | None:
    """The trial's `result.json` as a dict, or None when missing or unreadable. Every
    module reads the file through this function, so they agree on its shape."""
    try:
        payload = json.loads((Path(trial) / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
