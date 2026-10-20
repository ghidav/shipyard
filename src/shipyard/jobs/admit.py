from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor.agents.installed.base import (
    AgentAuthenticationError,
    AgentSafetyRefusalError,
    ApiError,
    ContextWindowExceededError,
    ModelNotFoundError,
    NetworkConnectionError,
    OutputTokenExceededError,
)
from harbor.agents.installed.vibe import ApiConnectionError

from harness_proxy.wire import BUDGET, CONTEXT, pictured

if TYPE_CHECKING:
    from harbor.models.trial.result import TrialResult

    from shipyard.groups import Trial
    from shipyard.jobs.engine import Job

#: Mask reasons.
ENV_ERROR = "env_error"
GRADING_ERROR = "grading_error"
TIMEOUT = "timeout"
API_ERROR = "api_error"
CONTEXT_OVERFLOW = "context_overflow"
MULTIMODAL = "multimodal"

#: Harbor's exception type for a trial that hit its timeout.
TIMED_OUT = "AgentTimeoutError"
#: The `ApiError`s the policy causes itself: an output too long, a context too full, a refusal.
POLICY_CAUSED = (OutputTokenExceededError, ContextWindowExceededError, AgentSafetyRefusalError)


def _below(*roots: type) -> frozenset[str]:
    found, stack = set(), list(roots)
    while stack:
        one = stack.pop()
        found.add(one.__name__)
        stack.extend(one.__subclasses__())
    return frozenset(found)


#: Harbor's exception types that indicate an endpoint failure, read from its class tree: an
#: `ApiError` the policy did not cause, or the agent's CLI unable to log in, find the model or
#: reach the network, which Harbor raises beside `ApiError` rather than under it. A type Harbor
#: adds later is classified by its place in the tree, and `endpoint_failed` also takes any
#: name ending in "ApiError".
ENDPOINT_FAILED = _below(
    ApiError,
    AgentAuthenticationError,
    ModelNotFoundError,
    NetworkConnectionError,
    ApiConnectionError,
) - _below(*POLICY_CAUSED)
#: Harbor's verifier-side exception types: the agent finished but grading failed. After the
#: context fills, these are not treated as the harness giving up.
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
    reward: float | None
    mask: str | None
    ended: str | None


def judge(job: Job) -> None:
    """Give each of the job's trials a verdict on whether its last model call failed."""
    for one in job.trials:
        failed = one.records is not None and failed_last(one)
        one.verdict = verdict(
            one.result, one.records, failed=failed, grade_timeouts=job.grade_timeouts
        )


def verdict(
    result: TrialResult | None,
    records: Sequence[Any] | None = None,
    *,
    failed: bool = False,
    grade_timeouts: bool = False,
) -> Verdict:
    """Judge one trial from Harbor's with either the verifier's reward or a mask."""
    ended = result.exception_info if result is not None else None
    why = ended.exception_type if ended is not None else None
    if records is not None:
        found = served_verdict(records, why, failed=failed)
        if found is not None:
            return found
    if result is None:
        return Verdict(None, ENV_ERROR, None)
    if why == TIMED_OUT and not grade_timeouts:
        return Verdict(None, TIMEOUT, why)
    if endpoint_failed(why):
        return Verdict(None, API_ERROR, why)
    rewards = result.verifier_result.rewards if result.verifier_result is not None else None
    reward = (rewards or {}).get("reward")
    if reward is None:
        return Verdict(None, GRADING_ERROR, why)
    return Verdict(float(reward), None, why)


def served_verdict(
    records: Sequence[Any], why: str | None, *, failed: bool = False
) -> Verdict | None:
    """Return the verdict the proxy's records decide before the verifier is read."""
    if not records:
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
    if any(not pictured(item) for item in records if item.error is None):
        return Verdict(None, MULTIMODAL, why)
    return None


def failed_last(trial: Trial) -> bool:
    """Return whether the trial's harness log says its last model call failed."""
    reader = trial.harness.profile.failed_last
    text = harness_log(trial.dir, trial.harness.name) if reader is not None else None
    return reader is not None and text is not None and reader(text)


def harness_log(trial: Path, harness: str) -> str | None:
    try:
        return (trial / "agent" / f"{harness}.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def endpoint_failed(why: str | None) -> bool:
    return why is not None and (why in ENDPOINT_FAILED or why.endswith("ApiError"))
