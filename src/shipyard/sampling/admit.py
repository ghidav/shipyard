"""Trial admission: decide whether each trial is graded or masked, and why.

Verdicts are read from Harbor's `result.json` and, for a served run, from the proxy's records
of the policy's model calls."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipyard.harnesses import bare_name
from shipyard.proxy.wire import BUDGET, CONTEXT, Record, pictured
from shipyard.types import Job, Trial, Verdict

__all__ = [
    "endpoint_failed",
    "harness_log",
    "judge",
    "read_logs",
    "result_of",
    "results",
    "verdict",
    "verdicts",
]

logger = logging.getLogger(__name__)

#: Mask reasons. A masked trial is excluded from every mean and every gradient.
ENV_ERROR = "env_error"
GRADING_ERROR = "grading_error"
TIMEOUT = "timeout"
API_ERROR = "api_error"
CONTEXT_OVERFLOW = "context_overflow"
MULTIMODAL = "multimodal"

#: Harbor's exception type for a trial that hit its timeout.
TIMED_OUT = "AgentTimeoutError"
#: Harbor's `ApiError` subclasses that indicate an endpoint failure, from
#: `harbor.agents.installed.base`. The three excluded subclasses are caused by the policy
#: (`OutputTokenExceededError`, `ContextWindowExceededError`, `AgentSafetyRefusalError`).
#: None of them ends in "ApiError", so `endpoint_failed` also matches that suffix.
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


def _ended(payload: dict[str, Any] | None) -> str | None:
    """Return Harbor's `exception_info.exception_type` from a result payload, or None."""
    found = payload.get("exception_info") if payload else None
    named = found.get("exception_type") if isinstance(found, dict) else None
    return str(named) if named else None


def endpoint_failed(why: str | None) -> bool:
    """Return whether the Harbor exception type `why` indicates an endpoint failure."""
    return why is not None and (why in ENDPOINT_FAILED or why.endswith("ApiError"))


def verdict(
    trial: Path,
    records: Sequence[Record] | None = None,
    *,
    asked: int | None = None,
    failed: bool = False,
) -> Verdict:
    """Judge one trial: return a reward with the verifier's numbers, or a mask.

    When `records` are given, `served_verdict` is checked first, then the trial's
    `result.json`."""
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
    if _number(reward):
        numbers = {key: float(one) for key, one in rewards.items() if _number(one)}
        return Verdict(float(reward), None, why, numbers)
    return Verdict(None, GRADING_ERROR, why)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def served_verdict(
    records: Sequence[Record],
    why: str | None,
    *,
    asked: int | None = None,
    failed: bool = False,
) -> Verdict | None:
    """Return the verdict the proxy's records decide before the verifier is read, or None.

    No records, or fewer than the harness asked for, is `env_error`. A first turn that never
    fit in the context is `context_overflow`. A trial cut by the token budget scores 0, and so
    does one that ended in an error (other than a timeout or a grading failure) after its
    context filled. A harness that quit after a failed call is `api_error`. So is a trial
    whose harness log says its last call failed (`failed`) while the last record shows no
    failure: the proxy never saw that call, so it failed on the way, not in the model. A call
    with an image the record did not keep is `multimodal`, since nothing can train on it."""
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
    if any(not pictured(item) for item in records if item.error is None):
        return Verdict(None, MULTIMODAL, why)
    return None


def judge(job: Job, *, check_turns: bool) -> None:
    """Set `job.verdicts`, one per trial.

    For a served job (one with records), first read the harness logs into `job.asked`, the
    turns each trial asked for, and `job.failed`, the trials whose last model call failed."""
    if job.records is not None:
        job.asked, job.failed = read_logs(job.trials, job.records, check_turns=check_turns)
    job.verdicts = verdicts(job)


def verdicts(job: Job) -> list[Verdict]:
    """Return one verdict per trial, in plan order.

    For a served job, each trial is judged with its own records and its harness's turn
    count."""
    if job.records is None:
        return [verdict(one.dir) for one in job.trials]
    asked, failed = job.asked or {}, job.failed or set()
    return [
        verdict(
            one.dir,
            list(job.records.get(one.name) or []),
            asked=asked.get(one.name),
            failed=one.name in failed,
        )
        for one in job.trials
    ]


def read_logs(
    trials: Sequence[Trial], records: Mapping[str, Sequence[Record]], *, check_turns: bool
) -> tuple[dict[str, int], set[str]]:
    """Read each trial's harness log for its turn count and its last call's outcome.

    Return the number of turns each harness asked for, by trial name (counted only with
    `check_turns`), and the names of the trials whose last model call failed."""
    asked: dict[str, int] = {}
    failed: set[str] = set()
    for trial in trials:
        profile = trial.harness.profile
        counter = profile.turns if check_turns else None
        reader = profile.failed_last
        if counter is None and reader is None:
            continue
        text = harness_log(trial.dir, trial.harness.name)
        if text is None:
            continue
        if counter is not None:
            asked[trial.name] = turns = int(counter(text))
            if turns == 0 and records.get(trial.name):
                logger.warning(
                    "%s: the %s turn counter found no turns beside %d record(s); its log "
                    "format may have moved, so this trial is not checked",
                    trial.name,
                    trial.harness.name,
                    len(records[trial.name]),
                )
        if reader is not None and reader(text):
            failed.add(trial.name)
    return asked, failed


def harness_log(trial: Path, harness: str) -> str | None:
    """Return the text of the trial's `agent/<harness>.txt`, or None if it cannot be read."""
    try:
        return (trial / "agent" / f"{bare_name(harness)}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None


def results(job: Job) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield each trial directory with a readable `result.json`, and its payload, in plan order.

    Every reader of a job's results uses this function, so they agree on which trials
    count."""
    for one in job.trials:
        payload = result_of(one.dir)
        if payload is not None:
            yield one.dir, payload


def result_of(trial: Path) -> dict[str, Any] | None:
    """Return the trial's `result.json` as a dict, or None if it is missing or unreadable.

    Every module reads the file through this function, so they agree on its shape."""
    try:
        payload = json.loads((Path(trial) / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
