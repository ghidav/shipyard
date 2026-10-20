"""Tests for `shipyard.jobs.admit`: verdicts read from Harbor's `result.json` and the proxy's
records."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from harness_proxy.wire import BUDGET, CONTEXT, IMAGE_TOKEN, Picture
from shipyard.groups import Trial, result_in
from shipyard.jobs import admit
from shipyard.jobs.admit import (
    API_ERROR,
    CONTEXT_OVERFLOW,
    ENDPOINT_FAILED,
    ENV_ERROR,
    GRADING_ERROR,
    MULTIMODAL,
    TIMEOUT,
    Verdict,
    endpoint_failed,
    failed_last,
    judge,
)
from shipyard.jobs.engine import Job
from shipyard.jobs.harnesses import PROFILES, harness, last_assistant_failed
from tests.fakes import made, write_result


def verdict(
    trial: Path,
    records: list[Any] | None = None,
    *,
    failed: bool = False,
    grade_timeouts: bool = False,
) -> Verdict:
    """Judge the trial directory `trial` from the result Harbor left in it, as `judge` does."""
    return admit.verdict(result_in(trial), records, failed=failed, grade_timeouts=grade_timeouts)


#: Harbor `ApiError` subclasses caused by the policy itself, which keep the verifier's reward.
POLICY_OWN = frozenset(
    {"OutputTokenExceededError", "ContextWindowExceededError", "AgentSafetyRefusalError"}
)


def verified(reward: float, ended: str | None = None) -> Verdict:
    return Verdict(reward, None, ended)


def _job(*trials: Path) -> Job:
    pi = harness("pi")
    return Job("j", trials=[Trial("j", Path("t"), pi, one) for one in trials])


def test_no_result_or_an_unreadable_one_is_an_env_error(tmp_path: Path) -> None:
    assert verdict(tmp_path / "never-written") == Verdict(None, ENV_ERROR, None)
    broken = write_result(tmp_path / "broken", raw="{not json")
    assert verdict(broken) == Verdict(None, ENV_ERROR, None)
    listed = write_result(tmp_path / "listed", raw="[1, 2]")
    assert verdict(listed) == Verdict(None, ENV_ERROR, None)


def test_harbors_clock_cut_is_a_timeout_whatever_the_verifier_wrote(tmp_path: Path) -> None:
    cut = write_result(tmp_path / "cut", reward=1.0, exception="AgentTimeoutError")
    assert verdict(cut) == Verdict(None, TIMEOUT, "AgentTimeoutError")
    graded = verdict(cut, grade_timeouts=True)
    assert graded == verified(1.0, "AgentTimeoutError"), "a recipe that grades them"


@pytest.mark.parametrize("named", sorted(ENDPOINT_FAILED))
def test_every_failure_the_endpoint_causes_is_masked(tmp_path: Path, named: str) -> None:
    """Harbor runs the verifier after these too, so the reward it wrote is not the policy's."""
    trial = write_result(tmp_path / "t", reward=0.0, exception=named)
    assert verdict(trial) == Verdict(None, API_ERROR, named)


def test_an_api_error_harbor_has_not_named_yet_is_masked_by_its_suffix(tmp_path: Path) -> None:
    assert endpoint_failed("GatewayApiError") and not endpoint_failed(None)
    assert not endpoint_failed("NonZeroAgentExitCodeError")
    trial = write_result(tmp_path / "t", reward=0.0, exception="GatewayApiError")
    assert verdict(trial).mask == API_ERROR


@pytest.mark.parametrize("named", sorted(POLICY_OWN))
def test_an_api_error_the_policy_caused_keeps_the_verifiers_number(
    tmp_path: Path, named: str
) -> None:
    trial = write_result(tmp_path / "t", reward=0.0, exception=named)
    assert verdict(trial) == verified(0.0, named)


def test_harbors_error_tree_decides_which_agent_failures_are_the_endpoints() -> None:
    """Every `ApiError` but the policy's own, and the agent's CLI unable to log in, find the
    model or reach the network, are the endpoint's; a bare non-zero exit is the agent's. Fails
    when Harbor adds an agent error that is neither."""
    from harbor.agents.installed import base

    def under(root: type) -> set[str]:
        return {
            name for name, cls in inspect.getmembers(base, inspect.isclass) if issubclass(cls, root)
        }

    beside = {"AgentAuthenticationError", "ModelNotFoundError", "NetworkConnectionError"}
    vibes = {"ApiConnectionError"}  # harbor.agents.installed.vibe's own network failure
    assert ENDPOINT_FAILED == (under(base.ApiError) - POLICY_OWN) | beside | vibes
    assert not ENDPOINT_FAILED & POLICY_OWN
    left = under(base.NonZeroAgentExitCodeError) - ENDPOINT_FAILED - POLICY_OWN
    assert left == {"NonZeroAgentExitCodeError"}


def test_a_numeric_reward_is_the_verdict_and_anything_else_a_grading_error(
    tmp_path: Path,
) -> None:
    assert verdict(write_result(tmp_path / "one", reward=1.0)) == verified(1.0)
    assert verdict(write_result(tmp_path / "half", rewards={"reward": 0.5, "style": 2})) == (
        Verdict(0.5, None, None)
    )
    ended = write_result(tmp_path / "ended", reward=0.0, exception="NonZeroAgentExitCodeError")
    assert verdict(ended) == verified(0.0, "NonZeroAgentExitCodeError")
    assert verdict(write_result(tmp_path / "none")) == Verdict(None, GRADING_ERROR, None)
    lost = write_result(tmp_path / "lost", exception="RewardFileNotFoundError")
    assert verdict(lost) == Verdict(None, GRADING_ERROR, "RewardFileNotFoundError")


def test_judge_gives_each_trial_its_verdict_in_plan_order(tmp_path: Path) -> None:
    won = write_result(tmp_path / "won", reward=1.0)
    lost = write_result(tmp_path / "lost", reward=0.0)
    cut = write_result(tmp_path / "cut", reward=1.0, exception="AgentTimeoutError")
    job = _job(won, tmp_path / "gone", lost, cut)
    judge(job)
    assert [one.verdict.mask for one in job.trials if one.verdict] == [
        None,
        ENV_ERROR,
        None,
        TIMEOUT,
    ]
    served = _job(won, lost)
    served.trials[0].records, served.trials[1].records = [made()], []
    judge(served)
    assert [one.verdict for one in served.trials] == [
        verified(1.0),
        Verdict(None, ENV_ERROR, None),
    ], "each trial is judged with its own records"


def test_no_records_is_an_env_error_whatever_the_verifier_wrote(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    assert verdict(graded, []) == Verdict(None, ENV_ERROR, None)
    assert verdict(graded, [made()]) == verified(1.0)


def test_a_first_turn_that_never_fit_is_a_context_overflow(tmp_path: Path) -> None:
    gave_up = write_result(tmp_path / "t", reward=0.0, exception="UnknownApiError")
    assert verdict(gave_up, [made(error=CONTEXT)]) == Verdict(
        None, CONTEXT_OVERFLOW, "UnknownApiError"
    )


def test_a_rollout_cut_by_its_budget_is_measured_zero_not_masked(tmp_path: Path) -> None:
    records = [made(seq=1), made(seq=2, error=BUDGET)]
    assert verdict(write_result(tmp_path / "t", reward=1.0), records) == Verdict(0.0, None, None)
    timed = write_result(tmp_path / "u", reward=1.0, exception="AgentTimeoutError")
    assert verdict(timed, records) == Verdict(0.0, None, "AgentTimeoutError")


def test_a_context_the_policy_filled_before_its_harness_quit_is_zero(tmp_path: Path) -> None:
    records = [made(seq=1), made(seq=2, error=CONTEXT)]
    quit_ = write_result(tmp_path / "quit", reward=1.0, exception="UnknownApiError")
    assert verdict(quit_, records) == Verdict(0.0, None, "UnknownApiError")
    finished = write_result(tmp_path / "finished", reward=1.0)
    assert verdict(finished, records) == verified(1.0)
    timed = write_result(tmp_path / "timed", reward=1.0, exception="AgentTimeoutError")
    assert verdict(timed, records) == Verdict(None, TIMEOUT, "AgentTimeoutError")


def test_a_harness_that_quit_on_a_failed_call_is_an_api_error(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=0.0)
    failed = [made(seq=1), made(seq=2, error="sampler: ConnectError")]
    assert verdict(graded, failed) == Verdict(None, API_ERROR, None)
    assert verdict(graded, [*failed, made(seq=3)]) == verified(0.0), "it recovered"


def test_a_last_call_the_harness_logged_as_failed_and_the_proxy_never_saw_is_masked(
    tmp_path: Path,
) -> None:
    """The call failed on the way, not in the model: the verifier's 0 is not the policy's."""
    graded = write_result(tmp_path / "t", reward=0.0)
    clean = [made(seq=1), made(seq=2)]
    assert verdict(graded, clean, failed=True) == Verdict(None, API_ERROR, None)
    assert verdict(graded, clean) == verified(0.0)
    refused = [made(seq=1), made(seq=2, error=CONTEXT)]
    assert verdict(graded, refused, failed=True) == verified(0.0), "the proxy saw it"


def _pi_log(*stops: str) -> str:
    """Build a pi JSON log ending each assistant message with the given stop reasons. Each
    message gets `message_start` and `message_end` lines followed by a tool result line."""
    lines = []
    for stop in stops:
        message = {"role": "assistant", "content": [], "stopReason": stop}
        lines.append(json.dumps({"type": "message_start", "message": message}))
        lines.append(json.dumps({"type": "message_end", "message": message}))
        lines.append(json.dumps({"type": "message_end", "message": {"role": "toolResult"}}))
    return "\n".join([*lines, '{"type":"agent_settled"}'])


def test_pi_says_its_last_call_failed_only_when_its_last_assistant_message_did() -> None:
    assert PROFILES["pi"].failed_last is last_assistant_failed
    assert last_assistant_failed(_pi_log("toolUse", "error")) is True
    assert last_assistant_failed(_pi_log("toolUse", "error", "stop")) is False, "pi recovered"
    assert last_assistant_failed(_pi_log("toolUse", "stop")) is False
    assert last_assistant_failed(_pi_log("toolUse", "length")) is False
    assert last_assistant_failed("") is False and last_assistant_failed("not json") is False
    assert last_assistant_failed('"message_end"') is False, "a line that is not an object"


def test_judge_masks_a_served_pi_trial_whose_log_ends_on_a_failed_call(tmp_path: Path) -> None:
    failed, settled, unlogged = (
        write_result(tmp_path / name, reward=0.0) for name in ("failed", "settled", "unlogged")
    )
    for one, stop in ((failed, "error"), (settled, "stop")):
        (one / "agent").mkdir()
        (one / "agent" / "pi.txt").write_text(_pi_log("toolUse", stop))
    served = _job(failed, settled, unlogged)
    for one in served.trials:
        one.records = [made()]
    judge(served)
    assert [one.verdict for one in served.trials] == [
        Verdict(None, API_ERROR, None),
        verified(0.0),
        verified(0.0),
    ]
    provided = _job(failed)
    judge(provided)
    assert provided.trials[0].verdict == verified(0.0), "no records: the provider served it"
    claude = Trial("j", Path("t"), harness("claude-code"), failed, [made()])
    assert not failed_last(claude), "only a harness with a reader says so"


def test_an_image_the_record_did_not_keep_masks_the_rollout_multimodal(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    seen = [made(seq=1), made((1, IMAGE_TOKEN, IMAGE_TOKEN, 2), seq=2)]
    assert verdict(graded, seen) == Verdict(None, MULTIMODAL, None)
    shot = Picture(1, 2, "png", "cG5nIQ==")
    kept = [made(seq=1), made((1, IMAGE_TOKEN, IMAGE_TOKEN, 2), seq=2, images=(shot,))]
    assert verdict(graded, kept) == verified(1.0), "a kept image trains"
