"""Every verdict branch, read from `result.json` files in Harbor's shape. Every `ApiError`
Harbor defines is classed as the endpoint's or the policy's."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from shipyard.admit import (
    API_ERROR,
    BUDGET,
    CONTEXT,
    CONTEXT_OVERFLOW,
    ENDPOINT_FAILED,
    ENV_ERROR,
    GRADING_ERROR,
    MULTIMODAL,
    TIMEOUT,
    Verdict,
    endpoint_failed,
    scores,
    served_verdict,
    verdict,
    verdicts,
)
from shipyard.proxy.wire import IMAGE_TOKEN
from shipyard.rollout import Rollouts
from tests.records import made
from tests.trials import write_result

#: Harbor's `ApiError` subclasses that are the policy's own doing, and keep the
#: verifier's number. Kept here so a new Harbor class must be put in one list or the other.
POLICY_OWN = frozenset(
    {"OutputTokenExceededError", "ContextWindowExceededError", "AgentSafetyRefusalError"}
)


def test_no_result_is_an_env_error(tmp_path: Path) -> None:
    assert verdict(tmp_path / "never-written") == Verdict(None, ENV_ERROR, None)
    (tmp_path / "empty").mkdir()
    assert verdict(tmp_path / "empty") == Verdict(None, ENV_ERROR, None)


def test_an_unreadable_result_is_an_env_error(tmp_path: Path) -> None:
    broken = write_result(tmp_path / "broken", raw="{not json")
    assert verdict(broken) == Verdict(None, ENV_ERROR, None)
    listed = write_result(tmp_path / "listed", raw="[1, 2]")
    assert verdict(listed) == Verdict(None, ENV_ERROR, None)


def test_harbors_clock_cut_is_a_timeout_whatever_the_verifier_wrote(tmp_path: Path) -> None:
    cut = write_result(tmp_path / "cut", reward=1.0, exception="AgentTimeoutError")
    assert verdict(cut) == Verdict(None, TIMEOUT, "AgentTimeoutError")


@pytest.mark.parametrize("named", sorted(ENDPOINT_FAILED))
def test_every_api_error_the_endpoint_causes_is_masked(tmp_path: Path, named: str) -> None:
    trial = write_result(tmp_path / "t", reward=0.0, exception=named)
    assert verdict(trial) == Verdict(None, API_ERROR, named)


def test_an_api_error_harbor_has_not_named_yet_is_masked_by_its_suffix(tmp_path: Path) -> None:
    assert endpoint_failed("GatewayApiError")
    assert not endpoint_failed("NonZeroAgentExitCodeError")
    assert not endpoint_failed(None)
    trial = write_result(tmp_path / "t", reward=0.0, exception="GatewayApiError")
    assert verdict(trial).mask == API_ERROR


@pytest.mark.parametrize("named", sorted(POLICY_OWN))
def test_an_api_error_the_policy_caused_keeps_the_verifiers_number(
    tmp_path: Path, named: str
) -> None:
    trial = write_result(tmp_path / "t", reward=0.0, exception=named)
    assert verdict(trial) == Verdict(0.0, None, named)
    assert not endpoint_failed(named)


def test_every_api_error_harbor_defines_is_decided_one_way_or_the_other() -> None:
    """A new `ApiError` subclass in Harbor fails this until someone decides whose fault
    it is. Left undecided, the suffix rule would mask a policy failure."""
    from harbor.agents.installed import base

    defined = {
        name
        for name, cls in inspect.getmembers(base, inspect.isclass)
        if issubclass(cls, base.ApiError)
    }
    assert defined, "Harbor's ApiError classes moved; this list is read from them"
    undecided = defined - ENDPOINT_FAILED - POLICY_OWN
    assert not undecided, f"decide whether these are the endpoint's or the policy's: {undecided}"
    assert not (ENDPOINT_FAILED & POLICY_OWN)
    assert ENDPOINT_FAILED <= defined


def test_a_numeric_reward_is_the_verdict(tmp_path: Path) -> None:
    assert verdict(write_result(tmp_path / "one", reward=1.0)) == Verdict(1.0, None, None)
    assert verdict(write_result(tmp_path / "zero", reward=0)) == Verdict(0.0, None, None)
    assert verdict(write_result(tmp_path / "half", rewards={"reward": 0.5, "style": 2})) == (
        Verdict(0.5, None, None)
    )


def test_a_graded_trial_that_ended_badly_keeps_its_ending(tmp_path: Path) -> None:
    trial = write_result(tmp_path / "t", reward=0.0, exception="NonZeroAgentExitCodeError")
    assert verdict(trial) == Verdict(0.0, None, "NonZeroAgentExitCodeError")


def test_no_number_is_a_grading_error(tmp_path: Path) -> None:
    assert verdict(write_result(tmp_path / "none")) == Verdict(None, GRADING_ERROR, None)
    rubric = write_result(tmp_path / "rubric", rewards={"style": 1.0})
    assert verdict(rubric) == Verdict(None, GRADING_ERROR, None)
    text = write_result(tmp_path / "text", rewards={"reward": "1"})
    assert verdict(text) == Verdict(None, GRADING_ERROR, None)
    flag = write_result(tmp_path / "flag", rewards={"reward": True})
    assert verdict(flag) == Verdict(None, GRADING_ERROR, None)
    lost = write_result(tmp_path / "lost", exception="RewardFileNotFoundError")
    assert verdict(lost) == Verdict(None, GRADING_ERROR, "RewardFileNotFoundError")


def test_verdicts_follow_plan_order_and_scores_leave_the_masked_out(tmp_path: Path) -> None:
    won = write_result(tmp_path / "won", reward=1.0)
    lost = write_result(tmp_path / "lost", reward=0.0)
    cut = write_result(tmp_path / "cut", reward=1.0, exception="AgentTimeoutError")
    gone = tmp_path / "gone"
    rollouts = Rollouts(job="j", trials=[won, gone, lost, cut], plan=[])
    assert [found.mask for found in verdicts(rollouts)] == [None, ENV_ERROR, None, TIMEOUT]
    assert scores(rollouts) == [1.0, 0.0]
    assert scores(Rollouts(job="j", trials=[gone, cut], plan=[])) == []


# ----------------------------------------------------------------- the served endings


def test_no_records_is_an_env_error_whatever_the_verifier_wrote(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    assert verdict(graded, []) == Verdict(None, ENV_ERROR, None)
    assert verdict(tmp_path / "gone", []) == Verdict(None, ENV_ERROR, None)
    assert verdict(graded) == Verdict(1.0, None, None), "no records given is the block-2 rule"
    assert verdict(graded, [made()]) == Verdict(1.0, None, None)


def test_a_first_turn_that_never_fit_is_a_context_overflow(tmp_path: Path) -> None:
    gave_up = write_result(tmp_path / "t", reward=0.0, exception="UnknownApiError")
    assert verdict(gave_up, [made(error=CONTEXT)]) == Verdict(
        None, CONTEXT_OVERFLOW, "UnknownApiError"
    )
    twice = [made(error=CONTEXT, seq=1), made(error=CONTEXT, seq=2)]
    assert verdict(gave_up, twice).mask == CONTEXT_OVERFLOW


def test_a_rollout_cut_by_its_budget_is_measured_zero_not_masked(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    records = [made(seq=1), made(seq=2, error=BUDGET)]
    assert verdict(graded, records) == Verdict(0.0, None, None)
    timed = write_result(tmp_path / "u", reward=1.0, exception="AgentTimeoutError")
    assert verdict(timed, records) == Verdict(0.0, None, "AgentTimeoutError")


def test_a_context_the_policy_filled_before_its_harness_quit_is_zero(tmp_path: Path) -> None:
    records = [made(seq=1), made(seq=2, error=CONTEXT)]
    quit_ = write_result(tmp_path / "quit", reward=1.0, exception="UnknownApiError")
    assert verdict(quit_, records) == Verdict(0.0, None, "UnknownApiError")
    crashed = write_result(tmp_path / "crashed", exception="NonZeroAgentExitCodeError")
    assert verdict(crashed, records) == Verdict(0.0, None, "NonZeroAgentExitCodeError")
    # Compacted and finished: the verifier's number stands.
    finished = write_result(tmp_path / "finished", reward=1.0)
    assert verdict(finished, [*records, made(seq=3)]) == Verdict(1.0, None, None)
    assert verdict(finished, records) == Verdict(1.0, None, None)
    # The clock's cut and the grader's failure are not the harness giving up.
    timed = write_result(tmp_path / "timed", reward=1.0, exception="AgentTimeoutError")
    assert verdict(timed, records) == Verdict(None, TIMEOUT, "AgentTimeoutError")
    ungraded = write_result(tmp_path / "ungraded", exception="RewardFileNotFoundError")
    assert verdict(ungraded, records) == Verdict(None, GRADING_ERROR, "RewardFileNotFoundError")


def test_a_harness_that_quit_on_a_failed_call_is_an_api_error(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=0.0)
    failed = [made(seq=1), made(seq=2, error="sampler: ConnectError")]
    assert verdict(graded, failed) == Verdict(None, API_ERROR, None)
    recovered = [*failed, made(seq=3)]
    assert verdict(graded, recovered) == Verdict(0.0, None, None)
    only = [made(seq=1, error="sampler: ConnectError")]
    assert verdict(graded, only) == Verdict(None, API_ERROR, None)
    # Budget and context endings are decided above.
    assert verdict(graded, [made(seq=1), made(seq=2, error=BUDGET)]).mask is None


def test_an_image_in_a_prompt_masks_the_rollout_multimodal(tmp_path: Path) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    seen = [made(seq=1), made((1, IMAGE_TOKEN, IMAGE_TOKEN, 2), seq=2)]
    assert verdict(graded, seen) == Verdict(None, MULTIMODAL, None)
    assert verdict(graded, [made(seq=1, error=BUDGET), *seen]) == Verdict(0.0, None, None)


def test_a_harness_that_asked_more_turns_than_were_recorded_is_an_env_error(
    tmp_path: Path,
) -> None:
    graded = write_result(tmp_path / "t", reward=1.0)
    records = [made(seq=1), made(seq=2, error=CONTEXT)]
    assert verdict(graded, records, asked=3) == Verdict(None, ENV_ERROR, None)
    assert verdict(graded, records, asked=2) == Verdict(1.0, None, None)
    assert verdict(graded, records, asked=1) == Verdict(1.0, None, None)
    assert verdict(graded, records) == Verdict(1.0, None, None)
    assert served_verdict([], None, asked=0) == Verdict(None, ENV_ERROR, None)


def test_a_last_call_the_harness_logged_as_failed_and_the_proxy_never_saw_is_masked(
    tmp_path: Path,
) -> None:
    graded = write_result(tmp_path / "t", reward=0.0)
    clean = [made(seq=1), made(seq=2)]
    assert verdict(graded, clean, failed=True) == Verdict(None, API_ERROR, None)
    assert verdict(graded, clean) == Verdict(0.0, None, None)
    refused = [made(seq=1), made(seq=2, error=CONTEXT)]
    assert verdict(graded, refused, failed=True) == Verdict(0.0, None, None), "the proxy saw it"


def test_the_served_endings_come_before_the_block_2_rules(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    assert verdict(gone, [made(error=BUDGET)]) == Verdict(0.0, None, None)
    assert verdict(gone, [made()]) == Verdict(None, ENV_ERROR, None)
    timed = write_result(tmp_path / "timed", reward=1.0, exception="AgentTimeoutError")
    assert verdict(timed, [made()]) == Verdict(None, TIMEOUT, "AgentTimeoutError")
    api = write_result(tmp_path / "api", reward=0.0, exception="ApiOverloadedError")
    assert verdict(api, [made()]) == Verdict(None, API_ERROR, "ApiOverloadedError")


def test_verdicts_read_the_records_and_turn_counts_attached_to_the_rollouts(
    tmp_path: Path,
) -> None:
    won = write_result(tmp_path / "won__0000001", reward=1.0)
    cut = write_result(tmp_path / "cut__0000002", reward=1.0)
    quiet = write_result(tmp_path / "quiet__0000003", reward=1.0)
    short = write_result(tmp_path / "short__0000004", reward=1.0)
    records = {
        won.name: [made()],
        cut.name: [made(seq=1), made(seq=2, error=BUDGET)],
        quiet.name: [],
        short.name: [made()],
    }
    rollouts = Rollouts(job="j", trials=[won, cut, quiet, short], plan=[], records=records)
    assert [v.mask for v in verdicts(rollouts)] == [None, None, ENV_ERROR, None]
    assert scores(rollouts) == [1.0, 0.0, 1.0]
    asked = Rollouts(job="j", trials=[won, short], plan=[], records=records, asked={short.name: 2})
    assert [v.mask for v in verdicts(asked)] == [None, ENV_ERROR]
    failed = Rollouts(job="j", trials=[won, short], plan=[], records=records, failed={won.name})
    assert [v.mask for v in verdicts(failed)] == [API_ERROR, None]
    plain = Rollouts(job="j", trials=[won, cut, quiet], plan=[])
    assert scores(plain) == [1.0, 1.0, 1.0], "without records, the files alone decide"
