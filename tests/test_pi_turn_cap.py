"""A pi trial that Harbor's `max_turns` ends is graded: pi logs the call it aborted as an error,
and that call is not one that failed."""

from __future__ import annotations

import json
from pathlib import Path

from shipyard.groups import Trial
from shipyard.jobs.admit import API_ERROR, Verdict, judge
from shipyard.jobs.engine import Job
from shipyard.jobs.harnesses import PI_ABORTED, Harness, last_assistant_failed
from shipyard.records import Records
from tests.fakes import made, write_result

#: The assistant message pi 0.85.1 logged after the 17th turn of a trial capped at 17, as
#: Harbor's extension aborted the call that would have been its 18th.
CAPPED = {
    "role": "assistant",
    "content": [],
    "api": "openai-completions",
    "provider": "harbor-endpoint",
    "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
    "stopReason": "error",
    "errorMessage": "This operation was aborted",
}


def _log(*last: dict) -> str:
    """Build a pi JSON log of one turn that called a tool, then the assistant messages `last`."""
    turn = {"role": "assistant", "content": [], "stopReason": "toolUse"}
    lines = [json.dumps({"type": "message_end", "message": one}) for one in (turn, *last)]
    return "\n".join([*lines, '{"type":"agent_settled"}'])


def test_a_call_pi_aborted_itself_did_not_fail() -> None:
    assert CAPPED["errorMessage"] == PI_ABORTED
    assert last_assistant_failed(_log(CAPPED)) is False
    lost = {**CAPPED, "errorMessage": "fetch failed"}
    assert last_assistant_failed(_log(lost)) is True, "a call lost on its way still failed"
    unexplained = {key: value for key, value in CAPPED.items() if key != "errorMessage"}
    assert last_assistant_failed(_log(unexplained)) is True, "an error pi gives no reason for"
    assert last_assistant_failed(_log(CAPPED, lost)) is True, "the last message decides"


def test_judge_grades_a_served_pi_trial_cut_at_its_turn_cap(tmp_path: Path) -> None:
    capped = write_result(tmp_path / "capped", reward=0.0)
    answered = write_result(tmp_path / "answered", reward=1.0)
    lost = write_result(tmp_path / "lost", reward=0.0)
    for one, last in (
        (capped, CAPPED),
        (answered, CAPPED),
        (lost, {**CAPPED, "errorMessage": "x"}),
    ):
        (one / "agent").mkdir()
        (one / "agent" / "pi.txt").write_text(_log(last))
    pi = Harness(name="pi@0.85.1", kwargs={"max_turns": 17})
    job = Job("j", trials=[Trial("j", Path("t"), pi, one) for one in (capped, answered, lost)])
    for one in job.trials:
        one.records = Records([made()])
    judge(job)
    assert [one.verdict for one in job.trials] == [
        Verdict(0.0, None, None),
        Verdict(1.0, None, None),
        Verdict(None, API_ERROR, None),
    ]
