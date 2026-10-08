"""The per-harness profiles: which wire, which extras, and how turns are counted."""

from __future__ import annotations

import json

from shipyard.proxy.profiles import (
    PROFILES,
    Profile,
    distinct_requests,
    last_assistant_failed,
    lines_with,
    profile_for,
    slug_of,
)


def test_a_version_is_stripped_and_an_unknown_harness_gets_the_generic_profile() -> None:
    assert profile_for("pi@0.85.1") is PROFILES["pi"]
    assert profile_for("claude-code") is PROFILES["claude-code"]
    assert profile_for("nobody-wired-this") == Profile()
    assert profile_for("acp:some.module") == Profile()
    assert profile_for("") == Profile()


def test_the_generic_profile_is_the_openai_wire_with_nothing_extra() -> None:
    generic = Profile()
    assert generic.dialect == "openai" and slug_of(generic) == "openai"
    assert generic.kwargs == {} and generic.env == {} and generic.volatile == ()
    assert generic.strip_v1 is False and generic.turns is None


def test_pi_names_its_model_api_and_terminus_is_generic() -> None:
    assert PROFILES["pi"].kwargs == {"model_api": "openai-completions"}
    assert PROFILES["pi"].dialect == "openai"
    assert PROFILES["terminus-2"] == Profile()


def test_claude_code_speaks_anthropic_appends_v1_itself_and_rewrites_its_budget_line() -> None:
    profile = PROFILES["claude-code"]
    assert profile.dialect == "anthropic" and slug_of(profile) == "anthropic"
    assert profile.strip_v1 is True
    assert profile.env == {"DISABLE_COMPACT": "1", "DISABLE_AUTO_COMPACT": "1"}
    assert profile.turns is distinct_requests
    assert len(profile.volatile) == 1
    import re

    cut = re.sub(profile.volatile[0], "", "be good\n\n<total_tokens>9 left</total_tokens>")
    assert cut == "be good"


def _event(kind: str, request_id: str | None, model: str = "m", message_id: str = "msg") -> str:
    event = {"type": kind, "message": {"id": message_id, "model": model}}
    if request_id is not None:
        event["request_id"] = request_id
    return json.dumps(event)


def test_claude_code_turns_are_distinct_requests_not_assistant_lines() -> None:
    log = "\n".join(
        [
            _event("assistant", "r1"),
            _event("assistant", "r1"),  # a second content block of the same reply
            _event("assistant", "r2"),
            _event("assistant", "r3", model="<synthetic>"),  # Claude Code wrote this itself
            _event("user", "r4"),
            _event("assistant", None, message_id="msg-5"),  # keyed by message id instead
            "not json at all",
            '{"type": "assistant"}',
        ]
    )
    assert distinct_requests(log) == 3
    assert distinct_requests("") == 0


def test_opencode_turns_are_step_start_lines_whatever_the_spacing() -> None:
    count = PROFILES["opencode"].turns
    assert count is not None
    assert count('{"type": "step-start"}\n{"type":"step-finish"}\n{ "type" : "step-start" }') == 2
    assert lines_with("a", "b")("a b\nab\nb\n") == 2


def _ended(*stops: str) -> str:
    """pi's JSON log: a message_start and message_end per assistant message, tool lines between."""
    lines = []
    for stop in stops:
        message = {"role": "assistant", "content": [], "stopReason": stop}
        lines.append(json.dumps({"type": "message_start", "message": message}))
        lines.append(json.dumps({"type": "message_end", "message": message}))
        lines.append(json.dumps({"type": "message_end", "message": {"role": "toolResult"}}))
    return "\n".join([*lines, '{"type":"agent_settled"}'])


def test_pi_says_its_last_call_failed_only_when_its_last_assistant_message_did() -> None:
    assert PROFILES["pi"].failed_last is last_assistant_failed
    assert last_assistant_failed(_ended("toolUse", "error")) is True
    assert last_assistant_failed(_ended("toolUse", "error", "stop")) is False, "pi recovered"
    assert last_assistant_failed(_ended("toolUse", "stop")) is False
    assert last_assistant_failed(_ended("toolUse", "length")) is False
    assert last_assistant_failed("") is False and last_assistant_failed("not json") is False
