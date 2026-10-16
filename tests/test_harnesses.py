"""Tests for the per-harness profiles: wire format, extras, and turn counting."""

from __future__ import annotations

import json

from shipyard.harnesses import (
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
    assert generic.kwargs == {} and generic.env == {}
    assert generic.strip_v1 is False and generic.turns is None and generic.provider is None


def test_pi_names_its_model_api_and_terminus_is_generic() -> None:
    assert PROFILES["pi"].kwargs == {"model_api": "openai-completions"}
    assert PROFILES["pi"].dialect == "openai"
    assert PROFILES["terminus-2"] == Profile()


def test_claude_code_speaks_anthropic_and_appends_v1_itself() -> None:
    profile = PROFILES["claude-code"]
    assert profile.dialect == "anthropic" and slug_of(profile) == "anthropic"
    assert profile.strip_v1 is True
    assert profile.env == {"DISABLE_COMPACT": "1", "DISABLE_AUTO_COMPACT": "1"}
    assert profile.turns is distinct_requests


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


def test_opencode_poses_as_its_own_chat_provider_with_no_title_call() -> None:
    """Under the `openai` provider, opencode posts to `/responses`. Under its own provider,
    which uses the chat package opencode bundles, it posts to `/chat/completions` with the
    key from the env that shipyard sets."""
    profile = PROFILES["opencode"]
    assert profile.provider == "shipyard" and slug_of(profile) == "shipyard"
    assert profile.dialect == "openai" and profile.env == {}
    config = profile.kwargs["opencode_config"]
    assert config["provider"] == {
        "shipyard": {
            "npm": "@ai-sdk/openai-compatible",
            "env": ["SHIPYARD_API_KEY"],
            "options": {"baseURL": "${SHIPYARD_BASE_URL}"},
        }
    }
    assert config["agent"] == {"title": {"disable": True}}


def test_a_provider_of_the_profiles_own_wins_over_the_dialects() -> None:
    assert slug_of(Profile(provider="mine")) == "mine"
    assert slug_of(Profile(dialect="anthropic", provider="mine")) == "mine"
    assert slug_of(Profile(provider="")) == "openai"


def test_opencode_turns_are_step_start_lines_whatever_the_spacing() -> None:
    count = PROFILES["opencode"].turns
    assert count is not None
    assert count('{"type": "step-start"}\n{"type":"step-finish"}\n{ "type" : "step-start" }') == 2
    assert lines_with("a", "b")("a b\nab\nb\n") == 2


def _ended(*stops: str) -> str:
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
    assert last_assistant_failed(_ended("toolUse", "error")) is True
    assert last_assistant_failed(_ended("toolUse", "error", "stop")) is False, "pi recovered"
    assert last_assistant_failed(_ended("toolUse", "stop")) is False
    assert last_assistant_failed(_ended("toolUse", "length")) is False
    assert last_assistant_failed("") is False and last_assistant_failed("not json") is False
    assert last_assistant_failed('"message_end"') is False, "a line that is not an object"
