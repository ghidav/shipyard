"""Per-harness profiles: the API each harness speaks and what else it needs to reach the model.

A harness without a profile gets the generic one."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """How shipyard connects one harness to the model and reads its log."""

    #: The API the harness speaks: `openai` or `anthropic`.
    dialect: str = "openai"
    #: Provider id for a harness that the dialect's default id would route wrongly. It forms
    #: the model name `<provider>/<model>` and the trial env vars `<PROVIDER>_BASE_URL` and
    #: `<PROVIDER>_API_KEY`.
    provider: str | None = None
    #: Extra Harbor agent kwargs.
    kwargs: dict = field(default_factory=dict)
    #: Trial env that turns off the harness's own context compaction. It is set only when
    #: `[rollout] fill_context` is on, which instead gives a call that would overflow the
    #: context whatever room is left.
    env: dict = field(default_factory=dict)
    #: Whether the harness appends `/v1` to its base URL itself, so the proxy address is
    #: given without it.
    strip_v1: bool = False
    #: Counts the model calls in the harness's log, for `[rollout] check_turns`.
    turns: Callable[[str], int] | None = None
    #: Reports from the harness's log whether its last model call failed. Read on every
    #: trial, regardless of `check_turns`: a last call that failed where the proxy never saw
    #: it (in transit, or refused before recording) must not be graded.
    failed_last: Callable[[str], bool] | None = None


#: Model name Claude Code gives an assistant message it wrote locally, without a model call.
SYNTHETIC_MODEL = "<synthetic>"


def distinct_requests(text: str) -> int:
    """Count the model calls in Claude Code's stream-json log, one per distinct request id.

    Claude Code writes one assistant line per content block, so lines are not counted.
    Synthetic messages are skipped."""
    seen: set[str] = set()
    for line in text.splitlines():
        if '"assistant"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        if message.get("model") == SYNTHETIC_MODEL:
            continue
        key = event.get("request_id") or message.get("id")
        if key:
            seen.add(str(key))
    return len(seen)


def last_assistant_failed(text: str) -> bool:
    """Return whether the last assistant `message_end` in pi's JSON log stopped on `error`.

    pi writes that stop reason when it gives up on a call, and still exits 0."""
    stop = None
    for line in text.splitlines():
        if '"message_end"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            stop = message.get("stopReason")
    return stop == "error"


def lines_with(*markers: str) -> Callable[[str], int]:
    """Return a counter of the lines that contain every marker, ignoring spaces.

    Ignoring spaces keeps a harness that starts pretty-printing its log from dropping the
    count to zero, which would silently disable the turn check."""

    def count(text: str) -> int:
        return sum(
            1
            for line in text.splitlines()
            if all(marker in line.replace(" ", "") for marker in markers)
        )

    return count


PROFILES: dict[str, Profile] = {
    "pi": Profile(kwargs={"model_api": "openai-completions"}, failed_last=last_assistant_failed),
    "claude-code": Profile(
        dialect="anthropic",
        strip_v1=True,
        env={"DISABLE_COMPACT": "1", "DISABLE_AUTO_COMPACT": "1"},
        turns=distinct_requests,
    ),
    "opencode": Profile(
        provider="shipyard",
        kwargs={
            "opencode_config": {
                # Under `openai` it posts every call to `/responses`, which the proxy does
                # not serve; this package, which it bundles, posts to `/chat/completions`.
                "provider": {
                    "shipyard": {
                        "npm": "@ai-sdk/openai-compatible",
                        "env": ["SHIPYARD_API_KEY"],
                        "options": {"baseURL": "${SHIPYARD_BASE_URL}"},
                    }
                },
                # Its title call would be sampled from the policy and trained as a chain
                # of its own, though no turn reads it.
                "agent": {"title": {"disable": True}},
            }
        },
        turns=lines_with("step-start"),
    ),
    # Its model calls are litellm's, made in the Harbor process from its `api_base` option
    # and the host env, so the trial env does not reach them. A per-trial kwarg comes in a
    # later release.
    "terminus-2": Profile(),
}


def bare_name(harness: str) -> str:
    """Return the harness name without its version: `pi@0.85.1` becomes `pi`. An import
    path, which contains a colon (`acp:x.y:Z@1`), is returned whole."""
    return harness if ":" in harness else harness.partition("@")[0]


def profile_for(harness: str) -> Profile:
    """Return the profile for `harness` by its bare name, or the generic profile."""
    return PROFILES.get(bare_name(harness).strip(), Profile())


def slug_of(profile: Profile) -> str:
    """Return the provider id a harness is configured for: the profile's `provider` if set,
    else `anthropic` for the Anthropic dialect and `openai` otherwise."""
    if profile.provider:
        return profile.provider
    return "anthropic" if profile.dialect == "anthropic" else "openai"
