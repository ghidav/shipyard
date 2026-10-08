"""Per-harness profiles: which wire a harness speaks and what it needs beyond Harbor's
model connection. A harness without a profile gets the generic one."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """One harness: its dialect, the provider it is posed to, the extra agent kwargs and
    trial env it takes, whether it appends `/v1` itself, the system lines it rewrites per
    request, its turn counter, and a log reader that reports whether its last model call
    failed."""

    dialect: str = "openai"
    #: A provider id for a harness the dialect's id would route wrongly. It forms the model
    #: name `<provider>/<model>` and the trial env `<PROVIDER>_BASE_URL` and
    #: `<PROVIDER>_API_KEY`.
    provider: str | None = None
    kwargs: dict = field(default_factory=dict)
    #: Extra env for the trial. It turns off the harness's compaction, so rollout sets it
    #: only when `[rollout] fill_context` is on, which gives an overflowing call what is
    #: left of the context.
    env: dict = field(default_factory=dict)
    strip_v1: bool = False
    volatile: tuple[str, ...] = ()
    turns: Callable[[str], int] | None = None
    #: Read on every trial, regardless of `check_turns`. A last call that failed where the
    #: proxy never saw it (in transit, or refused before recording) must not be graded.
    failed_last: Callable[[str], bool] | None = None


#: The model name Claude Code gives an assistant message it wrote locally, with no model call.
SYNTHETIC_MODEL = "<synthetic>"


def distinct_requests(text: str) -> int:
    """Count the API replies in Claude Code's stream-json log, one per distinct request id.
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
    """Whether the last assistant `message_end` in pi's JSON log stopped on `error`. pi
    writes that when it gives up on a call, and it still exits 0."""
    stop = None
    for line in text.splitlines():
        if '"message_end"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if event.get("type") == "message_end" and isinstance(message, dict):
            if message.get("role") == "assistant":
                stop = message.get("stopReason")
    return stop == "error"


def lines_with(*markers: str) -> Callable[[str], int]:
    """Count the lines that contain every marker, ignoring whitespace. A harness that
    starts pretty-printing would otherwise drop the count to zero and silently disable the
    guard."""

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
        volatile=(r"\n*<total_tokens>[^<]*</total_tokens>",),
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
    """Strip the version: `pi@0.85.1` becomes `pi`. An import path with a colon
    (`acp:x.y:Z@1`) is returned whole."""
    return harness if ":" in harness else harness.partition("@")[0]


def profile_for(harness: str) -> Profile:
    """The profile of `harness` by its bare name, or the generic profile if there is none."""
    return PROFILES.get(bare_name(harness).strip(), Profile())


def slug_of(profile: Profile) -> str:
    """The provider a harness is posed to: the profile's `provider` if set, else
    `anthropic` for the Anthropic dialect and `openai` otherwise."""
    if profile.provider:
        return profile.provider
    return "anthropic" if profile.dialect == "anthropic" else "openai"
