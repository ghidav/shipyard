"""Per-harness profiles: which wire a harness speaks and what it needs beyond Harbor's
model connection. A harness nobody has watched gets the generic profile, not a guess."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """One harness: its dialect, the extra agent kwargs and trial env it takes, whether it
    appends `/v1` itself, the system lines it rewrites per request, its turn counter, and
    the reader of its log that says whether its last model call failed."""

    dialect: str = "openai"
    kwargs: dict = field(default_factory=dict)
    #: Extra env for the trial. Rollout lays it only when `[rollout] fill_context` is on:
    #: it switches the harness's compaction off, which only a filled context can afford.
    env: dict = field(default_factory=dict)
    strip_v1: bool = False
    volatile: tuple[str, ...] = ()
    turns: Callable[[str], int] | None = None
    #: Read on every trial, whatever `check_turns` says: a last call that failed where the
    #: proxy never saw it (on the way, or refused before recording) must not be graded.
    failed_last: Callable[[str], bool] | None = None


#: The model Claude Code names on an assistant message it wrote itself, without a model.
SYNTHETIC_MODEL = "<synthetic>"


def distinct_requests(text: str) -> int:
    """Claude Code's stream-json log: one API reply per distinct request id. Not one per
    assistant line, which it writes once per content block, and never a synthetic one."""
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
    """pi's JSON log: whether its last assistant `message_end` stopped on `error`. pi writes
    that when it gives up on a call, and exits 0 all the same."""
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
    """Count the lines carrying every marker, whitespace ignored: a count that fell to zero
    when a harness started pretty-printing would disable a guard rather than fail it."""

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
    "opencode": Profile(turns=lines_with("step-start")),
    # Its model calls are litellm's in the Harbor process, from its `api_base` option and
    # the host env, so the trial env does not reach them; a per-trial kwarg is a later release.
    "terminus-2": Profile(),
}


def bare_name(harness: str) -> str:
    """`pi@0.85.1` as `pi`; an import path with a colon (`acp:x.y:Z@1`) is left whole."""
    return harness if ":" in harness else harness.partition("@")[0]


def profile_for(harness: str) -> Profile:
    """The profile of `harness` by its bare name; the generic one for a harness nobody has
    profiled."""
    return PROFILES.get(bare_name(harness).strip(), Profile())


def slug_of(profile: Profile) -> str:
    """The provider a harness is posed to: `anthropic` for that wire, `openai` otherwise."""
    return "anthropic" if profile.dialect == "anthropic" else "openai"
