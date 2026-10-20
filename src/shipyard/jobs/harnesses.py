from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """How a harness connects to the model."""

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
    #: Reports from the harness's log whether its last model call failed. A last call that
    #: failed where the proxy never saw it (in transit, or refused before recording) must not
    #: be graded.
    failed_last: Callable[[str], bool] | None = None
    #: For a harness whose model calls Harbor makes in its own process, not in the trial's
    #: container: its Harbor agent kwargs from the proxy's address on this machine and the
    #: proxy's token, since the trial env does not reach those calls.
    in_process: Callable[[str, str], dict] | None = None


def litellm_options(address: str, token: str) -> dict:
    """Return terminus-2's options for a proxy at `address`: its litellm base URL and token."""
    return {"api_base": address, "llm_call_kwargs": {"api_key": token}}


def last_assistant_failed(text: str) -> bool:
    """Return whether the last assistant `message_end` in pi's JSON log stopped on `error`."""
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


PROFILES: dict[str, Profile] = {
    "pi": Profile(kwargs={"model_api": "openai-completions"}, failed_last=last_assistant_failed),
    "claude-code": Profile(
        dialect="anthropic",
        strip_v1=True,
        env={"DISABLE_COMPACT": "1", "DISABLE_AUTO_COMPACT": "1"},
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
    ),
    # Its model calls are litellm's, made in the Harbor process, so the trial env does not
    # reach them: it takes the proxy's address and token as options instead.
    "terminus-2": Profile(in_process=litellm_options),
}


@dataclass(frozen=True)
class Harness:
    """The agent program a trial runs."""

    name: str
    #: The Harbor agent to install: a name with an optional version, or an import path.
    agent: str
    profile: Profile = field(default_factory=Profile)


def harness(named: str) -> Harness:
    """Return the harness for a name as `[agent] harness` gives it, such as `pi@0.85.1`."""
    return Harness(bare_name(named).strip(), named, profile_for(named))


def bare_name(harness: str) -> str:
    """Return the name without its version: `pi@0.85.1` is `pi`, an import path stays whole."""
    return harness if ":" in harness else harness.partition("@")[0]


def profile_for(harness: str) -> Profile:
    """Return the profile for `harness` by its bare name, or the generic profile."""
    return PROFILES.get(bare_name(harness).strip(), Profile())


def slug_of(profile: Profile) -> str:
    """Return the provider id: the profile's `provider`, else `anthropic` or `openai` by dialect."""
    if profile.provider:
        return profile.provider
    return "anthropic" if profile.dialect == "anthropic" else "openai"
