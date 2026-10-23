from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shipyard.groups import Connection, Where


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
    #: Trial env that turns off the harness's own context compaction. It is set only when the
    #: model's `fill_context` is on, which instead gives a call that would overflow the
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
                # Its title call would be sampled from the agent and trained as a chain
                # of its own, though no turn reads it.
                "agent": {"title": {"disable": True}},
            }
        },
    ),
    # Its model calls are litellm's, made in the Harbor process, so the trial env does not
    # reach them: it takes the proxy's address and token as options instead.
    "terminus-2": Profile(in_process=litellm_options),
}


class Harness(BaseModel):
    """The agent program a trial runs, at an optional pinned version, with the environment and
    keyword arguments Harbor starts it with: what it dials to reach the model, whether it can run
    with no network, and whether its log says its last model call failed. A recipe declares the
    harnesses it runs as keys of its own; `run.toml` gives one as a name, `"pi@0.85.1"`, or as a
    table with `name`, `env` and `kwargs`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The Harbor agent to install: a name with an optional version, such as `pi@0.85.1`, or an
    #: import path.
    name: str
    env: dict[str, str] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _named(cls, value: Any) -> Any:
        """Read a harness given as its name alone."""
        return {"name": value} if isinstance(value, str) else value

    @field_validator("name")
    @classmethod
    def _unblank(cls, value: str) -> str:
        """Refuse a blank name: Harbor reads an unnamed agent as its oracle, which runs the
        task's reference solution instead of the agent."""
        if not value.strip():
            raise ValueError("a harness needs its name, such as `pi@0.85.1`")
        return value.strip()

    @property
    def bare(self) -> str:
        """Its name without the version: the Harbor agent, such as `pi` of `pi@0.85.1`."""
        return bare_name(self.name)

    @property
    def version(self) -> str | None:
        """The version `name` pins, such as `0.85.1` of `pi@0.85.1`; None for none."""
        bare = self.bare
        return (self.name[len(bare) + 1 :] or None) if bare != self.name else None

    @property
    def profile(self) -> Profile:
        """How it connects to the model, by its bare name."""
        return profile_for(self.name)

    @property
    def runs_without_network(self) -> bool:
        """Whether it can run in a task with no network: its model calls leave from this
        machine, not from the trial's container."""
        return self.profile.in_process is not None

    def dial(self, where: Where, model: str, *, fill_context: bool) -> Connection:
        """Return what a trial of it dials to reach `model` at the proxy `where` is: the model
        under the provider id it routes by, and the env and kwargs that point it there. With
        `fill_context`, it is also told to leave its context uncompacted."""
        profile = self.profile
        address = where.address if profile.in_process is None else where.local
        if profile.strip_v1:
            address = address.removesuffix("/v1")
        slug = slug_of(profile)
        env = {
            **(dict(profile.env) if fill_context else {}),
            f"{slug.upper()}_BASE_URL": address,
            f"{slug.upper()}_API_KEY": where.token,
        }
        kwargs = dict(profile.kwargs)
        if profile.in_process is not None:
            kwargs |= profile.in_process(address, where.token)
        return Connection(model=f"{slug}/{model}", env=env, kwargs=kwargs, host=where.host)

    def last_call_failed(self, trial: Path) -> bool:
        """Return whether the log it kept in the trial directory `trial` says its last model
        call failed; False for a harness whose log is not read."""
        reader = self.profile.failed_last
        if reader is None:
            return False
        try:
            text = (trial / "agent" / f"{self.bare}.txt").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return False
        return reader(text)


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
