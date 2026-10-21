from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness_proxy.cookbook import Message, ToolCall, ToolSpec


class Refused(Exception):
    """A request the proxy will not serve, answered with a 400 in the wire's own shape."""


@dataclass(frozen=True)
class Caller:
    """Who made a call: the trial, lane and renderer its address names, the wire it came on,
    and the SDK's retry headers."""

    trial: str = ""
    #: The lane whose model answers the call; "" for a proxy's only lane.
    lane: str = ""
    #: The renderer that renders the call; "" for the lane's first.
    renderer: str = ""
    #: `anthropic` for a call on `/messages`, otherwise `openai`. Determines the form of the
    #: tool call ids the proxy mints.
    wire: str = "openai"
    retry: int = 0
    idempotency_key: str | None = None
    request_id: str | None = None


@dataclass
class Call:
    """A request as its dialect reads it: the conversation, and what the harness asked for."""

    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    #: `tool_choice` "none": the prompt gets no tools and the reply's calls are dropped, because
    #: a model can write its learned call syntax without a catalog.
    no_tools: bool = False
    model: str | None = None
    #: None for the server's default.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    #: The harness's own stop strings.
    stop: list[str] | None = None
    stream: bool = False
    #: OpenAI's `stream_options.include_usage`.
    include_usage: bool = False


@dataclass
class Reply:
    """A sampled reply, parsed, before a dialect writes it."""

    text: str
    thinking: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    #: Tinker's stop reason: "stop" or "length".
    stop_reason: str
    #: Whether sampling ended on one of the call's own stop strings, and which one if known.
    client_stop: bool = False
    matched_stop: str | None = None


def thinking_of(message: Any) -> str:
    """Return the joined thinking parts of a parsed message, or `""` if it has none."""
    content = message.get("content") if isinstance(message, dict) else None
    parts = content if isinstance(content, list) else []
    return "".join(
        str(p.get("thinking") or "")
        for p in parts
        if isinstance(p, dict) and p.get("type") == "thinking"
    )


# ------------------------------------------------------------------- reading a request


def text_part(text: Any) -> dict[str, str]:
    return {"type": "text", "text": str(text)}


def content_of(parts: Sequence[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Return a message's content from its parts, in order: text alone as one string."""
    if all(part.get("type") == "text" for part in parts):
        return "".join(str(part.get("text") or "") for part in parts)
    merged: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") == "text":
            if not part.get("text"):
                continue
            if merged and merged[-1].get("type") == "text":
                merged[-1] = text_part(merged[-1]["text"] + str(part["text"]))
                continue
        merged.append(part)
    return merged


def reject(body: dict[str, Any], keys: Sequence[str]) -> None:
    """Refuse a request that sets a field the proxy would silently ignore."""
    for key in keys:
        if body.get(key):
            raise Refused(f"{key!r} is not supported by this proxy")


def positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Refused(f"{key!r} must be a positive integer")
    return value


def integer(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refused(f"{key!r} must be an integer")
    return value


def number(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{key!r} must be a number")
    return float(value)


def stop_strings(value: Any, key: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise Refused(f"{key!r} must be a string or a list of strings")


def check_tool_choice(value: Any) -> None:
    """Accept no choice, auto or none. A forced choice has no renderer mechanism."""
    if value is None or value in ("auto", "none"):
        return
    if isinstance(value, dict) and value.get("type") in ("auto", "none"):
        return
    raise Refused("forced 'tool_choice' is not supported by this proxy; omit it or use auto/none")


def forbids_tools(value: Any) -> bool:
    """Return whether the tool choice is OpenAI's `"none"` or Anthropic's `{"type": "none"}`."""
    return value == "none" or (isinstance(value, dict) and value.get("type") == "none")


def tool_spec(name: Any, description: Any, parameters: Any) -> ToolSpec:
    if not isinstance(name, str) or not name:
        raise Refused("each tool requires a non-empty string 'name'")
    if parameters is not None and not isinstance(parameters, dict):
        raise Refused(f"tool {name!r} parameter schema must be an object")
    return ToolSpec(
        name=name,
        description=str(description or ""),
        parameters=parameters or {"type": "object", "properties": {}},
    )


def tool_call(call_id: Any, name: Any, arguments: str) -> ToolCall:
    if not isinstance(name, str) or not name:
        raise Refused("each tool call requires a non-empty string tool name")
    return ToolCall(
        id=str(call_id) if call_id else f"toolu_{uuid.uuid4().hex}",
        function=ToolCall.FunctionBody(name=name, arguments=arguments),
    )
