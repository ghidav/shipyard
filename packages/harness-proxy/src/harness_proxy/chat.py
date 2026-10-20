"""One harness call, whichever wire it came in on.

A dialect reads a request into a `Call`; `complete` renders it, samples through the recorder and
parses the reply into a `Reply`; the dialect writes that back. A request the proxy will not serve
raises `Refused`, which the dialect answers with a 400 in its wire's shape. The checks that both
wires share live here too."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import tinker

from harness_proxy.cookbook import Message, ToolCall, ToolSpec, get_text_content

#: What a 400 for a prompt that does not fit says, on a Tinker refusal or the recorder's. A
#: harness compacts its history on exactly this, so no other refusal may say it.
_OVERFLOW = (
    re.compile(r"prompt length plus max_tokens exceeds", re.IGNORECASE),
    re.compile(r"\d+\s*prompt tokens\s*\+\s*\d+\s*max_tokens\s*>\s*\d+", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
)


class Refused(Exception):
    """A request the proxy will not serve, answered with a 400 in the wire's own shape."""


def too_long(message: str) -> tinker.BadRequestError:
    """Return the error Tinker raises for a prompt that does not fit the context.

    `complete` answers it, Tinker's own or the recorder's, with the "prompt is too long" 400."""
    request = httpx.Request("POST", "http://harness-proxy.invalid")
    return tinker.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


@dataclass
class Call:
    """A request as a dialect reads it: the conversation in the cookbook's message format, and
    what the harness asked of the sample."""

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


async def complete(call: Call, renderer: Any, sampler: Any, default_max_tokens: int) -> Reply:
    """Render the call, sample one reply, and parse it.

    The renderer writes the tools into the system prompt; without tools, the system text is a
    system message. Raise `Refused` for a request that cannot be sampled as asked."""
    if call.tools:
        try:
            prefix = renderer.create_conversation_prefix_with_tools(
                call.tools, system_prompt=call.system or ""
            )
        except (NotImplementedError, AttributeError):
            raise Refused(
                "the configured renderer does not support tool calling; omit 'tools'"
            ) from None
        messages = [*prefix, *call.messages]
    elif call.system is not None:
        messages = [Message(role="system", content=call.system), *call.messages]
    else:
        messages = list(call.messages)
    prompt = renderer.build_generation_prompt(messages)
    stop: list[str] | list[int] = renderer.get_stop_sequences()
    string_stops = bool(stop) and isinstance(stop[0], str)
    if call.stop:
        if stop and not string_stops:
            # SamplingParams.stop holds strings or ids, not both.
            raise Refused(
                "client stop sequences are not supported with this model's renderer, which "
                "uses token-id stop conditions; omit stop/stop_sequences"
            )
        stop = [*[s for s in stop if isinstance(s, str)], *call.stop]
    max_tokens = call.max_tokens or default_max_tokens
    asked = {
        "temperature": call.temperature,
        "top_p": call.top_p,
        "top_k": call.top_k,
        "seed": call.seed,
    }
    try:
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            stop=stop,
            **{key: value for key, value in asked.items() if value is not None},
        )
    except (ValueError, TypeError) as bad:
        raise Refused(f"invalid sampling parameters: {bad}") from None
    try:
        response = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
    except tinker.BadRequestError as failed:
        if any(pattern.search(str(failed)) for pattern in _OVERFLOW):
            raise Refused(
                f"prompt is too long: {prompt.length} prompt tokens plus {max_tokens} "
                "max_tokens exceed the model's context window"
            ) from failed
        raise
    sequence = response.sequences[0]
    tokens = list(sequence.tokens)
    message, termination = renderer.parse_response(tokens)
    text = get_text_content(message)
    calls = [] if call.no_tools else list(message.get("tool_calls") or [])
    stop_reason = str(sequence.stop_reason or "stop")
    client_stop, matched = False, None
    if call.stop and not calls and stop_reason == "stop" and not termination.is_clean:
        # A stop the sampler excluded parses as unclean too, so only a stop seen in the text,
        # or the only kind of stop sampled with, says the harness's own stop fired.
        hits = [(text.find(s), s) for s in call.stop if s in text]
        if hits:
            at, matched = min(hits)
            text, client_stop = text[:at], True
        elif not string_stops:
            client_stop = True
            matched = call.stop[0] if len(call.stop) == 1 else None
    return Reply(
        text=text,
        thinking=thinking_of(message),
        tool_calls=calls,
        prompt_tokens=prompt.length,
        completion_tokens=len(tokens),
        stop_reason=stop_reason,
        client_stop=client_stop,
        matched_stop=matched,
    )


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
    """Return a message's content from its parts, in order.

    Text alone is one string. Otherwise the parts are kept, with adjacent text joined and empty
    text dropped."""
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
