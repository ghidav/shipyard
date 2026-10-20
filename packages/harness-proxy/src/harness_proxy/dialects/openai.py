from __future__ import annotations

import json
import time
import uuid
from typing import Any

from aiohttp import web

from harness_proxy import vision
from harness_proxy.chat import (
    Call,
    Refused,
    Reply,
    check_tool_choice,
    content_of,
    forbids_tools,
    integer,
    number,
    positive_int,
    reject,
    stop_strings,
    text_part,
    tool_call,
    tool_spec,
)
from harness_proxy.cookbook import Message, ToolCall

PATH = "/chat/completions"
UNSUPPORTED = (
    "functions",
    "function_call",
    "response_format",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "modalities",
    "audio",
    "prediction",
    "presence_penalty",
    "frequency_penalty",
)
#: Message fields that may carry reasoning sent back. The first one set is read.
REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
DETAILS = "reasoning_details"
#: The field a reply carries its reasoning in.
REPLY_FIELD = "reasoning"


def read(body: dict[str, Any]) -> Call:
    """Read a request body into a `Call`. Raise `Refused` for one the proxy will not serve."""
    reject(body, UNSUPPORTED)
    if body.get("n") not in (None, 1):
        raise Refused("'n' values other than 1 are not supported by this proxy")
    check_tool_choice(body.get("tool_choice"))
    no_tools = forbids_tools(body.get("tool_choice"))
    tools = [] if no_tools else _tools(body.get("tools"))
    options = body.get("stream_options")
    if options is not None and not isinstance(options, dict):
        raise Refused("'stream_options' must be an object")
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise Refused("'messages' must be a non-empty list")
    opening = next((i for i, m in enumerate(raw) if not _is_system(m)), len(raw))
    systems: list[str] = []
    messages: list[Message] = []
    names: dict[str, str] = {}  # tool call id -> tool name, for the results that answer it
    for at, one in enumerate(raw):
        if not isinstance(one, dict):
            raise Refused("each message must be a JSON object")
        role = one.get("role")
        if role == "system" and at < opening:
            parts = _parts(one.get("content"))
            if tools:
                # The renderer writes the tools into the system prompt, which takes this text.
                systems.append(vision.as_text(parts))
            else:
                messages.append(Message(role="system", content=content_of(parts)))
        elif role == "tool":
            messages.append(_result(one, names))
        elif role in ("user", "assistant", "system"):
            messages.append(_message("user" if role == "system" else role, one, names))
        else:
            raise Refused(f"unsupported message role {role!r}: only system/user/assistant/tool")
    limit, key = body.get("max_completion_tokens"), "max_completion_tokens"
    if limit is None:
        limit, key = body.get("max_tokens"), "max_tokens"
    return Call(
        messages=messages,
        system="\n\n".join(systems) if systems else None,
        tools=tools,
        no_tools=no_tools,
        model=body.get("model"),
        max_tokens=positive_int(limit, key),
        temperature=number(body.get("temperature"), "temperature"),
        top_p=number(body.get("top_p"), "top_p"),
        top_k=positive_int(body.get("top_k"), "top_k"),
        seed=integer(body.get("seed"), "seed"),
        stop=stop_strings(body.get("stop"), "stop"),
        stream=bool(body.get("stream")),
        include_usage=bool(options.get("include_usage")) if options else False,
    )


def _is_system(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"


def _message(role: str, one: dict[str, Any], names: dict[str, str]) -> Message:
    parts = _parts(one.get("content") or "")
    reasoning = reasoning_of(one) if role == "assistant" else ""
    if reasoning:
        parts = [{"type": "thinking", "thinking": reasoning}, *parts]
    message = Message(role=role, content=content_of(parts))
    raw_calls = one.get("tool_calls")
    if raw_calls:
        if role != "assistant" or not isinstance(raw_calls, list):
            raise Refused("'tool_calls' is only valid on assistant messages")
        calls: list[ToolCall] = []
        for entry in raw_calls:
            if not isinstance(entry, dict) or not isinstance(entry.get("function"), dict):
                raise Refused("each tool_call must have a 'function' object")
            function = entry["function"]
            call = tool_call(
                entry.get("id"), function.get("name"), str(function.get("arguments") or "{}")
            )
            names[str(call.id)] = call.function.name
            calls.append(call)
        message["tool_calls"] = calls
    return message


def _result(one: dict[str, Any], names: dict[str, str]) -> Message:
    call_id = one.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise Refused("tool messages require a string 'tool_call_id'")
    result = Message(role="tool", content=content_of(_parts(one.get("content") or "")))
    result["tool_call_id"] = call_id
    name = one.get("name") or names.get(call_id)
    if name:
        result["name"] = str(name)
    return result


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [text_part(content)]
    if not isinstance(content, list):
        raise Refused(f"unsupported content type {type(content).__name__!r}")
    parts: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else type(block)
        if kind == "text":
            parts.append(text_part(block.get("text", "")))
        elif kind in ("image_url", "file"):
            parts.extend(vision.openai_media(block))
        else:
            raise Refused(
                f"unsupported content part type {kind!r}: only text, image_url and file parts"
            )
    return parts


def reasoning_of(message: dict[str, Any]) -> str:
    """Return a message's reasoning from the first field that carries it."""
    for name in REASONING_FIELDS:
        value = message.get(name)
        if isinstance(value, str) and value:
            return value
    details = message.get(DETAILS)
    if isinstance(details, list):
        return "".join(
            str(entry.get("text") or entry.get("summary") or "")
            for entry in details
            if isinstance(entry, dict)
            and entry.get("type") in ("reasoning.text", "reasoning.summary")
        )
    return ""


def _tools(raw: Any) -> list[Any]:
    raw = raw or []
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise Refused("'tools' must be a list of tool objects")
    specs = []
    for entry in raw:
        function = entry.get("function")
        if entry.get("type") != "function" or not isinstance(function, dict):
            raise Refused("each OpenAI tool must be {'type': 'function', 'function': {...}}")
        specs.append(
            tool_spec(function.get("name"), function.get("description"), function.get("parameters"))
        )
    return specs


# --------------------------------------------------------------------------- writing


def answer(reply: Reply, model: str, call: Call) -> dict[str, Any]:
    """Return the reply as a chat completion."""
    calls = _calls(reply)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reply.text if reply.text or not calls else None,
    }
    if calls:
        message["tool_calls"] = calls
    if reply.thinking:
        message.update(_reasoning(reply.thinking))
    return {
        **_head(model, "chat.completion"),
        "choices": [{"index": 0, "message": message, "finish_reason": _finish(reply)}],
        "usage": _usage(reply),
    }


def events(reply: Reply, model: str, call: Call) -> list[bytes]:
    """Return the reply as SSE chunks: reasoning, text, tool calls, finish, and usage if asked."""
    head = _head(model, "chat.completion.chunk")
    deltas: list[tuple[dict[str, Any], str | None]] = []
    if reply.thinking:
        deltas.append(({"role": "assistant", **_reasoning(reply.thinking)}, None))
    deltas += [({"role": "assistant", "content": ""}, None), ({"content": reply.text}, None)]
    calls = _calls(reply)
    if calls:
        deltas.append(({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}, None))
    deltas.append(({}, _finish(reply)))
    chunks = []
    for delta, finish in deltas:
        chunk = {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if call.include_usage:
            chunk["usage"] = None
        chunks.append(_data(chunk))
    if call.include_usage:
        chunks.append(_data({**head, "choices": [], "usage": _usage(reply)}))
    chunks.append(_data("[DONE]"))
    return chunks


def error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"type": "invalid_request_error", "message": message, "code": None}},
        status=status,
    )


def _head(model: str, kind: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": kind,
        "created": int(time.time()),
        "model": model,
    }


def _calls(reply: Reply) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id or f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in reply.tool_calls
    ]


def _reasoning(thinking: str) -> dict[str, Any]:
    entry = {"type": "reasoning.text", "text": thinking, "format": "unknown", "index": 0}
    return {REPLY_FIELD: thinking, DETAILS: [entry]}


def _finish(reply: Reply) -> str:
    if reply.tool_calls:
        return "tool_calls"
    return "length" if reply.stop_reason == "length" else "stop"


def _usage(reply: Reply) -> dict[str, int]:
    return {
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "total_tokens": reply.prompt_tokens + reply.completion_tokens,
    }


def _data(payload: Any) -> bytes:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n".encode()
