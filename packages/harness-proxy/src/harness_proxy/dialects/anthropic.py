"""The Anthropic Messages wire (`/v1/messages`), as Claude Code speaks it.

Reading: thinking blocks the harness sends back become thinking parts, so the history message
digests the same as the sampled reply and the bridge can match it; redacted ones hold nothing a
model can read and are dropped. Images and documents become image parts. A system-role message
inside the list, which Claude Code sends on every call, is a user turn where it stands: folded into
the system prompt, a block appended at the end would land at the front of the prompt, and no call
would extend the one before it. The request's own `system` is the system prompt.

Writing: the reply's thinking is a signed `thinking` block before its text and tool calls."""

from __future__ import annotations

import base64
import hashlib
import json
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
    number,
    positive_int,
    reject,
    stop_strings,
    text_part,
    tool_call,
    tool_spec,
)
from harness_proxy.cookbook import Message, ToolCall

PATH = "/messages"
UNSUPPORTED = ("mcp_servers",)


def read(body: dict[str, Any]) -> Call:
    """Read a request body into a `Call`. Raise `Refused` for one the proxy will not serve."""
    reject(body, UNSUPPORTED)
    # Claude Code sends `thinking` on by default; the model thinks as its renderer has it.
    if body.get("thinking") is not None and not isinstance(body.get("thinking"), dict):
        raise Refused("'thinking' must be an object")
    check_tool_choice(body.get("tool_choice"))
    no_tools = forbids_tools(body.get("tool_choice"))
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise Refused("'messages' must be a non-empty list")
    messages: list[Message] = []
    names: dict[str, str] = {}  # tool_use id -> tool name, for the results that answer it
    for one in raw:
        if not isinstance(one, dict):
            raise Refused("each message must be a JSON object")
        role = "user" if one.get("role") == "system" else one.get("role")
        if role not in ("user", "assistant"):
            raise Refused(
                f"unsupported message role {one.get('role')!r}: only user/assistant/system"
            )
        messages.extend(_message(role, one.get("content"), names))
    system = body.get("system")
    return Call(
        messages=messages,
        system=(_system(system) or None) if system is not None else None,
        tools=[] if no_tools else _tools(body.get("tools")),
        no_tools=no_tools,
        model=body.get("model"),
        max_tokens=positive_int(body.get("max_tokens"), "max_tokens"),
        temperature=number(body.get("temperature"), "temperature"),
        top_p=number(body.get("top_p"), "top_p"),
        top_k=positive_int(body.get("top_k"), "top_k"),
        stop=stop_strings(body.get("stop_sequences"), "stop_sequences"),
        stream=bool(body.get("stream")),
    )


def _message(role: str, content: Any, names: dict[str, str]) -> list[Message]:
    """Return the messages one wire message becomes: a tool message per `tool_result`, then the
    message itself if it has anything left to say."""
    if isinstance(content, str):
        return [Message(role=role, content=content)]
    if not isinstance(content, list):
        raise Refused(f"unsupported content type {type(content).__name__!r}")
    out: list[Message] = []
    parts: list[dict[str, Any]] = []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            raise Refused("each content block must be a JSON object")
        kind = block.get("type")
        if kind == "text":
            parts.append(text_part(block.get("text", "")))
        elif kind in ("image", "document"):
            parts.extend(vision.anthropic_media(block))
        elif kind == "thinking" and role == "assistant":
            if block.get("thinking"):
                parts.append({"type": "thinking", "thinking": str(block["thinking"])})
        elif kind == "redacted_thinking" and role == "assistant":
            continue
        elif kind == "tool_use" and role == "assistant":
            call = tool_call(
                block.get("id"), block.get("name"), json.dumps(block.get("input") or {})
            )
            names[str(call.id)] = call.function.name
            calls.append(call)
        elif kind == "tool_result" and role == "user":
            out.append(_result(block, names))
        else:
            raise Refused(
                f"unsupported content block type {kind!r} for role {role!r}: only text, image, "
                "document, thinking and tool_use (assistant), and tool_result (user) are supported"
            )
    if parts or calls:
        made = Message(role=role, content=content_of(parts))
        if calls:
            made["tool_calls"] = calls
        out.append(made)
    return out


def _result(block: dict[str, Any], names: dict[str, str]) -> Message:
    call_id = block.get("tool_use_id")
    if not isinstance(call_id, str) or not call_id:
        raise Refused("'tool_result' blocks require a string tool_use_id")
    result = Message(role="tool", content=content_of(_parts(block.get("content") or "")))
    result["tool_call_id"] = call_id
    if call_id in names:
        # Some renderers (gpt-oss) render a result by its function's name.
        result["name"] = names[call_id]
    return result


def _parts(content: Any) -> list[dict[str, Any]]:
    """Return the parts of a `system` or `tool_result` content: text, images and documents."""
    if isinstance(content, str):
        return [text_part(content)]
    if not isinstance(content, list):
        raise Refused(f"unsupported content type {type(content).__name__!r}")
    parts: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else type(block)
        if kind == "text":
            parts.append(text_part(block.get("text", "")))
        elif kind in ("image", "document"):
            parts.extend(vision.anthropic_media(block))
        else:
            raise Refused(
                f"unsupported content block type {kind!r}: only text, image and document blocks"
            )
    return parts


def _system(system: Any) -> str:
    return vision.as_text(_parts(system))


def _tools(raw: Any) -> list[Any]:
    raw = raw or []
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise Refused("'tools' must be a list of tool objects")
    return [tool_spec(t.get("name"), t.get("description"), t.get("input_schema")) for t in raw]


# --------------------------------------------------------------------------- writing


def signature(thinking: str) -> str:
    """Return the signature of a thinking block this proxy writes: a base64 SHA-256 digest of
    the text. A client sends it back unchanged, and the proxy does not verify it."""
    return base64.b64encode(hashlib.sha256(thinking.encode("utf-8")).digest()).decode("ascii")


def answer(reply: Reply, model: str, call: Call) -> dict[str, Any]:
    """Return the reply as a message."""
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _blocks(reply),
        "stop_reason": _stop_reason(reply),
        "stop_sequence": reply.matched_stop,
        "usage": {"input_tokens": reply.prompt_tokens, "output_tokens": reply.completion_tokens},
    }


def events(reply: Reply, model: str, call: Call) -> list[bytes]:
    """Return the reply as SSE events, each block in one delta."""
    started = {
        **answer(reply, model, call),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": reply.prompt_tokens, "output_tokens": 0},
    }
    out = [_event("message_start", {"message": started})]
    for index, block in enumerate(_blocks(reply)):
        if block["type"] == "thinking":
            opened = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif block["type"] == "text":
            opened = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        else:
            opened = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        out.append(_event("content_block_start", {"index": index, "content_block": opened}))
        out += [_event("content_block_delta", {"index": index, "delta": d}) for d in deltas]
        out.append(_event("content_block_stop", {"index": index}))
    finished = {"stop_reason": _stop_reason(reply), "stop_sequence": reply.matched_stop}
    usage = {"output_tokens": reply.completion_tokens}
    out.append(_event("message_delta", {"delta": finished, "usage": usage}))
    out.append(_event("message_stop", {}))
    return out


def error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        status=status,
    )


def _blocks(reply: Reply) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if reply.thinking:
        blocks.append(
            {"type": "thinking", "thinking": reply.thinking, "signature": signature(reply.thinking)}
        )
    if reply.text or not reply.tool_calls:
        blocks.append({"type": "text", "text": reply.text})
    for call in reply.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id or f"toolu_{uuid.uuid4().hex}",
                "name": call.function.name,
                "input": _input(call),
            }
        )
    return blocks


def _input(call: ToolCall) -> dict[str, Any]:
    """Return a call's arguments as the object a `tool_use` block carries. Arguments that are not
    an object are passed as they were sampled, never reshaped."""
    try:
        parsed = json.loads(call.function.arguments)
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": call.function.arguments}


def _stop_reason(reply: Reply) -> str:
    if reply.tool_calls:
        return "tool_use"
    if reply.stop_reason == "length":
        return "max_tokens"
    return "stop_sequence" if reply.client_stop else "end_turn"


def _event(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n".encode()
