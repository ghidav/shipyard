"""A model's thinking on the wire, in both directions. Replies carry the thinking to the
harness in the wire's own form. The proxy reads back whatever thinking the harness kept in
a request's history, so the history digests as that reply and the bridge can match it."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from aiohttp import web

from shipyard.proxy import cookbook, keepalive
from shipyard.proxy.exchange import exchange

TAG = "thinking"
_OPEN = "\x00shipyard-thinking:"
_CLOSE = "\x00"
_SENTINEL = re.compile(re.escape(_OPEN) + r"(\d+)" + re.escape(_CLOSE))

#: The OpenAI-wire fields that may carry reasoning. The first one set is used.
OPENAI_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
DETAILS = "reasoning_details"
#: A reply carries `reasoning`, which pi, litellm and OpenRouter's SDK all read, and one
#: `reasoning_details` entry, which OpenRouter's SDK prefers and sends back.
REPLY_FIELD = "reasoning"


def thinking_of(message: Any) -> str:
    """The thinking of a parsed message: its thinking parts, joined, or ``""``."""
    content = message.get("content") if isinstance(message, dict) else None
    parts = content if isinstance(content, list) else []
    return "".join(
        str(p.get("thinking") or "")
        for p in parts
        if isinstance(p, dict) and p.get("type") == "thinking"
    )


def signature(thinking: str) -> str:
    """The Anthropic signature for a thinking block this proxy writes: a digest of the
    text. A client sends it back unchanged and the proxy does not verify it."""
    return base64.b64encode(hashlib.sha256(thinking.encode("utf-8")).digest()).decode("ascii")


def current() -> str:
    """The thinking of the reply to the current request."""
    found = exchange.get()
    return found.thinking if found is not None else ""


# ----------------------------------------------------------------------- installing


def install() -> None:
    """Wrap the cookbook's parsers and its stream writer, once per process. The cookbook's
    parser refuses thinking blocks, so the wrap swaps them for sentinel text before the
    parser runs and restores them afterwards. The keepalive's stream wrap goes in first,
    beneath this one, so a stream it opened takes the chunks made here."""
    keepalive.install()
    if cookbook.wrapped_by("_parse_openai", TAG):
        return
    anthropic, openai = cookbook.private("_parse_anthropic"), cookbook.private("_parse_openai")
    serve_sse = cookbook.private("_serve_sse")

    def parse_anthropic(body: dict[str, Any]) -> Any:
        bank: list[str] = []
        return restore(anthropic(lift_anthropic(body, bank)), bank)

    def parse_openai(body: dict[str, Any]) -> Any:
        bank: list[str] = []
        return restore(openai(lift_openai(body, bank)), bank)

    async def serve(request: Any, chunks: Sequence[bytes]) -> Any:
        thinking = current()
        return await serve_sse(request, streamed(chunks, thinking) if thinking else chunks)

    cookbook.patch(
        _parse_anthropic=cookbook.tagged(parse_anthropic, anthropic, TAG),
        _parse_openai=cookbook.tagged(parse_openai, openai, TAG),
        _serve_sse=cookbook.tagged(serve, serve_sse, TAG),
    )


def middleware() -> Any:
    """Add the reply's thinking to a JSON response. The `_serve_sse` wrap does the same
    for a stream, which the handler writes itself."""

    @web.middleware
    async def thought(request: Any, handler: Any) -> Any:
        response = await handler(request)
        thinking = current()
        is_json = response.status == 200 and response.content_type == "application/json"
        if thinking and is_json and isinstance(getattr(response, "body", None), bytes):
            response = web.json_response(answered(json.loads(response.body), thinking))
        return response

    return thought


# ----------------------------------------------------------------------- the way in


def _placed(text: str, bank: list[str]) -> dict[str, str]:
    bank.append(text)
    return {"type": "text", "text": f"{_OPEN}{len(bank) - 1}{_CLOSE}"}


def lift_anthropic(body: dict[str, Any], bank: list[str]) -> dict[str, Any]:
    """The body with each assistant `thinking` block replaced by a sentinel text block.
    `redacted_thinking` blocks are dropped because they hold nothing a model can read."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    lifted: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or message.get("role") != "assistant":
            lifted.append(message)
            continue
        blocks: list[Any] = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "thinking":
                text = str(block.get("thinking") or "")
                if text:
                    blocks.append(_placed(text, bank))
            elif kind != "redacted_thinking":
                blocks.append(block)
        lifted.append({**message, "content": blocks})
    return {**body, "messages": lifted}


def lift_openai(body: dict[str, Any], bank: list[str]) -> dict[str, Any]:
    """The body with each assistant message's reasoning moved into its content as a
    sentinel text block, ahead of its text."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    lifted: list[Any] = []
    for message in messages:
        text = _reasoning(message) if isinstance(message, dict) else ""
        if not text or message.get("role") != "assistant":
            lifted.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        elif not isinstance(content, list):
            content = []
        lifted.append({**message, "content": [_placed(text, bank), *content]})
    return {**body, "messages": lifted}


def _reasoning(message: dict[str, Any]) -> str:
    """An OpenAI-wire message's reasoning, from the first field that carries it."""
    for name in OPENAI_FIELDS:
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


def restore(parsed: Any, bank: list[str]) -> Any:
    """Turn every sentinel in the parsed chat back into a thinking part, in place."""
    if not bank:
        return parsed
    for message in parsed.messages:
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if isinstance(parts, list) and any(_OPEN in str(_field(p, "text")) for p in parts):
            message["content"] = [
                found
                for part in parts
                for found in (_split(str(part["text"]), bank) if _field(part, "text") else [part])
            ]
    return parsed


def _field(part: Any, name: str) -> Any:
    return part.get(name) if isinstance(part, dict) else None


def _split(text: str, bank: list[str]) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    at = 0
    for match in _SENTINEL.finditer(text):
        if match.start() > at:
            parts.append({"type": "text", "text": text[at : match.start()]})
        parts.append({"type": "thinking", "thinking": bank[int(match.group(1))]})
        at = match.end()
    if at < len(text):
        parts.append({"type": "text", "text": text[at:]})
    return parts


# ---------------------------------------------------------------------- the way out


def answered(payload: dict[str, Any], thinking: str) -> dict[str, Any]:
    """A JSON response body with the reply's thinking added in the wire's form."""
    if not thinking:
        return payload
    if payload.get("type") == "message" and isinstance(payload.get("content"), list):
        block = {"type": "thinking", "thinking": thinking, "signature": signature(thinking)}
        return {**payload, "content": [block, *payload["content"]]}
    choices = payload.get("choices")
    if payload.get("object") == "chat.completion" and isinstance(choices, list) and choices:
        first = dict(choices[0])
        first["message"] = {**(first.get("message") or {}), **_fields(thinking)}
        return {**payload, "choices": [first, *choices[1:]]}
    return payload


def _fields(thinking: str) -> dict[str, Any]:
    return {
        REPLY_FIELD: thinking,
        DETAILS: [{"type": "reasoning.text", "text": thinking, "format": "unknown", "index": 0}],
    }


def streamed(chunks: Sequence[bytes], thinking: str) -> list[bytes]:
    """A stream's events with the reply's thinking ahead of its text: a `thinking` block
    on the Anthropic wire, a reasoning delta on the OpenAI wire. A stream in another
    format passes through unchanged."""
    if not chunks:
        return list(chunks)
    if chunks[0].startswith(b"event:"):
        return _anthropic_stream(chunks, thinking)
    if chunks[0].startswith(b"data:"):
        return _openai_stream(chunks, thinking)
    return list(chunks)


def _event(chunk: bytes) -> tuple[str, dict[str, Any]] | None:
    head, _, rest = chunk.decode("utf-8").partition("\n")
    if not head.startswith("event: ") or not rest.startswith("data: "):
        return None
    return head[len("event: ") :], json.loads(rest[len("data: ") :])


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_stream(chunks: Sequence[bytes], thinking: str) -> list[bytes]:
    out: list[bytes] = []
    for chunk in chunks:
        parsed = _event(chunk)
        if parsed is None:
            return list(chunks)
        event, data = parsed
        if event.startswith("content_block_") and isinstance(data.get("index"), int):
            out.append(_sse(event, {**data, "index": data["index"] + 1}))
        else:
            out.append(chunk)
        if event == "message_start":
            out.extend(_thinking_block(thinking))
    return out


def _thinking_block(thinking: str) -> list[bytes]:
    block = {"type": "thinking", "thinking": "", "signature": ""}
    delta = {"type": "thinking_delta", "thinking": thinking}
    signed = {"type": "signature_delta", "signature": signature(thinking)}
    events = [
        ("content_block_start", {"content_block": block}),
        ("content_block_delta", {"delta": delta}),
        ("content_block_delta", {"delta": signed}),
        ("content_block_stop", {}),
    ]
    return [_sse(event, {"type": event, "index": 0, **data}) for event, data in events]


def _openai_stream(chunks: Sequence[bytes], thinking: str) -> list[bytes]:
    """The reasoning goes before the chunk that opens the text, because OpenRouter's SDK
    does not show `reasoning` that arrives after text has started."""
    head = chunks[0].decode("utf-8")
    try:
        first = json.loads(head[len("data: ") :])
    except ValueError:
        return list(chunks)
    if not isinstance(first, dict) or not first.get("choices"):
        return list(chunks)
    opening = {"role": "assistant", **_fields(thinking)}
    delta = {
        **{key: value for key, value in first.items() if key not in ("choices", "usage")},
        "choices": [{"index": 0, "delta": opening, "finish_reason": None}],
    }
    if "usage" in first:
        delta["usage"] = None
    return [f"data: {json.dumps(delta)}\n\n".encode(), *chunks]
