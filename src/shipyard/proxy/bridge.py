"""Message digests and the narrow bridge: a prompt whose history quotes a reply this
proxy sampled is built from that reply's tokens, with only the tail after it rendered
anew, so a tool loop trains as one sequence and not as its re-rendered square."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shipyard.proxy.vision import image_key

# ----------------------------------------------------------------------- the digest


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None) if value is not None else None


def _arguments(raw: Any) -> Any:
    """Tool arguments as a value when they parse, so formatting alone cannot fork; the
    proxy's `{"_raw_arguments": s}` wrapper is unwrapped to the string sampled."""
    if not isinstance(raw, str):
        return raw
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict) and set(parsed) == {"_raw_arguments"}:
        return parsed["_raw_arguments"]
    return parsed


def _call(call: Any) -> list[Any]:
    function = _field(call, "function")
    named = function if function is not None else call
    return [str(_field(named, "name") or ""), _arguments(_field(named, "arguments"))]


def normalize(message: Any) -> dict[str, Any]:
    """What identifies a message: role, text, thinking (a harness that strips it has
    changed the message), images by their pixels, tool calls by name and value, a tool
    result's tool name. Never a tool-call id: the proxy mints those after parsing."""
    content = _field(message, "content")
    text: list[str] = []
    thinking: list[str] = []
    images: list[str] = []
    if isinstance(content, str):
        text.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text.append(str(part.get("text") or ""))
            elif part.get("type") == "thinking":
                thinking.append(str(part.get("thinking") or ""))
            elif part.get("type") == "image":
                images.append(image_key(part))
    return {
        "role": str(_field(message, "role") or ""),
        "text": "".join(text),
        "thinking": "".join(thinking),
        "images": images,
        "calls": [_call(call) for call in (_field(message, "tool_calls") or [])],
        "name": str(_field(message, "name") or ""),
    }


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(message: Any) -> str:
    return hashlib.sha256(canon(normalize(message)).encode("utf-8")).hexdigest()[:16]


def digests_of(messages: Sequence[Any]) -> tuple[str, ...]:
    return tuple(digest(message) for message in messages)


def chains(digests: Sequence[str]) -> list[str]:
    """The chain of every prefix, rolled: the key of `digests[:i]` extends the key of
    `digests[:i-1]`, so every prefix of a history is keyed in one pass."""
    out: list[str] = []
    last = ""
    for one in digests:
        last = hashlib.sha256(f"{last}/{one}".encode()).hexdigest()[:16]
        out.append(last)
    return out


def chain(digests: Sequence[str]) -> str:
    return chains(digests)[-1] if digests else ""


# ------------------------------------------------------------------------ the index


@dataclass(frozen=True)
class Reply:
    """A sampled reply as the bridge needs it: its call's prompt and completion ids,
    whether the completion closed on a stop token, and the fresh render of the call's
    messages, which is its prompt unless that prompt was bridged too."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    ended_with_stop: bool
    rendered: tuple[int, ...]


class Index:
    """One trial's sampled replies, each under the chain of digests that leads to it."""

    def __init__(self) -> None:
        self._replies: dict[str, Reply] = {}

    def __len__(self) -> int:
        return len(self._replies)

    def add(self, prompt_digests: Sequence[str], reply_digest: str, reply: Reply) -> None:
        self._replies[chain([*prompt_digests, reply_digest])] = reply

    def find(self, digests: Sequence[str]) -> tuple[int, Reply] | None:
        """The longest prefix of the digests that ends in a sampled reply: where that
        reply sits among the messages, and the reply."""
        keys = chains(digests)
        for end in range(len(keys), 0, -1):
            reply = self._replies.get(keys[end - 1])
            if reply is not None:
                return end - 1, reply
        return None


# ----------------------------------------------------------------------- the bridge


@dataclass(frozen=True)
class Bridged:
    ids: tuple[int, ...]
    #: The fresh render of the messages: what a later head must equal to extend this.
    rendered: tuple[int, ...]


def bridge(
    messages: Sequence[Any],
    renderer: Any,
    index: Index,
    digests: Sequence[str] | None = None,
) -> Bridged | None:
    """The prompt built from a quoted reply's tokens (the stored prompt and completion, its
    stop, the tail's ids), or None to render afresh: each `return None` below is one of
    the five reasons, in order."""
    digests = digests_of(messages) if digests is None else digests
    found = index.find(digests)
    if found is None:
        return None
    at, reply = found
    if at < 1 or not tail_extends(messages[at + 1 :]):
        return None
    stops = {int(s) for s in (renderer.get_stop_sequences() or []) if isinstance(s, int)}
    if not stops:
        return None
    head = ids_in(renderer.build_generation_prompt(list(messages[:at])))
    if head is None or head != reply.rendered:
        return None
    full = ids_in(renderer.build_generation_prompt(list(messages)))
    if full is None or full[: len(head)] != head:
        return None
    end = next((i for i in range(len(head), len(full)) if full[i] in stops), None)
    if end is None:
        return None
    closing = () if reply.ended_with_stop else (full[end],)
    return Bridged((*reply.prompt_ids, *reply.completion_ids, *closing, *full[end + 1 :]), full)


def tail_extends(tail: Sequence[Any]) -> bool:
    """prime-rl's tail rule: tool results, optionally closed by one user message, and
    nothing else. An assistant message there would be re-tokenized, and nothing there
    is a resample, not an extension."""
    roles = [str(_field(message, "role") or "") for message in tail]
    if not roles:
        return False
    body = roles[:-1] if roles[-1] == "user" else roles
    return all(role == "tool" for role in body)


def ids_in(prompt: Any) -> tuple[int, ...] | None:
    """A rendered prompt as ids, or None when a chunk is not text: an image chunk has
    no ids to compare with or to splice onto."""
    ids: list[int] = []
    for chunk in getattr(prompt, "chunks", None) or []:
        tokens = getattr(chunk, "tokens", None)
        if tokens is None:
            return None
        ids.extend(int(token) for token in tokens)
    return tuple(ids)
