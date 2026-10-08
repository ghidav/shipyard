"""Message digests and the bridge. A prompt whose history quotes a reply this proxy
sampled is built from that reply's tokens. Only the tail after the reply is rendered
anew, so a tool loop trains as one sequence."""

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


def _call(call: Any) -> list[str]:
    """A tool call as its name and id. The id identifies the reply that made the call, so a
    call the harness rewrote before echoing it (a default filled in, a type coerced)
    still matches that reply."""
    function = _field(call, "function")
    named = function if function is not None else call
    return [str(_field(named, "name") or ""), str(_field(call, "id") or "")]


def normalize(message: Any) -> dict[str, Any]:
    """The fields that identify a message: role, text with whitespace removed (a harness
    may drop a block of it), thinking (a harness that strips it has changed the message),
    images by their pixels, tool calls by name and id (`with_call_ids` gives every call
    an id), and a tool result's tool name. Call arguments and a result's call id are
    left out."""
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
        "text": "".join("".join(text).split()),
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
    """The key of every prefix of the digests. The key of `digests[:i]` extends the key of
    `digests[:i-1]`, so one pass keys every prefix."""
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
    """A sampled reply: its call's prompt and completion ids, whether the completion ended
    on a stop token, and the fresh render of the call's messages. The render equals the
    prompt unless the prompt was bridged."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    ended_with_stop: bool
    rendered: tuple[int, ...]


class Index:
    """One trial's sampled replies, each keyed by the chain of digests that leads to it.
    Two different replies to one history are ambiguous when they share a call id (some
    models, such as Kimi, number a call the same way in every sample) or when they make no
    call and say the same thing. A trimmed echo of one may match the other, so the bridge
    uses neither."""

    def __init__(self) -> None:
        self._replies: dict[str, Reply] = {}
        self._ambiguous: set[str] = set()
        #: Per history and identity (a call id, or the digest of a reply making no call),
        #: the replies that carry it.
        self._carried: dict[tuple[str, str], list[tuple[str, Reply]]] = {}

    def __len__(self) -> int:
        return len(self._replies)

    def add(
        self,
        prompt_digests: Sequence[str],
        reply_digest: str,
        reply: Reply,
        call_ids: Sequence[str] = (),
    ) -> None:
        key, history = chain([*prompt_digests, reply_digest]), chain(prompt_digests)
        identities = [f"call:{one}" for one in call_ids] or [f"reply:{reply_digest}"]
        for identity in identities:
            carried = self._carried.setdefault((history, identity), [])
            for other_key, other in carried:
                if other.completion_ids != reply.completion_ids:
                    self._ambiguous.update((key, other_key))
            carried.append((key, reply))
        self._replies[key] = reply

    def find(self, digests: Sequence[str]) -> tuple[int, Reply] | None:
        """The longest prefix of the digests that ends in a sampled reply, as that reply's
        position among the messages and the reply. None when the reply is ambiguous."""
        keys = chains(digests)
        for end in range(len(keys), 0, -1):
            if keys[end - 1] in self._ambiguous:
                return None
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
    stop, the tail's ids), or None when the caller must render afresh."""
    digests = digests_of(messages) if digests is None else digests
    found = index.find(digests)
    if found is None:
        return None
    at, reply = found
    if at < 1 or not tail_extends(messages[at + 1 :]):
        return None
    # The template drops earlier thinking at a new user turn. Render such a prompt afresh.
    if getattr(renderer, "strip_thinking_from_history", False) and any(
        str(_field(message, "role") or "") == "user" for message in messages[at + 1 :]
    ):
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
    """prime-rl's tail rule: the tail is tool results, optionally followed by one user
    message. An assistant message in the tail would be re-tokenized. An empty tail means
    the call is a resample, so it returns False."""
    roles = [str(_field(message, "role") or "") for message in tail]
    if not roles:
        return False
    body = roles[:-1] if roles[-1] == "user" else roles
    return all(role == "tool" for role in body)


def ids_in(prompt: Any) -> tuple[int, ...] | None:
    """A rendered prompt as ids, or None when a chunk is not text. An image chunk has no
    ids to compare or splice."""
    ids: list[int] = []
    for chunk in getattr(prompt, "chunks", None) or []:
        tokens = getattr(chunk, "tokens", None)
        if tokens is None:
            return None
        ids.extend(int(token) for token in tokens)
    return tuple(ids)
