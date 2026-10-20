from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from harness_proxy.vision import image_key
from harness_proxy.wire import Picture, ids_of, pictures_of

# ----------------------------------------------------------------------- the digest


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None) if value is not None else None


def _call(call: Any) -> list[str]:
    """Return a tool call's name and id, without its arguments, which a harness may rewrite."""
    function = _field(call, "function")
    named = function if function is not None else call
    return [str(_field(named, "name") or ""), str(_field(call, "id") or "")]


def normalize(message: Any) -> dict[str, Any]:
    """Return the fields that identify a message, for digesting."""
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
    """Return a key for every prefix of `digests`, each derived from the one before."""
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
    """A sampled reply, stored for the bridge: its prompt, completion, stop, render and pictures."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    ended_with_stop: bool
    rendered: tuple[int, ...]
    pictures: tuple[Picture, ...] = ()


class Index:
    """The sampled replies of one trial, each keyed by the digest chain that leads to it."""

    def __init__(self) -> None:
        self._replies: dict[str, Reply] = {}
        self._ambiguous: set[str] = set()
        #: Replies by (history, identity), where the identity is a call id or, for a reply
        #: that makes no call, its digest.
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
        """Find the longest prefix of `digests` that ends in a sampled reply, or None."""
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
    #: Fresh render of the messages, which the head of a later prompt must equal to
    #: extend this one.
    rendered: tuple[int, ...]
    #: Pictures behind the image tokens in `ids`: the reply's first, then the tail's.
    pictures: tuple[Picture, ...] = ()


def bridge(
    messages: Sequence[Any],
    renderer: Any,
    index: Index,
    digests: Sequence[str] | None = None,
) -> Bridged | None:
    """Build the prompt for `messages` from the tokens of a sampled reply they quote, or None."""
    digests = digests_of(messages) if digests is None else digests
    found = index.find(digests)
    if found is None:
        return None
    at, reply = found
    if at < 1 or not tail_extends(messages[at + 1 :]):
        return None
    stops = int_stops(renderer)
    if not stops:
        return None
    head = ids_of(renderer.build_generation_prompt(list(messages[:at])))
    if head != reply.rendered:
        return None
    fresh = renderer.build_generation_prompt(list(messages))
    full = ids_of(fresh)
    if full[: len(head)] != head:
        return None
    end = next((i for i in range(len(head), len(full)) if full[i] in stops), None)
    if end is None or not _ends_reply(full, end, messages, at, renderer, stops):
        return None
    closing = () if reply.ended_with_stop else (full[end],)
    spliced = (*reply.prompt_ids, *reply.completion_ids, *closing, *full[end + 1 :])
    # An image in the head is the reply's picture; one in the tail moves with the tail.
    shift = len(spliced) - len(full)
    tail = tuple(
        Picture(one.offset + shift, one.length, one.format, one.data)
        for one in pictures_of(fresh)
        if one.offset > end
    )
    return Bridged(spliced, full, (*reply.pictures, *tail))


def _ends_reply(
    full: Sequence[int], end: int, messages: Sequence[Any], at: int, renderer: Any, stops: set[int]
) -> bool:
    """Return whether the stop at `end` of the fresh render closes the reply at `at`."""
    calls = [_bare(call) for call in _field(messages[at], "tool_calls") or []]
    stub: dict[str, Any] = {"role": "assistant", "content": "."}
    if calls:
        stub["tool_calls"] = calls
    held = renderer.build_generation_prompt([*messages[:at], stub, *messages[at + 1 :]])
    alt, tail = ids_of(held), tuple(full[end + 1 :])
    cut = len(alt) - len(tail)
    return cut >= 1 and alt[cut:] == tail and alt[cut - 1] in stops


def _bare(call: Any) -> Any:
    """Return the tool call with `{}` for arguments, which hold text the model wrote."""
    function = _field(call, "function")
    if isinstance(call, dict):
        return {**call, "function": {**(function or {}), "arguments": "{}"}}
    return call.model_copy(update={"function": function.model_copy(update={"arguments": "{}"})})


def tail_extends(tail: Sequence[Any]) -> bool:
    """Return whether the messages after a quoted reply can be bridged (prime-rl's tail rule)."""
    roles = [str(_field(message, "role") or "") for message in tail]
    if not roles:
        return False
    body = roles[:-1] if roles[-1] == "user" else roles
    return all(role == "tool" for role in body)


def int_stops(renderer: Any) -> set[int]:
    """Return the renderer's stop sequences that are token ids, skipping string stops."""
    return {int(s) for s in (renderer.get_stop_sequences() or []) if isinstance(s, int)}
