"""The renderer wrapper the app uses. Before rendering it cuts volatile lines from system
messages, refuses images for a model that takes none, and tries the bridge. After parsing
it digests the reply onto its record and keeps the reply's thinking for the wire."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import tinker

from shipyard.proxy import cookbook
from shipyard.proxy.bridge import bridge, chain, digest, digests_of
from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.thinking import thinking_of
from shipyard.proxy.vision import has_images, takes_images


class Rendering:
    """Wraps a renderer. It overrides `build_generation_prompt` and `parse_response`.
    Every other attribute comes from the inner renderer."""

    def __init__(self, inner: Any, volatile: Sequence[str] = (), recorder: Any = None) -> None:
        self._inner = inner
        self._volatile = tuple(re.compile(pattern) for pattern in volatile)
        self._recorder = recorder

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        found: Exchange | None = exchange.get()
        messages = list(messages)
        if self._volatile:
            messages, cut = drop_volatile(messages, self._volatile)
            if found is not None:
                found.cut = cut
        if has_images(messages) and not takes_images(self._inner):
            # Answer with a 400 so the harness sees the request was the problem.
            raise cookbook.refused(
                "this request carries an image and the model this endpoint serves takes "
                "none: its renderer was built without an image processor"
            )
        digests = digests_of(messages)
        if found is not None:
            found.prompt_digests = digests
            if self._recorder is not None:
                index = self._recorder.index_for(found.trial)
                built = bridge(messages, self._inner, index, digests)
                if built is not None:
                    found.bridged, found.rendered = True, built.rendered
                    return tinker.ModelInput.from_ints(list(built.ids))
        return self._inner.build_generation_prompt(messages, **kwargs)

    def parse_response(self, tokens: Any) -> Any:
        found: Exchange | None = exchange.get()
        message, termination = self._inner.parse_response(tokens)
        message = with_call_ids(message, tokens, found)
        if found is not None:
            found.thinking = thinking_of(message)
            if self._recorder is not None and found.seq is not None:
                ended = bool(tokens) and int(tokens[-1]) in int_stops(self._inner)
                self._recorder.reply(
                    found.trial,
                    found.seq,
                    digest(message),
                    ended_with_stop=ended,
                    rendered=found.rendered,
                    call_ids=tuple(str(_call_id(c)) for c in message.get("tool_calls") or []),
                )
        return message, termination

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def with_call_ids(message: Any, tokens: Any, found: Exchange | None) -> Any:
    """Give every tool call that has no id one, in the wire's own form, before the reply
    is digested. The cookbook keeps an id that is set and the harness echoes it, so the
    bridge recognises the reply by its ids. The id is derived from the trial, the
    request's digests and the reply's tokens, so a replayed retry gets the ids its first
    answer got."""
    calls = list(message.get("tool_calls") or [])
    if all(_call_id(call) for call in calls):
        return message
    prefix = "toolu_" if found is not None and found.wire == "anthropic" else "call_"
    seed = "|".join(
        [
            found.trial if found is not None else "",
            chain(found.prompt_digests) if found is not None else "",
            ",".join(str(int(token)) for token in tokens),
        ]
    )
    named = []
    for at, call in enumerate(calls):
        if _call_id(call):
            named.append(call)
            continue
        minted = prefix + hashlib.sha256(f"{seed}|{at}".encode()).hexdigest()[:32]
        named.append(
            {**call, "id": minted}
            if isinstance(call, dict)
            else call.model_copy(update={"id": minted})
        )
    return {**message, "tool_calls": named}


def _call_id(call: Any) -> str | None:
    return call.get("id") if isinstance(call, dict) else getattr(call, "id", None)


def int_stops(renderer: Any) -> set[int]:
    """The renderer's stop tokens that are token ids. String stops are skipped."""
    return {int(s) for s in (renderer.get_stop_sequences() or []) if isinstance(s, int)}


def drop_volatile(
    messages: Sequence[Any], patterns: Sequence[re.Pattern[str]]
) -> tuple[list[Any], int]:
    """The messages with every pattern match cut from their system text, and the number of
    matches cut. The input is copied, not modified. A per-request line makes every prompt
    unique and defeats the cache."""
    out: list[Any] = []
    dropped = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(message, dict) or message.get("role") != "system":
            out.append(message)
        elif isinstance(content, str):
            text, count = _cut(content, patterns)
            dropped += count
            out.append({**message, "content": text} if count else message)
        elif isinstance(content, list):
            parts, count = [], 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text, found = _cut(str(part.get("text", "")), patterns)
                    count += found
                    parts.append({**part, "text": text} if found else part)
                else:
                    parts.append(part)
            dropped += count
            out.append({**message, "content": parts} if count else message)
        else:
            out.append(message)
    return out, dropped


def _cut(text: str, patterns: Sequence[re.Pattern[str]]) -> tuple[str, int]:
    total = 0
    for pattern in patterns:
        text, count = pattern.subn("", text)
        total += count
    return text, total
