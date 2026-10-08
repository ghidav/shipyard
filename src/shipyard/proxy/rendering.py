"""The renderer as the app sees it: volatile lines cut from system messages, images
refused for a model that takes none, the bridge tried, then the prompt rendered; and the
parsed reply digested onto its record, its thinking kept for the wire."""

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
    """One method in, one method out, everything else the inner renderer's."""

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
            # A clean 400 rather than a renderer error dressed as a 500: the harness then
            # knows the request was the problem, and says so in its own log.
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
    """Every tool call the model wrote no id for given one, the wire's own form, before
    the reply is digested: the cookbook keeps an id that is set, and the harness echoes it,
    so the bridge knows the reply by its ids. Drawn from the trial, the request's digests
    and the reply's tokens, so a replayed retry carries the ids its first answer did."""
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
    """The renderer's stop tokens that are token ids; a string stop has no id to find."""
    return {int(s) for s in (renderer.get_stop_sequences() or []) if isinstance(s, int)}


def drop_volatile(
    messages: Sequence[Any], patterns: Sequence[re.Pattern[str]]
) -> tuple[list[Any], int]:
    """The messages with every match cut from their system text, copied not edited, and
    how many were cut: a per-request line makes every prompt unique and caches nothing."""
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
