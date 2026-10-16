"""Renderer wrapper the cookbook app uses, linking rendering to the bridge and the recorder.

Before rendering, it refuses images for a model that takes none and tries the bridge.
After parsing a reply, it sets the reply's digest on its record and keeps the reply's
thinking for the wire."""

from __future__ import annotations

import hashlib
from typing import Any

from shipyard.proxy import cookbook
from shipyard.proxy.bridge import bridge, chain, digest, digests_of, int_stops
from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.thinking import thinking_of
from shipyard.proxy.vision import has_images, takes_images
from shipyard.proxy.wire import model_input


class Rendering:
    """Renderer wrapper that overrides `build_generation_prompt` and `parse_response`.

    All other attributes are delegated to the inner renderer."""

    def __init__(self, inner: Any, recorder: Any) -> None:
        self._inner = inner
        self._recorder = recorder

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        found: Exchange | None = exchange.get()
        messages = list(messages)
        if has_images(messages) and not takes_images(self._inner):
            # Answer with a 400 so the harness sees the request was the problem.
            raise cookbook.refused(
                "this request carries an image and the model this endpoint serves takes "
                "none: its renderer was built without an image processor"
            )
        digests = digests_of(messages)
        if found is not None:
            found.prompt_digests = digests
            index = self._recorder.index_for(found.trial)
            built = bridge(messages, self._inner, index, digests)
            if built is not None:
                found.bridged, found.rendered = True, built.rendered
                return model_input(built.ids, built.pictures)
        return self._inner.build_generation_prompt(messages, **kwargs)

    def parse_response(self, tokens: Any) -> Any:
        found: Exchange | None = exchange.get()
        message, termination = self._inner.parse_response(tokens)
        message = with_call_ids(message, tokens, found)
        if found is not None:
            found.thinking = thinking_of(message)
            if found.seq is not None:
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
    """Give every tool call that has no id one, in the wire's format.

    This runs before the reply is digested. The cookbook keeps an id that is set and the
    harness echoes it, so the bridge recognises the reply by its call ids. Each id is derived
    from the trial, the request's message digests and the reply's tokens, so a replayed retry
    gets the same ids as the first answer."""
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
