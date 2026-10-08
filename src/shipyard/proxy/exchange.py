"""One request through the app, shared by the middleware, the renderer wrapper and the
recorder, each of which sees a slice of it in the task the handler runs in."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from aiohttp import web

#: The SDK's retry counter, on both the OpenAI and the Anthropic clients.
RETRY_HEADER = "x-stainless-retry-count"


@dataclass
class Exchange:
    """What the three callers share: the trial off the address, the SDK's headers, and what
    the wrapper cut or bridged."""

    trial: str = ""
    #: `anthropic` for a request on `/messages`, `openai` otherwise: the form of the ids.
    wire: str = "openai"
    retry: int = 0
    idempotency_key: str | None = None
    request_id: str | None = None
    #: Volatile lines cut from this request's system messages before rendering.
    cut: int = 0
    #: Whether the prompt was built by the bridge, the digests of its messages, and the
    #: fresh render behind a bridged prompt (what a later head must equal to extend it).
    bridged: bool = False
    prompt_digests: tuple[str, ...] = ()
    rendered: tuple[int, ...] | None = None
    #: The record this request produced; None for a refused or replayed one.
    seq: int | None = None
    #: The reply's thinking, handed back on the wire in its own form.
    thinking: str = ""


exchange: ContextVar[Exchange | None] = ContextVar("shipyard_exchange", default=None)


def trial_in(address: str) -> str:
    """The trial named in a `/r/key/value/...` address, read as the cookbook reads it."""
    parts = [part for part in address.strip("/").split("/") if part]
    return str(dict(zip(parts[0::2], parts[1::2], strict=False)).get("trial") or "")


def middleware() -> Any:
    """Open an `Exchange` per request off its headers and its `/r/trial/<name>/...` address,
    and reset it on the way out so nothing leaks into a task the loop reuses."""

    @web.middleware
    async def opened(request: Any, handler: Any) -> Any:
        found = Exchange(
            trial=trial_in(str(request.match_info.get("address") or "")),
            wire="anthropic" if request.path.endswith("/messages") else "openai",
            retry=_count(request.headers.get(RETRY_HEADER)),
            idempotency_key=request.headers.get("Idempotency-Key") or None,
            request_id=request.headers.get("x-request-id") or None,
        )
        token = exchange.set(found)
        try:
            return await handler(request)
        finally:
            exchange.reset(token)

    return opened


def _count(header: str | None) -> int:
    """The retry header as a count; absent or not a number is 0."""
    try:
        return int(header) if header else 0
    except ValueError:
        return 0
