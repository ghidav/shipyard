"""Per-request state shared by the middleware, the renderer wrapper and the recorder.

The state is held in a context variable, so each of them reads and writes it in the task
the handler runs in."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from aiohttp import web

#: Retry count header sent by both the OpenAI and the Anthropic SDKs.
RETRY_HEADER = "x-stainless-retry-count"


@dataclass
class Exchange:
    """State of one request: the trial, the SDK headers, and what the renderer wrapper bridged."""

    trial: str = ""
    #: `anthropic` for a request on `/messages`, otherwise `openai`. Determines the form
    #: of generated tool call ids.
    wire: str = "openai"
    retry: int = 0
    idempotency_key: str | None = None
    request_id: str | None = None
    #: The model the harness asked for, kept on the record. The proxy serves its own.
    requested_model: str | None = None
    #: Whether the bridge built the prompt, the digests of the prompt's messages, and for a
    #: bridged prompt the fresh render that a later prompt must match to extend it.
    bridged: bool = False
    prompt_digests: tuple[str, ...] = ()
    rendered: tuple[int, ...] | None = None
    #: Sequence number of the record this request produced; None if the request was
    #: refused, failed or replayed.
    seq: int | None = None


exchange: ContextVar[Exchange | None] = ContextVar("harness_proxy_exchange", default=None)


def trial_in(address: str) -> str:
    """Return the trial named in a `/r/key/value/...` address: the value after `trial`."""
    parts = [part for part in address.strip("/").split("/") if part]
    return str(dict(zip(parts[0::2], parts[1::2], strict=False)).get("trial") or "")


def middleware() -> Any:
    """Return middleware that sets an `Exchange` for each request.

    The exchange is built from the request's headers and its `/r/trial/<name>/...` address,
    and reset afterwards so it does not leak into a task the event loop reuses."""

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
    try:
        return int(header) if header else 0
    except ValueError:
        return 0
