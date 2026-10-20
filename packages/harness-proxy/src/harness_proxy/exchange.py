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
    #: The renderer the trial's address names, which renders the call; "" for the proxy's
    #: first.
    renderer: str = ""
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


def keys_in(address: str) -> dict[str, str]:
    """Return the keys of a `/r/key/value/...` address, such as its `trial` and `renderer`."""
    parts = [part for part in address.strip("/").split("/") if part]
    return dict(zip(parts[0::2], parts[1::2], strict=False))


def middleware() -> Any:
    """Return middleware that sets an `Exchange` for each request, and resets it afterwards."""

    @web.middleware
    async def opened(request: Any, handler: Any) -> Any:
        keys = keys_in(str(request.match_info.get("address") or ""))
        found = Exchange(
            trial=keys.get("trial") or "",
            renderer=keys.get("renderer") or "",
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
