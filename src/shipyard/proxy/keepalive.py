"""Keepalives on a streamed reply: a sampler working past the grace leaves a tunnel silent
long enough to cut the connection, and the harness's call with it. An SSE comment frame
every so often says the reply is coming; the record is untouched by any of it."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from aiohttp import web

from shipyard.proxy import cookbook

logger = logging.getLogger(__name__)

TAG = "keepalive"
#: Silence tolerated on a stream before the first comment frame, then the gap between them.
KEEPALIVE_GRACE = 30.0
KEEPALIVE_EVERY = 15.0
#: An SSE comment: every client skips it, and the bytes keep the connection open.
FRAME = b": keepalive\n\n"
SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}

#: The running request's keepalive, for the stream writer the cookbook calls.
current: ContextVar[Keepalive | None] = ContextVar("shipyard_keepalive", default=None)


class Keepalive:
    """One streamed request's frames: a task that opens the response once the grace has
    passed and writes a comment per interval, until the handler's chunks arrive."""

    def __init__(self, request: Any, *, grace: float, every: float) -> None:
        self.request = request
        self.grace, self.every = grace, every
        #: The response the frames went out on; None until the first frame.
        self.response: web.StreamResponse | None = None
        self.sent = 0
        self._lock = asyncio.Lock()
        self._stopped = False
        self._task = asyncio.create_task(self._frames())

    async def _frames(self) -> None:
        pause = self.grace
        while True:
            await asyncio.sleep(pause)
            async with self._lock:
                if self._stopped:
                    return
                try:
                    if self.response is None:
                        self.response = web.StreamResponse(headers=SSE_HEADERS)
                        await self.response.prepare(self.request)
                    await self.response.write(FRAME)
                except Exception:  # noqa: BLE001 - a client gone is not the sampler's problem
                    logger.debug("keepalive to %s ended", self.request.path, exc_info=True)
                    self._stopped = True
                    return
                self.sent += 1
            pause = self.every

    async def opened(self) -> web.StreamResponse | None:
        """Stop the frames and hand back the response they went out on, if any did: the
        chunks must go there, since a second response cannot be started on the request."""
        async with self._lock:
            self._stopped = True
        self._task.cancel()
        return self.response

    def cancel(self) -> None:
        """Stop the frames without waiting, for a request ending whatever happened to it."""
        self._stopped = True
        self._task.cancel()


def install() -> None:
    """Wrap the cookbook's stream writer, once per process: the chunks go onto the response
    a keepalive opened when one did, through the cookbook's own writer otherwise. Beneath
    thinking's wrap (thinking installs this first), so the chunks arrive transformed."""
    # `_serve_sse` (tinker_cookbook/capture/proxy/app.py:691) opens a fresh response and
    # writes every chunk at once, and both handlers call it only once sampling is done
    # (app.py:872 and :991): the first byte of a stream waits on the whole sample.
    if cookbook.wrapped_by("_serve_sse", TAG):
        return
    serve_sse = cookbook.private("_serve_sse")

    async def serve(request: Any, chunks: Sequence[bytes]) -> Any:
        kept_alive = current.get()
        response = await kept_alive.opened() if kept_alive is not None else None
        if kept_alive is None or response is None:
            return await serve_sse(request, chunks)
        logger.info("a reply to %s was kept alive by %d frame(s)", request.path, kept_alive.sent)
        try:
            for chunk in chunks:
                await response.write(chunk)
            await response.write_eof()
        except ConnectionResetError:
            logger.debug("client disconnected during SSE response: path=%s", request.path)
        return response

    cookbook.patch(_serve_sse=cookbook.tagged(serve, serve_sse, TAG))


def middleware(grace: float = KEEPALIVE_GRACE, every: float = KEEPALIVE_EVERY) -> Any:
    """A `Keepalive` per request whose body asks to stream, stopped when the handler
    answers. An answer that is not the stream the keepalive opened (an error after the
    grace) is written onto that stream as an error frame, since its status went out."""

    @web.middleware
    async def kept(request: Any, handler: Any) -> Any:
        if request.method != "POST" or not await streaming(request):
            return await handler(request)
        kept_alive = Keepalive(request, grace=grace, every=every)
        token = current.set(kept_alive)
        try:
            try:
                response = await handler(request)
            except Exception as failed:
                opened = await kept_alive.opened()
                if opened is None:
                    raise
                logger.warning("the request to %s failed after its stream opened", request.path)
                await error_frame(opened, request.path, f"{type(failed).__name__}: {failed}")
                return opened
            opened = await kept_alive.opened()
            if opened is None or response is opened:
                return response
            await error_frame(opened, request.path, message_of(response))
            return opened
        finally:
            kept_alive.cancel()
            current.reset(token)

    return kept


async def streaming(request: Any) -> bool:
    """Whether the body asks for a stream; the bytes stay cached for the handler's read."""
    try:
        raw = await request.read()
    except Exception:  # noqa: BLE001 - an unreadable body is the handler's refusal to make
        return False
    if b'"stream"' not in raw:
        return False
    try:
        body = json.loads(raw)
    except ValueError:
        return False
    return isinstance(body, dict) and bool(body.get("stream"))


def message_of(response: Any) -> str:
    """What an error response said, off its JSON body on either wire, else its status."""
    text = getattr(response, "text", None)
    try:
        body = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return f"{response.status} {response.reason or ''}".strip()


async def error_frame(response: web.StreamResponse, path: str, message: str) -> None:
    """End an opened stream with the wire's error event: Anthropic's `event: error` on
    `/messages`, OpenAI's `data: {"error": ...}` then `[DONE]` elsewhere."""
    if path.endswith("/messages"):
        payload = {"type": "error", "error": {"type": "api_error", "message": message}}
        frames = [f"event: error\ndata: {json.dumps(payload)}\n\n"]
    else:
        payload = {"error": {"message": message, "type": "server_error"}}
        frames = [f"data: {json.dumps(payload)}\n\n", "data: [DONE]\n\n"]
    try:
        for frame in frames:
            await response.write(frame.encode())
        await response.write_eof()
    except ConnectionResetError:
        logger.debug("client disconnected before the error frame: path=%s", path)
