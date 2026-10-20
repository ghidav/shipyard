"""HTTP server that harnesses call, on a Tinker sampling client.

It serves both wires at `/v1/...` and under a trial's `/r/trial/<name>/v1/...`, plus the records
route and the control route. A call is read by its dialect, completed by `chat.complete` and
written back by the dialect, with keepalives on slow streams. Every route except `/healthz`
requires a token, because sampling spends Tinker credit."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any

from aiohttp import web

from harness_proxy import cookbook, keepalive
from harness_proxy.chat import Refused, complete
from harness_proxy.dialects import anthropic, openai
from harness_proxy.exchange import exchange, middleware
from harness_proxy.keepalive import KEEPALIVE_EVERY, KEEPALIVE_GRACE
from harness_proxy.recorder import Recorder
from harness_proxy.rendering import Rendering
from harness_proxy.wire import (
    CONTROL_PATH,
    HEALTH_PATH,
    RECORDS_PATH,
    Record,
    delta_encoded,
    new_token,
)

logger = logging.getLogger(__name__)

#: Seconds a trial's last records payload is kept for `again=1`. A fetch drains the
#: records, so a lost answer cannot be rebuilt otherwise.
SERVED_KEPT_SECONDS = 1800.0
#: `max_tokens` for a request that sets none. The cookbook's default of 1024 cuts off a
#: coding turn.
DEFAULT_MAX_TOKENS = 8192
#: Largest request body the proxy reads, in bytes. aiohttp's default is 1 MiB.
MAX_BODY_BYTES = 256 * 1024**2

#: Makes a sampling client on a `tinker://` path, or on the base model for None.
ClientFactory = Callable[[str | None], Awaitable[Any]]


@dataclass
class Server:
    """Local server that stands in for a model provider.

    `host` is the hostname a sandbox uses to reach this machine. The token is required
    regardless of `bind`."""

    base_model: str
    model_path: str | None = None
    #: A cookbook renderer name, a renderer object (tests), or None for the model's default.
    renderer: Any = None
    host: str = "host.docker.internal"
    bind: str = "0.0.0.0"
    bind_port: int = 0
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = -1
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_context: int | None = None
    fill_context: bool = False
    #: Seconds a stream may stay silent before SSE keepalive frames start, one every
    #: `keepalive_every` seconds until the reply. None turns them off.
    keepalive_grace: float | None = KEEPALIVE_GRACE
    keepalive_every: float = KEEPALIVE_EVERY
    token: str = field(default_factory=new_token, repr=False)
    control_token: str | None = field(default=None, repr=False)
    #: `user_metadata` of the Tinker session, naming the run and recipe it samples for.
    metadata: dict[str, str] | None = None
    #: Sampling client factory for tests. None uses Tinker's service client.
    client_factory: ClientFactory | None = field(default=None, repr=False)

    _service: Any = field(default=None, repr=False)
    _recorder: Recorder | None = field(default=None, repr=False)
    _rendering: Rendering | None = field(default=None, repr=False)
    _runner: Any = field(default=None, repr=False)
    _port: int | None = field(default=None, repr=False)
    _served_last: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Generate a token if it is empty, as an unset env var arrives here."""
        if not self.token:
            self.token = new_token()

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def serving(self) -> str:
        """The model a reply names: the served weights, or the base model."""
        return self.model_path or self.base_model

    @property
    def recorder(self) -> Recorder:
        if self._recorder is None:
            raise RuntimeError("This endpoint is not started, so nothing is recorded.")
        return self._recorder

    def records_for(self, trial: str) -> list[Record]:
        return self.recorder.records_for(trial)

    async def start(self) -> None:
        """Build the renderer, the sampling client, the app and its routes, then bind the socket.

        The sampling client is on `model_path`, or on the base model when it is None. The context
        length is read from Tinker when not set."""
        renderer = (
            cookbook.renderer_for(self.base_model, self.renderer)
            if self.renderer is None or isinstance(self.renderer, str)
            else self.renderer
        )
        if self.client_factory is None and self._service is None:
            import tinker

            self._service = tinker.ServiceClient(user_metadata=self.metadata)
        client = await self._make(self.model_path)
        if self.max_context is None and self._service is not None:
            self.max_context = await context_of(self._service, self.base_model)
        self._recorder = Recorder(
            client,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_context=self.max_context,
            budget=self.max_context,
            fill_context=self.fill_context,
            served=self.model_path,
        )
        self._rendering = Rendering(renderer, recorder=self._recorder)
        # Control outermost, so only the control token opens the control route. The exchange
        # after auth, so a refused request opens none.
        middlewares = [self._control_middleware(), self._auth_middleware(), middleware()]
        if self.keepalive_grace is not None:
            middlewares.append(keepalive.middleware(self.keepalive_grace, self.keepalive_every))
        # aiohttp answers a body over its 1 MiB default with a 413 before any handler, so the
        # request would not be recorded. The context refusal is the cap.
        app = web.Application(middlewares=middlewares, client_max_size=MAX_BODY_BYTES)
        for dialect in (anthropic, openai):
            serve = self._serving(dialect)
            app.router.add_post(f"/v1{dialect.PATH}", serve)
            app.router.add_post(f"/r/{{address:.+}}/v1{dialect.PATH}", serve)
        app.router.add_get(f"{RECORDS_PATH}/{{trial}}", self._serve_records)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.bind, self.bind_port).start()
        self._runner = runner
        self._port = _bound_port(runner)
        logger.info("serving %s on port %s as %s", self.serving, self._port, self.host)

    async def swap(self, model_path: str | None) -> None:
        """Switch to a sampling client on `model_path`, or on the base model for None.

        The server stays up, so addresses already given to running trials keep answering."""
        client = await self._make(model_path)
        self.model_path = model_path
        if self._recorder is not None:
            self._recorder.client, self._recorder.served = client, model_path

    async def _make(self, model_path: str | None) -> Any:
        if model_path is not None and not model_path.startswith("tinker://"):
            raise ValueError(f"{model_path!r} is not a tinker:// path")
        if self.client_factory is not None:
            return await self.client_factory(model_path)
        if model_path is None:
            return await self._service.create_sampling_client_async(base_model=self.base_model)
        return await self._service.create_sampling_client_async(model_path=model_path)

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()
        self._port = None
        service, self._service = self._service, None
        if service is not None:
            with contextlib.suppress(Exception):
                await service.close("success")

    async def __aenter__(self) -> Server:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _serving(self, dialect: ModuleType) -> Callable[[web.Request], Awaitable[Any]]:
        """Return the handler for `dialect`'s route: read the call, complete it, write the reply.

        A refusal is a 400 in the wire's shape, and costs no sample when it comes before one.
        Any other failure is the server's own and leaves aiohttp's 500."""

        async def serve(request: web.Request) -> Any:
            try:
                # Off the event loop: images and PDF pages decode here, and other trials wait
                # on the loop.
                call = await asyncio.to_thread(dialect.read, await _body(request))
                found = exchange.get()
                if found is not None:
                    found.requested_model = call.model
                reply = await complete(call, self._rendering, self.recorder, int(self.max_tokens))
            except Refused as refused:
                logger.warning("refused a request to %s: %s", request.path, refused)
                return dialect.error(400, str(refused))
            if not call.stream:
                return web.json_response(dialect.answer(reply, self.serving, call))
            return await keepalive.serve_sse(request, dialect.events(reply, self.serving, call))

        return serve

    def _auth_middleware(self) -> Any:
        """Return middleware that requires the harness token, as `x-api-key` (Anthropic's SDK)
        or `Authorization: Bearer` (OpenAI's), on every route the control middleware passes on."""

        @web.middleware
        async def auth(request: Any, handler: Any) -> Any:
            presented = (request.headers.get("x-api-key"), _bearer(request))
            if not any(_matches(one, self.token) for one in presented):
                error = {"type": "authentication_error", "message": "invalid proxy token"}
                return web.json_response({"type": "error", "error": error}, status=401)
            return await handler(request)

        return auth

    async def _serve_records(self, request: Any) -> Any:
        """Return the trial's records in request order and drain them.

        With `delta=1`, each prompt is sent as an extension of an earlier record where possible.
        With `again=1`, the last payload built for the trial is re-sent."""
        trial = request.match_info["trial"]
        now = time.monotonic()
        for name, (at, _payload) in list(self._served_last.items()):
            if now - at > SERVED_KEPT_SECONDS:
                del self._served_last[name]
        kept = self._served_last.get(trial)
        if request.query.get("again") == "1" and kept is not None:
            payload = kept[1]
        else:
            payload = self._records_payload(trial, request.query.get("delta") == "1")
            self._served_last[trial] = (now, payload)
        response = web.json_response(payload, dumps=lambda data: json.dumps(data, default=str))
        response.enable_compression()
        return response

    def _records_payload(self, trial: str, delta: bool) -> dict[str, Any]:
        """Take the trial's records and build the response payload.

        An unknown trial gets an empty list and a 200: a rollout whose container never reached the
        proxy made no requests."""
        records = self.recorder.take(trial)
        return {"records": delta_encoded(records) if delta else [asdict(r) for r in records]}

    def _control_middleware(self) -> Any:
        """Return middleware that handles the run's routes before any other.

        `POST /control/weights` and `GET /records/<trial>` require the control token: the
        harness token is in every sandbox, and the model under training must neither re-point its
        own proxy nor read or drain any trial's records. Without a control token both are closed.
        `GET /healthz` needs no token and answers `status`, `model` and `serving` (the served
        model) and `budget` (the tokens each trial may sample, null when none is enforced)."""

        @web.middleware
        async def control(request: Any, handler: Any) -> Any:
            if request.path == HEALTH_PATH and request.method == "GET":
                answer = {"status": "ok", "model": self.serving, "serving": self.serving}
                return web.json_response({**answer, "budget": self.max_context or None})
            weights = request.path == CONTROL_PATH
            if not weights and not request.path.startswith(f"{RECORDS_PATH}/"):
                return await handler(request)
            if self.control_token is None:
                raise web.HTTPNotFound()
            if weights and request.method != "POST":
                raise web.HTTPMethodNotAllowed(request.method, ["POST"])
            if not _matches(_bearer(request), self.control_token):
                return web.json_response({"error": "invalid control token"}, status=401)
            if weights:
                return await self._serve_control(request)
            # The route's own handler, past the harness token's middleware.
            return await request.match_info.handler(request)

        return control

    async def _serve_control(self, request: Any) -> Any:
        """Switch to the body's `model_path` (a `tinker://` path or null).

        A malformed body, or a path that cannot be served, gets a 400."""
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(reason="expected a JSON object with a 'model_path'") from None
        if not isinstance(body, dict) or "model_path" not in body:
            raise web.HTTPBadRequest(reason="expected a JSON object with a 'model_path'")
        path = body["model_path"]
        if path is not None and (not isinstance(path, str) or not path.startswith("tinker://")):
            raise web.HTTPBadRequest(reason="'model_path' must be a tinker:// path or null")
        try:
            await self.swap(path)
        except Exception as refused:
            raise web.HTTPBadRequest(reason=f"cannot serve {path}: {refused}") from refused
        logger.info("serving %s", path or self.base_model)
        return web.json_response({"serving": path})


def _bearer(request: Any) -> str | None:
    """Return the token in an `Authorization: Bearer` header, or None."""
    header = request.headers.get("Authorization", "")
    return header.removeprefix("Bearer ") if header.startswith("Bearer ") else None


def _matches(presented: str | None, token: str) -> bool:
    """Compare a presented token with `token` in constant time.

    As bytes: `compare_digest` refuses a str with non-ASCII characters, which a caller controls.
    aiohttp keeps header bytes that are not UTF-8 as surrogates; `surrogateescape` restores
    them."""
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "surrogateescape"), token.encode("utf-8"))


async def _body(request: web.Request) -> dict[str, Any]:
    """Return the request body as a JSON object. Raise `Refused` for anything else."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise Refused("request body must be valid JSON") from None
    if not isinstance(body, dict):
        raise Refused("request body must be a JSON object")
    return body


def _bound_port(runner: Any) -> int:
    for address in getattr(runner, "addresses", None) or []:
        if isinstance(address, tuple) and len(address) >= 2:
            return int(address[1])
    raise RuntimeError("The endpoint bound no port.")


async def context_of(service: Any, base_model: str) -> int | None:
    """Return the model's context length from `get_server_capabilities`, or None on any error.

    Without a context length nothing is refused, so a failed lookup does not fail a run."""
    try:
        caps = await service.get_server_capabilities_async()
        for model in getattr(caps, "supported_models", None) or []:
            if getattr(model, "model_name", None) == base_model:
                return int(getattr(model, "max_context_length", 0) or 0) or None
    except Exception:  # noqa: BLE001 - a failed capability read must not fail a run
        logger.warning(
            "could not read the context length of %s: the budget and the context fit are "
            "not enforced",
            base_model,
            exc_info=True,
        )
    return None
