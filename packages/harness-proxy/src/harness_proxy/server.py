from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from aiohttp import web

from harness_proxy import backends, cookbook, keepalive
from harness_proxy.chat import Caller, Refused
from harness_proxy.dialects import anthropic, openai
from harness_proxy.keepalive import KEEPALIVE_EVERY, KEEPALIVE_GRACE
from harness_proxy.recorder import Recorder
from harness_proxy.rendering import complete
from harness_proxy.wire import (
    CONTROL_PATH,
    HEALTH_PATH,
    RECORDS_PATH,
    Address,
    Fetch,
    Record,
    Served,
    Swap,
    health,
    new_token,
    records_answer,
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

#: Retry count header sent by both the OpenAI and the Anthropic SDKs.
RETRY_HEADER = "x-stainless-retry-count"

#: Makes a sampling client on named weights, or on the base model for None.
ClientFactory = Callable[[str | None], Awaitable[Any]]


@dataclass
class Lane:
    """One model the proxy serves, which a call's address names: its weights, the backend that
    samples them, its renderers, and the recorder its calls sample through, which keeps their
    records."""

    #: The name a call's address gives it: `/r/lane/<name>/...`.
    name: str
    #: The base model it serves.
    model: str
    #: The weights it serves, under its backend's prefix; None for the base model.
    weights: str | None = None
    #: The renderers calls are rendered with: a cookbook renderer name, or a list of them that a
    #: trial's address picks from by name; a renderer object, or a dict of them by name (tests);
    #: or None for the model's default. A call whose address names none takes the first.
    renderer: Any = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = -1
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_context: int | None = None
    fill_context: bool = False
    #: `user_metadata` of the Tinker session, naming the run and recipe it samples for.
    metadata: dict[str, str] | None = None
    #: The backend that samples: `tinker`, or `together` for a session the run opened.
    backend: str = "tinker"
    #: The training session a `together` backend samples from, which the run opened.
    session: str | None = None
    #: Sampling client factory for tests. None uses the backend's service.
    client_factory: ClientFactory | None = field(default=None, repr=False)

    #: The backend's service, opened on `open`; a test may pass a fake one.
    _service: Any = field(default=None, repr=False)
    _recorder: Recorder | None = field(default=None, repr=False)
    #: The renderers, by the name a trial's address gives each; "" for the model's default.
    _renderers: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Refuse a name an address cannot hold, or an unknown backend."""
        if not self.name or "/" in self.name:
            raise ValueError(f"a lane is named in a call's address: {self.name!r} cannot be")
        backends.named(self.backend)

    @property
    def sampling(self) -> ModuleType:
        """The backend module the lane samples through."""
        return backends.named(self.backend)

    @property
    def serving(self) -> str:
        """The model a reply names: the served weights, or the base model."""
        return self.weights or self.model

    @property
    def renderers(self) -> list[str]:
        """The renderers a trial's address may name, in order; none for the model's default."""
        return [name for name in self._renderers if name]

    @property
    def recorder(self) -> Recorder:
        if self._recorder is None:
            raise RuntimeError("This endpoint is not started, so nothing is recorded.")
        return self._recorder

    def records_for(self, trial: str) -> list[Record]:
        return self.recorder.records_for(trial)

    async def open(self) -> None:
        """Build the renderers, the backend's service and the sampling client, and the recorder
        that every call samples through."""
        renderers = self._built()
        if self.client_factory is None and self._service is None:
            self._service = await self.sampling.connect(
                metadata=self.metadata, session=self.session
            )
        client = await self._make(self.weights)
        if self.max_context is None and self._service is not None:
            self.max_context = await self.sampling.context_of(self._service, self.model)
        self._recorder = Recorder(
            client,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_context=self.max_context,
            budget=self.max_context,
            fill_context=self.fill_context,
            served=self.weights,
        )
        self._renderers = renderers

    async def close(self) -> None:
        """Close the backend's service, if the lane opened one."""
        service, self._service = self._service, None
        if service is not None:
            with contextlib.suppress(Exception):
                await self.sampling.close(service)

    async def swap(self, weights: str | None) -> None:
        """Switch to a sampling client on `weights`, or on the base model for None."""
        client = await self._make(weights)
        self.weights = weights
        if self._recorder is not None:
            self._recorder.client, self._recorder.served = client, weights

    def _built(self) -> dict[str, Any]:
        """Return the renderers by the name a trial's address gives each, "" for the default."""
        given = self.renderer
        if isinstance(given, dict):
            return dict(given)
        names = (given if isinstance(given, list) else [given]) or [None]
        if not all(one is None or isinstance(one, str) for one in names):
            return {"": given}
        return {one or "": cookbook.renderer_for(self.model, one or None) for one in names}

    def renderer_for(self, named: str) -> Any:
        """Return the renderer a call's address names, the first for ""; `Refused` if unknown."""
        if not named:
            return next(iter(self._renderers.values()))
        renderer = self._renderers.get(named)
        if renderer is None:
            served = ", ".join(self.renderers) or "its model's default alone"
            raise Refused(f"this proxy does not render with {named!r}: it renders with {served}")
        return renderer

    async def _make(self, weights: str | None) -> Any:
        prefix = self.sampling.PREFIX
        if weights is not None and not weights.startswith(prefix):
            raise ValueError(f"{weights!r} is not a {prefix} path")
        if self.client_factory is not None:
            return await self.client_factory(weights)
        return await self.sampling.sampler(self._service, self.model, weights)

    def reported(self) -> Served:
        """Return the lane as `/healthz` reports it: its model, what it serves, the tokens one
        trial may sample, and the renderers an address may name."""
        return Served(self.model, self.serving, self.max_context or None, tuple(self.renderers))


@dataclass
class Server:
    """Local server that stands in for a model provider. It serves each of its `lanes` to the
    calls whose address names it (`/r/lane/<name>/...`), and its only lane, if it has one, to a
    call whose address names none."""

    lanes: list[Lane]
    host: str = "host.docker.internal"
    bind: str = "0.0.0.0"
    bind_port: int = 0
    #: Seconds a stream may stay silent before SSE keepalive frames start, one every
    #: `keepalive_every` seconds until the reply. None turns them off.
    keepalive_grace: float | None = KEEPALIVE_GRACE
    keepalive_every: float = KEEPALIVE_EVERY
    token: str = field(default_factory=new_token, repr=False)
    control_token: str | None = field(default=None, repr=False)

    _runner: Any = field(default=None, repr=False)
    _port: int | None = field(default=None, repr=False)
    _served_last: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)
    #: The lanes, by name.
    _named: dict[str, Lane] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Generate a token if it is empty; refuse no lane, or two lanes of one name."""
        if not self.token:
            self.token = new_token()
        if not self.lanes:
            raise ValueError("a proxy serves at least one lane")
        for one in self.lanes:
            if one.name in self._named:
                raise ValueError(f"two lanes are named {one.name!r}: an address names one")
            self._named[one.name] = one

    @property
    def port(self) -> int | None:
        return self._port

    async def start(self) -> None:
        """Open every lane, then build the app and its routes, and bind the socket."""
        for lane in self.lanes:
            await lane.open()
        # Control outermost, so only the control token opens the control route.
        middlewares = [self._control_middleware(), self._auth_middleware()]
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
        logger.info("listening on port %s as %s", self._port, self.host)
        for lane in self.lanes:
            logger.info("serving %s in lane %s", lane.serving, lane.name)

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()
        self._port = None
        for lane in self.lanes:
            await lane.close()

    async def __aenter__(self) -> Server:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def lane(self, name: str) -> Lane:
        """Return the lane `name`, or the only lane for ""; `Refused` for any other, and for ""
        when there are several."""
        if not name and len(self.lanes) == 1:
            return self.lanes[0]
        found = self._named.get(name)
        if found is None:
            served = ", ".join(self._named)
            if not name:
                raise Refused(f"this proxy serves several lanes, so a call names one: {served}")
            raise Refused(f"this proxy serves no lane {name!r}: it serves {served}")
        return found

    def _serving(self, dialect: ModuleType) -> Callable[[web.Request], Awaitable[Any]]:
        """Return the handler for `dialect`'s route: read the call, complete it, write the reply."""

        async def serve(request: web.Request) -> Any:
            try:
                caller = caller_of(request)
                lane = self.lane(caller.lane)
                renderer = lane.renderer_for(caller.renderer)
                # Off the event loop: images and PDF pages decode here, and other trials wait
                # on the loop.
                call = await asyncio.to_thread(dialect.read, await _body(request))
                reply = await complete(call, caller, renderer, lane.recorder, int(lane.max_tokens))
            except Refused as refused:
                logger.warning("refused a request to %s: %s", request.path, refused)
                return dialect.error(400, str(refused))
            if not call.stream:
                return web.json_response(dialect.answer(reply, lane.serving, call))
            return await keepalive.serve_sse(request, dialect.events(reply, lane.serving, call))

        return serve

    def _auth_middleware(self) -> Any:
        """Return middleware that requires the harness token, as `x-api-key` or a Bearer header."""

        @web.middleware
        async def auth(request: Any, handler: Any) -> Any:
            presented = (request.headers.get("x-api-key"), _bearer(request))
            if not any(_matches(one, self.token) for one in presented):
                error = {"type": "authentication_error", "message": "invalid proxy token"}
                return web.json_response({"type": "error", "error": error}, status=401)
            return await handler(request)

        return auth

    async def _serve_records(self, request: Any) -> Any:
        """Return the trial's records in request order and drain them."""
        trial = request.match_info["trial"]
        now = time.monotonic()
        for name, (at, _payload) in list(self._served_last.items()):
            if now - at > SERVED_KEPT_SECONDS:
                del self._served_last[name]
        kept, asked = self._served_last.get(trial), Fetch.of(request.query)
        if asked.again and kept is not None:
            payload = kept[1]
        else:
            payload = self._records_payload(trial, asked.delta)
            self._served_last[trial] = (now, payload)
        response = web.json_response(payload, dumps=lambda data: json.dumps(data, default=str))
        response.enable_compression()
        return response

    def _records_payload(self, trial: str, delta: bool) -> dict[str, Any]:
        """Take the trial's records and build the response payload; [] for an unknown trial."""
        # A trial calls one lane; each lane's recorder takes, and drains, its own records.
        records = [one for lane in self.lanes for one in lane.recorder.take(trial)]
        return records_answer(records, delta=delta)

    def _control_middleware(self) -> Any:
        """Return the first middleware: health, then the run's routes behind the control token."""

        @web.middleware
        async def control(request: Any, handler: Any) -> Any:
            if request.path == HEALTH_PATH and request.method == "GET":
                return web.json_response(health({one.name: one.reported() for one in self.lanes}))
            weights = request.path == CONTROL_PATH
            if not weights and not request.path.startswith(f"{RECORDS_PATH}/"):
                return await handler(request)
            if self.control_token is None:
                raise web.HTTPNotFound()
            if weights and request.method != "POST":
                raise web.HTTPMethodNotAllowed(request.method, ["POST"])
            # Every sandbox holds the harness token, so the model under training must not be able
            # to re-point the proxy, or read and drain records: those take the run's own token.
            if not _matches(_bearer(request), self.control_token):
                return web.json_response({"error": "invalid control token"}, status=401)
            if weights:
                return await self._serve_control(request)
            # The route's own handler, past the harness token's middleware.
            return await request.match_info.handler(request)

        return control

    async def _serve_control(self, request: Any) -> Any:
        """Switch the lane the body's `lane` names, the only one if it names none, to the body's
        `weights`, under its backend's prefix or null; 400 otherwise."""
        try:
            body = await request.json()
        except Exception:
            body = None  # which `Swap.of` refuses as it does any body that is not a swap
        try:
            asked = Swap.of(body)
            lane = self.lane(asked.lane or "")
        except (ValueError, Refused) as refused:
            raise web.HTTPBadRequest(reason=str(refused)) from None
        path, prefix = asked.weights, lane.sampling.PREFIX
        if path is not None and not path.startswith(prefix):
            raise web.HTTPBadRequest(reason=f"'weights' must be a {prefix} path or null")
        try:
            await lane.swap(path)
        except Exception as refused:
            raise web.HTTPBadRequest(reason=f"cannot serve {path}: {refused}") from refused
        logger.info("serving %s in lane %s", path or lane.model, lane.name)
        return web.json_response({"serving": path})


def caller_of(request: Any) -> Caller:
    """Return who made a call: what its address names, its wire, and the SDK's retry headers;
    `Refused` for an address that names no trial."""
    address = None
    if "address" in request.match_info:
        try:
            address = Address.of(request.rel_url.raw_path)
        except ValueError as unread:
            raise Refused(str(unread)) from None
    return Caller(
        trial=address.trial if address is not None else "",
        lane=(address.lane if address is not None else None) or "",
        renderer=(address.renderer if address is not None else None) or "",
        wire="anthropic" if request.path.endswith("/messages") else "openai",
        retry=_count(request.headers.get(RETRY_HEADER)),
        idempotency_key=request.headers.get("Idempotency-Key") or None,
        request_id=request.headers.get("x-request-id") or None,
    )


def _count(header: str | None) -> int:
    try:
        return int(header) if header else 0
    except ValueError:
        return 0


def _bearer(request: Any) -> str | None:
    """Return the token in an `Authorization: Bearer` header, or None."""
    header = request.headers.get("Authorization", "")
    return header.removeprefix("Bearer ") if header.startswith("Bearer ") else None


def _matches(presented: str | None, token: str) -> bool:
    """Compare a presented token with `token` in constant time, as bytes."""
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
