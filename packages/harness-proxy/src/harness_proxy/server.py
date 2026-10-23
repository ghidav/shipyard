from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
from harness_proxy.rendering import complete
from harness_proxy.trial import Trial
from harness_proxy.wire import (
    HEALTH_PATH,
    TRIALS_PATH,
    Opening,
    Served,
    health,
    new_token,
    records_answer,
)

logger = logging.getLogger(__name__)

#: Seconds a closed trial's records are kept, so a close whose answer was lost is answered
#: again: closing hands the records over, and the trial is gone.
CLOSED_KEPT_SECONDS = 1800.0
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
    """One model the proxy serves, which a trial is opened on: the backend that samples it, its
    renderers, the run's sampling settings, and a sampling client on each weights its open
    trials were opened on."""

    #: The name a trial is opened on it by.
    name: str
    #: The base model it serves.
    model: str
    #: The renderers calls are rendered with: a cookbook renderer name, or a list of them that a
    #: trial is opened with by name; a renderer object, or a dict of them by name (tests); or
    #: None for the model's default. A trial opened with none takes the first.
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
    #: The renderers, by the name a trial is opened with; "" for the model's default.
    _renderers: dict[str, Any] = field(default_factory=dict, repr=False)
    #: The sampling client on each weights open trials use, and how many of them use it.
    _clients: dict[str | None, list[Any]] = field(default_factory=dict, repr=False)
    _making: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        """Refuse a nameless lane, or an unknown backend."""
        if not self.name:
            raise ValueError("a lane is named, since a trial is opened on it by name")
        backends.named(self.backend)

    @property
    def sampling(self) -> ModuleType:
        """The backend module the lane samples through."""
        return backends.named(self.backend)

    @property
    def renderers(self) -> list[str]:
        """The renderers a trial may be opened with, in order; none for the model's default."""
        return [name for name in self._renderers if name]

    async def open(self) -> None:
        """Build the renderers and the backend's service, and read the model's context length;
        `RuntimeError` if its backend cannot say it and the lane gives none, since neither the
        budget nor the context could be enforced."""
        renderers = self._built()
        if self.client_factory is None and self._service is None:
            self._service = await self.sampling.connect(
                metadata=self.metadata, session=self.session
            )
        if self.max_context is None and self._service is not None:
            self.max_context = await self.sampling.context_of(self._service, self.model)
            if self.max_context is None:
                raise RuntimeError(
                    f"lane {self.name}: could not read the context length of {self.model}, so "
                    "neither a trial's budget nor the context could be enforced: give the lane "
                    "its max_context"
                )
        self._renderers = renderers

    async def close(self) -> None:
        """Close the backend's service, if the lane opened one."""
        service, self._service = self._service, None
        if service is not None:
            with contextlib.suppress(Exception):
                await self.sampling.close(service)

    async def hold(self, weights: str | None) -> Any:
        """Return a sampling client on `weights`, or on the base model for None, for a trial
        opened on them: the one the lane holds if an open trial uses them already. `ValueError`
        for weights under another backend's prefix."""
        prefix = self.sampling.PREFIX
        if weights is not None and not weights.startswith(prefix):
            raise ValueError(f"{weights!r} is not a {prefix} path")
        async with self._making:
            held = self._clients.get(weights)
            if held is None:
                held = self._clients[weights] = [await self._make(weights), 0]
            held[1] += 1
            return held[0]

    def release(self, weights: str | None) -> None:
        """Let go of the sampling client on `weights` once no open trial uses them."""
        held = self._clients.get(weights)
        if held is not None:
            held[1] -= 1
            if held[1] <= 0:
                del self._clients[weights]

    def trial(self, opening: Opening, client: Any) -> Trial:
        """Return the trial `opening` opens on the lane, which samples through `client`."""
        return Trial(
            opening.trial,
            client,
            renderer=self.renderer_for(opening.renderer or ""),
            weights=opening.weights,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_context=self.max_context,
            budget=self.max_context,
            fill_context=self.fill_context,
        )

    def _built(self) -> dict[str, Any]:
        """Return the renderers by the name a trial is opened with, "" for the default."""
        given = self.renderer
        if isinstance(given, dict):
            return dict(given)
        names = (given if isinstance(given, list) else [given]) or [None]
        if not all(one is None or isinstance(one, str) for one in names):
            return {"": given}
        return {one or "": cookbook.renderer_for(self.model, one or None) for one in names}

    def renderer_for(self, named: str) -> Any:
        """Return the renderer `named`, the first for ""; `Refused` if unknown."""
        if not named:
            return next(iter(self._renderers.values()))
        renderer = self._renderers.get(named)
        if renderer is None:
            served = ", ".join(self.renderers) or "its model's default alone"
            raise Refused(f"this proxy does not render with {named!r}: it renders with {served}")
        return renderer

    async def _make(self, weights: str | None) -> Any:
        if self.client_factory is not None:
            return await self.client_factory(weights)
        return await self.sampling.sampler(self._service, self.model, weights)

    def reported(self) -> Served:
        """Return the lane as `/healthz` reports it: its model, the tokens one trial may sample,
        and the renderers a trial may be opened with."""
        return Served(self.model, self.max_context or None, tuple(self.renderers))


@dataclass(frozen=True)
class Opened:
    """An open trial: what the run opened it with, the token its calls carry, and its lane."""

    opening: Opening
    token: str = field(repr=False)
    trial: Trial
    lane: Lane


#: Where the auth middleware leaves the open trial a call's token names, for the call's handler.
OPENED = web.RequestKey("opened", Opened)


@dataclass
class Server:
    """Local server that stands in for a model provider. The run opens each trial on one of its
    `lanes` with the control token, and the trial's calls carry the token the opening returned;
    closing the trial hands its records back."""

    lanes: list[Lane]
    host: str = "host.docker.internal"
    bind: str = "0.0.0.0"
    bind_port: int = 0
    #: Seconds a stream may stay silent before SSE keepalive frames start, one every
    #: `keepalive_every` seconds until the reply. None turns them off.
    keepalive_grace: float | None = KEEPALIVE_GRACE
    keepalive_every: float = KEEPALIVE_EVERY
    #: The run's token, which opens and closes trials; generated if empty.
    control_token: str = field(default_factory=new_token, repr=False)

    _runner: Any = field(default=None, repr=False)
    _port: int | None = field(default=None, repr=False)
    #: The lanes, by name.
    _named: dict[str, Lane] = field(default_factory=dict, repr=False)
    #: The open trials, by name, and each one's name by the digest of its token.
    _open: dict[str, Opened] = field(default_factory=dict, repr=False)
    _by_token: dict[str, str] = field(default_factory=dict, repr=False)
    #: Each recently closed trial's answer, by name, with when it closed.
    _closed: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)
    _opening: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        """Generate the control token if it is empty; refuse no lane, or two lanes of one name."""
        if not self.control_token:
            self.control_token = new_token()
        if not self.lanes:
            raise ValueError("a proxy serves at least one lane")
        for one in self.lanes:
            if one.name in self._named:
                raise ValueError(f"two lanes are named {one.name!r}: a trial is opened on one")
            self._named[one.name] = one

    @property
    def port(self) -> int | None:
        return self._port

    async def start(self) -> None:
        """Open every lane, then build the app and its routes, and bind the socket."""
        try:
            for lane in self.lanes:
                await lane.open()
        except BaseException:
            for lane in self.lanes:
                await lane.close()  # the sessions of the lanes that opened
            raise
        # Control outermost, so only the control token opens and closes trials.
        middlewares = [self._control_middleware(), self._auth_middleware()]
        if self.keepalive_grace is not None:
            middlewares.append(keepalive.middleware(self.keepalive_grace, self.keepalive_every))
        # aiohttp answers a body over its 1 MiB default with a 413 before any handler, so the
        # request would not be recorded. The context refusal is the cap.
        app = web.Application(middlewares=middlewares, client_max_size=MAX_BODY_BYTES)
        for dialect in (anthropic, openai):
            app.router.add_post(f"/v1{dialect.PATH}", self._serving(dialect))
        app.router.add_post(TRIALS_PATH, self._serve_open)
        app.router.add_post(f"{TRIALS_PATH}/{{trial}}/close", self._serve_close)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.bind, self.bind_port).start()
        self._runner = runner
        self._port = _bound_port(runner)
        logger.info("listening on port %s as %s", self._port, self.host)
        for lane in self.lanes:
            logger.info("serving %s in lane %s", lane.model, lane.name)

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
                raise Refused(f"this proxy serves several lanes, so a trial names one: {served}")
            raise Refused(f"this proxy serves no lane {name!r}: it serves {served}")
        return found

    def trial(self, name: str) -> Trial | None:
        """Return the open trial `name`, or None."""
        held = self._open.get(name)
        return held.trial if held is not None else None

    async def open(self, opening: Opening) -> str:
        """Open the trial `opening` names and return the token its calls carry; the same token
        for the same opening asked again. `Refused` for a lane or renderer the proxy does not
        serve, or a trial open already on others; `ValueError` for weights it cannot serve."""
        lane = self.lane(opening.lane or "")
        lane.renderer_for(opening.renderer or "")  # refuses an unknown one
        async with self._opening:
            held = self._open.get(opening.trial)
            if held is not None:
                if held.opening == opening:
                    return held.token  # an open whose answer was lost, asked again
                raise Refused(f"trial {opening.trial} is open already, as {held.opening}")
            client = await lane.hold(opening.weights)
            token = new_token()
            self._open[opening.trial] = Opened(opening, token, lane.trial(opening, client), lane)
            self._by_token[_digest(token)] = opening.trial
            self._closed.pop(opening.trial, None)
        logger.info(
            "opened trial %s in lane %s on %s",
            opening.trial,
            lane.name,
            opening.weights or lane.model,
        )
        return token

    def close(self, name: str) -> dict[str, Any]:
        """Close the trial `name` and return its records as the close route answers them, or the
        same answer for one closed in the last `CLOSED_KEPT_SECONDS`; `KeyError` for neither.
        A call of the trial still in flight is answered, and recorded nowhere."""
        now = time.monotonic()
        for one, (at, _answer) in list(self._closed.items()):
            if now - at > CLOSED_KEPT_SECONDS:
                del self._closed[one]
        held = self._open.pop(name, None)
        if held is None:
            return self._closed[name][1]
        self._by_token.pop(_digest(held.token), None)
        held.lane.release(held.opening.weights)
        answer = records_answer(held.trial.records)
        self._closed[name] = (now, answer)
        logger.info("closed trial %s with %d records", name, len(held.trial.records))
        return answer

    def _serving(self, dialect: ModuleType) -> Callable[[web.Request], Awaitable[Any]]:
        """Return the handler for `dialect`'s route: read the call, complete it, write the reply."""

        async def serve(request: web.Request) -> Any:
            held = request[OPENED]
            trial = held.trial
            try:
                caller = caller_of(request, trial.name)
                # Off the event loop: images and PDF pages decode here, and other trials wait
                # on the loop.
                call = await asyncio.to_thread(dialect.read, await _body(request))
                reply = await complete(call, caller, trial, int(held.lane.max_tokens))
            except Refused as refused:
                logger.warning("refused a call of trial %s: %s", trial.name, refused)
                return dialect.error(400, str(refused))
            serving = trial.weights or held.lane.model
            if not call.stream:
                return web.json_response(dialect.answer(reply, serving, call))
            return await keepalive.serve_sse(request, dialect.events(reply, serving, call))

        return serve

    def _auth_middleware(self) -> Any:
        """Return middleware that finds the open trial whose token a call carries, as
        `x-api-key` or a Bearer header, and turns away a call that carries none."""

        @web.middleware
        async def auth(request: Any, handler: Any) -> Any:
            for presented in (request.headers.get("x-api-key"), _bearer(request)):
                name = self._by_token.get(_digest(presented)) if presented else None
                held = self._open.get(name) if name is not None else None
                if held is not None:
                    request[OPENED] = held
                    return await handler(request)
            message = "no open trial holds this token: the run opens a trial before it calls"
            error = {"type": "authentication_error", "message": message}
            return web.json_response({"type": "error", "error": error}, status=401)

        return auth

    async def _serve_open(self, request: Any) -> Any:
        """Open the trial the body names and answer its token; 400 for one it cannot open."""
        try:
            body = await request.json()
        except Exception:
            body = None  # which `Opening.of` refuses as it does any body that is not an opening
        try:
            token = await self.open(Opening.of(body))
        except (ValueError, Refused) as refused:
            raise web.HTTPBadRequest(reason=str(refused)) from None
        except Exception as failed:
            raise web.HTTPBadRequest(reason=f"cannot serve those weights: {failed}") from failed
        return web.json_response({"token": token})

    async def _serve_close(self, request: Any) -> Any:
        """Close the trial the path names and answer its records; 404 if none is open by it."""
        name = request.match_info["trial"]
        try:
            answer = self.close(name)
        except KeyError:
            minutes = round(CLOSED_KEPT_SECONDS / 60)
            raise web.HTTPNotFound(
                reason=f"no trial {name} is open, or closed in the last {minutes} minutes"
            ) from None
        response = web.json_response(answer, dumps=lambda data: json.dumps(data, default=str))
        response.enable_compression()
        return response

    def _control_middleware(self) -> Any:
        """Return the first middleware: health, then the run's routes behind the control token."""

        @web.middleware
        async def control(request: Any, handler: Any) -> Any:
            if request.path == HEALTH_PATH and request.method == "GET":
                return web.json_response(health({one.name: one.reported() for one in self.lanes}))
            if request.path != TRIALS_PATH and not request.path.startswith(f"{TRIALS_PATH}/"):
                return await handler(request)
            # Every sandbox holds a trial's token, so the model under training must not be able
            # to open or close trials, or read records: those take the run's own token.
            if not _matches(_bearer(request), self.control_token):
                return web.json_response({"error": "invalid control token"}, status=401)
            # The route's own handler, past the trial token's middleware.
            return await request.match_info.handler(request)

        return control


def caller_of(request: Any, trial: str) -> Caller:
    """Return who made a call: the trial its token opened, its wire, and the SDK's retry
    headers."""
    return Caller(
        trial=trial,
        wire="anthropic" if request.path.endswith("/messages") else "openai",
        retry=_count(request.headers.get(RETRY_HEADER)),
        idempotency_key=request.headers.get("Idempotency-Key") or None,
        request_id=request.headers.get("x-request-id") or None,
    )


def _digest(token: str) -> str:
    """Return what the server keeps a token by: its digest, which a lookup's timing cannot
    give away."""
    return hashlib.sha256(token.encode("utf-8", "surrogateescape")).hexdigest()


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
