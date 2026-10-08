"""The served endpoint: the cookbook's app on a Tinker sampling client, with the records
route, the control route, the Exchange middleware and keepalives on streams, behind a token
on every route but `/healthz`, since what sits behind it spends Tinker credit."""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from aiohttp import web

from shipyard.proxy import cookbook, keepalive, thinking, vision
from shipyard.proxy.exchange import middleware
from shipyard.proxy.keepalive import KEEPALIVE_EVERY, KEEPALIVE_GRACE
from shipyard.proxy.recorder import Record, Recorder
from shipyard.proxy.rendering import Rendering
from shipyard.proxy.wire import CONTROL_PATH, HEALTH_PATH, RECORDS_PATH, delta_encoded, new_token

logger = logging.getLogger(__name__)

#: How long the last records payload of a trial is kept for a caller whose answer was
#: lost on the way (`again=1`): the fetch drains, so a lost answer cannot be rebuilt.
SERVED_KEPT_SECONDS = 1800.0
#: What a request that names no limit is allowed; the cookbook's 1024 cuts a coding turn.
DEFAULT_MAX_TOKENS = 8192
#: The largest request body the proxy reads; aiohttp's own default is 1 MiB.
MAX_BODY_BYTES = 256 * 1024**2

#: How a sampling client is made for a `tinker://` path, or None for the base model.
ClientFactory = Callable[[str | None], Awaitable[Any]]


@dataclass
class Endpoint:
    """A local server standing in for a provider. `host` is the name a sandbox resolves this
    machine by; the token is the gate, whatever `bind` is narrowed to."""

    base_model: str
    model_path: str | None = None
    #: A cookbook renderer's name, an object (a test's), or None for the model's own.
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
    #: Regexes cut from system messages before rendering; off unless the run asks.
    volatile: tuple[str, ...] = ()
    #: A stream silent this long gets SSE comment frames, one per `keepalive_every`
    #: seconds, until its reply; None turns them off.
    keepalive_grace: float | None = KEEPALIVE_GRACE
    keepalive_every: float = KEEPALIVE_EVERY
    token: str = field(default_factory=new_token)
    control_token: str | None = None
    #: The Tinker session's `user_metadata`: the run and recipe it samples for.
    metadata: dict[str, str] | None = None
    #: A test's way of making sampling clients; None opens Tinker's service client.
    client_factory: ClientFactory | None = field(default=None, repr=False)

    _service: Any = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)
    _recorder: Recorder | None = field(default=None, repr=False)
    _deps: Any = field(default=None, repr=False)
    _runner: Any = field(default=None, repr=False)
    _port: int | None = field(default=None, repr=False)
    _served_last: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """An empty token is replaced: the cookbook installs its auth on a truthy token and
        none at all otherwise, and an unset env var arrives here as empty."""
        if not self.token:
            self.token = new_token()

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def recorder(self) -> Recorder:
        if self._recorder is None:
            raise RuntimeError("This endpoint is not started; there is nothing recorded yet.")
        return self._recorder

    def address_for(self, trial: str) -> str:
        """`http://<host>:<port>/r/trial/<trial>/v1`. The name must be non-empty and free of
        `/`: the cookbook reads the path as key/value pairs and would file the tokens
        elsewhere. Refused before `start`, which is a harness talking to nothing."""
        if not trial or "/" in trial:
            raise ValueError(
                f"A trial name rides in the URL path, so {trial!r} cannot be addressed: "
                "it must be non-empty and free of '/'."
            )
        if self._port is None:
            raise RuntimeError("This endpoint has no port yet; call start() before address_for.")
        return f"http://{self.host}:{self._port}/r/trial/{trial}/v1"

    def records_for(self, trial: str) -> list[Record]:
        return self.recorder.records_for(trial)

    async def start(self) -> None:
        """Bind and serve: a sampling client on `model_path` or the base, the context from
        Tinker when unknown, the renderer, the app and its routes, then the socket."""
        # Images and thinking on the wire, wrapped over the cookbook's parsers once per
        # process; thinking's wrap after vision's, so no thinking block reaches a parser.
        # thinking.install puts the keepalive's stream wrap beneath its own.
        vision.install()
        thinking.install()
        if self.client_factory is None and self._service is None:
            import tinker

            self._service = tinker.ServiceClient(user_metadata=self.metadata)
        self._client = await self._make(self.model_path)
        if self.max_context is None and self._service is not None:
            self.max_context = await context_of(self._service, self.base_model)
        renderer = (
            cookbook.renderer_for(self.base_model, self.renderer)
            if self.renderer is None or isinstance(self.renderer, str)
            else self.renderer
        )
        self._recorder = Recorder(
            self._client,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_context=self.max_context,
            budget=self.max_context,
            fill_context=self.fill_context,
            served=self.model_path,
        )
        self._deps = cookbook.ProxyDeps(
            renderer=Rendering(renderer, volatile=self.volatile, recorder=self._recorder),
            sampling_client=self._recorder,
            model_label=self.model_path or self.base_model,
            default_max_tokens=int(self.max_tokens),
        )
        app = cookbook.make_app(self._deps, auth_token=self.token)
        # aiohttp answers a body over its 1 MiB default with a 413 before any handler, so a
        # long history or a few images would fail unrecorded; the context refusal is the cap.
        app._client_max_size = MAX_BODY_BYTES
        # Outermost, so the control token and not the harness's opens the control route;
        # the Exchange after the cookbook's auth, so a refused request opens nothing.
        app.middlewares.insert(0, self._control_middleware())
        app.middlewares.append(middleware())
        if self.keepalive_grace is not None:
            app.middlewares.append(keepalive.middleware(self.keepalive_grace, self.keepalive_every))
        app.middlewares.append(thinking.middleware())
        app.router.add_get(f"{RECORDS_PATH}/{{trial}}", self._serve_records)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.bind, self.bind_port).start()
        self._runner = runner
        self._port = _bound_port(runner)
        logger.info("serving %s on port %s as %s", self._deps.model_label, self._port, self.host)

    async def swap(self, model_path: str | None) -> None:
        """A new sampling client on the path, or on the base for None; the server stays up
        so the addresses already handed to running trials keep answering."""
        self._client = await self._make(model_path)
        self.model_path = model_path
        if self._recorder is not None:
            self._recorder.client, self._recorder.served = self._client, model_path
        if self._deps is not None:
            self._deps.model_label = model_path or self.base_model

    async def _make(self, model_path: str | None) -> Any:
        if model_path is not None and not model_path.startswith("tinker://"):
            raise ValueError(f"{model_path!r} is not a tinker:// path; a bare name names nothing")
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

    async def __aenter__(self) -> Endpoint:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def _serve_records(self, request: Any) -> Any:
        """The trial's records in request order, drained; `delta=1` sends each prompt as
        an extension of an earlier record; `again=1` re-sends the last payload built."""
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
        """An unknown trial is an empty list and a 200: a rollout whose container never
        reached the proxy made no requests, which is a true answer."""
        recorder = self.recorder
        turned_away = recorder.turned_away.get(trial, 0)
        cut, spoke = recorder.cut.get(trial, 0), recorder.spoke.get(trial, 0)
        records = recorder.take(trial)
        return {
            "trial": trial,
            "records": delta_encoded(records) if delta else [asdict(r) for r in records],
            "turned_away": turned_away,
            "cut": cut,
            "spoke": spoke,
        }

    def _control_middleware(self) -> Any:
        """`POST /control/weights` behind the control token alone: the harness token is in
        every sandbox, and the model under training must not re-point its own proxy. And
        `GET /healthz` naming what is served, the cookbook's answer with `serving` on it,
        and the token budget each trial may sample, null when none is enforced."""

        @web.middleware
        async def control(request: Any, handler: Any) -> Any:
            if request.path == HEALTH_PATH and request.method == "GET":
                served = self.model_path or self.base_model
                answer = {"status": "ok", "model": served, "serving": served}
                return web.json_response({**answer, "budget": self.max_context or None})
            if request.path != CONTROL_PATH:
                return await handler(request)
            if self.control_token is None:
                raise web.HTTPNotFound()
            if request.method != "POST":
                raise web.HTTPMethodNotAllowed(request.method, ["POST"])
            presented = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not presented or not hmac.compare_digest(presented, self.control_token):
                return web.json_response({"error": "invalid control token"}, status=401)
            return await self._serve_control(request)

        return control

    async def _serve_control(self, request: Any) -> Any:
        """Point this endpoint at `{"model_path": <tinker:// path or null>}`; resolving is
        what can fail, and it fails here where the caller is listening."""
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


def _bound_port(runner: Any) -> int:
    for address in getattr(runner, "addresses", None) or []:
        if isinstance(address, tuple) and len(address) >= 2:
            return int(address[1])
    raise RuntimeError("The endpoint bound no port.")


async def context_of(service: Any, base_model: str) -> int | None:
    """The model's context length off `get_server_capabilities`, or None on anything
    unexpected: nothing is refused without a number, so a lost answer costs no run."""
    try:
        caps = await service.get_server_capabilities_async()
        for model in getattr(caps, "supported_models", None) or []:
            if getattr(model, "model_name", None) == base_model:
                return int(getattr(model, "max_context_length", 0) or 0) or None
    except Exception:  # noqa: BLE001 - a capability read is never worth failing a run over
        logger.debug("could not read the context length of %s", base_model, exc_info=True)
    return None
