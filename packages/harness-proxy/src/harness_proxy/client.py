from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, Any
from urllib.parse import quote, urlsplit

import httpx

# `started` (wait for a matching line) and `end` (stop a process) are defined in tunnel.py
# and also serve the process started here.
from harness_proxy.tunnel import STARTUP_SECONDS as TUNNEL_SECONDS
from harness_proxy.tunnel import end, started
from harness_proxy.wire import (
    CONTROL_PATH,
    CONTROL_TOKEN_ENV,
    HEALTH_PATH,
    PROXY_TOKEN_ENV,
    RECORDS_PATH,
    Record,
    new_token,
    records_of,
)

logger = logging.getLogger(__name__)

#: Command that starts `harness-proxy`. Tests replace it with a stub.
SERVE: list[str] = [sys.executable, "-m", "harness_proxy.serve"]
#: First line `harness-proxy` prints, which names the port the kernel chose.
BANNER = re.compile(
    r"^serving (?P<model>\S+) at http://(?P<host>[^:/]+):(?P<port>\d+) "
    r"\(control: (?P<control>on|off)\)$"
)
#: The line serve prints once its tunnel is up, after the banner.
TUNNEL_LINE = re.compile(r"^tunnel at (?P<origin>https://\S+)$")
STARTUP_SECONDS = 180.0
#: Seconds `close` gives serve to stop its tunnel and close its session before killing it.
STOP_SECONDS = 30.0
#: Attempts per records fetch, so one tunnel stall does not lose a trial.
RECORDS_ATTEMPTS = 4
RECORDS_TIMEOUT = 120.0
CONTROL_TIMEOUT = 120.0
PAUSES = (1.0, 2.0, 4.0)
#: Settings passed to serve as flags. All others go in `--settings`.
FLAGGED = ("model", "weights", "renderer", "bind", "bind_port")
#: Seconds the proxy has to answer `/healthz` before the run is refused, and the pause
#: between probes. A quick tunnel's hostname can resolve late, and a resolver that
#: looked it up too early caches the miss for trycloudflare.com's negative TTL of 60 s,
#: so the window is longer than that.
PROBE_SECONDS = 120.0
PROBE_PAUSE = 1.0


class Unreachable(RuntimeError):
    """The proxy does not answer at the address sandboxes will dial, before any sandbox opens."""


def serve_command(settings: Mapping[str, Any], *, host: str, tunnel: bool = False) -> list[str]:
    """Return the `python -m harness_proxy.serve` command for `settings`, with `tunnel` if asked."""
    made = [*SERVE, "--model", str(settings["model"])]
    made += ["--bind", str(settings.get("bind") or "0.0.0.0")]
    made += ["--port", str(int(settings.get("bind_port") or 0)), "--advertise", host]
    made += ["--parent-pid", str(os.getpid()), *(["--tunnel"] if tunnel else [])]
    if settings.get("weights"):
        made += ["--weights", str(settings["weights"])]
    renderers = settings.get("renderer") or []
    for one in [renderers] if isinstance(renderers, str) else renderers:
        made += ["--renderer", str(one)]
    rest = {key: value for key, value in settings.items() if key not in FLAGGED}
    if rest:
        made += ["--settings", json.dumps(rest, sort_keys=True)]
    return made


@dataclass
class Proxy:
    """A proxy as seen from the run: its addresses, its tokens, and the process it started."""

    origin: str
    token: str = field(repr=False)
    control_token: str | None = field(default=None, repr=False)
    loopback: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    #: Whether the proxy opened a quick tunnel, whose address `origin` is.
    tunnel: bool = False
    #: Tokens a trial may sample, as last reported by `/healthz`. None when no budget is
    #: enforced.
    budget: int | None = None
    #: Renderers a trial's address may name, as last reported by `/healthz`. None for a proxy
    #: that does not say, as one from before a proxy rendered with several.
    renderers: list[str] | None = None
    #: Transport for the HTTP client, set by tests. None uses httpx's default.
    transport: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.origin = self.origin.rstrip("/")
        self.loopback = (self.loopback or self.origin).rstrip("/")

    @property
    def host(self) -> str:
        """Hostname a sandbox resolves for this proxy, for a network allowlist."""
        return urlsplit(self.origin).hostname or ""

    @classmethod
    async def local(
        cls,
        endpoint_settings: Mapping[str, Any],
        *,
        host: str = "host.docker.internal",
        environ: Mapping[str, str] | None = None,
        log: IO[str] | None = None,
        seconds: float = STARTUP_SECONDS,
    ) -> Proxy:
        """Start `harness-proxy` as a subprocess advertising `host`; it stops with this process."""
        return await cls._started(
            endpoint_settings, host=host, environ=environ, log=log, seconds=seconds
        )

    @classmethod
    async def tunnelled(
        cls,
        endpoint_settings: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        log: IO[str] | None = None,
    ) -> Proxy:
        """Start the proxy as `local` does, behind a Cloudflare quick tunnel it opens and closes."""
        return await cls._started(
            endpoint_settings, host="127.0.0.1", environ=environ, log=log, tunnel=True
        )

    @classmethod
    async def _started(
        cls,
        endpoint_settings: Mapping[str, Any],
        *,
        host: str,
        environ: Mapping[str, str] | None,
        log: IO[str] | None,
        seconds: float = STARTUP_SECONDS,
        tunnel: bool = False,
    ) -> Proxy:
        environ = dict(os.environ if environ is None else environ)
        token = environ.get(PROXY_TOKEN_ENV) or new_token()
        control = environ.get(CONTROL_TOKEN_ENV) or new_token()
        environ.update({PROXY_TOKEN_ENV: token, CONTROL_TOKEN_ENV: control})
        environ.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            serve_command(endpoint_settings, host=host, tunnel=tunnel),
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            match = await started(
                process, BANNER, seconds=seconds, what="harness-proxy", log=log, pump=not tunnel
            )
            opened = (
                await started(
                    process,
                    TUNNEL_LINE,
                    seconds=TUNNEL_SECONDS + 10.0,
                    what="harness-proxy's tunnel",
                    log=log,
                )
                if tunnel
                else None
            )
        except BaseException:
            end(process)
            raise
        port = int(match.group("port"))
        logger.info("proxy serving %s on port %s as %s", match.group("model"), port, host)
        return cls(
            origin=opened.group("origin") if opened else f"http://{match.group('host')}:{port}",
            token=token,
            control_token=control,
            loopback=f"http://127.0.0.1:{port}",
            process=process,
            tunnel=opened is not None,
        )

    @classmethod
    def remote(cls, url: str, token: str, control_token: str | None = None) -> Proxy:
        """Return a proxy running elsewhere at `url`, reached with `token`."""
        if not url.strip():
            raise ValueError("a remote proxy needs a URL")
        if not token:
            raise ValueError(
                f"{PROXY_TOKEN_ENV} is unset, so the token that harnesses must present to the "
                f"proxy at {url} is unknown"
            )
        return cls(origin=url, token=token, control_token=control_token or None)

    async def ready(self, seconds: float = PROBE_SECONDS) -> str:
        """Wait for `/healthz` to answer, and return the model the proxy serves."""
        what, base = ("tunnel", self.origin) if self.tunnel else ("proxy", self.loopback)
        url = f"{base}{HEALTH_PATH}"
        try:
            answer = await answers(url, seconds=seconds, transport=self.transport)
        except Exception as failed:
            raise Unreachable(
                f"the {what} at {base} does not answer: {type(failed).__name__}: {failed}"
            ) from failed
        budget = answer.get("budget")
        known = isinstance(budget, int) and not isinstance(budget, bool) and budget > 0
        self.budget = budget if known else None
        named = answer.get("renderers")
        self.renderers = [str(one) for one in named] if isinstance(named, list) else None
        return str(answer.get("serving") or answer.get("model") or "")

    def gone(self) -> str | None:
        """Return why a proxy this run started can no longer answer, or None."""
        if self.process is not None and (code := self.process.poll()) is not None:
            return f"harness-proxy exited with {code}"
        return None

    def address_for(self, trial: str, renderer: str | None = None) -> str:
        """Return a trial's base URL on the proxy, naming `renderer` if given."""
        for what, name in (("trial name", trial), ("renderer", renderer)):
            if name is not None and (not name or "/" in name):
                raise ValueError(
                    f"{name!r} cannot be addressed: a {what} goes in the URL path, "
                    "so it must be non-empty and contain no '/'."
                )
        named = f"/renderer/{renderer}" if renderer is not None else ""
        return f"{self.origin}/r/trial/{trial}{named}/v1"

    async def swap(self, model_path: str | None) -> None:
        """Switch the proxy to the named weights, or to the base model for None."""
        if self.control_token is None:
            raise RuntimeError(
                f"the proxy at {self.origin} opened no control route to this run: set "
                f"{CONTROL_TOKEN_ENV} to the token it was started with"
            )
        async with httpx.AsyncClient(timeout=CONTROL_TIMEOUT, transport=self.transport) as client:
            answered = await client.post(
                f"{self.loopback}{CONTROL_PATH}",
                json={"model_path": model_path},
                headers={"Authorization": f"Bearer {self.control_token}"},
            )
        if answered.status_code != 200:
            raise RuntimeError(
                f"the proxy at {self.origin} refused to serve {model_path}: "
                f"{answered.status_code} {answered.text[:200]}"
            )

    async def records(self, trial: str) -> list[Record]:
        """Fetch the trial's records, which drains them, retrying transport errors and 5xx."""
        if self.control_token is None:
            raise RuntimeError(
                f"the proxy at {self.origin} opened no records route to this run: set "
                f"{CONTROL_TOKEN_ENV} to the token it was started with"
            )
        url = f"{self.loopback}{RECORDS_PATH}/{quote(trial, safe='')}"
        params = {"delta": "1"}
        for attempt in range(RECORDS_ATTEMPTS):
            if attempt:
                params["again"] = "1"
            try:
                async with httpx.AsyncClient(
                    timeout=RECORDS_TIMEOUT, transport=self.transport
                ) as client:
                    answered = await client.get(
                        url,
                        params=params,
                        headers={"Authorization": f"Bearer {self.control_token}"},
                    )
                answered.raise_for_status()
                payload = answered.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as failed:
                status = getattr(getattr(failed, "response", None), "status_code", 0)
                last = attempt == RECORDS_ATTEMPTS - 1
                if (isinstance(failed, httpx.HTTPStatusError) and status < 500) or last:
                    raise
                logger.warning(
                    "records of %s: %s on try %d of %d, retrying",
                    trial,
                    type(failed).__name__,
                    attempt + 1,
                    RECORDS_ATTEMPTS,
                )
                await asyncio.sleep(PAUSES[min(attempt, len(PAUSES) - 1)])
                continue
            return records_of(payload.get("records") or [])
        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        """Stop the serve process, if this run started it, which stops its tunnel."""
        process, self.process = self.process, None
        if process is not None:
            end(process, seconds=STOP_SECONDS)


async def answers(
    url: str, *, seconds: float = PROBE_SECONDS, transport: Any = None
) -> dict[str, Any]:
    """Poll `url` until it answers 200 and return its JSON object."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=max(remaining, 0.1), transport=transport
            ) as client:
                answered = await client.get(url)
            answered.raise_for_status()
            payload = answered.json()
            return payload if isinstance(payload, dict) else {}
        except (httpx.HTTPError, ValueError):
            if time.monotonic() + PROBE_PAUSE >= deadline:
                raise
        await asyncio.sleep(PROBE_PAUSE)
