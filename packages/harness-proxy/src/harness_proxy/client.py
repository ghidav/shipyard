from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, Any
from urllib.parse import urlsplit

import httpx

# `started` (wait for a matching line) and `end` (stop a process) are defined in tunnel.py
# and also serve the process started here.
from harness_proxy.tunnel import STARTUP_SECONDS as TUNNEL_SECONDS
from harness_proxy.tunnel import end, started
from harness_proxy.wire import (
    BANNER,
    CONTROL_PATH,
    CONTROL_TOKEN_ENV,
    HEALTH_PATH,
    PROXY_TOKEN_ENV,
    TUNNEL_LINE,
    Address,
    Fetch,
    Record,
    Served,
    Swap,
    lanes_in,
    new_token,
    records_in,
    records_path,
)

logger = logging.getLogger(__name__)

#: Command that starts `harness-proxy`. Tests replace it with a stub.
SERVE: list[str] = [sys.executable, "-m", "harness_proxy.serve"]
STARTUP_SECONDS = 180.0
#: Seconds `close` gives serve to stop its tunnel and close its session before killing it.
STOP_SECONDS = 30.0
#: Attempts per records fetch, so one tunnel stall does not lose a trial.
RECORDS_ATTEMPTS = 4
RECORDS_TIMEOUT = 120.0
CONTROL_TIMEOUT = 120.0
PAUSES = (1.0, 2.0, 4.0)
#: Settings passed to serve as flags, which describe the process. Its lanes, and the rest,
#: go in `--settings`.
FLAGGED = ("bind", "bind_port")
#: Seconds the proxy has to answer `/healthz` before the run is refused, and the pause
#: between probes. A quick tunnel's hostname can resolve late, and a resolver that
#: looked it up too early caches the miss for trycloudflare.com's negative TTL of 60 s,
#: so the window is longer than that.
PROBE_SECONDS = 120.0
PROBE_PAUSE = 1.0


class Unreachable(RuntimeError):
    """The proxy does not answer at the address sandboxes will dial, before any sandbox opens."""


def serve_command(settings: Mapping[str, Any], *, host: str, tunnel: bool = False) -> list[str]:
    """Return the `python -m harness_proxy.serve` command for `settings`, its `bind`,
    `bind_port` and `lanes`, with `tunnel` if asked."""
    made = [*SERVE, "--bind", str(settings.get("bind") or "0.0.0.0")]
    made += ["--port", str(int(settings.get("bind_port") or 0)), "--advertise", host]
    made += ["--parent-pid", str(os.getpid()), *(["--tunnel"] if tunnel else [])]
    rest = {key: value for key, value in settings.items() if key not in FLAGGED}
    return [*made, "--settings", json.dumps(rest, sort_keys=True)]


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
    #: The lanes it serves, by the name a trial's address gives each, as `/healthz` last
    #: reported them.
    lanes: dict[str, Served] = field(default_factory=dict)
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
        logger.info("proxy serving %s on port %s as %s", match.group("lanes"), port, host)
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

    async def ready(self, seconds: float = PROBE_SECONDS) -> dict[str, Served]:
        """Wait for `/healthz` to answer, and return the lanes the proxy serves."""
        what, base = ("tunnel", self.origin) if self.tunnel else ("proxy", self.loopback)
        url = f"{base}{HEALTH_PATH}"
        try:
            answer = await answers(url, seconds=seconds, transport=self.transport)
        except Exception as failed:
            raise Unreachable(
                f"the {what} at {base} does not answer: {type(failed).__name__}: {failed}"
            ) from failed
        self.lanes = lanes_in(answer)
        return self.lanes

    def gone(self) -> str | None:
        """Return why a proxy this run started can no longer answer, or None."""
        if self.process is not None and (code := self.process.poll()) is not None:
            return f"harness-proxy exited with {code}"
        return None

    def address_for(self, trial: str, renderer: str | None = None, lane: str | None = None) -> str:
        """Return a trial's base URL on the proxy, naming `renderer` and `lane` if given; a
        proxy with one lane answers an address that names none."""
        return self.origin + Address(trial, lane, renderer).path

    async def swap(self, weights: str | None, lane: str | None = None) -> None:
        """Switch `lane`, or a proxy's one lane, to `weights`, or to its base model for None."""
        if self.control_token is None:
            raise RuntimeError(
                f"the proxy at {self.origin} opened no control route to this run: set "
                f"{CONTROL_TOKEN_ENV} to the token it was started with"
            )
        async with httpx.AsyncClient(timeout=CONTROL_TIMEOUT, transport=self.transport) as client:
            answered = await client.post(
                f"{self.loopback}{CONTROL_PATH}",
                json=Swap(weights, lane).body,
                headers={"Authorization": f"Bearer {self.control_token}"},
            )
        if answered.status_code != 200:
            raise RuntimeError(
                f"the proxy at {self.origin} refused to serve {weights}: "
                f"{answered.status_code} {answered.text[:200]}"
            )

    async def records(self, trial: str) -> list[Record]:
        """Fetch the trial's records, which drains them, retrying transport errors and 5xx."""
        if self.control_token is None:
            raise RuntimeError(
                f"the proxy at {self.origin} opened no records route to this run: set "
                f"{CONTROL_TOKEN_ENV} to the token it was started with"
            )
        url = f"{self.loopback}{records_path(trial)}"
        for attempt in range(RECORDS_ATTEMPTS):
            # A retry asks again for the answer the lost fetch drained.
            params = Fetch(delta=True, again=attempt > 0).query
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
            return records_in(payload)
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
