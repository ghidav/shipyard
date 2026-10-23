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
    CONTROL_TOKEN_ENV,
    HEALTH_PATH,
    TRIALS_PATH,
    TUNNEL_LINE,
    Opening,
    Record,
    Served,
    closing_path,
    lanes_in,
    new_token,
    records_in,
)

logger = logging.getLogger(__name__)

#: Command that starts `harness-proxy`. Tests replace it with a stub.
SERVE: list[str] = [sys.executable, "-m", "harness_proxy.serve"]
STARTUP_SECONDS = 180.0
#: Seconds `stop` gives serve to stop its tunnel and close its session before killing it.
STOP_SECONDS = 30.0
#: Attempts at opening or closing a trial, so one network stall to a remote proxy does not
#: lose a trial: the proxy answers an open or a close asked again as it answered the first.
CONTROL_ATTEMPTS = 4
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
    """A proxy as seen from the run: its addresses, the control token that opens and closes its
    trials, and the process it started."""

    origin: str
    control_token: str = field(repr=False)
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

    @property
    def address(self) -> str:
        """The base URL every trial's harness dials, with the token its trial was opened with."""
        return f"{self.origin}/v1"

    @property
    def local_address(self) -> str:
        """The same address from this machine, for a harness whose calls leave from here."""
        return f"{self.loopback}/v1"

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
        control = environ[CONTROL_TOKEN_ENV] = environ.get(CONTROL_TOKEN_ENV) or new_token()
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
            control_token=control,
            loopback=f"http://127.0.0.1:{port}",
            process=process,
            tunnel=opened is not None,
        )

    @classmethod
    def remote(cls, url: str, control_token: str) -> Proxy:
        """Return a proxy running elsewhere at `url`, whose trials the run opens with
        `control_token`."""
        if not url.strip():
            raise ValueError("a remote proxy needs a URL")
        if not control_token:
            raise ValueError(
                f"{CONTROL_TOKEN_ENV} is unset, so the run cannot open trials on the proxy at "
                f"{url}: export the token it was started with"
            )
        return cls(origin=url, control_token=control_token)

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

    async def open(
        self,
        trial: str,
        *,
        lane: str | None = None,
        renderer: str | None = None,
        weights: str | None = None,
    ) -> str:
        """Open `trial` on `lane`, the only one for None, rendered with `renderer`, the lane's
        first for None, and sampled from `weights`, its base model for None; return the token
        its harness calls with. `RuntimeError` if the proxy refuses it."""
        opening = Opening(trial, lane, renderer, weights)
        answer = await self._control(TRIALS_PATH, opening.body, what=f"open trial {trial}")
        return str(answer["token"])

    async def close(self, trial: str) -> list[Record]:
        """Close `trial` and return its records, in request order. `RuntimeError` for a trial
        the proxy does not hold."""
        return records_in(await self._control(closing_path(trial), None, what=f"close {trial}"))

    async def _control(self, path: str, body: Any, *, what: str) -> dict[str, Any]:
        """Post `body` to the control route `path` and return its answer, retrying transport
        errors and 5xx; `RuntimeError` for a refusal."""
        url = f"{self.loopback}{path}"
        for attempt in range(CONTROL_ATTEMPTS):
            try:
                async with httpx.AsyncClient(
                    timeout=CONTROL_TIMEOUT, transport=self.transport
                ) as client:
                    answered = await client.post(
                        url, json=body, headers={"Authorization": f"Bearer {self.control_token}"}
                    )
                answered.raise_for_status()
                payload = answered.json()
            except httpx.HTTPStatusError as failed:
                status = failed.response.status_code
                if status < 500 or attempt == CONTROL_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"the proxy at {self.origin} refused to {what}: {status} "
                        f"{failed.response.text[:300]}"
                    ) from failed
            except httpx.TransportError:
                if attempt == CONTROL_ATTEMPTS - 1:
                    raise
            else:
                return payload if isinstance(payload, dict) else {}
            logger.warning("%s: try %d of %d failed, retrying", what, attempt + 1, CONTROL_ATTEMPTS)
            await asyncio.sleep(PAUSES[min(attempt, len(PAUSES) - 1)])
        raise AssertionError("unreachable")  # pragma: no cover

    def stop(self) -> None:
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
