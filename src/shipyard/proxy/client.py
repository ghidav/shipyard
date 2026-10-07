"""The run's side of the proxy, whichever placement: `shipyard serve` as a subprocess on
this machine, the same behind a Cloudflare tunnel, or one running elsewhere. A fetch
that fails raises; an empty answer is reserved for a trial that asked for nothing."""

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

# The subprocess helpers (the first matching line, the end) are tunnel.py's, shared with
# the serve process started here.
from shipyard.proxy.tunnel import Tunnel, end, started
from shipyard.proxy.wire import (
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

#: How `shipyard serve` is started; a test puts its stub here.
SERVE: list[str] = [sys.executable, "-m", "shipyard.serve"]
#: The one line serve prints first, which names the port the kernel chose.
BANNER = re.compile(
    r"^serving (?P<model>\S+) at http://(?P<host>[^:/]+):(?P<port>\d+) "
    r"\(control: (?P<control>on|off)\)$"
)
STARTUP_SECONDS = 180.0
#: A records fetch is tried this many times; a tunnel that stalls once is not a lost trial.
RECORDS_ATTEMPTS = 4
RECORDS_TIMEOUT = 120.0
CONTROL_TIMEOUT = 120.0
PAUSES = (1.0, 2.0, 4.0)
#: What the serve command takes as flags; every other setting rides in `--settings`.
FLAGGED = ("model", "weights", "renderer", "bind", "bind_port")
#: How long the proxy has to answer `/healthz` before the run is refused, and the pause
#: between asks: a quick tunnel's name resolves a little late.
PROBE_SECONDS = 30.0
PROBE_PAUSE = 1.0


class Unreachable(RuntimeError):
    """A proxy that does not answer where its sandboxes will dial it, before one is opened."""


def serve_command(settings: Mapping[str, Any], *, host: str) -> list[str]:
    """`python -m shipyard.serve` as the settings describe it: `model`, `weights`,
    `renderer`, `bind` and `bind_port` as flags, everything else as `--settings` JSON."""
    made = [*SERVE, "--model", str(settings["model"])]
    made += ["--bind", str(settings.get("bind") or "0.0.0.0")]
    made += ["--port", str(int(settings.get("bind_port") or 0)), "--advertise", host]
    if settings.get("weights"):
        made += ["--weights", str(settings["weights"])]
    if settings.get("renderer"):
        made += ["--renderer", str(settings["renderer"])]
    rest = {key: value for key, value in settings.items() if key not in FLAGGED}
    if rest:
        made += ["--settings", json.dumps(rest, sort_keys=True)]
    return made


@dataclass
class Proxy:
    """What the run holds: the origin a sandbox dials, the loopback origin this process
    dials for records and control, the harness token and, when it is ours, the control
    token and the processes to stop."""

    origin: str
    token: str = field(repr=False)
    control_token: str | None = field(default=None, repr=False)
    loopback: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    tunnel: Tunnel | None = field(default=None, repr=False)
    #: What each fetch brought beside the records, by trial: turned_away, cut, spoke.
    counters: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    #: A test's transport for the HTTP client; None is httpx's own.
    transport: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.origin = self.origin.rstrip("/")
        self.loopback = (self.loopback or self.origin).rstrip("/")

    @property
    def host(self) -> str:
        """The name a sandbox resolves this proxy by, for an allowlist."""
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
        """`shipyard serve` as a subprocess advertising `host`, its port read off its first
        line; the tokens from the environment, or made here and handed to it."""
        environ = dict(os.environ if environ is None else environ)
        token = environ.get(PROXY_TOKEN_ENV) or new_token()
        control = environ.get(CONTROL_TOKEN_ENV) or new_token()
        environ.update({PROXY_TOKEN_ENV: token, CONTROL_TOKEN_ENV: control})
        environ.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            serve_command(endpoint_settings, host=host),
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            match = await started(process, BANNER, seconds=seconds, what="shipyard serve", log=log)
        except BaseException:
            end(process)
            raise
        port = int(match.group("port"))
        logger.info("proxy serving %s on port %s as %s", match.group("model"), port, host)
        return cls(
            origin=f"http://{match.group('host')}:{port}",
            token=token,
            control_token=control,
            loopback=f"http://127.0.0.1:{port}",
            process=process,
        )

    @classmethod
    async def tunnelled(
        cls,
        endpoint_settings: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        log: IO[str] | None = None,
        tunnel_log: IO[str] | None = None,
    ) -> Proxy:
        """`local` on loopback with a quick tunnel in front; the origin is the tunnel's."""
        made = await cls.local(endpoint_settings, host="127.0.0.1", environ=environ, log=log)
        try:
            tunnel = await Tunnel.start(
                int(urlsplit(made.loopback or "").port or 0), log=tunnel_log
            )
        except BaseException:
            made.close()
            raise
        made.origin, made.tunnel = tunnel.origin, tunnel
        return made

    @classmethod
    def remote(cls, url: str, token: str, control_token: str | None = None) -> Proxy:
        """A proxy somebody else runs at `url`, behind `token`; nothing of it is stopped."""
        if not url.strip():
            raise ValueError("a remote proxy needs a URL")
        if not token:
            raise ValueError(
                f"{PROXY_TOKEN_ENV} is unset, so nothing here knows the token the harnesses "
                f"must present to the proxy at {url}"
            )
        return cls(origin=url, token=token, control_token=control_token or None)

    async def ready(self, seconds: float = PROBE_SECONDS) -> str:
        """What the proxy serves, read off `/healthz` through its tunnel when it has one,
        else on loopback (a remote one's origin); `Unreachable` names the address when
        nothing answers, before a sandbox is opened to dial it."""
        what, base = ("tunnel", self.origin) if self.tunnel else ("proxy", self.loopback)
        url = f"{base}{HEALTH_PATH}"
        try:
            answer = await answers(url, seconds=seconds, transport=self.transport)
        except Exception as failed:
            raise Unreachable(
                f"the {what} at {base} does not answer: {type(failed).__name__}: {failed}"
            ) from failed
        return str(answer.get("serving") or answer.get("model") or "")

    def gone(self) -> str | None:
        """Why a proxy this run started can no longer answer (its tunnel or its serve process
        exited), or None; one somebody else runs is never known to be gone."""
        if self.tunnel is not None and (code := self.tunnel.process.poll()) is not None:
            return f"the tunnel at {self.origin} exited with {code}"
        if self.process is not None and (code := self.process.poll()) is not None:
            return f"shipyard serve exited with {code}"
        return None

    def address_for(self, trial: str) -> str:
        """`<origin>/r/trial/<trial>/v1`; a name with `/` would file the tokens elsewhere."""
        if not trial or "/" in trial:
            raise ValueError(
                f"A trial name rides in the URL path, so {trial!r} cannot be addressed: "
                "it must be non-empty and free of '/'."
            )
        return f"{self.origin}/r/trial/{trial}/v1"

    async def swap(self, model_path: str | None) -> None:
        """Point the proxy at a `tinker://` path, or at its base for None, through the
        control route; refused without the control token, since the harness's opens nothing."""
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
        """This trial's records, drained from the proxy. Up to `RECORDS_ATTEMPTS` tries over
        a transport error or a 5xx, `again=1` after the first so a drained answer that was
        lost on the way is re-sent; a 4xx is not retried; the last failure raises."""
        url = f"{self.loopback}{RECORDS_PATH}/{quote(trial, safe='')}"
        params = {"delta": "1"}
        for attempt in range(RECORDS_ATTEMPTS):
            if attempt:
                params["again"] = "1"
            try:
                async with httpx.AsyncClient(
                    timeout=RECORDS_TIMEOUT, transport=self.transport
                ) as client:
                    answered = await client.get(url, params=params, headers=self._bearer())
                answered.raise_for_status()
                payload = answered.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as failed:
                status = getattr(getattr(failed, "response", None), "status_code", 0)
                last = attempt == RECORDS_ATTEMPTS - 1
                if (isinstance(failed, httpx.HTTPStatusError) and status < 500) or last:
                    raise
                logger.warning(
                    "records of %s: %s on try %d of %d; asking again for the same answer",
                    trial,
                    type(failed).__name__,
                    attempt + 1,
                    RECORDS_ATTEMPTS,
                )
                await asyncio.sleep(PAUSES[min(attempt, len(PAUSES) - 1)])
                continue
            self.counters[trial] = {
                key: int(payload.get(key) or 0) for key in ("turned_away", "cut", "spoke")
            }
            return records_of(payload.get("records") or [])
        raise AssertionError("unreachable")  # pragma: no cover

    def _bearer(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def close(self) -> None:
        """Stop what this run started: the tunnel, then the serve process. Synchronous, so
        a run closing outside its loop can still call it."""
        tunnel, self.tunnel = self.tunnel, None
        if tunnel is not None:
            tunnel.stop()
        process, self.process = self.process, None
        if process is not None:
            end(process)


async def answers(
    url: str, *, seconds: float = PROBE_SECONDS, transport: Any = None
) -> dict[str, Any]:
    """`url`'s JSON once it answers 200, asked again every `PROBE_PAUSE` until `seconds`
    have passed; the last failure is raised. A name that does not resolve yet is retried."""
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
