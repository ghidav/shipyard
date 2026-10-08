"""The run's client of the proxy. The proxy is a `shipyard serve` subprocess on this
machine, the same behind a Cloudflare tunnel, or one running elsewhere. A failed fetch
raises. An empty answer means the trial made no requests."""

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

#: The command that starts `shipyard serve`. Tests replace it with a stub.
SERVE: list[str] = [sys.executable, "-m", "shipyard.serve"]
#: The first line serve prints. It names the port the kernel chose.
BANNER = re.compile(
    r"^serving (?P<model>\S+) at http://(?P<host>[^:/]+):(?P<port>\d+) "
    r"\(control: (?P<control>on|off)\)$"
)
STARTUP_SECONDS = 180.0
#: Attempts per records fetch, so one tunnel stall does not lose a trial.
RECORDS_ATTEMPTS = 4
RECORDS_TIMEOUT = 120.0
CONTROL_TIMEOUT = 120.0
PAUSES = (1.0, 2.0, 4.0)
#: Settings passed to serve as flags. All other settings go in `--settings`.
FLAGGED = ("model", "weights", "renderer", "bind", "bind_port")
#: Seconds the proxy has to answer `/healthz` before the run is refused, and the pause
#: between probes. A quick tunnel's name can resolve late.
PROBE_SECONDS = 30.0
PROBE_PAUSE = 1.0


class Unreachable(RuntimeError):
    """The proxy does not answer at the address sandboxes will dial. Raised before any
    sandbox opens."""


def serve_command(settings: Mapping[str, Any], *, host: str) -> list[str]:
    """The `python -m shipyard.serve` command for `settings`. `model`, `weights`,
    `renderer`, `bind` and `bind_port` become flags. The rest go in `--settings` as JSON."""
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
    """A proxy as the run sees it: the origin a sandbox dials, the loopback origin this
    process dials for records and control, the harness token and, for a proxy this run
    started, the control token and the processes to stop."""

    origin: str
    token: str = field(repr=False)
    control_token: str | None = field(default=None, repr=False)
    loopback: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    tunnel: Tunnel | None = field(default=None, repr=False)
    #: Per trial, the counts each fetch returned with the records: turned_away, cut, spoke.
    counters: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    #: Tokens a trial may sample, as `/healthz` last reported. None when no budget is enforced.
    budget: int | None = None
    #: Transport for the HTTP client, set by tests. None uses httpx's default.
    transport: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.origin = self.origin.rstrip("/")
        self.loopback = (self.loopback or self.origin).rstrip("/")

    @property
    def host(self) -> str:
        """The hostname a sandbox resolves for this proxy, for an allowlist."""
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
        """Start `shipyard serve` as a subprocess advertising `host`. The port comes from its
        first output line. Tokens come from the environment, or are generated here and
        passed to it."""
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
        """`local` on loopback behind a quick tunnel. The origin is the tunnel's."""
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
        """A proxy run elsewhere at `url`, behind `token`. `close` stops nothing."""
        if not url.strip():
            raise ValueError("a remote proxy needs a URL")
        if not token:
            raise ValueError(
                f"{PROXY_TOKEN_ENV} is unset, so the token that harnesses must present to the "
                f"proxy at {url} is unknown"
            )
        return cls(origin=url, token=token, control_token=control_token or None)

    async def ready(self, seconds: float = PROBE_SECONDS) -> str:
        """Return what the proxy serves, read from `/healthz` through its tunnel if it has
        one, else on loopback (a remote proxy's origin). Keeps the token budget in `budget`.
        Raises `Unreachable` with the address when nothing answers."""
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
        return str(answer.get("serving") or answer.get("model") or "")

    def gone(self) -> str | None:
        """Why a proxy this run started can no longer answer (its tunnel or serve process
        exited), or None. A proxy run elsewhere always gives None."""
        if self.tunnel is not None and (code := self.tunnel.process.poll()) is not None:
            return f"the tunnel at {self.origin} exited with {code}"
        if self.process is not None and (code := self.process.poll()) is not None:
            return f"shipyard serve exited with {code}"
        return None

    def address_for(self, trial: str) -> str:
        """`<origin>/r/trial/<trial>/v1`. Raises ValueError for a name that is empty or
        contains `/`."""
        if not trial or "/" in trial:
            raise ValueError(
                f"{trial!r} cannot be addressed: a trial name goes in the URL path, "
                "so it must be non-empty and contain no '/'."
            )
        return f"{self.origin}/r/trial/{trial}/v1"

    async def swap(self, model_path: str | None) -> None:
        """Point the proxy at a `tinker://` path, or at the base model for None, through the
        control route. Raises without the control token, which the harness token cannot
        replace."""
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
        """Fetch and drain this trial's records from the proxy. Retries up to
        `RECORDS_ATTEMPTS` times on a transport error or a 5xx, with `again=1` after the
        first attempt so the proxy re-sends an answer lost in transit. A 4xx is not
        retried. The last failure raises."""
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
                    "records of %s: %s on try %d of %d, retrying",
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
        """Stop the tunnel, then the serve process, if this run started them. Synchronous,
        so it can be called outside the event loop."""
        tunnel, self.tunnel = self.tunnel, None
        if tunnel is not None:
            tunnel.stop()
        process, self.process = self.process, None
        if process is not None:
            end(process)


async def answers(
    url: str, *, seconds: float = PROBE_SECONDS, transport: Any = None
) -> dict[str, Any]:
    """Return `url`'s JSON once it answers 200. Retries every `PROBE_PAUSE`, including
    when the name does not resolve yet, until `seconds` have passed. Then raises the last
    failure."""
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
