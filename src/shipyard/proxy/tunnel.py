"""A Cloudflare quick tunnel in front of a proxy on this machine, for a sandbox with no
route back here. It runs `cloudflared` from PATH, else its image through docker. Without
a tunnel such a rollout cannot reach the proxy and records nothing."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import IO, Any

logger = logging.getLogger(__name__)

IMAGE = "cloudflare/cloudflared:latest"
#: The origin a quick tunnel prints once it is up (cloudflared writes it to stderr). It
#: matches one label and excludes `api.`, because the service's own host appears in the
#: error line of a refused request.
ORIGIN = re.compile(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com")
STARTUP_SECONDS = 60.0


def found() -> tuple[str, list[str]] | None:
    """How a tunnel can be opened here: the cloudflared path and its command with a
    `{port}` placeholder, else docker's path and command with the image, else None."""
    binary = shutil.which("cloudflared")
    if binary:
        return binary, [binary, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:{port}"]
    docker = shutil.which("docker")
    if docker:
        # Docker Desktop resolves the name itself. A Linux engine needs it mapped.
        url, host = "http://host.docker.internal:{port}", "host.docker.internal:host-gateway"
        run = [docker, "run", "--rm", f"--add-host={host}", IMAGE]
        return docker, [*run, "tunnel", "--no-autoupdate", "--url", url]
    return None


def command(port: int) -> list[str]:
    """The command that opens a tunnel to `port`. Raises RuntimeError when neither
    cloudflared nor docker is installed."""
    located = found()
    if located is None:
        raise RuntimeError(
            "cloudflared is not installed and neither is docker, so no tunnel can carry a "
            "remote sandbox to this proxy. Install cloudflared or set [rollout] endpoint_url "
            "to a proxy the sandbox reaches."
        )
    return [part.format(port=port) for part in located[1]]


@dataclass
class Tunnel:
    """One `cloudflared` process and the origin it answers at."""

    origin: str
    process: subprocess.Popen[str] = field(repr=False)

    @classmethod
    async def start(
        cls, port: int, *, seconds: float = STARTUP_SECONDS, log: IO[str] | None = None
    ) -> Tunnel:
        """Open a tunnel to `127.0.0.1:<port>` and wait for the first trycloudflare origin
        in its output. The command is logged and written to `log` before the tunnel's lines."""
        made = command(port)
        logger.info("opening a quick tunnel: %s", shlex.join(made))
        if log is not None:
            log.write(f"$ {shlex.join(made)}\n")
            log.flush()
        process = subprocess.Popen(
            made,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            match = await started(process, ORIGIN, seconds=seconds, what="the tunnel", log=log)
        except BaseException:
            end(process)
            raise
        origin = match.group(0).rstrip("/")
        logger.info("tunnel to port %s is up at %s", port, origin)
        return cls(origin=origin, process=process)

    def stop(self) -> None:
        end(self.process)


async def started(
    process: subprocess.Popen[str],
    pattern: re.Pattern[str],
    *,
    seconds: float,
    what: str,
    log: IO[str] | None = None,
) -> re.Match[str]:
    """Return the first stdout line of `process` that matches `pattern`, within `seconds`
    and before the process exits. Later lines are copied to `log` so a full pipe does not
    stall the process."""
    assert process.stdout is not None
    stream, seen, deadline = process.stdout, [], time.monotonic() + seconds
    while True:
        remaining = max(deadline - time.monotonic(), 0)
        try:
            line = await asyncio.wait_for(asyncio.to_thread(stream.readline), remaining)
        except TimeoutError:
            raise TimeoutError(
                f"{what} printed no address within {seconds:g} s:\n{_tail(seen)}"
            ) from None
        if not line:
            code = process.wait()
            raise RuntimeError(
                f"{what} exited with {code} before it printed an address:\n{_tail(seen)}"
            )
        seen.append(line.rstrip("\n"))
        if log is not None:
            log.write(line)
            log.flush()
        match = pattern.search(line.rstrip("\n"))
        if match is not None:
            threading.Thread(target=_pump, args=(stream, log), daemon=True).start()
            return match


def _pump(stream: IO[str], log: IO[str] | None) -> None:
    for line in stream:
        if log is not None:
            log.write(line)
            log.flush()


def _tail(lines: Sequence[str], count: int = 20) -> str:
    return "\n".join(lines[-count:])


def end(process: Any, *, seconds: float = 10.0) -> None:
    """Send SIGTERM, then SIGKILL after `seconds`, so a process that ignores SIGTERM cannot
    hold up the run's exit."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
