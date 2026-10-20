"""Cloudflare quick tunnel to a local proxy, for sandboxes with no route back to this machine.

It runs `cloudflared` from PATH, or else its image through Docker, in a container named for the
process that opened it so that stopping the tunnel removes it. The serve process opens the
tunnel and closes it when it stops. Without a tunnel, such a rollout cannot reach the proxy and
records nothing."""

from __future__ import annotations

import asyncio
import logging
import os
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
#: Origin a quick tunnel prints once it is up (cloudflared writes it to stderr). It
#: matches a single label and excludes `api.`, because the service's own host appears in
#: the error line of a refused request.
ORIGIN = re.compile(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com")
STARTUP_SECONDS = 60.0


def found() -> tuple[str, list[str]] | None:
    """Return how a tunnel can be opened on this machine, or None.

    The result is the cloudflared binary and its command, else the docker binary and a command
    that runs the cloudflared image. The command has a `{port}` placeholder, and Docker's a
    `{name}` one for its container."""
    binary = shutil.which("cloudflared")
    if binary:
        return binary, [binary, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:{port}"]
    docker = shutil.which("docker")
    if docker:
        # Docker Desktop resolves the name itself. A Linux engine needs it mapped.
        url, host = "http://host.docker.internal:{port}", "host.docker.internal:host-gateway"
        run = [docker, "run", "--rm", "--name", "{name}", f"--add-host={host}", IMAGE]
        return docker, [*run, "tunnel", "--no-autoupdate", "--url", url]
    return None


def command(port: int, name: str = "") -> list[str]:
    """Return the command that opens a tunnel to `port`, in a container named `name` when it
    runs through Docker.

    Raise RuntimeError when neither cloudflared nor docker is installed."""
    located = found()
    if located is None:
        raise RuntimeError(
            "cloudflared is not installed and neither is docker, so no tunnel can carry a "
            "remote sandbox to this proxy. Install cloudflared or set [rollout] endpoint_url "
            "to a proxy the sandbox reaches."
        )
    return [part.format(port=port, name=name) for part in located[1]]


@dataclass
class Tunnel:
    """A running `cloudflared` process and the origin it answers at, and its Docker container's
    name when it runs in one."""

    origin: str
    process: subprocess.Popen[str] = field(repr=False)
    container: str | None = None

    @classmethod
    async def start(
        cls, port: int, *, seconds: float = STARTUP_SECONDS, log: IO[str] | None = None
    ) -> Tunnel:
        """Open a tunnel to `127.0.0.1:<port>` and wait for the trycloudflare origin it prints.

        The command is logged, and written to `log` before the tunnel's output."""
        name = f"harness-proxy-tunnel-{os.getpid()}-{port}"
        made = command(port, name)
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
        return cls(origin=origin, process=process, container=name if name in made else None)

    def stop(self) -> None:
        """Stop the tunnel, and remove its container: the Docker daemon keeps a container
        running whatever becomes of the `docker run` that started it."""
        end(self.process)
        if self.container is None:
            return
        try:
            subprocess.run(
                [shutil.which("docker") or "docker", "rm", "--force", self.container],
                capture_output=True,
                timeout=STARTUP_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("could not remove the tunnel's container %s", self.container)


async def started(
    process: subprocess.Popen[str],
    pattern: re.Pattern[str],
    *,
    seconds: float,
    what: str,
    log: IO[str] | None = None,
    pump: bool = True,
) -> re.Match[str]:
    """Wait for a stdout line of `process` that matches `pattern`, and return the match.

    Raise TimeoutError if none appears within `seconds`, or RuntimeError if the process exits
    first. Every line is copied to `log`, and after the match a background thread keeps
    draining the output so a full pipe does not stall the process; without `pump`, the caller
    reads on, and waits for another line."""
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
            if pump:
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
    """Terminate `process` with SIGTERM, then SIGKILL if it is still running after `seconds`.

    This keeps a process that ignores SIGTERM from holding up the run's exit."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
