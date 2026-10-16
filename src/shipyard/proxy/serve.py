"""`shipyard serve`: the proxy as a standalone process, for a run whose sandbox is elsewhere.

It prints one line that the run parses, then serves until SIGTERM or SIGINT."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields
from typing import Any, TextIO

from shipyard.proxy.server import Server
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, new_token

#: Bind addresses that mean every interface: valid to listen on, but not to dial.
WILDCARD_BINDS = frozenset({"0.0.0.0", "::", "*", ""})
#: Server fields set by flags, which `--settings` may not name.
SET_HERE = frozenset(
    {"base_model", "model_path", "renderer", "host", "bind", "bind_port", "token"}
    | {"control_token", "client_factory"}
)


def advertised(bind: str, advertise: str | None) -> str:
    """Return the hostname a sandbox uses to reach this proxy.

    Raise ValueError for a wildcard bind without `--advertise`, because a URL with 0.0.0.0
    reaches nothing and every harness request would fail."""
    if advertise:
        return advertise
    if bind in WILDCARD_BINDS:
        raise ValueError(
            f"A bind of {bind!r} listens on every interface, and a sandbox cannot dial that. "
            "Name how this proxy is reached with --advertise: host.docker.internal when its "
            "port is published on the machine the sandbox's Docker runs on, or its hostname."
        )
    return bind


def server_for(
    model: str,
    *,
    weights: str | None,
    bind: str,
    port: int,
    advertise: str | None,
    renderer: str | None,
    settings: Mapping[str, Any] | None,
    token: str | None,
    control_token: str | None,
) -> Server:
    """Build the server the command-line flags describe.

    `settings` carries the run's other `[rollout]` keys. Raise ValueError for a key that is
    not a `Server` field or that a flag already sets."""
    named: dict[str, Any] = {
        "base_model": model,
        "model_path": weights or None,
        "host": advertised(bind, advertise),
        "bind": bind,
        "bind_port": int(port),
        "token": token or "",
        "control_token": control_token or None,
    }
    if renderer:
        named["renderer"] = renderer
    if settings:
        taken = {f.name for f in fields(Server) if not f.name.startswith("_")} - SET_HERE
        foreign = sorted(set(settings) - taken)
        if foreign:
            raise ValueError(
                f"--settings names {', '.join(foreign)}, which this server does not take "
                f"from a run. It takes {', '.join(sorted(taken))}."
            )
        named.update(settings)
    return Server(**named)


def configure_logging(stream: TextIO = sys.stderr) -> None:
    """Log shipyard's messages at INFO to `stream`, which a run keeps as `proxy.log`.

    Do nothing if logging is already configured, as it is under the CLI. aiohttp's
    per-request access log stays off."""
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("shipyard")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def banner(server: Server) -> str:
    """Return the first line serve prints, which the run parses.

    It names the served model, the address, and whether the control route is on."""
    served = server.model_path or server.base_model
    control = "on" if server.control_token else "off"
    return f"serving {served} at http://{server.host}:{server.port} (control: {control})"


def say(line: str) -> None:
    """Print `line` to stdout and flush, because a pipe is block-buffered."""
    print(line, flush=True)


@contextlib.contextmanager
def stopping_on_signals(stop: asyncio.Event) -> Iterator[None]:
    """Set `stop` on SIGINT or SIGTERM while the context is active.

    The handlers are installed on the event loop so the wait wakes up, and are removed on exit
    for a caller that keeps running."""
    loop = asyncio.get_running_loop()
    installed = []
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(number, stop.set)
            installed.append(number)
        yield
    finally:
        for number in installed:
            loop.remove_signal_handler(number)


async def serve(
    server: Server,
    *,
    out: Callable[[str], None] = say,
    stop: asyncio.Event | None = None,
    generated: Mapping[str, str] | None = None,
) -> None:
    """Start the server, print the banner and any tokens generated here, and wait for `stop`.

    The server (socket and Tinker session) is closed on exit or error."""
    stop = asyncio.Event() if stop is None else stop
    await server.start()
    try:
        out(banner(server))
        for name, value in (generated or {}).items():
            out(f"{name} {value}")
        await stop.wait()
    finally:
        await server.stop()


def tokens_from_env(environ: Mapping[str, str] = os.environ) -> tuple[str, str, dict[str, str]]:
    """Return `(token, control, generated)` for the harness and control tokens.

    A token missing from the environment is generated here and also listed in `generated`,
    so the caller can print it for the launcher to read."""
    generated: dict[str, str] = {}
    token = environ.get(PROXY_TOKEN_ENV) or ""
    if not token:
        token = generated[PROXY_TOKEN_ENV] = new_token()
    control = environ.get(CONTROL_TOKEN_ENV) or ""
    if not control:
        control = generated[CONTROL_TOKEN_ENV] = new_token()
    return token, control, generated


def parser() -> argparse.ArgumentParser:
    made = argparse.ArgumentParser(
        prog="shipyard serve",
        description="Serve a Tinker model to a harness in a sandbox, recording what it sampled.",
    )
    made.add_argument("--model", required=True, help="the base model the weights are of")
    made.add_argument("--weights", help="a tinker:// path to serve (default: the base model)")
    made.add_argument("--bind", default="0.0.0.0", help="the interface to listen on")
    made.add_argument("--port", type=int, default=0, help="the port to listen on; 0 for any")
    made.add_argument("--advertise", help="the name a sandbox reaches this by (default: the bind)")
    made.add_argument(
        "--renderer", metavar="NAME", help="a cookbook renderer that overrides the model's default"
    )
    made.add_argument(
        "--settings",
        type=json.loads,
        metavar="JSON",
        help="the run's other [rollout] endpoint keys as a JSON object: temperature, top_p, "
        "top_k, max_tokens, max_context, fill_context",
    )
    return made


def main(argv: list[str] | None = None) -> int:
    """Build the server and serve until signalled. Return 2 if the server cannot be built."""
    configure_logging()
    args = parser().parse_args(argv)
    token, control, generated = tokens_from_env()
    try:
        server = server_for(
            args.model,
            weights=args.weights,
            bind=args.bind,
            port=args.port,
            advertise=args.advertise,
            renderer=args.renderer,
            settings=args.settings,
            token=token,
            control_token=control,
        )
    except ValueError as refused:
        say(str(refused))
        return 2
    asyncio.run(_until_signalled(server, generated))
    return 0


async def _until_signalled(server: Server, generated: Mapping[str, str]) -> None:
    stop = asyncio.Event()
    with stopping_on_signals(stop):
        await serve(server, stop=stop, generated=generated)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
