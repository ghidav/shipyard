"""`harness-proxy`: the proxy as a standalone process, for a run whose sandbox is elsewhere.

It prints one line that the run parses, then serves until SIGTERM or SIGINT. With `--tunnel` it
also opens a Cloudflare quick tunnel to itself and prints its address on a second line. With
`--parent-pid` it stops once that process, the run that started it, is gone, however it ended:
a run killed outright cannot stop it, and the proxy would keep its port, its Tinker session and
its tunnel."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import fields
from typing import Any, TextIO

from harness_proxy.server import Server
from harness_proxy.tunnel import Tunnel
from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, new_token

logger = logging.getLogger(__name__)

#: Bind addresses that mean every interface: valid to listen on, but not to dial.
WILDCARD_BINDS = frozenset({"0.0.0.0", "::", "*", ""})
#: Seconds between checks that the run that started this process, and its tunnel, are there.
WATCH_SECONDS = 1.0
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
    renderer: str | Sequence[str] | None,
    settings: Mapping[str, Any] | None,
    token: str | None,
    control_token: str | None,
) -> Server:
    """Build the server the command-line flags describe.

    `renderer` is a renderer, or the several a trial's address picks from. `settings` carries
    the run's other `Server` settings. Raise ValueError for a key that is not a `Server` field or
    that a flag already sets."""
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
        named["renderer"] = renderer if isinstance(renderer, str) else list(renderer)
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
    """Log the proxy's messages at INFO to `stream`, which a run keeps as `proxy.log`.

    Do nothing if logging is already configured, as it is under the CLI. aiohttp's
    per-request access log stays off."""
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("harness_proxy")
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
    parent: int | None = None,
    tunnel: bool = False,
) -> int:
    """Start the server, print the banner and any tokens generated here, and wait for `stop`.

    With `tunnel`, open a quick tunnel to the server and print `tunnel at <origin>`. With
    `parent`, stop once this process's parent is no longer that process. Stop as well if the
    tunnel exits, since no sandbox can reach the server then. The tunnel, then the server
    (socket and Tinker session), are closed on exit or error. Return 1 if the tunnel exited,
    else 0."""
    stop = asyncio.Event() if stop is None else stop
    await server.start()
    opened: Tunnel | None = None
    watch: asyncio.Task[str | None] | None = None
    ended: str | None = None
    try:
        out(banner(server))
        for name, value in (generated or {}).items():
            out(f"{name} {value}")
        if tunnel:
            opened = await Tunnel.start(server.port, log=sys.stderr)
            out(f"tunnel at {opened.origin}")
        watch = asyncio.create_task(watching(stop, parent=parent, tunnel=opened))
        await stop.wait()
        ended = watch.result() if watch.done() else None
    finally:
        if watch is not None:
            watch.cancel()
        if opened is not None:
            await asyncio.to_thread(opened.stop)
        await server.stop()
    return 1 if ended == "tunnel" else 0


async def watching(
    stop: asyncio.Event,
    *,
    parent: int | None,
    tunnel: Tunnel | None,
    every: float = WATCH_SECONDS,
) -> str | None:
    """Set `stop` once the run that started this process is gone, which hands this process to
    another parent, or once the tunnel has exited. Return which, or None if `stop` was set
    first."""
    while not stop.is_set():
        if parent is not None and os.getppid() != parent:
            logger.warning("the run that started this proxy (pid %d) is gone; stopping", parent)
            stop.set()
            return "parent"
        if tunnel is not None and (code := tunnel.process.poll()) is not None:
            logger.error("the tunnel at %s exited with %s; stopping", tunnel.origin, code)
            stop.set()
            return "tunnel"
        await asyncio.sleep(every)
    return None


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


def parser(prog: str = "harness-proxy") -> argparse.ArgumentParser:
    made = argparse.ArgumentParser(
        prog=prog,
        description="Serve a Tinker model to a harness in a sandbox, recording what it sampled.",
    )
    made.add_argument("--model", required=True, help="the base model the weights are of")
    made.add_argument("--weights", help="a tinker:// path to serve (default: the base model)")
    made.add_argument("--bind", default="0.0.0.0", help="the interface to listen on")
    made.add_argument("--port", type=int, default=0, help="the port to listen on; 0 for any")
    made.add_argument("--advertise", help="the name a sandbox reaches this by (default: the bind)")
    made.add_argument(
        "--renderer",
        metavar="NAME",
        action="append",
        help="a cookbook renderer that overrides the model's default; given again, another, "
        "which a trial's address names to be rendered with it, the first rendering the rest",
    )
    made.add_argument(
        "--tunnel",
        action="store_true",
        help="open a Cloudflare quick tunnel to this proxy, and print `tunnel at <origin>`",
    )
    made.add_argument(
        "--parent-pid",
        type=int,
        metavar="PID",
        help="stop once this process, the one that started the proxy, is gone",
    )
    made.add_argument(
        "--settings",
        type=json.loads,
        metavar="JSON",
        help="the run's other sampling settings as a JSON object: temperature, top_p, "
        "top_k, max_tokens, max_context, fill_context",
    )
    return made


def main(argv: list[str] | None = None, prog: str = "harness-proxy") -> int:
    """Build the server and serve until signalled. Return 2 if the server cannot be built.

    `prog` names the command in help and errors, for a CLI that wraps this one."""
    configure_logging()
    args = parser(prog).parse_args(argv)
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
    return asyncio.run(_until_signalled(server, generated, args.parent_pid, args.tunnel))


async def _until_signalled(
    server: Server, generated: Mapping[str, str], parent: int | None, tunnel: bool
) -> int:
    stop = asyncio.Event()
    with stopping_on_signals(stop):
        return await serve(server, stop=stop, generated=generated, parent=parent, tunnel=tunnel)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
