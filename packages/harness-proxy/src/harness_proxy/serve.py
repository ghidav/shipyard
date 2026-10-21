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

from harness_proxy import wire
from harness_proxy.server import Lane, Server
from harness_proxy.tunnel import Tunnel
from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, new_token

logger = logging.getLogger(__name__)

#: Bind addresses that mean every interface: valid to listen on, but not to dial.
WILDCARD_BINDS = frozenset({"0.0.0.0", "::", "*", ""})
#: Seconds between checks that the run that started this process, and its tunnel, are there.
WATCH_SECONDS = 1.0
#: The lane the flags start a proxy with, unless `--lane` names another: the run's agent's.
LANE = "agent"
#: The server's own settings a run's `--settings` may give, beside its `lanes`.
SERVER_SETTINGS = ("keepalive_grace", "keepalive_every")
#: What a lane's entry in `--settings` may name: a lane's every field but a test's client.
LANE_KEYS = frozenset(f.name for f in fields(Lane) if not f.name.startswith("_")) - {
    "client_factory"
}


def advertised(bind: str, advertise: str | None) -> str:
    """Return the hostname a sandbox uses to reach this proxy; ValueError for a bare wildcard."""
    if advertise:
        return advertise
    if bind in WILDCARD_BINDS:
        raise ValueError(
            f"A bind of {bind!r} listens on every interface, and a sandbox cannot dial that. "
            "Name how this proxy is reached with --advertise: host.docker.internal when its "
            "port is published on the machine the sandbox's Docker runs on, or its hostname."
        )
    return bind


def document(
    settings: Mapping[str, Any] | None,
    *,
    lane: str,
    model: str | None,
    weights: str | None,
    renderer: str | Sequence[str] | None,
) -> dict[str, Any]:
    """Return the run's `settings`, or, for a proxy started by hand with `--model`, the
    document of its one lane, `lane`, which serves `weights` with `renderer`."""
    given = dict(settings or {})
    if not model:
        if weights or renderer:
            raise ValueError("--weights and --renderer describe the lane of a --model: name it")
        return given
    if "lanes" in given:
        raise ValueError(
            "--model starts the proxy with one lane, and --settings lists its lanes: give one"
        )
    one: dict[str, Any] = {"name": lane, "model": model}
    if weights:
        one["weights"] = weights
    if renderer:
        one["renderer"] = renderer if isinstance(renderer, str) else list(renderer)
    return {**given, "lanes": [one]}


def server_for(
    given: Mapping[str, Any],
    *,
    bind: str,
    port: int,
    advertise: str | None,
    token: str | None,
    control_token: str | None,
) -> Server:
    """Build the server the command-line flags and the run's document, `given`, describe: its
    `lanes`, and any of the server's own settings."""
    foreign = sorted(set(given) - {"lanes", *SERVER_SETTINGS})
    if foreign:
        laned = " A lane's settings go in its entry of lanes." if set(foreign) & LANE_KEYS else ""
        raise ValueError(
            f"--settings names {', '.join(foreign)}, which this server does not take from a "
            f"run. It takes lanes, {', '.join(SERVER_SETTINGS)}.{laned}"
        )
    return Server(
        lanes=lanes_of(given.get("lanes")),
        host=advertised(bind, advertise),
        bind=bind,
        bind_port=int(port),
        token=token or "",
        control_token=control_token or None,
        **{key: given[key] for key in SERVER_SETTINGS if key in given},
    )


def lanes_of(given: Any) -> list[Lane]:
    """Build the lanes `--settings` lists, each complete: a `name`, a `model`, and any other
    setting of its own, such as its `weights`, `renderer`, `temperature` or `backend`."""
    if not isinstance(given, list) or not given:
        raise ValueError(
            "a proxy serves a --model, or the lanes --settings lists, each with a name and a model"
        )
    found: list[Lane] = []
    for one in given:
        if not isinstance(one, Mapping) or not one.get("name") or not one.get("model"):
            raise ValueError("--settings lanes: each lane has a name and a model")
        foreign = sorted(set(one) - LANE_KEYS)
        if foreign:
            raise ValueError(
                f"--settings lanes: {one['name']} names {', '.join(foreign)}, which a lane "
                f"does not take. It takes {', '.join(sorted(LANE_KEYS))}."
            )
        found.append(Lane(**one))
    return found


def configure_logging(stream: TextIO = sys.stderr) -> None:
    """Log the proxy's messages at INFO to `stream`, which a run keeps as `proxy.log`."""
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("harness_proxy")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def banner(server: Server) -> str:
    """Return the first line serve prints, which the run parses: the lanes, and the address."""
    named = [lane.name for lane in server.lanes]
    return wire.banner(named, server.host, server.port, control=bool(server.control_token))


def say(line: str) -> None:
    """Print `line` to stdout and flush, because a pipe is block-buffered."""
    print(line, flush=True)


@contextlib.contextmanager
def stopping_on_signals(stop: asyncio.Event) -> Iterator[None]:
    """Set `stop` on SIGINT or SIGTERM while the context is active."""
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
    """Serve until `stop`, behind a tunnel if asked; return 1 if the tunnel exited, else 0."""
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
            out(wire.tunnel_line(opened.origin))
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
    """Set `stop` once the run that started this process is gone or the tunnel exits; say which."""
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
    """Return `(token, control, generated)`: the tokens from the environment, else generated."""
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
    made.add_argument(
        "--model",
        help="the base model of the one lane this proxy serves; or list lanes in --settings",
    )
    made.add_argument("--weights", help="a tinker:// path to serve (default: the base model)")
    made.add_argument(
        "--lane", default=LANE, help=f"the name an address gives that lane (default: {LANE})"
    )
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
        help="the run's document, as a JSON object: its lanes, each a name, a model and its "
        "own settings (weights, renderer, temperature, top_p, top_k, max_tokens, max_context, "
        "fill_context, metadata, backend, session); and keepalive_grace, keepalive_every",
    )
    return made


def main(argv: list[str] | None = None, prog: str = "harness-proxy") -> int:
    """Build the server and serve until signalled; return 2 if the server cannot be built."""
    configure_logging()
    args = parser(prog).parse_args(argv)
    token, control, generated = tokens_from_env()
    try:
        given = document(
            args.settings,
            lane=args.lane,
            model=args.model,
            weights=args.weights,
            renderer=args.renderer,
        )
        server = server_for(
            given,
            bind=args.bind,
            port=args.port,
            advertise=args.advertise,
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
