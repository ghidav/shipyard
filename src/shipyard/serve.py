"""`shipyard serve` runs the proxy as its own process, for a run whose sandbox is not on
this machine. It prints one line a run parses, then serves until SIGTERM or SIGINT."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields
from typing import Any

from shipyard.proxy.endpoint import Endpoint
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, new_token

#: Test-only: serve on a fake sampling client and renderer, with no Tinker and no tokenizer.
FAKE_SAMPLER_ENV = "SHIPYARD_FAKE_SAMPLER"
#: Bind values that mean every interface. Fine to listen on, but not an address to dial.
WILDCARD_BINDS = frozenset({"0.0.0.0", "::", "*", ""})
#: Endpoint fields set by flags. `--settings` may not name them.
SET_HERE = frozenset(
    {"base_model", "model_path", "renderer", "host", "bind", "bind_port", "token"}
    | {"control_token", "client_factory"}
)


def advertised(bind: str, advertise: str | None) -> str:
    """The name a sandbox resolves this proxy by. A wildcard bind with no `--advertise`
    raises, because a URL with 0.0.0.0 reaches nothing and every harness request fails."""
    if advertise:
        return advertise
    if bind in WILDCARD_BINDS:
        raise ValueError(
            f"A bind of {bind!r} listens on every interface, and a sandbox cannot dial that. "
            "Name how this proxy is reached with --advertise: host.docker.internal when its "
            "port is published on the machine the sandbox's Docker runs on, or its hostname."
        )
    return bind


def endpoint_for(
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
) -> Endpoint:
    """Build the endpoint the flags describe. `--settings` carries the run's other
    `[rollout]` keys. A key that is not an Endpoint field, or that a flag already sets,
    raises ValueError."""
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
        taken = {f.name for f in fields(Endpoint) if not f.name.startswith("_")} - SET_HERE
        foreign = sorted(set(settings) - taken)
        if foreign:
            raise ValueError(
                f"--settings names {', '.join(foreign)}, which this endpoint does not take "
                f"from a run. It takes {', '.join(sorted(taken))}."
            )
        named.update(settings)
        if "volatile" in named:
            named["volatile"] = tuple(named["volatile"] or ())
    if os.environ.get(FAKE_SAMPLER_ENV) == "1":
        named["client_factory"], named["renderer"] = fake_parts()
    return Endpoint(**named)


def fake_parts() -> tuple[Callable[[str | None], Any], Any]:
    """A sampling client that echoes the prompt's tokens as the completion, and a renderer
    with one character per token. A scripted test can run serve with them, without Tinker."""
    from types import SimpleNamespace

    import tinker

    from shipyard.proxy.cookbook import ParseTermination

    class Echo:
        def __init__(self, path: str | None) -> None:
            self.path = path

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            tokens = list(prompt.to_ints())[: int(sampling_params.max_tokens or 0) or None]
            return SimpleNamespace(
                sequences=[
                    SimpleNamespace(tokens=tokens, logprobs=[0.0] * len(tokens), stop_reason="stop")
                    for _ in range(num_samples)
                ],
                prompt_cache_hit_tokens=0,
            )

    class Chars:
        def get_stop_sequences(self) -> list[int]:
            return []

        def build_generation_prompt(self, messages: list[dict[str, Any]]) -> Any:
            text = "".join(str(message.get("content") or "") for message in messages)
            return tinker.ModelInput.from_ints([ord(character) for character in text])

        def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
            text = "".join(chr(token) for token in response)
            return {"role": "assistant", "content": text}, ParseTermination.STOP_SEQUENCE

    async def make(path: str | None) -> Echo:
        return Echo(path)

    return make, Chars()


def banner(endpoint: Endpoint) -> str:
    """The first line serve prints, which a run parses: what is served, where, and
    whether control is on."""
    served = endpoint.model_path or endpoint.base_model
    control = "on" if endpoint.control_token else "off"
    return f"serving {served} at http://{endpoint.host}:{endpoint.port} (control: {control})"


def say(line: str) -> None:
    """Print to stdout and flush, because a pipe is block-buffered."""
    print(line, flush=True)


@contextlib.contextmanager
def stopping_on_signals(stop: asyncio.Event) -> Iterator[None]:
    """Set `stop` on SIGINT or SIGTERM while entered. The handlers go through the event
    loop so the wait wakes, and are removed on exit for a caller that keeps running."""
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
    endpoint: Endpoint,
    *,
    out: Callable[[str], None] = say,
    stop: asyncio.Event | None = None,
    generated: Mapping[str, str] | None = None,
) -> None:
    """Start the endpoint, print the banner and then each token generated here, wait for
    `stop`, and close the endpoint (socket and Tinker session) on exit or error."""
    stop = asyncio.Event() if stop is None else stop
    await endpoint.start()
    try:
        out(banner(endpoint))
        for name, value in (generated or {}).items():
            out(f"{name} {value}")
        await stop.wait()
    finally:
        await endpoint.stop()


def tokens_from_env(environ: Mapping[str, str] = os.environ) -> tuple[str, str, dict[str, str]]:
    """Return `(token, control, generated)`. A token comes from the env, else it is
    generated here and listed in `generated` for the caller to print, so the launcher can
    read it."""
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
        "top_k, max_tokens, max_context, fill_context, volatile",
    )
    return made


def main(argv: list[str] | None = None) -> int:
    """Build the endpoint and serve until signalled. Returns 2 when it cannot be built."""
    args = parser().parse_args(argv)
    token, control, generated = tokens_from_env()
    try:
        endpoint = endpoint_for(
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
    asyncio.run(_until_signalled(endpoint, generated))
    return 0


async def _until_signalled(endpoint: Endpoint, generated: Mapping[str, str]) -> None:
    stop = asyncio.Event()
    with stopping_on_signals(stop):
        await serve(endpoint, stop=stop, generated=generated)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
