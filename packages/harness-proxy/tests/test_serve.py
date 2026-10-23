"""Tests for `harness-proxy`, the process, including an end-to-end run on a fake sampler."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import harness_proxy
from harness_proxy import serve
from harness_proxy.server import Server
from harness_proxy.wire import records_of
from proxies import server


@dataclass
class FakeEndpoint:
    """Fake `Server` with only the attributes `serve` uses. `called` records start and stop
    calls in order."""

    lanes: list[Any] = field(default_factory=lambda: [SimpleNamespace(name="agent")])
    host: str = "proxy-host"
    port: int | None = None
    called: list[str] = field(default_factory=list)

    async def start(self) -> None:
        self.port = 8000
        self.called.append("start")

    async def stop(self) -> None:
        self.port = None
        self.called.append("stop")


async def run_briefly(fake: Any, **overrides: Any) -> list[str]:
    stop = asyncio.Event()
    stop.set()
    printed: list[str] = []
    await serve.serve(fake, out=printed.append, stop=stop, **overrides)
    return printed


@dataclass
class FakeTunnel:
    """Fake `Tunnel` whose process runs while `code` is None, and has exited with it otherwise.
    Its stop is recorded in `called`, which the endpoint shares."""

    called: list[str]
    code: int | None = None
    origin: str = "https://fake-words.trycloudflare.com"

    @property
    def process(self) -> Any:
        return SimpleNamespace(poll=lambda: self.code)

    def stop(self) -> None:
        self.called.append("tunnel stop")


def opening(monkeypatch: pytest.MonkeyPatch, made: FakeTunnel) -> list[int]:
    """Make `Tunnel.start` return `made`, and return the ports it is asked for."""
    asked: list[int] = []

    async def start(port: int, *, log: Any = None) -> FakeTunnel:
        asked.append(port)
        return made

    monkeypatch.setattr(serve.Tunnel, "start", start)
    return asked


# ----------------------------------------------------------------- what is advertised


def test_a_wildcard_bind_with_nothing_named_is_refused() -> None:
    with pytest.raises(ValueError, match="--advertise"):
        serve.advertised("0.0.0.0", None)
    assert serve.advertised("10.0.0.4", None) == "10.0.0.4"
    assert serve.advertised("0.0.0.0", "proxy-host") == "proxy-host"


def test_the_flags_default_to_every_interface_and_any_port() -> None:
    args = serve.parser().parse_args(["--model", "a/b"])
    assert (args.bind, args.port, args.advertise, args.renderer, args.lane) == (
        "0.0.0.0",
        0,
        None,
        None,
        "agent",
    )
    assert args.settings is None
    parsed = serve.parser().parse_args(["--settings", '{"lanes": []}'])
    assert parsed.settings == {"lanes": []} and parsed.model is None


def test_main_says_what_is_wrong_rather_than_serving_nowhere(capsys: Any) -> None:
    assert serve.main(["--model", "a/b"]) == 2
    assert "--advertise" in capsys.readouterr().out


def test_main_configures_the_proxys_logging_unless_it_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.StringIO()
    root, ours = logging.getLogger(), logging.getLogger("harness_proxy")
    level, handlers = ours.level, list(ours.handlers)
    monkeypatch.setattr(root, "handlers", [])
    try:
        serve.configure_logging(out)
        logging.getLogger("harness_proxy.server").info("swapped to %s", "w")
        logging.getLogger("aiohttp.access").info("GET /v1/chat/completions")
    finally:
        ours.handlers[:] = handlers
        ours.setLevel(level)
    assert out.getvalue() == "swapped to w\n"
    monkeypatch.setattr(root, "handlers", [logging.NullHandler()])
    serve.configure_logging(out)
    assert ours.handlers == handlers, "configured already, as under the CLI's verb"


# ------------------------------------------------------------------- the server


def build(*, renderer: str | None = None, **overrides: Any) -> Server:
    """Build the server `harness-proxy --model Qwen/Qwen3-8B` and the other flags describe."""
    given = serve.document(None, lane="agent", model="Qwen/Qwen3-8B", renderer=renderer)
    named: dict[str, Any] = dict(bind="127.0.0.1", port=0, advertise=None, control_token="c")
    return serve.server_for(given, **{**named, **overrides})


def test_the_flags_reach_the_real_endpoint() -> None:
    made = build(bind="0.0.0.0", port=8000, advertise="proxy")
    assert isinstance(made, Server)
    (lane,) = made.lanes
    assert (lane.name, lane.model) == ("agent", "Qwen/Qwen3-8B")
    assert (made.bind, made.bind_port, made.host, made.control_token) == (
        "0.0.0.0",
        8000,
        "proxy",
        "c",
    )
    assert made.port is None, "nothing is bound yet"
    assert build(renderer="qwen3_disable_thinking").lanes[0].renderer == "qwen3_disable_thinking"
    assert build().lanes[0].renderer is None


def test_an_empty_control_token_is_replaced() -> None:
    assert build(control_token=None).control_token


def test_a_lane_carries_its_own_settings_and_nothing_else() -> None:
    def lane(**settings: Any) -> Any:
        given = {"lanes": [{"name": "agent", "model": "Qwen/Qwen3-8B", **settings}]}
        named = dict(bind="127.0.0.1", port=0, advertise=None, control_token="c")
        (made,) = serve.server_for(given, **named).lanes
        return made

    made = lane(temperature=0.5, max_tokens=64, max_context=4096, fill_context=True)
    assert (made.temperature, made.max_tokens, made.max_context) == (0.5, 64, 4096)
    assert made.fill_context is True
    assert lane(metadata={"run": "r"}).metadata == {"run": "r"}
    for foreign in ("graph_text", "bind", "client_factory", "weights"):
        with pytest.raises(ValueError, match=foreign):
            lane(**{foreign: None})


def test_the_control_token_comes_from_the_env_or_is_generated_and_said() -> None:
    assert serve.control_from_env({"HARNESS_PROXY_CONTROL_TOKEN": "c"}) == ("c", {})
    control, generated = serve.control_from_env({})
    assert control and generated == {"HARNESS_PROXY_CONTROL_TOKEN": control}


# ------------------------------------------------------------------ what is printed


async def test_the_banner_is_the_first_line_and_the_generated_tokens_follow() -> None:
    fake = FakeEndpoint(lanes=[SimpleNamespace(name="agent"), SimpleNamespace(name="teacher")])
    printed = await run_briefly(fake, generated={"HARNESS_PROXY_CONTROL_TOKEN": "made-up"})
    assert printed[0] == "serving agent,teacher at http://proxy-host:8000"
    assert printed[1] == "HARNESS_PROXY_CONTROL_TOKEN made-up"
    assert fake.called == ["start", "stop"]


async def test_a_token_the_caller_supplied_is_not_printed_back() -> None:
    assert await run_briefly(FakeEndpoint()) == ["serving agent at http://proxy-host:8000"]


async def test_the_banner_names_the_real_endpoints_port() -> None:
    async with server(control_token="c") as started:
        line, port = serve.banner(started), started.port
    assert line == f"serving agent at http://127.0.0.1:{port}"


async def test_the_endpoint_closes_even_when_printing_raises() -> None:
    fake = FakeEndpoint()

    def fail(_line: str) -> None:
        raise RuntimeError("the log went away")

    with pytest.raises(RuntimeError, match="the log went away"):
        await serve.serve(fake, out=fail)
    assert fake.called == ["start", "stop"]


# --------------------------------------------------------------- stopping


async def test_a_sigterm_is_what_ends_it_and_the_handlers_come_off_again() -> None:
    stop = asyncio.Event()
    with serve.stopping_on_signals(stop):
        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(stop.wait(), timeout=2)
    assert stop.is_set()
    assert asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM) is False


async def test_it_serves_while_the_run_that_started_it_is_there_and_stops_once_it_is_gone() -> None:
    fake, stop = FakeEndpoint(), asyncio.Event()
    serving = asyncio.create_task(serve.serve(fake, out=[].append, stop=stop, parent=os.getppid()))
    await asyncio.sleep(0.2)
    assert not serving.done(), "its parent is still the run"
    stop.set()
    assert await serving == 0
    handed_on = FakeEndpoint()
    # A parent other than its own: the run is gone and the process was handed on.
    gone = serve.serve(handed_on, out=[].append, parent=os.getppid() + 1)
    assert await asyncio.wait_for(gone, timeout=5) == 0
    assert handed_on.called == ["start", "stop"]


async def test_a_tunnel_opens_after_the_banner_and_closes_before_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEndpoint()
    asked = opening(monkeypatch, FakeTunnel(fake.called))
    printed = await run_briefly(fake, tunnel=True)
    assert printed == [
        "serving agent at http://proxy-host:8000",
        "tunnel at https://fake-words.trycloudflare.com",
    ]
    assert asked == [8000] and fake.called == ["start", "tunnel stop", "stop"]


async def test_a_tunnel_that_exits_stops_the_proxy_with_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeEndpoint()
    opening(monkeypatch, FakeTunnel(fake.called, code=1))
    status = await asyncio.wait_for(serve.serve(fake, out=[].append, tunnel=True), timeout=5)
    assert status == 1 and fake.called == ["start", "tunnel stop", "stop"]


# ------------------------------------------------------------- the whole process


def _first_line(process: subprocess.Popen[str], seconds: float = 30.0) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            return line.rstrip("\n")
        if process.poll() is not None:
            break
    raise AssertionError(
        f"serve printed nothing; stderr: {process.stderr.read() if process.stderr else ''}"
    )


#: Script for `python -c` in the process test. It runs `serve.main` with an echoing sampler
#: and the tests' character renderer, so neither Tinker nor a tokenizer is needed.
STUB = """\
import sys
from harness_proxy import serve
from proxies import Echo, FakeRenderer

built = serve.server_for


def with_fakes(*args, **kwargs):
    server = built(*args, **kwargs)
    for lane in server.lanes:
        lane.client_factory, lane.renderer = Echo.make, FakeRenderer()
    return server


serve.server_for = with_fakes
sys.exit(serve.main(sys.argv[1:]))
"""


def test_serve_runs_for_real_on_a_fake_sampler(tmp_path: Path) -> None:
    """Mirrors the acceptance script: start the process, open a trial and call both wires, open
    another on other weights beside it, close both for their records, and stop it with
    SIGTERM."""
    src = str(Path(harness_proxy.__file__).resolve().parent.parent)
    fakes = str(Path(__file__).resolve().parent)
    env = {
        **os.environ,
        "HARNESS_PROXY_CONTROL_TOKEN": "control-token",
        "PYTHONPATH": os.pathsep.join([src, fakes, os.environ.get("PYTHONPATH", "")]),
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", STUB, "--model", "Qwen/Qwen3-8B", "--bind", "127.0.0.1"],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _first_line(process)
        assert line.startswith("serving agent at http://127.0.0.1:")
        origin = line.split(" at ", 1)[1]
        control = {"Authorization": "Bearer control-token"}
        with httpx.Client(base_url=origin, timeout=10) as client:
            token = client.post("/trials", headers=control, json={"trial": "t-1"}).json()["token"]
            chat = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "m", "messages": [{"role": "user", "content": "echo"}]},
            )
            assert chat.status_code == 200, chat.text
            assert chat.json()["choices"][0]["message"]["content"] == "echo"
            messages = client.post(
                "/v1/messages",
                headers={"x-api-key": token},
                json={
                    "model": "m",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert messages.status_code == 200, messages.text
            assert messages.json()["content"][0]["text"] == "hi"
            weights = "tinker://run/sampler_weights/step-3"
            other = client.post(
                "/trials", headers=control, json={"trial": "t-2", "weights": weights}
            ).json()["token"]
            later = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {other}"},
                json={"model": "m", "messages": [{"role": "user", "content": "again"}]},
            )
            assert later.json()["model"] == weights
            first = client.post("/trials/t-1/close", headers=control)
            second = client.post("/trials/t-2/close", headers=control)
            assert first.status_code == second.status_code == 200
        records = records_of(first.json()["records"])
        assert [r.seq for r in records] == [1, 2]
        assert [tuple(r.completion_token_ids) for r in records] == [
            tuple(map(ord, "echo")),
            tuple(map(ord, "hi")),
        ]
        assert [r.weights for r in records] == [None, None]
        (again,) = records_of(second.json()["records"])
        assert (again.completion_token_ids, again.weights) == (tuple(map(ord, "again")), weights)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


#: A `cloudflared` that writes its pid where `STUB_TUNNEL_PID` says, prints a quick tunnel's
#: address as the real one does, and runs until SIGTERM.
CLOUDFLARED = """\
#!/bin/sh
echo $$ > "$STUB_TUNNEL_PID"
echo "INF |  https://orphan-check.trycloudflare.com  |" >&2
trap 'exit 0' TERM
while true; do sleep 0.1; done
"""
#: A run, as far as its proxy can tell: it starts a tunnelled proxy, prints the serve process's
#: pid, and waits. `argv[1]` is the serve command's script.
OWNER = """\
import asyncio, sys
from harness_proxy import client

client.SERVE = [sys.executable, "-c", sys.argv[1]]


async def main():
    lanes = [{"name": "agent", "model": "Qwen/Qwen3-8B"}]
    proxy = await client.Proxy.tunnelled({"lanes": lanes, "bind": "127.0.0.1"})
    print(proxy.process.pid, flush=True)
    await asyncio.sleep(600)


asyncio.run(main())
"""


def _alive(pid: int) -> bool:
    """Return whether `pid` runs: it exists and is not a zombie waiting to be reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def test_a_run_killed_outright_takes_its_proxy_and_tunnel_with_it(tmp_path: Path) -> None:
    """SIGKILL gives the run no chance to stop what it started: serve notices it is gone and
    stops, closing the tunnel it opened."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cloudflared").write_text(CLOUDFLARED, encoding="utf-8")
    (bin_dir / "cloudflared").chmod(0o755)
    src = str(Path(harness_proxy.__file__).resolve().parent.parent)
    fakes = str(Path(__file__).resolve().parent)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "STUB_TUNNEL_PID": str(tmp_path / "tunnel.pid"),
        "PYTHONPATH": os.pathsep.join([src, fakes, os.environ.get("PYTHONPATH", "")]),
        "PYTHONUNBUFFERED": "1",
    }
    owner = subprocess.Popen(
        [sys.executable, "-c", OWNER, STUB],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids: list[int] = []
    try:
        pids = [int(_first_line(owner)), int((tmp_path / "tunnel.pid").read_text())]
        assert all(_alive(pid) for pid in pids)
        owner.kill()
        owner.wait()
        deadline = time.monotonic() + 10
        while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(_alive(pid) for pid in pids), "serve or its tunnel outlived the run"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
