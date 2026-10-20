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

    base_model: str = "Qwen/Qwen3-8B"
    model_path: str | None = None
    host: str = "proxy-host"
    control_token: str | None = "c"
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
    assert (args.bind, args.port, args.advertise, args.weights, args.renderer) == (
        "0.0.0.0",
        0,
        None,
        None,
        None,
    )
    assert args.settings is None
    parsed = serve.parser().parse_args(["--model", "a/b", "--settings", '{"top_p": 0.5}'])
    assert parsed.settings == {"top_p": 0.5}


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


def build(**overrides: Any) -> Server:
    named: dict[str, Any] = dict(
        weights=None,
        bind="127.0.0.1",
        port=0,
        advertise=None,
        renderer=None,
        settings=None,
        token="k",
        control_token="c",
    )
    return serve.server_for("Qwen/Qwen3-8B", **{**named, **overrides})


def test_the_flags_reach_the_real_endpoint() -> None:
    made = build(weights="tinker://w/step-1", bind="0.0.0.0", port=8000, advertise="proxy")
    assert isinstance(made, Server)
    assert (made.base_model, made.model_path) == ("Qwen/Qwen3-8B", "tinker://w/step-1")
    assert (made.bind, made.bind_port, made.host) == ("0.0.0.0", 8000, "proxy")
    assert (made.token, made.control_token) == ("k", "c")
    assert made.port is None, "nothing is bound yet"
    assert build(renderer="qwen3_disable_thinking").renderer == "qwen3_disable_thinking"
    assert build().renderer is None


def test_an_empty_token_is_replaced_and_no_control_token_closes_the_route() -> None:
    made = build(token="", control_token=None)
    assert made.token and made.control_token is None


def test_settings_carry_the_runs_other_keys_and_nothing_else() -> None:
    made = build(
        settings={
            "temperature": 0.5,
            "max_tokens": 64,
            "max_context": 4096,
            "fill_context": True,
        }
    )
    assert (made.temperature, made.max_tokens, made.max_context) == (0.5, 64, 4096)
    assert made.fill_context is True
    assert build(settings={"metadata": {"run": "r"}}).metadata == {"run": "r"}
    with pytest.raises(ValueError, match="graph_text"):
        build(settings={"graph_text": True})
    with pytest.raises(ValueError, match="bind"):
        build(settings={"bind": "0.0.0.0"})
    with pytest.raises(ValueError, match="client_factory"):
        build(settings={"client_factory": None})


def test_the_tokens_come_from_the_env_or_are_generated_and_said() -> None:
    token, control, generated = serve.tokens_from_env({"HARNESS_PROXY_TOKEN": "t"})
    assert token == "t" and control and generated == {"HARNESS_PROXY_CONTROL_TOKEN": control}
    token, control, generated = serve.tokens_from_env(
        {"HARNESS_PROXY_TOKEN": "", "HARNESS_PROXY_CONTROL_TOKEN": "c"}
    )
    assert token and control == "c" and generated == {"HARNESS_PROXY_TOKEN": token}
    assert serve.tokens_from_env({})[2].keys() == {
        "HARNESS_PROXY_TOKEN",
        "HARNESS_PROXY_CONTROL_TOKEN",
    }


# ------------------------------------------------------------------ what is printed


async def test_the_banner_is_the_first_line_and_the_generated_tokens_follow() -> None:
    fake = FakeEndpoint(model_path="tinker://w/step-1")
    printed = await run_briefly(fake, generated={"HARNESS_PROXY_TOKEN": "made-up"})
    assert printed[0] == "serving tinker://w/step-1 at http://proxy-host:8000 (control: on)"
    assert printed[1] == "HARNESS_PROXY_TOKEN made-up"
    assert fake.called == ["start", "stop"]


async def test_a_token_the_caller_supplied_is_not_printed_back() -> None:
    printed = await run_briefly(FakeEndpoint())
    assert len(printed) == 1 and "control: on" in printed[0]
    assert "control: off" in (await run_briefly(FakeEndpoint(control_token=None)))[0]


async def test_the_banner_names_the_real_endpoints_port() -> None:
    async with server(control_token="c") as started:
        line, port = serve.banner(started), started.port
    assert line == f"serving Qwen/Qwen3-8B at http://127.0.0.1:{port} (control: on)"


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
        "serving Qwen/Qwen3-8B at http://proxy-host:8000 (control: on)",
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
    server.client_factory, server.renderer = Echo.make, FakeRenderer()
    return server


serve.server_for = with_fakes
sys.exit(serve.main(sys.argv[1:]))
"""


def test_serve_runs_for_real_on_a_fake_sampler(tmp_path: Path) -> None:
    """Mirrors the acceptance script: start the process, call both wires, read the records,
    swap weights through the control route, and stop it with SIGTERM."""
    src = str(Path(harness_proxy.__file__).resolve().parent.parent)
    fakes = str(Path(__file__).resolve().parent)
    env = {
        **os.environ,
        "HARNESS_PROXY_TOKEN": "harness-token",
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
        assert line.startswith("serving Qwen/Qwen3-8B at http://127.0.0.1:")
        assert line.endswith("(control: on)")
        origin = line.split(" at ", 1)[1].split(" ", 1)[0]
        with httpx.Client(base_url=origin, timeout=10) as client:
            chat = client.post(
                "/r/trial/t-1/v1/chat/completions",
                headers={"Authorization": "Bearer harness-token"},
                json={"model": "m", "messages": [{"role": "user", "content": "echo"}]},
            )
            assert chat.status_code == 200, chat.text
            assert chat.json()["choices"][0]["message"]["content"] == "echo"
            messages = client.post(
                "/r/trial/t-1/v1/messages",
                headers={"x-api-key": "harness-token"},
                json={
                    "model": "m",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert messages.status_code == 200, messages.text
            assert messages.json()["content"][0]["text"] == "hi"
            swapped = client.post(
                "/control/weights",
                headers={"Authorization": "Bearer control-token"},
                json={"model_path": "tinker://run/sampler_weights/step-3"},
            )
            assert swapped.status_code == 200 and swapped.json()["serving"].endswith("step-3")
            later = client.post(
                "/r/trial/t-1/v1/chat/completions",
                headers={"Authorization": "Bearer harness-token"},
                json={"model": "m", "messages": [{"role": "user", "content": "again"}]},
            )
            assert later.json()["model"] == "tinker://run/sampler_weights/step-3"
            fetched = client.get(
                "/records/t-1",
                params={"delta": "1"},
                headers={"Authorization": "Bearer control-token"},
            )
            assert fetched.status_code == 200
        records = records_of(fetched.json()["records"])
        assert [r.seq for r in records] == [1, 2, 3]
        assert [tuple(r.completion_token_ids) for r in records] == [
            tuple(map(ord, "echo")),
            tuple(map(ord, "hi")),
            tuple(map(ord, "again")),
        ]
        assert [r.served for r in records] == [None, None, "tinker://run/sampler_weights/step-3"]
        assert sum(len(r.completion_token_ids) for r in records) == len("echohiagain")
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
    proxy = await client.Proxy.tunnelled({"model": "Qwen/Qwen3-8B", "bind": "127.0.0.1"})
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
