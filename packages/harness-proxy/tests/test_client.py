"""Tests for how a run places its proxy: a local subprocess, a tunnel, or a remote URL."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from harness_proxy import client
from harness_proxy.client import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Proxy, serve_command

SERVE_STUB = textwrap.dedent(
    """
    import json, os, signal, sys, time
    from pathlib import Path

    args = sys.argv[1:]
    flag = lambda name: args[args.index(name) + 1] if name in args else None
    mode = os.environ.get("STUB_MODE", "serve")
    if os.environ.get("STUB_OUT"):
        env = {k: v for k, v in os.environ.items() if k.startswith("HARNESS_PROXY_")}
        Path(os.environ["STUB_OUT"]).write_text(json.dumps({"argv": args, "env": env}))
    print("warming up", file=sys.stderr, flush=True)
    if mode == "exit":
        print("could not bind", file=sys.stderr, flush=True)
        sys.exit(3)
    if mode == "silent":
        time.sleep(30)
        sys.exit(0)
    port = os.environ.get("STUB_PORT", "8765")
    line = f"serving {flag('--model')} at http://{flag('--advertise')}:{port} (control: on)"
    print(line, flush=True)
    print("serving on", file=sys.stderr, flush=True)
    if "--tunnel" in args:
        if mode == "no-tunnel":
            print("RuntimeError: cloudflared is not installed", file=sys.stderr, flush=True)
            sys.exit(1)
        print("INF |  https://fake-words.trycloudflare.com  |", file=sys.stderr, flush=True)
        print("tunnel at https://fake-words.trycloudflare.com", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
    """
)
SETTINGS = {"model": "Qwen/Qwen3-8B", "weights": None, "bind": "127.0.0.1", "bind_port": 0}


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace `harness-proxy` with a script that prints its banner and waits for SIGTERM.
    Returns the path where the script writes its arguments and environment."""
    script = tmp_path / "serve_stub.py"
    script.write_text(SERVE_STUB, encoding="utf-8")
    monkeypatch.setattr(client, "SERVE", [sys.executable, str(script)])
    monkeypatch.setenv("STUB_OUT", str(tmp_path / "stub.json"))
    return tmp_path / "stub.json"


# ------------------------------------------------------------------- the command


def test_the_serve_command_carries_the_flags_and_the_rest_as_settings() -> None:
    made = serve_command(
        {
            "model": "a/b",
            "weights": "tinker://w/step-1",
            "renderer": "qwen3",
            "bind": "0.0.0.0",
            "bind_port": 8000,
            "temperature": 0.5,
            "max_context": 4096,
        },
        host="proxy",
    )
    assert made[: len(client.SERVE)] == client.SERVE
    rest = made[len(client.SERVE) :]
    assert rest[:10] == [
        "--model",
        "a/b",
        "--bind",
        "0.0.0.0",
        "--port",
        "8000",
        "--advertise",
        "proxy",
        "--parent-pid",
        str(os.getpid()),
    ]
    assert rest[10:14] == ["--weights", "tinker://w/step-1", "--renderer", "qwen3"]
    assert rest[14] == "--settings" and json.loads(rest[15]) == {
        "temperature": 0.5,
        "max_context": 4096,
    }
    bare = serve_command({"model": "a/b"}, host="h", tunnel=True)[len(client.SERVE) :]
    assert bare == [
        "--model",
        "a/b",
        "--bind",
        "0.0.0.0",
        "--port",
        "0",
        "--advertise",
        "h",
        "--parent-pid",
        str(os.getpid()),
        "--tunnel",
    ]


# ------------------------------------------------------------------ a local proxy


async def test_local_reads_the_port_off_the_first_line_and_hands_the_tokens_down(
    stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_PORT", "4242")
    log = tmp_path / "proxy.log"
    with log.open("a", encoding="utf-8") as sink:
        proxy = await Proxy.local(SETTINGS, host="host.docker.internal", log=sink)
        try:
            assert proxy.origin == "http://host.docker.internal:4242"
            assert proxy.loopback == "http://127.0.0.1:4242"
            assert proxy.host == "host.docker.internal"
            assert proxy.address_for("t-1") == "http://host.docker.internal:4242/r/trial/t-1/v1"
            assert proxy.token and proxy.control_token and proxy.token != proxy.control_token
            seen = json.loads(stub.read_text())
            assert seen["env"][PROXY_TOKEN_ENV] == proxy.token
            assert seen["env"][CONTROL_TOKEN_ENV] == proxy.control_token
            assert seen["argv"][:6] == [
                "--model",
                "Qwen/Qwen3-8B",
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
            ]
            assert proxy.process is not None and proxy.process.poll() is None
        finally:
            proxy.close()
    assert proxy.process is None
    assert "warming up" in log.read_text() and "serving Qwen/Qwen3-8B" in log.read_text()


async def test_the_tokens_from_the_environment_are_the_ones_handed_down(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = {**os.environ, PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    proxy = await Proxy.local(SETTINGS, environ=environ)
    try:
        assert (proxy.token, proxy.control_token) == ("t", "c")
        assert json.loads(stub.read_text())["env"] == {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    finally:
        proxy.close()


async def test_a_serve_that_exits_before_its_line_raises_with_what_it_said(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "exit")
    with pytest.raises(RuntimeError, match="exited with 3") as caught:
        await Proxy.local(SETTINGS)
    assert "could not bind" in str(caught.value)


async def test_a_serve_that_never_prints_is_given_up_on_and_stopped(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "silent")
    with pytest.raises(TimeoutError, match="harness-proxy"):
        await Proxy.local(SETTINGS, seconds=0.5)


# ---------------------------------------------------------------- behind a tunnel


async def test_tunnelled_serves_on_loopback_and_answers_at_the_tunnel_it_opened(
    stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_PORT", "5151")
    log = (tmp_path / "proxy.log").open("w", encoding="utf-8")
    proxy = await Proxy.tunnelled(SETTINGS, log=log)
    try:
        assert proxy.origin == "https://fake-words.trycloudflare.com" and proxy.tunnel
        assert proxy.loopback == "http://127.0.0.1:5151"
        assert proxy.host == "fake-words.trycloudflare.com"
        assert proxy.address_for("t") == "https://fake-words.trycloudflare.com/r/trial/t/v1"
        argv = json.loads(stub.read_text())["argv"]
        assert argv[argv.index("--advertise") + 1] == "127.0.0.1" and "--tunnel" in argv
        assert argv[argv.index("--parent-pid") + 1] == str(os.getpid())
        process = proxy.process
    finally:
        proxy.close()
        log.close()
    assert process is not None and process.poll() is not None
    assert "trycloudflare.com  |" in (tmp_path / "proxy.log").read_text(), "its output is kept"


async def test_a_tunnel_that_cannot_open_stops_the_serve_it_fronted(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "no-tunnel")
    with pytest.raises(RuntimeError, match="cloudflared is not installed"):
        await Proxy.tunnelled(SETTINGS)


# ------------------------------------------------------------------- a remote one


def test_remote_is_addressed_by_url_and_needs_the_token() -> None:
    proxy = Proxy.remote("https://proxy.example/", "k")
    assert proxy.origin == proxy.loopback == "https://proxy.example"
    assert proxy.address_for("t-1") == "https://proxy.example/r/trial/t-1/v1"
    assert proxy.control_token is None and proxy.process is None and not proxy.tunnel
    assert Proxy.remote("https://proxy.example", "k", "c").control_token == "c"
    with pytest.raises(ValueError, match=PROXY_TOKEN_ENV):
        Proxy.remote("https://proxy.example", "")
    with pytest.raises(ValueError, match="URL"):
        Proxy.remote("  ", "k")


@pytest.mark.parametrize("name", ["", "org/name"])
def test_a_trial_name_that_cannot_ride_in_the_path_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        Proxy.remote("https://proxy.example", "k").address_for(name)


async def test_a_remote_proxy_is_closed_without_stopping_anything() -> None:
    proxy = Proxy.remote("https://proxy.example", "k")
    proxy.close()
    proxy.close()
