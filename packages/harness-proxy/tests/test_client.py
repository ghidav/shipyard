"""Tests for how a run places its proxy: a local subprocess, a tunnel, or a remote URL."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from harness_proxy import client
from harness_proxy.client import CONTROL_TOKEN_ENV, Proxy, serve_command
from harness_proxy.wire import Opening

SERVE_STUB = textwrap.dedent(
    """
    import json, os, signal, sys, time
    from pathlib import Path

    from harness_proxy.wire import banner, tunnel_line

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
    named = [one["name"] for one in json.loads(flag("--settings"))["lanes"]]
    print(banner(named, flag("--advertise"), int(port)), flush=True)
    print("serving on", file=sys.stderr, flush=True)
    if "--tunnel" in args:
        if mode == "no-tunnel":
            print("RuntimeError: cloudflared is not installed", file=sys.stderr, flush=True)
            sys.exit(1)
        print("INF |  https://fake-words.trycloudflare.com  |", file=sys.stderr, flush=True)
        print(tunnel_line("https://fake-words.trycloudflare.com"), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
    """
)
SETTINGS = {
    "lanes": [{"name": "agent", "model": "Qwen/Qwen3-8B"}],
    "bind": "127.0.0.1",
    "bind_port": 0,
}


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


def test_the_serve_command_carries_the_process_as_flags_and_the_lanes_as_settings() -> None:
    lanes = [{"name": "agent", "model": "a/b", "renderer": "q"}]
    made = serve_command(
        {"lanes": lanes, "bind": "0.0.0.0", "bind_port": 8000, "keepalive_every": 5.0},
        host="proxy",
    )
    assert made[: len(client.SERVE)] == client.SERVE
    rest = made[len(client.SERVE) :]
    assert rest[:8] == [
        "--bind",
        "0.0.0.0",
        "--port",
        "8000",
        "--advertise",
        "proxy",
        "--parent-pid",
        str(os.getpid()),
    ]
    assert rest[8] == "--settings" and json.loads(rest[9]) == {
        "lanes": lanes,
        "keepalive_every": 5.0,
    }
    bare = serve_command({"lanes": lanes}, host="h", tunnel=True)[len(client.SERVE) :]
    assert bare[:9] == [
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


async def test_local_reads_the_port_off_the_first_line_and_hands_the_control_token_down(
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
            assert proxy.address == "http://host.docker.internal:4242/v1"
            assert proxy.local_address == "http://127.0.0.1:4242/v1"
            seen = json.loads(stub.read_text())
            assert proxy.control_token and seen["env"] == {CONTROL_TOKEN_ENV: proxy.control_token}
            assert seen["argv"][:6] == [
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
                "--advertise",
                "host.docker.internal",
            ]
            assert proxy.process is not None and proxy.process.poll() is None
        finally:
            proxy.stop()
    assert proxy.process is None
    assert "warming up" in log.read_text() and "serving agent" in log.read_text()


async def test_the_control_token_from_the_environment_is_the_one_handed_down(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = await Proxy.local(SETTINGS, environ={**os.environ, CONTROL_TOKEN_ENV: "c"})
    try:
        assert proxy.control_token == "c"
        assert json.loads(stub.read_text())["env"] == {CONTROL_TOKEN_ENV: "c"}
    finally:
        proxy.stop()


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
        assert proxy.address == "https://fake-words.trycloudflare.com/v1"
        assert proxy.local_address == "http://127.0.0.1:5151/v1"
        argv = json.loads(stub.read_text())["argv"]
        assert argv[argv.index("--advertise") + 1] == "127.0.0.1" and "--tunnel" in argv
        assert argv[argv.index("--parent-pid") + 1] == str(os.getpid())
        process = proxy.process
    finally:
        proxy.stop()
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


def test_remote_is_addressed_by_url_and_needs_the_control_token() -> None:
    proxy = Proxy.remote("https://proxy.example/", "c")
    assert proxy.origin == proxy.loopback == "https://proxy.example"
    assert proxy.address == proxy.local_address == "https://proxy.example/v1"
    assert proxy.control_token == "c" and proxy.process is None and not proxy.tunnel
    with pytest.raises(ValueError, match=CONTROL_TOKEN_ENV):
        Proxy.remote("https://proxy.example", "")
    with pytest.raises(ValueError, match="URL"):
        Proxy.remote("  ", "c")


@pytest.mark.parametrize("name", ["", "org/name"])
async def test_a_trial_name_the_close_route_cannot_hold_is_refused_before_anything_is_sent(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be a trial's name"):
        Opening(name)
    with pytest.raises(ValueError, match="cannot be a trial's name"):
        await Proxy.remote("https://proxy.example", "c").open(name)


async def test_a_remote_proxy_is_stopped_without_stopping_anything() -> None:
    proxy = Proxy.remote("https://proxy.example", "c")
    proxy.stop()
    proxy.stop()
