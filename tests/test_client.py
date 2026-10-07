"""The run's side of the proxy, placed: `shipyard serve` started as a subprocess and read
off its first line (a stub stands in for it), a tunnel in front of it (a stub cloudflared),
and a remote one by URL. The fetch, the probe and the swap are in test_client_records."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from shipyard.proxy import client
from shipyard.proxy.client import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Proxy, serve_command

SERVE_STUB = textwrap.dedent(
    """
    import json, os, signal, sys, time
    from pathlib import Path

    args = sys.argv[1:]
    flag = lambda name: args[args.index(name) + 1] if name in args else None
    mode = os.environ.get("STUB_MODE", "serve")
    if os.environ.get("STUB_OUT"):
        env = {k: v for k, v in os.environ.items() if k.startswith("SHIPYARD_")}
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
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
    """
)
CLOUDFLARED_STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$@" > "$STUB_ARGS"
    echo "2026-10-07 INF Thank you for trying Cloudflare Tunnel." >&2
    echo "2026-10-07 INF +------------------------------------------+" >&2
    echo "2026-10-07 INF |  https://fake-words.trycloudflare.com  |" >&2
    echo "2026-10-07 INF +------------------------------------------+" >&2
    trap 'exit 0' TERM
    while true; do sleep 0.1; done
    """
)
SETTINGS = {"model": "Qwen/Qwen3-8B", "weights": None, "bind": "127.0.0.1", "bind_port": 0}


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`shipyard serve` replaced by a script printing the banner and waiting for SIGTERM."""
    script = tmp_path / "serve_stub.py"
    script.write_text(SERVE_STUB, encoding="utf-8")
    monkeypatch.setattr(client, "SERVE", [sys.executable, str(script)])
    monkeypatch.setenv("STUB_OUT", str(tmp_path / "stub.json"))
    return tmp_path / "stub.json"


@pytest.fixture
def cloudflared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `cloudflared` on PATH that prints a quick tunnel's origin the way the real one does."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "cloudflared"
    script.write_text(CLOUDFLARED_STUB, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("STUB_ARGS", str(tmp_path / "cloudflared.args"))
    return tmp_path / "cloudflared.args"


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
            "volatile": ["x"],
        },
        host="proxy",
    )
    assert made[: len(client.SERVE)] == client.SERVE
    rest = made[len(client.SERVE) :]
    assert rest[:8] == [
        "--model",
        "a/b",
        "--bind",
        "0.0.0.0",
        "--port",
        "8000",
        "--advertise",
        "proxy",
    ]
    assert rest[8:12] == ["--weights", "tinker://w/step-1", "--renderer", "qwen3"]
    assert rest[12] == "--settings" and json.loads(rest[13]) == {
        "temperature": 0.5,
        "volatile": ["x"],
    }
    bare = serve_command({"model": "a/b"}, host="h")[len(client.SERVE) :]
    assert bare == ["--model", "a/b", "--bind", "0.0.0.0", "--port", "0", "--advertise", "h"]


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
    with pytest.raises(TimeoutError, match="shipyard serve"):
        await Proxy.local(SETTINGS, seconds=0.5)


# ---------------------------------------------------------------- behind a tunnel


async def test_tunnelled_serves_on_loopback_and_answers_at_the_tunnels_origin(
    stub: Path, cloudflared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_PORT", "5151")
    proxy = await Proxy.tunnelled(SETTINGS)
    try:
        assert proxy.origin == "https://fake-words.trycloudflare.com"
        assert proxy.loopback == "http://127.0.0.1:5151"
        assert proxy.host == "fake-words.trycloudflare.com"
        assert proxy.address_for("t") == "https://fake-words.trycloudflare.com/r/trial/t/v1"
        assert json.loads(stub.read_text())["argv"][-2:] == ["--advertise", "127.0.0.1"]
        assert cloudflared.read_text().split() == [
            "tunnel",
            "--no-autoupdate",
            "--url",
            "http://127.0.0.1:5151",
        ]
        assert proxy.tunnel is not None and proxy.tunnel.process.poll() is None
        tunnel, process = proxy.tunnel.process, proxy.process
    finally:
        proxy.close()
    assert tunnel.poll() is not None and process is not None and process.poll() is not None


async def test_a_tunnel_that_cannot_open_stops_the_serve_it_fronted(
    stub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: None)
    with pytest.raises(RuntimeError, match="cloudflared is not installed"):
        await Proxy.tunnelled(SETTINGS)


# ------------------------------------------------------------------- a remote one


def test_remote_is_addressed_by_url_and_needs_the_token() -> None:
    proxy = Proxy.remote("https://proxy.example/", "k")
    assert proxy.origin == proxy.loopback == "https://proxy.example"
    assert proxy.address_for("t-1") == "https://proxy.example/r/trial/t-1/v1"
    assert proxy.control_token is None and proxy.process is None and proxy.tunnel is None
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
