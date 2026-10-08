"""The quick tunnel as a subprocess: found on PATH or through docker, its origin read from
its output, stopped with the run, and a clear error when nothing here can open one."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from shipyard.proxy import tunnel
from shipyard.proxy.tunnel import Tunnel, command, found

STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$@" > "$STUB_ARGS"
    case "$STUB_MODE" in
      exit) echo "failed to connect" >&2; exit 7 ;;
      silent) exec sleep 30 ;;
      refused)
        api='"https://api.trycloudflare.com/tunnel"'
        echo "ERR failed to request quick Tunnel: Post $api: dial tcp: i/o timeout" >&2
        exit 1 ;;
    esac
    echo "INF Requesting new quick Tunnel on trycloudflare.com..." >&2
    echo "INF |  https://some-words-here.trycloudflare.com  |" >&2
    trap 'exit 0' TERM
    while true; do sleep 0.1; done
    """
)


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "cloudflared"
    script.write_text(STUB, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("STUB_ARGS", str(tmp_path / "args"))
    return script


def test_the_binary_on_path_is_preferred_then_docker_then_nothing(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    located = found()
    assert located is not None and located[0] == str(stub)
    assert command(8000) == [
        str(stub),
        "tunnel",
        "--no-autoupdate",
        "--url",
        "http://127.0.0.1:8000",
    ]

    def only_docker(name: str) -> str | None:
        return "/usr/local/bin/docker" if name == "docker" else None

    monkeypatch.setattr(shutil, "which", only_docker)
    located = found()
    assert located is not None and located[0] == "/usr/local/bin/docker"
    assert command(8000) == [
        "/usr/local/bin/docker",
        "run",
        "--rm",
        "--add-host=host.docker.internal:host-gateway",
        tunnel.IMAGE,
        "tunnel",
        "--no-autoupdate",
        "--url",
        "http://host.docker.internal:8000",
    ]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert found() is None
    with pytest.raises(RuntimeError, match="endpoint_url"):
        command(8000)


async def test_the_origin_is_read_off_the_output_and_stop_ends_the_process(
    stub: Path, tmp_path: Path
) -> None:
    log = tmp_path / "tunnel.log"
    with log.open("a", encoding="utf-8") as sink:
        opened = await Tunnel.start(7001, log=sink)
        try:
            assert opened.origin == "https://some-words-here.trycloudflare.com"
            assert opened.process.poll() is None
            assert (tmp_path / "args").read_text().split() == [
                "tunnel",
                "--no-autoupdate",
                "--url",
                "http://127.0.0.1:7001",
            ]
        finally:
            opened.stop()
    assert opened.process.poll() is not None
    written = log.read_text()
    assert written.startswith(f"$ {stub} tunnel --no-autoupdate --url http://127.0.0.1:7001\n")
    assert "Requesting new quick Tunnel" in written


async def test_the_quick_tunnel_services_own_host_is_not_an_origin(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cloudflared prints `api.trycloudflare.com` when it could not get a tunnel. The run
    reports that error and does not treat the API's host as a tunnel origin."""
    monkeypatch.setenv("STUB_MODE", "refused")
    with pytest.raises(RuntimeError, match="exited with 1") as caught:
        await Tunnel.start(7004)
    assert "failed to request quick Tunnel" in str(caught.value)
    assert tunnel.ORIGIN.search("|  https://some-words-here.trycloudflare.com  |") is not None


async def test_a_tunnel_that_exits_first_raises_with_its_output(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "exit")
    with pytest.raises(RuntimeError, match="exited with 7") as caught:
        await Tunnel.start(7002)
    assert "failed to connect" in str(caught.value)


async def test_a_tunnel_that_never_prints_is_given_up_on(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "silent")
    with pytest.raises(TimeoutError, match="the tunnel"):
        await Tunnel.start(7003, seconds=0.5)


def test_end_kills_what_will_not_stop(tmp_path: Path) -> None:
    import subprocess

    stubborn = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.1)",
        ]
    )
    tunnel.end(stubborn, seconds=0.5)
    assert stubborn.poll() is not None
    tunnel.end(stubborn)  # already gone: nothing to do
