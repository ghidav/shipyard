"""The proxy's lifecycle in a run: probed at `/healthz` through the tunnel (or on loopback)
before the first trial, refused in words when nothing answers, and a tunnel or serve process
that exits stops the run before the next job instead of masking a batch. A stub cloudflared
and a stub `shipyard serve` that answers HTTP stand in for the real ones."""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest

from shipyard import record
from shipyard.proxy import client
from shipyard.proxy.client import Proxy, Unreachable
from shipyard.proxy.tunnel import Tunnel
from shipyard.run import Run
from shipyard.serving import ProxyGone, Serving
from tests.proxies import MODEL, FakeSampler, control, endpoint, root
from tests.records import FakeProxy
from tests.trials import FakeTrials, fixture_tasks

ORIGIN = "https://fake-words.trycloudflare.com"
SERVE_STUB = textwrap.dedent(
    """
    import json, signal, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    flag = lambda name: args[args.index(name) + 1] if name in args else None
    model = flag("--model")
    record = {"seq": 1, "prompt_token_ids": [1, 2], "completion_token_ids": [3]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/healthz":
                body = {"status": "ok", "model": model, "serving": model}
            elif path.startswith("/records/"):
                body = {"trial": path.rsplit("/", 1)[1], "records": [record], "spoke": 1}
            else:
                self.send_response(404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    print(f"serving {model} at http://{flag('--advertise')}:{port} (control: on)", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    server.serve_forever()
    """
)
CLOUDFLARED_STUB = textwrap.dedent(
    f"""\
    #!/bin/sh
    echo "INF Requesting new quick Tunnel on trycloudflare.com..." >&2
    echo "INF |  {ORIGIN}  |" >&2
    trap 'exit 0' TERM
    while true; do sleep 0.1; done
    """
)
SERVED = """
[model]
name = "Qwen/Qwen3-8B"
[data]
dataset = "fixture"
batch_size = 1
[rollout]
harness = "pi@0.85.1"
sandbox = "{sandbox}"
[recipe]
kind = "evaluate"
"""


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`shipyard serve` and `cloudflared` replaced, from a directory holding the fixtures."""
    script = tmp_path / "serve_stub.py"
    script.write_text(SERVE_STUB, encoding="utf-8")
    monkeypatch.setattr(client, "SERVE", [sys.executable, str(script)])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    binary = bin_dir / "cloudflared"
    binary.write_text(CLOUDFLARED_STUB, encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def opened(tmp_path: Path, sandbox: str) -> Run:
    home = tmp_path / "bp"
    home.mkdir(exist_ok=True)
    (home / "run.toml").write_text(SERVED.format(sandbox=sandbox), encoding="utf-8")
    run = Run.open(home, root=tmp_path / "runs")
    run.run_trial = FakeTrials()
    return run


def task(tmp_path: Path) -> list[Path]:
    return [tmp_path / "tasks" / "fixture" / "alpha"]


# ------------------------------------------------------------------- in a run


async def test_the_tunnel_is_probed_before_the_first_trial_and_its_exit_stops_the_next_job(
    stubs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []

    async def through_the_tunnel(url: str, **_: Any) -> dict[str, Any]:
        probed.append(url)
        return {"status": "ok", "serving": MODEL}

    monkeypatch.setattr(client, "answers", through_the_tunnel)
    run = opened(stubs, "modal")
    with pytest.raises(ProxyGone, match="the proxy is no longer reachable"), run:
        first = await run.sample(task(stubs), rollouts=1, index=0)
        assert probed == [f"{ORIGIN}/healthz"], "probed once, through the tunnel"
        assert first.records is not None and all(first.records.values())
        assert run.serving is not None and run.serving.proxy is not None
        process = run.serving.proxy.tunnel.process  # type: ignore[union-attr]
        process.kill()
        process.wait()
        await run.sample(task(stubs), rollouts=1, index=1)
    note = Run.read(run.directory)
    assert note["error"] == (
        f"ProxyGone: the proxy is no longer reachable: the tunnel at {ORIGIN} exited with -9"
    )
    assert note["proxy"] == {"placement": "tunnel", "origin": ORIGIN}
    assert [row["job"] for row in record.read(run.directory / record.JOBS)] == [f"{run.id}-0000"]
    assert sorted(p.name for p in (stubs / "jobs").iterdir()) == [f"{run.id}-0000"]
    assert "trycloudflare" in (run.directory / "tunnel.log").read_text()
    assert run.serving.proxy is None, "stopped with the run"


async def test_a_local_serve_is_probed_on_loopback_and_its_exit_stops_the_next_job(
    stubs: Path,
) -> None:
    run = opened(stubs, "docker")
    with pytest.raises(ProxyGone, match="shipyard serve exited with -9"), run:
        await run.sample(task(stubs), rollouts=1, index=0)
        assert run.serving is not None and run.serving.proxy is not None
        process = run.serving.proxy.process
        assert process is not None
        process.kill()
        process.wait()
        await run.sample(task(stubs), rollouts=1, index=1)
    assert Run.read(run.directory)["proxy"]["placement"] == "local"
    assert len(list(record.read(run.directory / record.JOBS))) == 1


async def test_a_tunnel_that_does_not_answer_refuses_the_run_before_any_trial(
    stubs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unresolved(url: str, **_: Any) -> dict[str, Any]:
        raise httpx.ConnectError("nodename nor servname provided")

    monkeypatch.setattr(client, "answers", unresolved)
    run = opened(stubs, "modal")
    with pytest.raises(Unreachable, match=f"the tunnel at {ORIGIN} does not answer"), run:
        await run.sample(task(stubs), rollouts=1, index=0)
    assert run.run_trial.configs == []  # type: ignore[union-attr]
    assert not (stubs / "jobs").exists() and not (run.directory / record.JOBS).exists()
    note = Run.read(run.directory)
    assert note["failed"] is True and "does not answer" in note["error"]


async def test_a_fetch_that_failed_because_the_proxy_died_says_so(tmp_path: Path) -> None:
    from shipyard.config import Blueprint
    from shipyard.rollout import Rollouts

    cfg = Blueprint.model_validate(
        {
            "model": {"name": MODEL},
            "data": {"dataset": "d", "batch_size": 1},
            "rollout": {"harness": "pi"},
            "recipe": {"kind": "evaluate"},
        }
    )
    proxy = FakeProxy(failing={"alpha"}, exited="shipyard serve exited with 1")
    held = Serving(cfg, tmp_path, proxy=proxy)
    batch = Rollouts(job="j", trials=[tmp_path / "alpha__0000000"], plan=[])
    with pytest.raises(ProxyGone, match="exited with 1"):
        await held.harvest(batch)
    with pytest.raises(ProxyGone):
        await held.point("tinker://w/step-1")
    proxy.exited = None
    with pytest.raises(httpx.ConnectError):
        await held.harvest(batch)


# --------------------------------------------------------------- the probe itself


async def test_healthz_names_what_is_served_without_a_token_and_follows_a_swap() -> None:
    async with endpoint(FakeSampler("ok"), control_token="c") as started:
        async with httpx.AsyncClient() as http:
            answer = (await http.get(f"{root(started)}/healthz")).json()
            assert answer == {"status": "ok", "model": MODEL, "serving": MODEL, "budget": None}
            await control(started, "tinker://w/step-3")
            answer = (await http.get(f"{root(started)}/healthz")).json()
        assert answer["serving"] == "tinker://w/step-3"
        proxy = Proxy(origin=root(started), token="not-needed")
        assert await proxy.ready(seconds=2) == "tinker://w/step-3"


async def test_healthz_names_the_token_budget_and_the_run_keeps_it() -> None:
    async with endpoint(FakeSampler("ok"), max_context=12288) as started:
        async with httpx.AsyncClient() as http:
            answer = (await http.get(f"{root(started)}/healthz")).json()
        assert answer["budget"] == 12288, "each trial may sample the context length in all"
        proxy = Proxy(origin=root(started), token="not-needed")
        assert proxy.budget is None
        await proxy.ready(seconds=2)
        assert proxy.budget == 12288


class LateName(httpx.AsyncBaseTransport):
    """A quick tunnel's name that resolves only from the `after`-th ask, at its origin."""

    def __init__(self, after: int) -> None:
        self.after, self.asked = after, []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(str(request.url))
        if len(self.asked) < self.after:
            raise httpx.ConnectError("nodename nor servname provided", request=request)
        return httpx.Response(200, json={"status": "ok", "model": MODEL})


class Running:
    """A process that has not exited, or one that has with `code`."""

    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


async def test_the_probe_goes_through_the_tunnel_and_waits_for_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PROBE_PAUSE", 0.01)
    late = LateName(after=3)
    held = Tunnel(origin=ORIGIN, process=Running())  # type: ignore[arg-type]
    proxy = Proxy(
        origin=ORIGIN, token="t", loopback="http://127.0.0.1:9", tunnel=held, transport=late
    )
    assert await proxy.ready(seconds=5) == MODEL
    assert late.asked == [f"{ORIGIN}/healthz"] * 3
    never = Proxy(
        origin=ORIGIN, token="t", tunnel=held, transport=LateName(after=10**6), loopback=None
    )
    with pytest.raises(Unreachable, match=f"the tunnel at {ORIGIN} does not answer: Connect"):
        await never.ready(seconds=0.2)
    alone = Proxy(origin="http://127.0.0.1:9", token="t", transport=LateName(after=10**6))
    with pytest.raises(Unreachable, match="the proxy at http://127.0.0.1:9 does not answer"):
        await alone.ready(seconds=0.05)


def test_gone_says_which_process_exited_and_nothing_for_one_somebody_else_runs() -> None:
    held = Tunnel(origin=ORIGIN, process=Running())  # type: ignore[arg-type]
    proxy = Proxy(origin=ORIGIN, token="t", tunnel=held, process=Running())  # type: ignore[arg-type]
    assert proxy.gone() is None
    proxy.process = Running(3)  # type: ignore[assignment]
    assert proxy.gone() == "shipyard serve exited with 3"
    held.process = Running(1)  # type: ignore[assignment]
    assert proxy.gone() == f"the tunnel at {ORIGIN} exited with 1"
    assert Proxy.remote("https://p.example", "k").gone() is None


async def test_the_tunnel_command_is_logged_once(
    stubs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="shipyard.proxy.tunnel"):
        opened_tunnel = await Tunnel.start(7001)
        opened_tunnel.stop()
    said = [r.getMessage() for r in caplog.records if "opening a quick tunnel" in r.getMessage()]
    assert len(said) == 1 and said[0].endswith("tunnel --no-autoupdate --url http://127.0.0.1:7001")


async def test_a_tunnel_that_dies_under_a_batch_stops_the_run_once_the_job_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shipyard import serving
    from shipyard.recipes import run_recipe

    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr(serving, "Proxy", proxy)
    run = opened(tmp_path, "modal")
    trials = FakeTrials()

    async def dying(config: Any) -> None:
        await trials(config)
        proxy.exited = f"the tunnel at {proxy.origin} exited with 1"

    run.run_trial = dying
    with pytest.raises(ProxyGone, match="exited with 1"), run:
        await run_recipe(run)
    assert len(list(record.read(run.directory / record.JOBS))) == 1, "the batch is on record"
    assert not (run.directory / record.METRICS).exists(), "and not measured"
