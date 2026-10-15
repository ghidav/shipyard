"""Test the endpoints: placement, settings, connections, records, refusals, the provider case."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shipyard import record
from shipyard.config import Blueprint
from shipyard.proxy.profiles import PROFILES, bare_name
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard.run import Run
from shipyard.sampling.endpoint import (
    PROXY_LOG,
    NothingServed,
    ProviderEndpoint,
    ProxyEndpoint,
    ProxyGone,
    answered_by_ours,
    backend,
    endpoint_settings,
    placement,
    tinker_base_url,
)
from shipyard.types import Connection, Trial, harness
from tests.records import FakeProxy, made
from tests.trials import write_result

DAPO = Path(__file__).parent / "blueprints" / "dapo"


def blueprint(**rollout: Any) -> Blueprint:
    model = {"name": "Qwen/Qwen3-8B"}
    if "from_checkpoint" in rollout:
        model["from_checkpoint"] = rollout.pop("from_checkpoint")
    return Blueprint.model_validate(
        {
            "model": model,
            "data": {"dataset": "d", "batch_size": 1},
            "rollout": {"harness": "pi@0.85.1", **rollout},
            "recipe": {"kind": "evaluate"},
        }
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", proxy)
    return proxy


def trials(
    tmp_path: Path, *outcomes: tuple[str, dict[str, Any] | None], deployed: str = "pi"
) -> list[Trial]:
    """A job's trials under `jobs/j`, each with its result, None for none."""
    found = []
    for n, (task, outcome) in enumerate(outcomes):
        home = tmp_path / "jobs" / "j" / f"{task}__{n:07d}"
        if outcome is not None:
            write_result(home, **outcome)
        found.append(Trial("j", Path(task), n, harness(deployed), home))
    return found


# ------------------------------------------------------------------------ placement


@pytest.mark.parametrize("sandbox", ["docker", "podman", "apple-container"])
def test_a_sandbox_on_this_machine_reaches_a_proxy_here(sandbox: str) -> None:
    assert placement(blueprint(sandbox=sandbox).rollout) == "local"


@pytest.mark.parametrize("sandbox", ["modal", "e2b", "daytona"])
def test_a_sandbox_elsewhere_needs_a_tunnel_unless_a_url_is_named(sandbox: str) -> None:
    assert placement(blueprint(sandbox=sandbox).rollout) == "tunnel"
    assert (
        placement(blueprint(sandbox=sandbox, endpoint_url="https://p.example").rollout) == "remote"
    )
    assert (
        placement(blueprint(sandbox="docker", endpoint_url="https://p.example").rollout) == "remote"
    )


def test_the_party_is_tinker_or_tinker_at_the_backends_host() -> None:
    assert backend({}) == "tinker"
    assert (
        backend({"TINKER_BASE_URL": "https://api.fireworks.ai/inference/v1"})
        == "tinker@api.fireworks.ai"
    )
    assert backend({"TINKER_BASE_URL": ""}) == "tinker"
    assert (
        tinker_base_url({"TINKER_BASE_URL": "https://api.fireworks.ai/"})
        == "https://api.fireworks.ai"
    )
    assert tinker_base_url({}).startswith("https://tinker.thinkingmachines.dev")
    assert bare_name("pi@0.85.1") == "pi" and bare_name("acp:x.y:Z@1") == "acp:x.y:Z@1"


def test_the_endpoint_settings_are_the_rollout_knobs_and_the_checkpoint() -> None:
    cfg = blueprint(
        from_checkpoint="tinker://w/step-9",
        renderer="tml_v0",
        bind="127.0.0.1",
        bind_port=8000,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=4096,
        max_context=32768,
        fill_context=True,
    )
    assert endpoint_settings(cfg) == {
        "model": "Qwen/Qwen3-8B",
        "weights": "tinker://w/step-9",
        "renderer": "tml_v0",
        "bind": "127.0.0.1",
        "bind_port": 8000,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 4096,
        "max_context": 32768,
        "fill_context": True,
    }
    bare = endpoint_settings(blueprint())
    assert bare["weights"] is None and bare["renderer"] is None and bare["max_context"] is None
    assert (bare["bind"], bare["bind_port"], bare["max_tokens"]) == ("0.0.0.0", 0, 8192)
    assert "metadata" not in bare


def test_the_proxys_tinker_session_is_tagged_with_the_run_and_its_recipe() -> None:
    cfg = blueprint()
    tagged = endpoint_settings(cfg, run="dapo-docker__abc")
    assert tagged["metadata"] == {
        "shipyard_run": "dapo-docker__abc",
        "shipyard_recipe": cfg.recipe.kind,
    }


# ------------------------------------------------------------------- open and close


async def test_a_local_sandbox_starts_serve_once_with_the_checkpoint_on_its_command_line(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = ProxyEndpoint(blueprint(from_checkpoint="tinker://w/step-2", host="proxy.lan"), tmp_path)
    await held.open()
    await held.open()
    [(how, settings, named)] = fake.placed
    assert (
        how == "local"
        and named["host"] == "proxy.lan"
        and settings["weights"] == "tinker://w/step-2"
    )
    assert named["log"].name == str(tmp_path / PROXY_LOG)
    assert held.told == "tinker://w/step-2"
    assert fake.swapped == [], "serve was started on the path; nothing to swap"
    assert held.proxy is fake and held.budget is None
    held.close()
    assert fake.closed == 1 and held.proxy is None
    assert (tmp_path / PROXY_LOG).is_file()
    held.close()
    assert fake.closed == 1


async def test_a_sandbox_elsewhere_gets_a_tunnel(fake: FakeProxy, tmp_path: Path) -> None:
    held = ProxyEndpoint(blueprint(sandbox="modal"), tmp_path)
    await held.open()
    [(how, settings, named)] = fake.placed
    assert how == "tunnel" and settings["weights"] is None and "tunnel_log" in named
    assert held.told is None
    assert (held.placement, held.origin) == ("tunnel", fake.origin)
    held.close()
    assert fake.closed == 1


async def test_a_proxy_that_exited_is_found_on_the_next_open(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = ProxyEndpoint(blueprint(), tmp_path)
    await held.open()
    fake.exited = "shipyard serve exited with 1"
    with pytest.raises(ProxyGone, match="exited with 1"):
        await held.open()
    assert len(fake.placed) == 1


async def test_a_remote_proxy_is_pointed_through_its_control_route(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    cfg = blueprint(endpoint_url="https://p.example/", from_checkpoint="tinker://w/step-4")
    held = ProxyEndpoint(cfg, tmp_path, environ=environ)
    await held.open()
    assert fake.placed == [("remote", "https://p.example/", {"token": "t", "control_token": "c"})]
    assert fake.probed == 1, "probed before any sandbox opens"
    assert fake.swapped == ["tinker://w/step-4"]
    assert held.told == "tinker://w/step-4"
    assert not (tmp_path / PROXY_LOG).exists()
    await held.serve("tinker://w/step-5")
    assert fake.swapped[-1] == "tinker://w/step-5" and held.told == "tinker://w/step-5"
    await held.serve(None)
    assert held.told is None


async def test_a_remote_proxy_without_a_control_token_serves_only_what_it_was_started_on(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t"}
    held = ProxyEndpoint(blueprint(endpoint_url="https://p.example"), tmp_path, environ=environ)
    await held.open()
    assert fake.swapped == [] and held.told is None, "base, checked"
    with pytest.raises(RuntimeError, match="not started"):
        await ProxyEndpoint(blueprint(endpoint_url="https://p.example"), tmp_path).serve(None)
    cfg = blueprint(endpoint_url="https://p.example", from_checkpoint="tinker://w/step-4")
    with pytest.raises(RuntimeError, match=CONTROL_TOKEN_ENV):
        await ProxyEndpoint(cfg, tmp_path, environ=environ).open()
    with pytest.raises(ValueError, match=PROXY_TOKEN_ENV):
        await ProxyEndpoint(
            blueprint(endpoint_url="https://p.example"), tmp_path, environ={}
        ).open()


# ---------------------------------------------------------------------- the connection


def test_each_trial_dials_the_address_that_names_it_on_its_harnesss_wire(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = ProxyEndpoint(blueprint(), tmp_path, proxy=fake)
    one, two = trials(tmp_path, ("alpha", None), ("alpha", None))
    first, second = held.connection(one), held.connection(two)
    assert first.model == second.model == "openai/Qwen/Qwen3-8B"
    assert first.address == f"{fake.origin}/r/trial/{one.name}/v1"
    assert second.address == f"{fake.origin}/r/trial/{two.name}/v1"
    assert first.env == {"OPENAI_BASE_URL": first.address, "OPENAI_API_KEY": "harness-token"}
    assert first.token == "harness-token" and first.host == "host.docker.internal"
    assert first.kwargs == {"model_api": "openai-completions"}
    assert first.kwargs is not PROFILES["pi"].kwargs
    assert fake.addressed == [one.name, two.name]
    with pytest.raises(RuntimeError, match="not started"):
        ProxyEndpoint(blueprint(), tmp_path).connection(one)


def test_the_wire_and_the_env_follow_the_trials_harness(fake: FakeProxy, tmp_path: Path) -> None:
    [trial] = trials(tmp_path, ("alpha", None), deployed="claude-code")
    filling = ProxyEndpoint(blueprint(fill_context=True), tmp_path, proxy=fake)
    found = filling.connection(trial)
    assert found.model == "anthropic/Qwen/Qwen3-8B"
    assert found.address == f"{fake.origin}/r/trial/{trial.name}", "no /v1 for the Anthropic wire"
    assert found.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": found.address,
        "ANTHROPIC_API_KEY": "harness-token",
    }
    plain = ProxyEndpoint(blueprint(), tmp_path, proxy=fake).connection(trial)
    assert "DISABLE_COMPACT" not in plain.env, "the profile's env only when filling"
    [trial] = trials(tmp_path, ("alpha", None), deployed="opencode@1.18.35")
    named = ProxyEndpoint(blueprint(), tmp_path, proxy=fake).connection(trial)
    assert named.model == "shipyard/Qwen/Qwen3-8B"
    assert set(named.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    assert "opencode_config" in named.kwargs


# ------------------------------------------------------------------------ the records


async def test_record_takes_each_trials_records_and_writes_a_row_per_record(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {
        "alpha": [
            made("one", "ok", seq=1, request_id="r-1"),
            made("oneoktwo", "fine", seq=2, cached=5, bridged=True),
        ],
        "beta": [made("go", error="context", seq=1)],
    }
    held = ProxyEndpoint(blueprint(), tmp_path)
    await held.open()
    batch = trials(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 0.0}), ("gamma", None))
    records = await held.record("j", batch)
    assert fake.fetched == [trial.name for trial in batch]
    assert [len(records[trial.name]) for trial in batch] == [2, 1, 1]
    rows = list(record.read(tmp_path / record.REQUESTS))
    assert [row["trial"] for row in rows] == [batch[0].name] * 2 + [t.name for t in batch[1:]]
    first = rows[0]
    assert set(first) == {
        "at",
        "job",
        "trial",
        "seq",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "stop_reason",
        "sample_ms",
        "served",
        "request_id",
        "bridged",
        "error",
    }
    assert (first["job"], first["seq"], first["prompt_tokens"], first["completion_tokens"]) == (
        "j",
        1,
        3,
        2,
    )
    assert (first["request_id"], first["served"], first["error"], first["bridged"]) == (
        "r-1",
        None,
        None,
        False,
    )
    assert (rows[1]["cached_tokens"], rows[1]["bridged"], rows[1]["sample_ms"]) == (5, True, 12.0)
    assert (rows[2]["error"], rows[2]["completion_tokens"], rows[2]["stop_reason"]) == (
        "context",
        0,
        "",
    )


async def test_a_two_trial_job_is_dialled_and_recorded_under_two_names(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made("one", "ok")]}
    held = ProxyEndpoint(blueprint(), tmp_path)
    await held.open()
    batch = trials(tmp_path, ("alpha", {"reward": 1.0}), ("alpha", {"reward": 0.0}))
    assert len({held.connection(trial).address for trial in batch}) == 2
    records = await held.record("j", batch)
    assert sorted(records) == sorted(trial.name for trial in batch)
    assert sorted(row["trial"] for row in record.read(tmp_path / record.REQUESTS)) == sorted(
        trial.name for trial in batch
    )


async def test_a_batch_answered_by_other_weights_is_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {
        "alpha": [made(served="tinker://w/step-2")],
        "beta": [made(served="tinker://intruder/step-9")],
    }
    held = ProxyEndpoint(blueprint(from_checkpoint="tinker://w/step-2"), tmp_path)
    await held.open()
    with pytest.raises(ValueError, match="intruder"):
        await held.record(
            "j", trials(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 1.0}))
        )
    fake.answers = {
        "alpha": [made(served="tinker://w/step-2")],
        "beta": [made(served="tinker://w/step-2")],
    }
    records = await held.record("j", trials(tmp_path, ("alpha", {"reward": 1.0})))
    assert len(records) == 1
    answered_by_ours("j", {"t": [made(served=None)]}, None)
    with pytest.raises(ValueError, match="pointed its proxy at None"):
        answered_by_ours("j", {"t": [made(served="tinker://x/step-1")]}, None)


async def test_a_remote_proxy_this_run_did_not_point_must_serve_the_base_model(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made(served="tinker://theirs/step-5")]}
    environ = {PROXY_TOKEN_ENV: "t"}
    held = ProxyEndpoint(blueprint(endpoint_url="https://p.example"), tmp_path, environ=environ)
    await held.open()
    with pytest.raises(ValueError, match="re-pointed"):
        await held.record("j", trials(tmp_path, ("alpha", {"reward": 1.0})))
    fake.answers = {"beta": [made()]}
    records = await held.record("j", trials(tmp_path, ("beta", {"reward": 1.0})))
    assert [len(found) for found in records.values()] == [1]


async def test_a_fetch_that_fails_fails_the_job_rather_than_masking_a_trial(
    fake: FakeProxy, tmp_path: Path
) -> None:
    import httpx

    fake.failing = {"beta"}
    held = ProxyEndpoint(blueprint(), tmp_path)
    await held.open()
    with pytest.raises(httpx.ConnectError):
        await held.record(
            "j", trials(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 1.0}))
        )


async def test_check_raises_the_stop_for_after_the_row(fake: FakeProxy, tmp_path: Path) -> None:
    fake.default = []
    held = ProxyEndpoint(blueprint(), tmp_path)
    await held.open()
    batch = trials(tmp_path, ("alpha", {"reward": 1.0}))
    records = await held.record("j", batch)
    with pytest.raises(NothingServed, match="no trial reached the proxy") as caught:
        held.check("j", records)
    assert caught.value.rolled is None
    assert str(caught.value) == (
        "job j: no trial reached the proxy; the endpoint is dead or unreachable"
    )
    fake.default = None
    held.check("j", await held.record("j", batch))
    fake.exited = "shipyard serve exited with 1"
    with pytest.raises(ProxyGone, match="exited with 1"):
        held.check("j", records)


# ----------------------------------------------------------------------- a provider


async def test_a_providers_client_dials_the_model_alone_and_records_nothing(tmp_path: Path) -> None:
    held = ProviderEndpoint("openrouter/some-model")
    await held.open()
    [trial] = trials(tmp_path, ("alpha", None))
    assert held.connection(trial) == Connection(model="openrouter/some-model")
    assert held.budget is None and await held.record("j", [trial]) is None
    with pytest.raises(RuntimeError, match="served by its provider"):
        await held.serve("tinker://w/step-1")
    held.close()
    assert ProviderEndpoint(None).connection(trial).model is None


# --------------------------------------------------------------------- the run


def test_a_served_run_notes_the_backend_and_stops_its_proxy_on_exit(
    fake: FakeProxy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    with Run.open(DAPO, root=tmp_path) as opened:
        assert isinstance(opened.endpoint, ProxyEndpoint)
        assert opened.endpoint.party == "tinker@api.fireworks.ai"
        note = Run.read(opened.directory)
        assert note["tinker_base_url"] == "https://api.fireworks.ai/inference/v1"
        assert fake.closed == 0
    assert fake.closed == 0, "nothing was started, so nothing was stopped"


async def test_a_started_proxy_is_stopped_when_the_run_closes(
    fake: FakeProxy, tmp_path: Path
) -> None:
    with Run.open(DAPO, root=tmp_path) as opened:
        await opened.endpoint.open()
        assert fake.placed and fake.closed == 0
    assert fake.closed == 1
