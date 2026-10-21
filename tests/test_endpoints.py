"""Tests for `shipyard.models.endpoints`: the run's proxy, which the trained agent's harnesses
call, and a provider's model, which code asks through LiteLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Served
from proxies import MODEL
from shipyard.config import Blueprint
from shipyard.groups import Connection, Trial
from shipyard.jobs.harnesses import harness
from shipyard.models.clients.tinker import TinkerClient
from shipyard.models.endpoints import (
    PROXY_LOG,
    LaneEndpoint,
    NothingServed,
    ProviderEndpoint,
    ProxyEndpoint,
    ProxyGone,
    answered_by_ours,
    placement,
)
from shipyard.records import Records
from shipyard.run.logs import AGENT
from tests.fakes import FakeProxy, made
from tests.service import FakeService

DEAD = "no trial reached the proxy; the endpoint is dead or unreachable"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    return proxy


def served(**rollout: Any) -> Blueprint:
    """A blueprint: `harness` and `from_checkpoint` go to `[agent]`, the other keywords to
    `[rollout]`."""
    agent = {"model": MODEL, "harness": rollout.pop("harness", "pi@0.85.1")}
    if "from_checkpoint" in rollout:
        agent["from_checkpoint"] = rollout.pop("from_checkpoint")
    return Blueprint.model_validate(
        {
            "agent": agent,
            "data": {"train": "d", "batch_size": 1},
            "rollout": rollout,
            "recipe": {"steps": 1, "learning_rate": 1e-5},
        }
    )


def trials(tmp_path: Path, *tasks: str, deployed: str = "pi") -> list[Trial]:
    """One trial of job `j` per task, named as Harbor names a trial's directory."""
    jobs = tmp_path / "jobs" / "j"
    return [
        Trial("j", Path(task), harness(deployed), jobs / f"{task}__{n:07d}")
        for n, task in enumerate(tasks)
    ]


def agent(config: Blueprint, directory: Path, **named: Any) -> LaneEndpoint:
    """The agent's lane of a run's proxy, made as the run makes it."""
    held, rollout = ProxyEndpoint(config, directory, **named), config.rollout
    return held.lane(
        AGENT,
        config.agent.model,
        renderers=rollout.renderer,
        max_context=rollout.max_context or None,
    )


# ------------------------------------------------------------------- where it runs


@pytest.mark.parametrize("sandbox", ["docker", "podman", "apple-container"])
def test_a_sandbox_on_this_machine_reaches_a_proxy_here(sandbox: str) -> None:
    assert placement(served(sandbox=sandbox).rollout) == "local"


@pytest.mark.parametrize("sandbox", ["modal", "e2b", "daytona"])
def test_a_sandbox_elsewhere_needs_a_tunnel_unless_a_url_is_named(sandbox: str) -> None:
    assert placement(served(sandbox=sandbox).rollout) == "tunnel"
    assert placement(served(sandbox=sandbox, endpoint_url="https://p.example").rollout) == "remote"
    assert placement(served(sandbox="docker", endpoint_url="https://p.example").rollout) == "remote"


def test_the_agents_lane_takes_the_rollout_knobs_and_its_sessions_tags(tmp_path: Path) -> None:
    cfg = served(
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
    held = ProxyEndpoint(cfg, tmp_path / "dapo__abc")
    rollout = cfg.rollout
    lane = held.lane(AGENT, MODEL, renderers=rollout.renderer, max_context=rollout.max_context)
    assert lane.document == {
        "name": AGENT,
        "model": MODEL,
        "renderer": ["tml_v0"],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 4096,
        "max_context": 32768,
        "fill_context": True,
        "metadata": {"shipyard_run": "dapo__abc"},
    }, "a checkpoint is not served from the start: the trained agent points the proxy next"
    bare = agent(served(), tmp_path).document
    assert "renderer" not in bare and "max_context" not in bare and "weights" not in bare
    assert bare["max_tokens"] == 8192


def test_a_lanes_tokens_are_billed_as_the_client_it_serves_was_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = agent(served(), tmp_path)
    assert (held.provider, held.kind) == (None, None), "nothing to bill until a model is opened"
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    held.opened = TinkerClient(MODEL, service=FakeService())
    assert (held.provider, held.kind, held.model) == ("tinker", "tinker", MODEL)
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    assert held.provider == "tinker", "fixed when the client was made"
    held.opened = TinkerClient(MODEL, service=FakeService())
    assert (held.provider, held.kind) == ("tinker@api.fireworks.ai", "tinker"), "named by kind"
    assert ProviderEndpoint(MODEL, "anthropic").kind == "api"


# ------------------------------------------------------------------- open and close


async def test_a_local_sandbox_starts_the_proxy_on_this_machine(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(host="proxy.lan"), tmp_path)
    await held.proxy.open()
    [(how, given, named)] = fake.placed
    assert how == "local" and named["host"] == "proxy.lan"
    assert (given["bind"], given["bind_port"], list(given)) == (
        "0.0.0.0",
        0,
        ["bind", "bind_port", "lanes"],
    )
    assert named["log"].name == str(tmp_path / PROXY_LOG)
    assert held.told is None and fake.swapped == [] and fake.probed == 1
    assert held.proxy.proxy is fake and held.budget is None
    held.proxy.close()
    assert fake.closed == 1 and held.proxy.proxy is None
    assert (tmp_path / PROXY_LOG).is_file()
    held.proxy.close()
    assert fake.closed == 1


async def test_a_sandbox_elsewhere_gets_a_tunnel(fake: FakeProxy, tmp_path: Path) -> None:
    held = agent(served(sandbox="modal"), tmp_path)
    await held.proxy.open()
    [(how, _, named)] = fake.placed
    assert how == "tunnel" and named["log"].name == str(tmp_path / PROXY_LOG)
    held.proxy.close()
    assert fake.closed == 1


async def test_a_remote_proxy_is_pointed_through_its_control_route(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    held = agent(served(endpoint_url="https://p.example/"), tmp_path, environ=environ)
    fake.lanes = {AGENT: Served(MODEL, MODEL, None, ())}
    await held.proxy.open()
    assert fake.placed == [("remote", "https://p.example/", {"token": "t", "control_token": "c"})]
    assert fake.probed == 1, "probed before any sandbox opens"
    assert held.told is None and fake.swapped == []
    assert not (tmp_path / PROXY_LOG).exists()
    await held.serve("tinker://w/step-5")
    assert fake.lane_swaps == [(AGENT, "tinker://w/step-5")] and held.told == "tinker://w/step-5"
    await held.serve(None)
    assert held.told is None


async def test_a_remote_proxy_needs_both_of_its_tokens(fake: FakeProxy, tmp_path: Path) -> None:
    remote = served(endpoint_url="https://p.example")
    with pytest.raises(RuntimeError, match=CONTROL_TOKEN_ENV):
        await agent(remote, tmp_path, environ={PROXY_TOKEN_ENV: "t"}).proxy.open()
    with pytest.raises(ValueError, match=PROXY_TOKEN_ENV):
        await agent(remote, tmp_path, environ={}).proxy.open()


# ---------------------------------------------------------------------- the connection


def test_each_trial_dials_the_address_that_names_it_on_its_harnesss_wire(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(), tmp_path, proxy=fake)
    one, two = trials(tmp_path, "alpha", "alpha")
    first, second = held.connection(one), held.connection(two)
    assert first.model == second.model == f"openai/{MODEL}"
    address = f"{fake.origin}/r/lane/{AGENT}/trial/{one.name}/v1"
    assert first.env == {"OPENAI_BASE_URL": address, "OPENAI_API_KEY": "harness-token"}
    assert second.env["OPENAI_BASE_URL"] == f"{fake.origin}/r/lane/{AGENT}/trial/{two.name}/v1"
    assert first.host == "host.docker.internal"
    assert first.kwargs == {"model_api": "openai-completions"}
    first.kwargs["model_api"] = "written into"
    assert held.connection(one).kwargs == {"model_api": "openai-completions"}, "each its own"
    assert fake.addressed == [one.name, two.name, one.name]


def test_the_wire_and_the_env_follow_the_trials_harness(fake: FakeProxy, tmp_path: Path) -> None:
    [trial] = trials(tmp_path, "alpha", deployed="claude-code")
    filling = agent(served(fill_context=True), tmp_path, proxy=fake)
    found = filling.connection(trial)
    assert found.model == f"anthropic/{MODEL}"
    assert found.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": f"{fake.origin}/r/lane/{AGENT}/trial/{trial.name}",  # no /v1
        "ANTHROPIC_API_KEY": "harness-token",
    }
    plain = agent(served(), tmp_path, proxy=fake).connection(trial)
    assert "DISABLE_COMPACT" not in plain.env, "the profile's env only when filling"
    [trial] = trials(tmp_path, "alpha", deployed="opencode@1.18.35")
    named = agent(served(), tmp_path, proxy=fake).connection(trial)
    assert named.model == f"shipyard/{MODEL}"
    assert set(named.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    assert "opencode_config" in named.kwargs


# ------------------------------------------------------------------------ the records


async def test_record_takes_each_trials_records(fake: FakeProxy, tmp_path: Path) -> None:
    fake.answers = {
        "alpha": [made("one", "ok", seq=1), made("oneoktwo", "fine", seq=2)],
        "beta": [made("go", error="context", seq=1)],
    }
    held = agent(served(), tmp_path)
    await held.proxy.open()
    batch = trials(tmp_path, "alpha", "beta", "gamma")
    records = await held.record("j", batch)
    assert fake.fetched == [trial.name for trial in batch]
    assert [len(records[trial.name]) for trial in batch] == [2, 1, 1]


async def test_a_two_trial_job_is_dialled_and_recorded_under_two_names(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made("one", "ok")]}
    held = agent(served(), tmp_path)
    await held.proxy.open()
    batch = trials(tmp_path, "alpha", "alpha")
    assert len({held.connection(trial).env["OPENAI_BASE_URL"] for trial in batch}) == 2
    records = await held.record("j", batch)
    assert sorted(records) == sorted(trial.name for trial in batch)


async def test_records_served_from_weights_the_run_did_not_point_at_are_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made(weights="tinker://theirs/step-5")]}
    held = agent(served(), tmp_path)
    await held.proxy.open()
    with pytest.raises(ValueError, match="re-pointed"):
        await held.record("j", trials(tmp_path, "alpha"))
    await held.serve("tinker://w/step-2")
    fake.answers = {
        "alpha": [made(weights="tinker://w/step-2")],
        "beta": [made(weights="tinker://intruder/step-9")],
    }
    with pytest.raises(ValueError, match="intruder"):
        await held.record("j", trials(tmp_path, "alpha", "beta"))
    assert len(await held.record("j", trials(tmp_path, "alpha"))) == 1
    answered_by_ours("j", {"t": [made(weights=None)]}, None, lane=AGENT)
    with pytest.raises(ValueError, match="in lane agent, and the run pointed that lane at None"):
        answered_by_ours("j", {"t": [made(weights="tinker://x/step-1")]}, None, lane=AGENT)


async def test_a_fetch_that_fails_fails_the_job_rather_than_masking_a_trial(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.failing = {"beta"}
    held = agent(served(), tmp_path)
    await held.proxy.open()
    with pytest.raises(httpx.ConnectError):
        await held.record("j", trials(tmp_path, "alpha", "beta"))


async def test_check_stops_a_job_the_proxy_served_nothing_or_died_during(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.default = []
    held = agent(served(), tmp_path)
    await held.proxy.open()
    batch = trials(tmp_path, "alpha")
    taken = await held.record("j", batch)
    batch[0].records = Records(taken[batch[0].name])
    with pytest.raises(NothingServed) as caught:
        held.check("j", batch)
    assert str(caught.value) == f"job j: {DEAD}"
    batch[0].records = Records([made()])
    held.check("j", batch)
    fake.exited = "harness-proxy exited with 1"
    with pytest.raises(ProxyGone, match="exited with 1"):
        held.check("j", batch)


# ----------------------------------------------------------------------- a provider


async def test_a_providers_endpoint_gives_a_trial_the_model_alone_and_records_nothing(
    tmp_path: Path,
) -> None:
    held = ProviderEndpoint("openrouter/some-model", "openrouter")
    [trial] = trials(tmp_path, "alpha")
    assert held.connection(trial) == Connection(model="openrouter/some-model")
    assert held.budget is None and await held.record("j", [trial]) is None
    held.check("j", [trial])
    named = ProviderEndpoint("claude-x", "anthropic").connection(trial).model
    assert named == "anthropic/claude-x", "Harbor's agents take the model under its provider"


@pytest.mark.parametrize(
    ("endpoint", "routed"),
    [
        (ProviderEndpoint("anthropic/claude-x", "anthropic"), "anthropic/claude-x"),
        (ProviderEndpoint("claude-x", "anthropic"), "anthropic/claude-x"),
        (ProviderEndpoint("anthropic/claude-x", "openrouter"), "openrouter/anthropic/claude-x"),
    ],
)
def test_a_providers_endpoint_names_the_model_under_its_provider(
    endpoint: ProviderEndpoint, routed: str
) -> None:
    """A harness asks for the model under its provider, unless its name already starts with
    it."""
    assert endpoint.routed == routed
