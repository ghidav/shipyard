"""Tests for `shipyard.models.endpoints`: the run's proxy, which the trained policy's harnesses
call, and a provider's model, which code asks through LiteLLM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from proxies import MODEL
from shipyard.config import Blueprint
from shipyard.groups import Trial
from shipyard.jobs.harnesses import PROFILES, harness
from shipyard.models.endpoints import (
    PROVIDER_CALLS,
    PROVIDER_RETRIES,
    PROXY_LOG,
    Connection,
    NothingServed,
    ProviderEndpoint,
    ProxyEndpoint,
    ProxyGone,
    answered_by_ours,
    placement,
    served_nothing,
    settings,
)
from tests.fakes import FakeProxy, made

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


# ------------------------------------------------------------------- where it runs


@pytest.mark.parametrize("sandbox", ["docker", "podman", "apple-container"])
def test_a_sandbox_on_this_machine_reaches_a_proxy_here(sandbox: str) -> None:
    assert placement(served(sandbox=sandbox).rollout) == "local"


@pytest.mark.parametrize("sandbox", ["modal", "e2b", "daytona"])
def test_a_sandbox_elsewhere_needs_a_tunnel_unless_a_url_is_named(sandbox: str) -> None:
    assert placement(served(sandbox=sandbox).rollout) == "tunnel"
    assert placement(served(sandbox=sandbox, endpoint_url="https://p.example").rollout) == "remote"
    assert placement(served(sandbox="docker", endpoint_url="https://p.example").rollout) == "remote"


def test_a_started_proxy_takes_the_rollout_knobs_and_its_sessions_tags_on_the_base_model() -> None:
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
    assert settings(cfg, "dapo__abc") == {
        "model": MODEL,
        "renderer": "tml_v0",
        "bind": "127.0.0.1",
        "bind_port": 8000,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 4096,
        "max_context": 32768,
        "fill_context": True,
        "metadata": {"shipyard_run": "dapo__abc"},
    }, "a checkpoint is not served from the start: the trained policy points the proxy next"
    bare = settings(served(), "r")
    assert bare["renderer"] is None and bare["max_context"] is None
    assert (bare["bind"], bare["bind_port"], bare["max_tokens"]) == ("0.0.0.0", 0, 8192)


def test_the_proxys_tokens_are_billed_to_tinker_or_to_tinker_at_its_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    held = ProxyEndpoint(served(), tmp_path)
    assert (held.provider, held.model) == ("tinker", MODEL)
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    assert held.provider == "tinker@api.fireworks.ai"


# ------------------------------------------------------------------- open and close


async def test_a_local_sandbox_starts_the_proxy_on_this_machine(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = ProxyEndpoint(served(host="proxy.lan"), tmp_path)
    await held.open()
    [(how, _, named)] = fake.placed
    assert how == "local" and named["host"] == "proxy.lan"
    assert named["log"].name == str(tmp_path / PROXY_LOG)
    assert held.told is None and fake.swapped == [] and fake.probed == 1
    assert held.proxy is fake and held.budget is None
    held.close()
    assert fake.closed == 1 and held.proxy is None
    assert (tmp_path / PROXY_LOG).is_file()
    held.close()
    assert fake.closed == 1


async def test_a_sandbox_elsewhere_gets_a_tunnel(fake: FakeProxy, tmp_path: Path) -> None:
    held = ProxyEndpoint(served(sandbox="modal"), tmp_path)
    await held.open()
    [(how, _, named)] = fake.placed
    assert how == "tunnel" and named["log"].name == str(tmp_path / PROXY_LOG)
    held.close()
    assert fake.closed == 1


async def test_a_remote_proxy_is_pointed_through_its_control_route(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    held = ProxyEndpoint(served(endpoint_url="https://p.example/"), tmp_path, environ=environ)
    await held.open()
    assert fake.placed == [("remote", "https://p.example/", {"token": "t", "control_token": "c"})]
    assert fake.probed == 1, "probed before any sandbox opens"
    assert held.told is None and fake.swapped == []
    assert not (tmp_path / PROXY_LOG).exists()
    await held.serve("tinker://w/step-5")
    assert fake.swapped == ["tinker://w/step-5"] and held.told == "tinker://w/step-5"
    await held.serve(None)
    assert held.told is None


async def test_a_remote_proxy_needs_both_of_its_tokens(fake: FakeProxy, tmp_path: Path) -> None:
    remote = served(endpoint_url="https://p.example")
    with pytest.raises(RuntimeError, match=CONTROL_TOKEN_ENV):
        await ProxyEndpoint(remote, tmp_path, environ={PROXY_TOKEN_ENV: "t"}).open()
    with pytest.raises(ValueError, match=PROXY_TOKEN_ENV):
        await ProxyEndpoint(remote, tmp_path, environ={}).open()


# ---------------------------------------------------------------------- the connection


def test_each_trial_dials_the_address_that_names_it_on_its_harnesss_wire(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = ProxyEndpoint(served(), tmp_path, proxy=fake)
    one, two = trials(tmp_path, "alpha", "alpha")
    first, second = held.connection(one), held.connection(two)
    assert first.model == second.model == f"openai/{MODEL}"
    address = f"{fake.origin}/r/trial/{one.name}/v1"
    assert first.env == {"OPENAI_BASE_URL": address, "OPENAI_API_KEY": "harness-token"}
    assert second.env["OPENAI_BASE_URL"] == f"{fake.origin}/r/trial/{two.name}/v1"
    assert first.host == "host.docker.internal"
    assert first.kwargs == {"model_api": "openai-completions"}
    assert first.kwargs is not PROFILES["pi"].kwargs
    assert fake.addressed == [one.name, two.name]


def test_the_wire_and_the_env_follow_the_trials_harness(fake: FakeProxy, tmp_path: Path) -> None:
    [trial] = trials(tmp_path, "alpha", deployed="claude-code")
    filling = ProxyEndpoint(served(fill_context=True), tmp_path, proxy=fake)
    found = filling.connection(trial)
    assert found.model == f"anthropic/{MODEL}"
    assert found.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": f"{fake.origin}/r/trial/{trial.name}",  # no /v1 on this wire
        "ANTHROPIC_API_KEY": "harness-token",
    }
    plain = ProxyEndpoint(served(), tmp_path, proxy=fake).connection(trial)
    assert "DISABLE_COMPACT" not in plain.env, "the profile's env only when filling"
    [trial] = trials(tmp_path, "alpha", deployed="opencode@1.18.35")
    named = ProxyEndpoint(served(), tmp_path, proxy=fake).connection(trial)
    assert named.model == f"shipyard/{MODEL}"
    assert set(named.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    assert "opencode_config" in named.kwargs


# ------------------------------------------------------------------------ the records


async def test_record_takes_each_trials_records(fake: FakeProxy, tmp_path: Path) -> None:
    fake.answers = {
        "alpha": [made("one", "ok", seq=1), made("oneoktwo", "fine", seq=2)],
        "beta": [made("go", error="context", seq=1)],
    }
    held = ProxyEndpoint(served(), tmp_path)
    await held.open()
    batch = trials(tmp_path, "alpha", "beta", "gamma")
    records = await held.record("j", batch)
    assert fake.fetched == [trial.name for trial in batch]
    assert [len(records[trial.name]) for trial in batch] == [2, 1, 1]


async def test_a_two_trial_job_is_dialled_and_recorded_under_two_names(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made("one", "ok")]}
    held = ProxyEndpoint(served(), tmp_path)
    await held.open()
    batch = trials(tmp_path, "alpha", "alpha")
    assert len({held.connection(trial).env["OPENAI_BASE_URL"] for trial in batch}) == 2
    records = await held.record("j", batch)
    assert sorted(records) == sorted(trial.name for trial in batch)


async def test_records_served_from_weights_the_run_did_not_point_at_are_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {"alpha": [made(served="tinker://theirs/step-5")]}
    held = ProxyEndpoint(served(), tmp_path)
    await held.open()
    with pytest.raises(ValueError, match="re-pointed"):
        await held.record("j", trials(tmp_path, "alpha"))
    await held.serve("tinker://w/step-2")
    fake.answers = {
        "alpha": [made(served="tinker://w/step-2")],
        "beta": [made(served="tinker://intruder/step-9")],
    }
    with pytest.raises(ValueError, match="intruder"):
        await held.record("j", trials(tmp_path, "alpha", "beta"))
    assert len(await held.record("j", trials(tmp_path, "alpha"))) == 1
    answered_by_ours("j", {"t": [made(served=None)]}, None)
    with pytest.raises(ValueError, match="pointed its proxy at None"):
        answered_by_ours("j", {"t": [made(served="tinker://x/step-1")]}, None)


async def test_a_fetch_that_fails_fails_the_job_rather_than_masking_a_trial(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.failing = {"beta"}
    held = ProxyEndpoint(served(), tmp_path)
    await held.open()
    with pytest.raises(httpx.ConnectError):
        await held.record("j", trials(tmp_path, "alpha", "beta"))


async def test_check_stops_a_job_the_proxy_served_nothing_or_died_during(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.default = []
    held = ProxyEndpoint(served(), tmp_path)
    await held.open()
    batch = trials(tmp_path, "alpha")
    taken = await held.record("j", batch)
    batch[0].records = taken[batch[0].name]
    with pytest.raises(NothingServed) as caught:
        held.check("j", batch)
    assert str(caught.value) == f"job j: {DEAD}"
    batch[0].records = [made()]
    held.check("j", batch)
    fake.exited = "harness-proxy exited with 1"
    with pytest.raises(ProxyGone, match="exited with 1"):
        held.check("j", batch)


def test_served_nothing_is_about_records_not_rewards(tmp_path: Path) -> None:
    def holding(*records: list[Any]) -> list[Trial]:
        found = trials(tmp_path, *(f"t{n}" for n in range(len(records))))
        for one, held in zip(found, records, strict=True):
            one.records = held
        return found

    assert served_nothing(holding([], [])) == DEAD
    failed = [made(error="sampler: ConnectError"), made(error="sampler: BadRequestError")]
    assert served_nothing(holding(failed)) == (
        "every request failed at the sampler (sampler: BadRequestError, sampler: ConnectError); "
        "the weights it serves are gone or the backend refuses them"
    )
    assert served_nothing(holding([made(error="context")])) is None, "a refusal is an answer"
    assert served_nothing(holding(failed, [made()])) is None, "one was served"
    assert served_nothing([]) is None, "no trials: nothing to serve"


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
async def test_a_providers_endpoint_asks_litellm_for_the_model_under_its_provider(
    endpoint: ProviderEndpoint, routed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is asked for under its provider, unless its name already starts with it. A
    cached count past the prompt is capped at it."""
    asked: list[tuple[str, dict[str, Any]]] = []

    async def acompletion(model: str, messages: list[Any], **params: Any) -> Any:
        asked.append((model, params))
        usage = SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=9),
        )
        message = SimpleNamespace(content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr("litellm.acompletion", acompletion)
    reply = await endpoint.sample([{"role": "user", "content": "hi"}], temperature=0.0)
    assert reply == ("", {"prefill": 0, "cache": 5, "sample": 2})
    assert asked == [(routed, {"num_retries": PROVIDER_RETRIES, "temperature": 0.0})]


async def test_a_providers_endpoint_has_only_a_few_calls_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search's burst of reflections meets no rate limit all at once."""
    flight = {"now": 0, "peak": 0}

    async def acompletion(model: str, messages: list[Any], **params: Any) -> Any:
        flight["now"] += 1
        flight["peak"] = max(flight["peak"], flight["now"])
        await asyncio.sleep(0.01)
        flight["now"] -= 1
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr("litellm.acompletion", acompletion)
    endpoint = ProviderEndpoint("claude-x", "anthropic")
    asked = [endpoint.sample([{"role": "user", "content": str(n)}]) for n in range(20)]
    assert len(await asyncio.gather(*asked)) == 20 and flight["peak"] == PROVIDER_CALLS
