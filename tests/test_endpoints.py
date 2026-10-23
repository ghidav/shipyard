"""Tests for `shipyard.models.endpoints`: the run's proxy, which serves each model on Tinker or
Together to its harnesses in a lane of its own, and a provider's model, which a harness calls
with its own credentials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from harness_proxy.wire import CONTROL_TOKEN_ENV, Served
from proxies import MODEL
from shipyard.config import Config, ModelTable
from shipyard.groups import Connection, Trial
from shipyard.jobs.harnesses import Harness
from shipyard.models.clients.tinker import TinkerClient
from shipyard.models.endpoints import (
    PREFLIGHT_TOKEN,
    PROXY_LOG,
    LaneEndpoint,
    NothingServed,
    ProviderEndpoint,
    ProxyEndpoint,
    ProxyGone,
    placement,
)
from shipyard.records import Records
from tests.fakes import FakeProxy, made
from tests.service import FakeService

#: The agent's role, which names its lane.
AGENT = "agent"
DEAD = "no trial reached the proxy; the endpoint is dead or unreachable"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    return proxy


def served(**keys: Any) -> Config:
    """A config: the keywords a model's table takes go to `[models.agent]`, the others to
    `[rollout]`."""
    model = {key: keys.pop(key) for key in list(keys) if key in ModelTable.model_fields}
    return Config.model_validate(
        {
            "models": {AGENT: {"model": MODEL, **model}},
            "data": {"train": "d", "batch_size": 1},
            "rollout": keys,
            "recipe": {"steps": 1},
        }
    )


def trials(tmp_path: Path, *tasks: str, deployed: str = "pi") -> list[Trial]:
    """One trial of job `j` per task, named as Harbor names a trial's directory."""
    jobs = tmp_path / "jobs" / "j"
    return [
        Trial("j", Path(task), Harness(name=deployed), jobs / f"{task}__{n:07d}")
        for n, task in enumerate(tasks)
    ]


def agent(config: Config, directory: Path, **named: Any) -> LaneEndpoint:
    """The agent's lane of a run's proxy, made from its table as `connect` makes it."""
    table = config.models[AGENT]
    return ProxyEndpoint(config, directory, **named).lane(
        AGENT,
        table.model,
        renderers=table.renderers,
        max_context=table.max_context or None,
        sampling=table.sampling,
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


def test_the_agents_lane_takes_its_models_sampling_and_its_sessions_tags(tmp_path: Path) -> None:
    cfg = served(
        from_checkpoint="tinker://w/step-9",
        renderers="tml_v0",
        bind="127.0.0.1",
        bind_port=8000,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=4096,
        max_context=32768,
        fill_context=True,
    )
    lane = agent(cfg, tmp_path / "dapo__abc")
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
    }, "a lane names no weights: each trial is opened on its own"
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


# ---------------------------------------------------------------- starting and stopping


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
    assert held.told is None and fake.opened == [] and fake.probed == 1
    assert held.proxy.proxy is fake and held.budget is None
    held.proxy.close()
    assert fake.stopped == 1 and held.proxy.proxy is None
    assert (tmp_path / PROXY_LOG).is_file()
    held.proxy.close()
    assert fake.stopped == 1


async def test_a_sandbox_elsewhere_gets_a_tunnel(fake: FakeProxy, tmp_path: Path) -> None:
    held = agent(served(sandbox="modal"), tmp_path)
    await held.proxy.open()
    [(how, _, named)] = fake.placed
    assert how == "tunnel" and named["log"].name == str(tmp_path / PROXY_LOG)
    held.proxy.close()
    assert fake.stopped == 1


async def test_a_remote_proxy_is_reached_with_its_control_token(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(
        served(endpoint_url="https://p.example/"), tmp_path, environ={CONTROL_TOKEN_ENV: "c"}
    )
    fake.lanes = {AGENT: Served(MODEL, None, ())}
    await held.proxy.open()
    assert fake.placed == [("remote", "https://p.example/", {"control_token": "c"})]
    assert fake.probed == 1, "probed before any sandbox opens"
    assert not (tmp_path / PROXY_LOG).exists()
    with pytest.raises(ValueError, match=CONTROL_TOKEN_ENV):
        await agent(served(endpoint_url="https://p.example"), tmp_path, environ={}).proxy.open()


# ------------------------------------------------------------------------ a trial's life


async def test_each_trial_dials_the_proxy_with_the_token_its_opening_returned(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(), tmp_path, proxy=fake)
    one, two = trials(tmp_path, "alpha", "alpha")
    first, second = await held.open(one), await held.open(two)
    assert first.model == second.model == f"openai/{MODEL}"
    assert first.env == {
        "OPENAI_BASE_URL": f"{fake.origin}/v1",
        "OPENAI_API_KEY": f"token-{one.name}",
    }
    assert second.env["OPENAI_API_KEY"] == f"token-{two.name}", "each trial its own token"
    assert first.host == "host.docker.internal"
    assert first.kwargs == {"model_api": "openai-completions"}
    first.kwargs["model_api"] = "written into"
    assert held.connection(one).kwargs == {"model_api": "openai-completions"}, "each its own"
    assert held.connection(one).env["OPENAI_API_KEY"] == PREFLIGHT_TOKEN, "a preflight opens none"
    assert fake.opened == [(one.name, AGENT, None, None), (two.name, AGENT, None, None)]


async def test_the_wire_and_the_env_follow_the_trials_harness(
    fake: FakeProxy, tmp_path: Path
) -> None:
    [trial] = trials(tmp_path, "alpha", deployed="claude-code")
    filling = agent(served(fill_context=True), tmp_path, proxy=fake)
    found = await filling.open(trial)
    assert found.model == f"anthropic/{MODEL}"
    assert found.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": fake.origin,  # no /v1
        "ANTHROPIC_API_KEY": f"token-{trial.name}",
    }
    plain = agent(served(), tmp_path, proxy=fake).connection(trial)
    assert "DISABLE_COMPACT" not in plain.env, "the profile's env only when filling"
    [trial] = trials(tmp_path, "alpha", deployed="opencode@1.18.35")
    named = agent(served(), tmp_path, proxy=fake).connection(trial)
    assert named.model == f"shipyard/{MODEL}"
    assert set(named.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    assert "opencode_config" in named.kwargs


async def test_closing_a_trial_takes_its_records(fake: FakeProxy, tmp_path: Path) -> None:
    fake.answers = {
        "alpha": [made("one", "ok", seq=1), made("oneoktwo", "fine", seq=2)],
        "beta": [made("go", error="context", seq=1)],
    }
    held = agent(served(), tmp_path)
    await held.proxy.open()
    batch = trials(tmp_path, "alpha", "beta", "gamma")
    for trial in batch:
        await held.open(trial)
    assert held.trials == {trial.name for trial in batch}
    taken = [await held.close(trial) for trial in batch]
    assert fake.closed == [trial.name for trial in batch] and held.trials == set()
    assert [len(records) for records in taken] == [2, 1, 1]


async def test_a_trial_keeps_the_weights_it_was_opened_on_and_one_opened_later_takes_the_new(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(), tmp_path)
    await held.proxy.open()
    early, late, last = trials(tmp_path, "alpha", "beta", "gamma")
    await held.open(early)
    held.serve("tinker://w/step-2")
    await held.open(late)
    held.serve(None)
    await held.open(last)
    assert fake.weights == [None, "tinker://w/step-2", None]
    held.publishing()  # Tinker serves several versions at once: a trial open is no matter


async def test_together_refuses_to_publish_while_one_of_its_trials_is_open(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(), tmp_path)
    held.backend = {"backend": "together", "session": "s1"}
    await held.proxy.open()
    [trial] = trials(tmp_path, "alpha")
    held.publishing()
    await held.open(trial)
    with pytest.raises(RuntimeError, match="1 of its trials are open on the ones before"):
        held.publishing()
    await held.close(trial)
    held.publishing()


async def test_a_close_that_fails_fails_the_run_rather_than_masking_a_trial(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.failing = {"beta"}
    held = agent(served(), tmp_path)
    await held.proxy.open()
    alpha, beta = trials(tmp_path, "alpha", "beta")
    await held.open(alpha)
    await held.open(beta)
    await held.close(alpha)
    with pytest.raises(httpx.ConnectError):
        await held.close(beta)
    assert held.trials == set()
    fake.exited = "harness-proxy exited with 1"
    with pytest.raises(ProxyGone, match="exited with 1"):
        await held.open(alpha)


async def test_check_stops_a_job_the_proxy_served_nothing_or_died_during(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.default = []
    held = agent(served(), tmp_path)
    await held.proxy.open()
    batch = trials(tmp_path, "alpha")
    await held.open(batch[0])
    batch[0].records = await held.close(batch[0])
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
    dialled = await held.open(trial)
    assert dialled == held.connection(trial) == Connection(model="openrouter/some-model")
    assert held.budget is None and not (await held.close(trial)).recorded
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
