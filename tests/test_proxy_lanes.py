"""Tests for the proxy's lanes as the run uses them: the run starts its proxy with a lane for
each model on Tinker or Together, each whole and sampled and rendered as its own model's table
says; a trial of a model is opened on its lane, on the weights the model serves then; a proxy
that does not serve a lane as the run made it is refused before any trial."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_proxy.serve import server_for
from harness_proxy.wire import CONTROL_TOKEN_ENV, Served
from proxies import MODEL
from tests.fakes import FakeProxy
from tests.service import FakeService
from tests.test_connect import agent_keys, connected, table
from tests.test_endpoints import AGENT, agent, served, trials

#: A model the run opened on Tinker: a checkpoint's weights, and the base model they are of.
WEIGHTS = "tinker://r/sampler_weights/final"
TEACHER = SimpleNamespace(model="Qwen/Qwen3-32B", provider="tinker", kind="tinker")
#: What a remote proxy reports when it serves the agent's lane and the teacher's.
REPORTED = {AGENT: Served(MODEL, None, ()), "teacher": Served("Qwen/Qwen3-32B", 32768, ())}
#: What a model's table sets for every call its lane samples.
SAMPLING = ("temperature", "top_p", "top_k", "max_tokens", "fill_context")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    return proxy


async def test_the_proxy_starts_with_each_lane_whole_and_a_trial_is_opened_on_its_own(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(max_context=4096), tmp_path).proxy
    lane = held.lane("teacher", WEIGHTS, opened=TEACHER)
    await held.open()
    [(_, settings, _)] = fake.placed
    ours, theirs = settings["lanes"]
    assert (ours["name"], ours["model"], ours["max_context"]) == (AGENT, MODEL, 4096)
    assert (theirs["name"], theirs["model"]) == ("teacher", "Qwen/Qwen3-32B")
    assert "max_context" not in theirs and "renderer" not in theirs, "the teacher's own"
    assert "weights" not in theirs, "each trial names the weights it is opened on"
    assert ours["metadata"] == theirs["metadata"] == {"shipyard_run": tmp_path.name}
    first, second = trials(tmp_path, "alpha", "beta")
    dialled = await lane.open(first)
    assert dialled.model == "openai/Qwen/Qwen3-32B"
    assert dialled.env["OPENAI_API_KEY"] == f"token-{first.name}"
    assert (lane.provider, lane.model, lane.told) == ("tinker", "Qwen/Qwen3-32B", WEIGHTS)
    lane.serve("tinker://r/sampler_weights/step-2")
    await lane.open(second)
    assert fake.opened == [
        (first.name, "teacher", None, WEIGHTS),
        (second.name, "teacher", None, "tinker://r/sampler_weights/step-2"),
    ]
    with pytest.raises(ValueError, match="two models would serve in the proxy's lane 'teacher'"):
        held.lane("teacher", WEIGHTS, opened=TEACHER)


async def test_each_lane_samples_and_renders_as_its_own_models_table_says(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy, _ = fakes
    own = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 2048, "fill_context": True}
    teacher = table("teacher", TEACHER.model, renderers=["low", "high"], **own)
    ours = agent_keys(temperature=0.7, renderers="tml_v0")
    held, _ = connected(tmp_path, teacher, **ours)
    await held.proxy.open()
    ((_, settings, _),) = proxy.placed
    agents, teachers = settings["lanes"]
    assert {key: teachers[key] for key in SAMPLING} == own
    assert teachers["renderer"] == ["low", "high"]
    assert {key: agents[key] for key in SAMPLING} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 8192,
        "fill_context": False,
    }, "the agent's table, its defaults for what it leaves out"
    assert agents["renderer"] == ["tml_v0"]
    held.proxy.close()
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "c")
    remote = {"concurrency = 2": 'concurrency = 2\nendpoint_url = "https://p.example"'}
    held, _ = connected(tmp_path, teacher, **ours, **remote)
    proxy.lanes = {
        AGENT: Served(MODEL, None, ("tml_v0",)),
        "teacher": Served(TEACHER.model, None, ("low",)),
    }
    with pytest.raises(
        RuntimeError, match=r"lane teacher with low, and \[models.teacher\] renderers names high"
    ):
        await held.proxy.open()


async def test_a_remote_proxy_that_does_not_serve_a_lane_as_the_run_made_it_is_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {CONTROL_TOKEN_ENV: "c"}
    held = agent(served(endpoint_url="https://p.example/"), tmp_path, environ=environ).proxy
    held.lane("teacher", WEIGHTS, opened=TEACHER)
    fake.lanes = {AGENT: REPORTED[AGENT]}
    with pytest.raises(RuntimeError, match="serves no lane for teacher"):
        await held.open()
    fake.lanes = {**REPORTED, AGENT: Served("Qwen/Qwen3-4B", None, ())}
    with pytest.raises(RuntimeError, match=f"serves Qwen/Qwen3-4B in lane agent, .* {MODEL}"):
        await held.open()
    fake.lanes = dict(REPORTED)
    await held.open()
    assert held.lanes["teacher"].budget == 32768 and held.lanes[AGENT].budget is None


def test_the_proxy_reads_each_lane_as_the_run_writes_it(tmp_path: Path) -> None:
    held = agent(served(renderers=["low", "high"], max_context=4096, top_k=40), tmp_path).proxy
    held.lanes[AGENT].backend = {"backend": "together", "session": "s1"}
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 2048}
    held.lane("teacher", WEIGHTS, opened=TEACHER, sampling={**sampling, "fill_context": True})
    given = {"lanes": [one.document for one in held.lanes.values()]}
    built = server_for(given, bind="127.0.0.1", port=0, advertise=None, control_token="c")
    ours, theirs = built.lanes
    assert (ours.name, ours.model, ours.renderer, ours.max_context) == (
        AGENT,
        MODEL,
        ["low", "high"],
        4096,
    )
    assert (ours.backend, ours.session, ours.metadata) == (
        "together",
        "s1",
        {"shipyard_run": tmp_path.name},
    )
    assert (ours.top_k, ours.max_tokens, ours.fill_context) == (40, 8192, False)
    assert (theirs.name, theirs.model, theirs.max_context) == ("teacher", "Qwen/Qwen3-32B", None)
    found = (theirs.temperature, theirs.top_p, theirs.top_k, theirs.max_tokens)
    assert found == tuple(sampling.values()) and theirs.fill_context, "the teacher's own"
