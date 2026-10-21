"""Tests for the proxy's lanes as the run uses them: the run starts its proxy with a lane for
each model on Tinker, each whole and alike; a trial of a model dials its lane by name, and its
records are checked against the lane's weights; a proxy that does not serve a lane as the run
made it is refused before any trial."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_proxy.serve import server_for
from harness_proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV, Served
from proxies import MODEL, made
from shipyard.run.logs import AGENT
from tests.fakes import FakeProxy
from tests.test_endpoints import agent, served, trials

#: A model the run opened on Tinker: a checkpoint's weights, and the base model they are of.
WEIGHTS = "tinker://r/sampler_weights/final"
TEACHER = SimpleNamespace(model="Qwen/Qwen3-32B", provider="tinker", kind="tinker")
#: What a remote proxy reports when it serves the agent's lane and the teacher's.
REPORTED = {
    AGENT: Served(MODEL, MODEL, None, ()),
    "teacher": Served("Qwen/Qwen3-32B", WEIGHTS, 32768, ()),
}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    return proxy


async def test_the_proxy_starts_with_each_lane_whole_and_a_trial_dials_its_own(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = agent(served(max_context=4096), tmp_path).proxy
    lane = held.lane("teacher", WEIGHTS, opened=TEACHER)
    await held.open()
    [(_, settings, _)] = fake.placed
    ours, theirs = settings["lanes"]
    assert (ours["name"], ours["model"], ours["max_context"]) == (AGENT, MODEL, 4096)
    assert (theirs["name"], theirs["model"], theirs["weights"]) == (
        "teacher",
        "Qwen/Qwen3-32B",
        WEIGHTS,
    )
    assert "max_context" not in theirs and "renderer" not in theirs, "the agent's own"
    shared = ("temperature", "top_p", "top_k", "max_tokens", "fill_context", "metadata")
    assert all(theirs[key] == ours[key] for key in shared), "the run's sampling, in each"
    (trial,) = trials(tmp_path, "alpha")
    dialled = lane.connection(trial)
    assert dialled.model == "openai/Qwen/Qwen3-32B"
    assert "/r/lane/teacher/trial/" in dialled.env["OPENAI_BASE_URL"]
    assert (lane.provider, lane.model, lane.told) == ("tinker", "Qwen/Qwen3-32B", WEIGHTS)
    await lane.serve("tinker://r/sampler_weights/step-2")
    assert fake.lane_swaps == [("teacher", "tinker://r/sampler_weights/step-2")]
    fake.default = [made(served="tinker://r/sampler_weights/step-2")]
    assert await lane.record("j", [trial]), "answered by the lane's own weights"
    fake.default = [made(served="tinker://w/sample-1")]
    with pytest.raises(ValueError, match="in lane teacher, and the run pointed that lane"):
        await lane.record("j", [trial])
    with pytest.raises(ValueError, match="two models would serve in the proxy's lane 'teacher'"):
        held.lane("teacher", WEIGHTS, opened=TEACHER)


async def test_a_remote_proxy_that_does_not_serve_a_lane_as_the_run_made_it_is_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    held = agent(served(endpoint_url="https://p.example/"), tmp_path, environ=environ).proxy
    held.lane("teacher", WEIGHTS, opened=TEACHER)
    fake.lanes = {AGENT: REPORTED[AGENT]}
    with pytest.raises(RuntimeError, match="serves no lane for teacher"):
        await held.open()
    fake.lanes = {**REPORTED, AGENT: Served("Qwen/Qwen3-4B", "Qwen/Qwen3-4B", None, ())}
    with pytest.raises(RuntimeError, match=f"serves Qwen/Qwen3-4B in lane agent, .* {MODEL}"):
        await held.open()
    fake.lanes = dict(REPORTED)
    await held.open()
    assert held.lanes["teacher"].budget == 32768 and held.lanes[AGENT].budget is None


def test_the_proxy_reads_each_lane_as_the_run_writes_it(tmp_path: Path) -> None:
    held = agent(served(renderer=["low", "high"], max_context=4096), tmp_path).proxy
    held.lanes[AGENT].backend = {"backend": "together", "session": "s1"}
    held.lane("teacher", WEIGHTS, opened=TEACHER)
    given = {"lanes": [one.document for one in held.lanes.values()]}
    built = server_for(
        given, bind="127.0.0.1", port=0, advertise=None, token="t", control_token="c"
    )
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
    assert (theirs.name, theirs.model, theirs.weights, theirs.max_context) == (
        "teacher",
        "Qwen/Qwen3-32B",
        WEIGHTS,
        None,
    )
