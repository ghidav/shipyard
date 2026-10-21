"""Tests for the proxy's lanes: one proxy serves several models, each a lane a call's address
names, with its own weights, sampler and records; a call that names none reaches the only lane,
and is refused when there are several."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from harness_proxy.client import Proxy
from harness_proxy.serve import document, server_for
from harness_proxy.server import Lane
from harness_proxy.wire import CONTROL_PATH, HEALTH_PATH, Address, Served
from proxies import MODEL, FakeRenderer, FakeSampler, ask, by_path, fetch, ids, root, server

TEACHER = "Qwen/Qwen3-32B"


def teacher(sampler: Any = None, **overrides: Any) -> Lane:
    """A lane on fake sampling clients: `sampler`, or one per weights path, as `by_path`."""

    async def factory(path: str | None) -> Any:
        return sampler if sampler is not None else by_path(path)

    fields = {
        "name": "teacher",
        "model": TEACHER,
        "renderer": FakeRenderer(),
        "client_factory": factory,
    }
    return Lane(**{**fields, **overrides})


async def ask_lane(started: Any, lane: str, trial: str) -> httpx.Response:
    """Send one OpenAI chat completion for `trial` to the lane `lane`, as a harness would."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}{Address(trial, lane).path}/chat/completions",
            headers={"Authorization": f"Bearer {started.token}"},
            json={"model": "m", "messages": [{"role": "user", "content": "go"}]},
        )


async def swap(started: Any, body: dict[str, Any]) -> httpx.Response:
    """Post `body` to the control route, as the run does."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}{CONTROL_PATH}",
            headers={"Authorization": f"Bearer {started.control_token}"},
            json=body,
        )


def said(answered: httpx.Response) -> str:
    return str(answered.json()["choices"][0]["message"]["content"])


async def test_a_call_is_answered_by_the_lane_its_address_names_and_recorded_there() -> None:
    async with server(FakeSampler("ours"), others=[teacher(FakeSampler("taught"))]) as started:
        assert said(await ask_lane(started, "agent", "a")) == "ours"
        assert said(await ask_lane(started, "teacher", "b")) == "taught"
        unknown = await ask_lane(started, "critic", "c")
        assert unknown.status_code == 400 and "serves no lane 'critic'" in unknown.text
        bare = await ask(started, "d")
        assert bare.status_code == 400 and "a call names one: agent, teacher" in bare.text
        agent, taught = started.lane("agent"), started.lane("teacher")
        assert [r.completion_token_ids for r in agent.records_for("a")] == [ids("ours")]
        assert agent.records_for("b") == [], "the teacher's call is the teacher's"
        assert [r.completion_token_ids for r in taught.records_for("b")] == [ids("taught")]
        fetched = (await fetch(started, "b")).json()["records"]
        assert len(fetched) == 1, "the records route finds a trial in whichever lane it called"
        assert taught.records_for("b") == [], "and drains them"


async def test_a_proxy_with_one_lane_answers_an_address_that_names_none() -> None:
    async with server(FakeSampler("ours")) as started:
        assert said(await ask(started, "a")) == "ours"
        assert said(await ask_lane(started, "agent", "b")) == "ours"


async def test_the_control_route_swaps_the_lane_its_body_names() -> None:
    async with server(by_path=by_path, others=[teacher()]) as started:
        swapped = await swap(started, {"lane": "teacher", "weights": "tinker://w/step-3"})
        unknown = await swap(started, {"lane": "critic", "weights": None})
        unnamed = await swap(started, {"weights": None})
        assert swapped.status_code == 200
        assert unknown.status_code == unnamed.status_code == 400
        assert said(await ask_lane(started, "teacher", "t")) == "step-3"
        assert said(await ask_lane(started, "agent", "t")) == "base", "the agent did not move"
        assert started.lane("agent").weights is None
        assert started.lane("teacher").weights == "tinker://w/step-3"
        proxy = Proxy(origin=root(started), token=started.token, control_token="ctl")
        await proxy.swap("tinker://w/step-4", lane="teacher")
        assert said(await ask_lane(started, "teacher", "t")) == "step-4"


async def test_the_health_check_reports_each_lane_alike() -> None:
    lanes = [teacher(weights="tinker://w/final", max_context=4096)]
    async with server(others=lanes) as started, httpx.AsyncClient() as client:
        answer = (await client.get(f"{root(started)}{HEALTH_PATH}")).json()
        proxy = Proxy(origin=root(started), token=started.token, control_token="ctl")
        await proxy.ready()
    assert answer == {
        "status": "ok",
        "lanes": {
            "agent": {"model": MODEL, "serving": MODEL, "budget": None, "renderers": []},
            "teacher": {
                "model": TEACHER,
                "serving": "tinker://w/final",
                "budget": 4096,
                "renderers": [],
            },
        },
    }
    assert proxy.lanes == {
        "agent": Served(MODEL, MODEL, None, ()),
        "teacher": Served(TEACHER, "tinker://w/final", 4096, ()),
    }


def test_the_run_gives_each_lane_whole_and_nothing_is_filled_in() -> None:
    lanes = [
        {"name": "teacher", "model": TEACHER, "weights": "tinker://w/final", "temperature": 0.7},
        {"name": "critic", "model": "Qwen/Qwen3-4B"},
    ]
    built = server_for(
        {"lanes": lanes, "keepalive_every": 5.0},
        bind="127.0.0.1",
        port=0,
        advertise=None,
        token="t",
        control_token="c",
    )
    taught, critic = built.lane("teacher"), built.lane("critic")
    assert (taught.model, taught.weights, taught.temperature) == (TEACHER, "tinker://w/final", 0.7)
    assert (critic.temperature, critic.weights) == (1.0, None), "a lane's own, else its default"
    assert built.keepalive_every == 5.0
    with pytest.raises(ValueError, match="names cpus, which a lane does not take"):
        server_for(
            {"lanes": [{"name": "x", "model": "y", "cpus": 2}]},
            bind="127.0.0.1",
            port=0,
            advertise=None,
            token="t",
            control_token="c",
        )
    with pytest.raises(ValueError, match="A lane's settings go in its entry of lanes"):
        server_for(
            {"lanes": lanes, "temperature": 0.5},
            bind="127.0.0.1",
            port=0,
            advertise=None,
            token="t",
            control_token="c",
        )


def test_the_flags_are_a_document_of_one_lane() -> None:
    assert document(None, lane="agent", model="m", weights="tinker://w/1", renderer="low") == {
        "lanes": [{"name": "agent", "model": "m", "weights": "tinker://w/1", "renderer": "low"}]
    }
    assert document(None, lane="teacher", model="m", weights=None, renderer=None) == {
        "lanes": [{"name": "teacher", "model": "m"}]
    }
    given = {"lanes": [{"name": "a", "model": "m"}]}
    assert document(given, lane="agent", model=None, weights=None, renderer=None) == given
    with pytest.raises(ValueError, match="give one"):
        document(given, lane="agent", model="m", weights=None, renderer=None)
    with pytest.raises(ValueError, match="describe the lane of a --model"):
        document(None, lane="agent", model=None, weights="tinker://w/1", renderer=None)
    with pytest.raises(ValueError, match="a proxy serves a --model, or the lanes"):
        server_for(
            document(None, lane="agent", model=None, weights=None, renderer=None),
            bind="127.0.0.1",
            port=0,
            advertise=None,
            token="t",
            control_token="c",
        )


def test_the_client_addresses_a_lane_and_names_it_when_it_swaps() -> None:
    proxy = Proxy(origin="http://h:1", token="t", control_token="c")
    assert proxy.address_for("t1") == "http://h:1/r/trial/t1/v1"
    assert proxy.address_for("t1", "low", lane="teacher") == (
        "http://h:1/r/lane/teacher/trial/t1/renderer/low/v1"
    )
    with pytest.raises(ValueError, match="a lane goes in the URL path"):
        proxy.address_for("t1", lane="a/b")
