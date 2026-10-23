"""Tests for the proxy's lanes: one proxy serves several models, each a lane a trial is opened
on, with its own sampler and settings; a trial that names no lane is opened on the only one,
and refused when there are several."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from harness_proxy.client import Proxy
from harness_proxy.serve import document, server_for
from harness_proxy.server import Lane
from harness_proxy.wire import HEALTH_PATH, Served
from proxies import (
    MODEL,
    FakeRenderer,
    FakeSampler,
    ask,
    by_path,
    close,
    ids,
    opened,
    recorded,
    root,
    server,
)

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


async def ask_lane(
    started: Any, lane: str, trial: str, *, weights: str | None = None
) -> httpx.Response:
    """Open `trial` on the lane `lane`, at `weights`, and send one OpenAI chat completion with
    its token, as a harness would."""
    token = await opened(started, trial, lane=lane, weights=weights)
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "messages": [{"role": "user", "content": "go"}]},
        )


def said(answered: httpx.Response) -> str:
    return str(answered.json()["choices"][0]["message"]["content"])


async def test_a_call_is_answered_by_the_lane_its_trial_was_opened_on_and_recorded_there() -> None:
    async with server(FakeSampler("ours"), others=[teacher(FakeSampler("taught"))]) as started:
        assert said(await ask_lane(started, "agent", "a")) == "ours"
        assert said(await ask_lane(started, "teacher", "b")) == "taught"
        proxy = Proxy(origin=root(started), control_token=started.control_token)
        with pytest.raises(RuntimeError, match="serves no lane 'critic'"):
            await proxy.open("c", lane="critic")
        with pytest.raises(RuntimeError, match="a trial names one: agent, teacher"):
            await proxy.open("d")
        assert [r.completion_token_ids for r in recorded(started, "a")] == [ids("ours")]
        assert [r.completion_token_ids for r in recorded(started, "b")] == [ids("taught")]
        closed = (await close(started, "b")).json()["records"]
        assert len(closed) == 1 and started.trial("b") is None


async def test_a_proxy_with_one_lane_opens_a_trial_that_names_none_on_it() -> None:
    async with server(FakeSampler("ours")) as started:
        assert said(await ask(started, "a")) == "ours"
        assert said(await ask_lane(started, "agent", "b")) == "ours"


async def test_a_trial_samples_the_weights_it_was_opened_on_in_its_own_lane() -> None:
    async with server(by_path=by_path, others=[teacher()]) as started:
        assert said(await ask_lane(started, "teacher", "t", weights="tinker://w/step-3")) == (
            "step-3"
        )
        assert said(await ask_lane(started, "agent", "a")) == "base", "the agent's are its own"
        assert said(await ask_lane(started, "teacher", "u", weights="tinker://w/step-4")) == (
            "step-4"
        )
        assert said(await ask_lane(started, "teacher", "t", weights="tinker://w/step-3")) == (
            "step-3"
        ), "a trial open on older weights keeps them"


async def test_the_health_check_reports_each_lane_alike() -> None:
    lanes = [teacher(max_context=4096)]
    async with server(others=lanes) as started, httpx.AsyncClient() as client:
        answer = (await client.get(f"{root(started)}{HEALTH_PATH}")).json()
        proxy = Proxy(origin=root(started), control_token="ctl")
        await proxy.ready()
    assert answer == {
        "status": "ok",
        "lanes": {
            "agent": {"model": MODEL, "budget": None, "renderers": []},
            "teacher": {"model": TEACHER, "budget": 4096, "renderers": []},
        },
    }
    assert proxy.lanes == {"agent": Served(MODEL, None, ()), "teacher": Served(TEACHER, 4096, ())}


def test_the_run_gives_each_lane_whole_and_nothing_is_filled_in() -> None:
    lanes = [
        {"name": "teacher", "model": TEACHER, "temperature": 0.7},
        {"name": "critic", "model": "Qwen/Qwen3-4B"},
    ]
    named: dict[str, Any] = dict(bind="127.0.0.1", port=0, advertise=None, control_token="c")
    built = server_for({"lanes": lanes, "keepalive_every": 5.0}, **named)
    taught, critic = built.lane("teacher"), built.lane("critic")
    assert (taught.model, taught.temperature) == (TEACHER, 0.7)
    assert critic.temperature == 1.0, "a lane's own, else its default"
    assert built.keepalive_every == 5.0
    with pytest.raises(ValueError, match="names cpus, which a lane does not take"):
        server_for({"lanes": [{"name": "x", "model": "y", "cpus": 2}]}, **named)
    with pytest.raises(ValueError, match="A lane's settings go in its entry of lanes"):
        server_for({"lanes": lanes, "temperature": 0.5}, **named)


def test_the_flags_are_a_document_of_one_lane() -> None:
    assert document(None, lane="agent", model="m", renderer="low") == {
        "lanes": [{"name": "agent", "model": "m", "renderer": "low"}]
    }
    assert document(None, lane="teacher", model="m", renderer=None) == {
        "lanes": [{"name": "teacher", "model": "m"}]
    }
    given = {"lanes": [{"name": "a", "model": "m"}]}
    assert document(given, lane="agent", model=None, renderer=None) == given
    with pytest.raises(ValueError, match="give one"):
        document(given, lane="agent", model="m", renderer=None)
    with pytest.raises(ValueError, match="describes the lane of a --model"):
        document(None, lane="agent", model=None, renderer="low")
    with pytest.raises(ValueError, match="a proxy serves a --model, or the lanes"):
        server_for(
            document(None, lane="agent", model=None, renderer=None),
            bind="127.0.0.1",
            port=0,
            advertise=None,
            control_token="c",
        )
