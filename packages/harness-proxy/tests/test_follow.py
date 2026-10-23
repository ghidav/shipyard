"""Tests of a trial that follows its lane: each call takes the weights the lane's model
published last when the call starts, a call in flight keeps the weights it started with, and
each record names the weights that answered it. A trial opened on weights stays on them, as an
evaluation's does."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx

from harness_proxy.server import Lane, Server
from harness_proxy.wire import TRIALS_PATH, Opening, Published, serving_path
from proxies import (
    MODEL,
    FakeRenderer,
    FakeResponse,
    FakeSampler,
    FakeSequence,
    by_path,
    only,
    opened,
    recorded,
    root,
    server,
)


class Held:
    """Fake sampling client whose reply waits until the test lets it go."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.started, self.go = asyncio.Event(), asyncio.Event()

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.started.set()
        await self.go.wait()
        tokens = [ord(one) for one in self.answer]
        return FakeResponse([FakeSequence(tokens, [-0.5] * len(tokens))])


async def following(started: Server, trial: str) -> str:
    """Open `trial` to follow the only lane, as the run opens a training trial; its token."""
    async with httpx.AsyncClient() as client:
        answered = await client.post(
            f"{root(started)}{TRIALS_PATH}",
            headers={"Authorization": f"Bearer {started.control_token}"},
            json=Opening(trial, follow=True).body,
        )
    assert answered.status_code == 200, answered.text
    return str(answered.json()["token"])


async def publish(
    started: Server,
    weights: str | None,
    version: int | None = None,
    *,
    token: str | None = None,
    lane: str = "agent",
    body: Any = None,
) -> httpx.Response:
    """Publish a lane's new weights on the control route, as the run does after an update."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{root(started)}{serving_path(lane)}",
            headers={"Authorization": f"Bearer {token or started.control_token}"},
            json=Published(weights, version).body if body is None else body,
        )


async def call(started: Server, token: str) -> str:
    """Make one call with the trial's token, as its harness would; the reply's text."""
    async with httpx.AsyncClient() as client:
        answered = await client.post(
            f"{root(started)}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "messages": [{"role": "user", "content": "go"}]},
        )
    assert answered.status_code == 200, answered.text
    return str(answered.json()["choices"][0]["message"]["content"])


async def test_a_trial_that_follows_its_lane_takes_the_weights_published_last_at_each_call() -> (
    None
):
    async with server(by_path=by_path) as started:
        assert (await publish(started, "tinker://w/start")).status_code == 200
        token = await following(started, "t")
        first = await call(started, token)
        await publish(started, "tinker://w/sample-1")
        second = await call(started, token)
        weights = [one.weights for one in recorded(started, "t")]
    assert (first, second) == ("start", "sample-1")
    assert weights == ["tinker://w/start", "tinker://w/sample-1"], "each call's own weights"


async def test_a_call_in_flight_keeps_the_weights_it_started_with_and_lets_them_go_after() -> None:
    held = Held("start")
    samplers = {"tinker://w/start": held, "tinker://w/sample-1": FakeSampler("sample-1")}
    async with server(by_path=samplers.__getitem__) as started:
        await publish(started, "tinker://w/start")
        token = await following(started, "t")
        flying = asyncio.create_task(call(started, token))
        await held.started.wait()
        await publish(started, "tinker://w/sample-1")
        after = await call(started, token)
        held.go.set()
        before = await flying
        weights = {one.weights for one in recorded(started, "t")}
        clients = set(only(started)._clients)
    assert (before, after) == ("start", "sample-1")
    assert weights == {"tinker://w/start", "tinker://w/sample-1"}
    assert clients == set(), "a following trial holds no weights between its calls"


async def test_a_trial_opened_on_weights_stays_on_them_after_a_publish() -> None:
    async with server(by_path=by_path) as started:
        await publish(started, "tinker://w/sample-1")
        token = await opened(started, "evaluation", weights="tinker://w/start")
        await publish(started, "tinker://w/sample-2")
        reply = await call(started, token)
        weights = [one.weights for one in recorded(started, "evaluation")]
    assert reply == "start" and weights == ["tinker://w/start"]


async def test_only_the_run_moves_a_lanes_weights_and_only_to_weights_it_can_serve() -> None:
    async with server() as started:
        assert (await publish(started, "tinker://w/1", token="a-trials-token")).status_code == 401
        assert (await publish(started, "tinker://w/1", lane="critic")).status_code == 400
        assert (await publish(started, "elsewhere://w/1")).status_code == 400
        assert (await publish(started, None, body={"weights": 3})).status_code == 400
        assert only(started).served is None, "nothing moved"


def test_a_lane_that_serves_one_version_names_a_calls_weights_by_the_version_that_wrote_it() -> (
    None
):
    lane = Lane(name="agent", model=MODEL, backend="together", renderer=FakeRenderer())
    lane.publish(Published("together://s/start"))  # the weights the run starts from: no version
    lane.publish(Published("together://s/sample-1", version=4))
    lane.publish(Published("together://s/sample-2", version=8))

    def named(*versions: int) -> str | None:
        return lane.answered("together://s/sample-2", SimpleNamespace(versions=versions))

    assert named(0) == "together://s/start", "a version before every one given"
    assert named(4) == named(6) == "together://s/sample-1"
    assert named(8) == named(9) == "together://s/sample-2"
    assert named() == "together://s/sample-2", "no version said: the weights the call took"
    fresh = Lane(name="agent", model=MODEL, backend="together", renderer=FakeRenderer())
    fresh.publish(Published("together://s/sample-1", version=4))
    answer = SimpleNamespace(versions=(2,))
    assert fresh.answered("together://s/sample-1", answer) is None, "the base model, before it"
