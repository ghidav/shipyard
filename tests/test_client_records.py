"""The run's client against a real endpoint on this loop: the records fetch with its retry
and `again=1`, the readiness probe, and the swap through the control route."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from shipyard.proxy import client
from shipyard.proxy.client import CONTROL_TOKEN_ENV, Proxy, Unreachable
from shipyard.proxy.wire import Record
from tests.proxies import FakeSampler, ask, endpoint, ids, root


def of(started: Any, **named: Any) -> Proxy:
    """The run's client for a real endpoint on this loop."""
    fields = {
        "origin": root(started),
        "token": started.token,
        "control_token": started.control_token,
    }
    return Proxy(**{**fields, **named})


async def test_records_come_back_as_records_with_their_ids_and_the_counters() -> None:
    async with endpoint(FakeSampler("ok")) as started:
        await ask(started, "t-1", messages=[{"role": "user", "content": "one"}])
        await ask(
            started,
            "t-1",
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "two"},
            ],
        )
        proxy = of(started)
        records = await proxy.records("t-1")
        again = await proxy.records("t-1")
        unknown = await proxy.records("nobody")
    assert [type(found) for found in records] == [Record, Record]
    assert [found.seq for found in records] == [1, 2]
    assert records[1].prompt_token_ids == ids("oneoktwo"), "the delta is read back whole"
    assert records[0].completion_token_ids == ids("ok")
    assert proxy.counters["t-1"] == {"turned_away": 0, "cut": 0, "spoke": 0}
    assert again == [] and unknown == [], "drained, and an unknown trial asked for nothing"


class LosingFirst(httpx.AsyncBaseTransport):
    """The proxy answers, and the first answer is lost on the way back."""

    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.asked: list[dict[str, str]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        answered = await self.inner.handle_async_request(request)
        self.asked.append(dict(request.url.params))
        if len(self.asked) == 1:
            await answered.aread()
            await answered.aclose()
            raise httpx.ReadTimeout("the read timed out", request=request)
        return answered


async def test_an_answer_lost_after_the_drain_is_asked_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PAUSES", (0.0,))
    async with endpoint(FakeSampler("ok")) as started:
        await ask(started, "t-1")
        await ask(started, "t-1")
        transport = LosingFirst()
        records = await of(started, transport=transport).records("t-1")
    assert len(records) == 2, "the drained records, not an empty retry"
    assert [asked.get("again") for asked in transport.asked] == [None, "1"]
    assert all(asked.get("delta") == "1" for asked in transport.asked)


async def test_a_refused_token_is_not_retried_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "PAUSES", (0.0,))
    async with endpoint(FakeSampler("ok")) as started:
        await ask(started, "t-1")
        transport = LosingFirst()
        transport.asked.append({})  # nothing is lost; the 401 comes straight back
        with pytest.raises(httpx.HTTPStatusError):
            await of(started, token="not-the-token", transport=transport).records("t-1")
        assert len(transport.asked) == 2, "one request, then the 401 raised"
        assert len(started.records_for("t-1")) == 1, "nothing was drained"


class NeverAnswers(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.asked = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.asked += 1
        raise httpx.ConnectError("nothing there", request=request)


async def test_every_attempt_failing_raises_rather_than_answering_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PAUSES", (0.0,))
    transport = NeverAnswers()
    proxy = Proxy(origin="http://127.0.0.1:9", token="k", transport=transport)
    with pytest.raises(httpx.ConnectError):
        await proxy.records("t-1")
    assert transport.asked == client.RECORDS_ATTEMPTS == 4
    assert "t-1" not in proxy.counters


# ----------------------------------------------------------------------- the probe


async def test_ready_passes_against_a_proxy_that_answers_and_needs_no_token() -> None:
    async with endpoint(FakeSampler("ok")) as started:
        await of(started, token="not-the-token").ready()


async def test_ready_raises_with_the_address_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PROBE_PAUSE", 0.01)
    proxy = Proxy(origin="http://127.0.0.1:9", token="k", transport=NeverAnswers())
    with pytest.raises(Unreachable, match="the proxy at http://127.0.0.1:9 does not answer"):
        await proxy.ready(seconds=0.05)


# ------------------------------------------------------------------------ the swap


def by_path(path: str | None) -> FakeSampler:
    return FakeSampler("base" if path is None else path.rsplit("/", 1)[-1])


async def test_swap_goes_through_the_control_route_and_changes_what_answers() -> None:
    async with endpoint(by_path=by_path, control_token="c") as started:
        proxy = of(started)
        await proxy.swap("tinker://run/sampler_weights/step-3")
        answered = await ask(started, "t")
        assert answered.json()["choices"][0]["message"]["content"] == "step-3"
        await proxy.swap(None)
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "base"
        assert [r.served for r in started.records_for("t")] == [
            "tinker://run/sampler_weights/step-3",
            None,
        ]


async def test_swap_is_refused_without_the_control_token_and_on_a_wrong_one() -> None:
    async with endpoint(by_path=by_path, control_token="c") as started:
        with pytest.raises(RuntimeError, match=CONTROL_TOKEN_ENV):
            await of(started, control_token=None).swap("tinker://w/step-1")
        with pytest.raises(RuntimeError, match="401"):
            await of(started, control_token="wrong").swap("tinker://w/step-1")
        with pytest.raises(RuntimeError, match="refused to serve"):
            await of(started).swap("not-a-path")
        assert started.records_for("t") == []
