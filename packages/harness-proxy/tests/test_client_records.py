"""Tests for the proxy client against a real server: opening and closing trials, readiness, and
trials on weights of their own."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from harness_proxy import client
from harness_proxy.client import CONTROL_TOKEN_ENV, Proxy, Unreachable
from harness_proxy.wire import Record
from proxies import MODEL, FakeSampler, by_path, ids, root, server


def of(started: Any, **named: Any) -> Proxy:
    """Build a `Proxy` client for a server started in this event loop."""
    fields = {"origin": root(started), "control_token": started.control_token}
    return Proxy(**{**fields, **named})


async def call(proxy: Proxy, token: str, *messages: dict[str, Any]) -> httpx.Response:
    """Send one OpenAI chat completion with `token` to the address every trial dials."""
    sent = list(messages) or [{"role": "user", "content": "go"}]
    async with httpx.AsyncClient() as http:
        return await http.post(
            f"{proxy.address}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "m", "messages": sent},
        )


async def test_closing_hands_the_records_back_whole_and_ends_the_trial() -> None:
    async with server(FakeSampler("ok")) as started:
        proxy = of(started)
        token = await proxy.open("t-1")
        await call(proxy, token, {"role": "user", "content": "one"})
        await call(
            proxy,
            token,
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "two"},
        )
        records = await proxy.close("t-1")
        again = await proxy.close("t-1")
        later = await call(proxy, token)
    assert [type(found) for found in records] == [Record, Record]
    assert [found.seq for found in records] == [1, 2]
    assert records[1].prompt_token_ids == ids("oneoktwo"), "the delta is read back whole"
    assert records[0].completion_token_ids == ids("ok")
    assert again == records, "a close asked again is answered as the first"
    assert later.status_code == 401, "a closed trial's token opens nothing"


async def test_a_trial_never_opened_can_be_neither_called_nor_closed() -> None:
    async with server(FakeSampler("ok")) as started:
        proxy = of(started)
        refused = await call(proxy, "made-up")
        assert refused.status_code == 401 and "no open trial holds this token" in refused.text
        with pytest.raises(RuntimeError, match="404"):
            await proxy.close("nobody")


async def test_an_open_asked_again_gets_its_token_and_one_on_other_terms_is_refused() -> None:
    async with server(by_path=by_path) as started:
        proxy = of(started)
        token = await proxy.open("t")
        assert await proxy.open("t") == token
        with pytest.raises(RuntimeError, match="open already"):
            await proxy.open("t", weights="tinker://w/step-1")


@pytest.mark.parametrize("trial", ["a b#c?d__x", "50%__y", "a%2Fb__z"])
async def test_a_trial_named_with_reserved_characters_is_served_whole(trial: str) -> None:
    async with server(FakeSampler("ok")) as started:
        proxy = of(started)
        assert (await call(proxy, await proxy.open(trial))).status_code == 200
        (record,) = await proxy.close(trial)
        assert record.completion_token_ids == ids("ok") and started.trial(trial) is None


class Counting(httpx.AsyncBaseTransport):
    """HTTP transport that counts its requests, and drops the first `lose` responses after the
    server has sent them."""

    def __init__(self, lose: int = 0) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.asked = 0
        self.lose = lose

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        answered = await self.inner.handle_async_request(request)
        self.asked += 1
        if self.asked <= self.lose:
            await answered.aread()
            await answered.aclose()
            raise httpx.ReadTimeout("the read timed out", request=request)
        return answered


async def test_an_open_or_a_close_whose_answer_was_lost_is_answered_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PAUSES", (0.0,))
    async with server(FakeSampler("ok")) as started:
        opening = Counting(lose=1)
        token = await of(started, transport=opening).open("t-1")
        await call(of(started), token)
        await call(of(started), token)
        closing = Counting(lose=1)
        records = await of(started, transport=closing).close("t-1")
    assert (opening.asked, closing.asked) == (2, 2)
    assert len(records) == 2, "the trial's records, not an empty or refused retry"


async def test_a_refused_token_is_not_retried_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "PAUSES", (0.0,))
    async with server(FakeSampler("ok")) as started:
        transport = Counting()
        with pytest.raises(RuntimeError, match="401"):
            await of(started, control_token="not-it", transport=transport).open("t-1")
        assert transport.asked == 1, "one request, then the 401 raised"
        assert started.trial("t-1") is None


def test_a_remote_proxy_needs_the_control_token() -> None:
    with pytest.raises(ValueError, match=CONTROL_TOKEN_ENV):
        Proxy.remote("http://127.0.0.1:9", "")


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
    proxy = Proxy(origin="http://127.0.0.1:9", control_token="c", transport=transport)
    with pytest.raises(httpx.ConnectError):
        await proxy.close("t-1")
    assert transport.asked == client.CONTROL_ATTEMPTS == 4


# ----------------------------------------------------------------------- the probe


async def test_ready_passes_against_a_proxy_that_answers_and_needs_no_token() -> None:
    async with server(FakeSampler("ok")) as started:
        await of(started, control_token="not-the-token").ready()


async def test_ready_raises_with_the_address_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "PROBE_PAUSE", 0.01)
    proxy = Proxy(origin="http://127.0.0.1:9", control_token="c", transport=NeverAnswers())
    with pytest.raises(Unreachable, match="the proxy at http://127.0.0.1:9 does not answer"):
        await proxy.ready(seconds=0.05)


# ------------------------------------------------------------- weights of their own


async def test_trials_open_on_other_weights_are_each_answered_from_theirs_at_once() -> None:
    async with server(by_path=by_path) as started:
        proxy = of(started)
        weights = "tinker://run/sampler_weights/step-3"
        old, new = await proxy.open("old"), await proxy.open("new", weights=weights)
        answered = await call(proxy, old)
        assert answered.json()["choices"][0]["message"]["content"] == "base"
        assert answered.json()["model"] == MODEL, "a reply names the weights that answered it"
        answered = await call(proxy, new)
        assert answered.json()["choices"][0]["message"]["content"] == "step-3"
        assert answered.json()["model"] == weights
        assert [r.weights for r in await proxy.close("old")] == [None]
        assert [r.weights for r in await proxy.close("new")] == [weights]


async def test_a_lane_holds_one_client_on_weights_while_a_trial_is_open_on_them() -> None:
    made: list[str | None] = []

    def counted(path: str | None) -> Any:
        made.append(path)
        return by_path(path)

    async with server(by_path=counted) as started:
        proxy = of(started)
        for name in ("a", "b"):
            await proxy.open(name, weights="tinker://w/step-1")
        assert made == ["tinker://w/step-1"], "one client, shared by the trials on them"
        await proxy.close("a")
        await proxy.close("b")
        await proxy.open("c", weights="tinker://w/step-1")
    assert made == ["tinker://w/step-1"] * 2, "let go with its last trial, then made again"


async def test_an_open_needs_the_control_token_and_weights_its_lane_can_serve() -> None:
    async with server(by_path=by_path, control_token="c") as started:
        with pytest.raises(RuntimeError, match="401"):
            await of(started, control_token="wrong").open("t")
        with pytest.raises(RuntimeError, match="is not a tinker:// path"):
            await of(started).open("t", weights="not-a-path")
        assert started.trial("t") is None
