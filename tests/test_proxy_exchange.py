"""One request through the app: the SDK's headers on the record, and a retry or idempotent
request replayed from the stored reply even after a budget cut."""

from __future__ import annotations

from tests.proxies import FakeSampler, ask, endpoint

# ------------------------------------------------------------------- the exchange


async def test_the_headers_the_sdk_sends_land_on_the_record() -> None:
    async with endpoint(FakeSampler("ok")) as started:
        await ask(started, "t", headers={"x-request-id": "req-1"})
        await ask(started, "t", headers={"Idempotency-Key": "idem-2", "x-request-id": "req-2"})
        await ask(started, "t")
    assert [r.request_id for r in started.records_for("t")] == ["req-1", "idem-2", None]


async def test_a_retry_is_answered_with_the_stored_reply_not_a_second_sample() -> None:
    sampler = FakeSampler("ok")
    async with endpoint(sampler) as started:
        first = await ask(started, "t")
        again = await ask(started, "t", headers={"x-stainless-retry-count": "1"})
        other = await ask(started, "t", headers={"x-stainless-retry-count": "not-a-number"})
    assert first.json()["choices"] == again.json()["choices"] == other.json()["choices"]
    assert len(sampler.asked) == 2, "the retry replayed; the bad header counted as none"
    assert len(started.records_for("t")) == 2


async def test_an_idempotency_key_replays_without_a_retry_header() -> None:
    sampler = FakeSampler("ok")
    async with endpoint(sampler) as started:
        await ask(started, "t", headers={"Idempotency-Key": "k"})
        await ask(started, "t", headers={"Idempotency-Key": "k"}, max_tokens=7)
        await ask(started, "t", headers={"Idempotency-Key": "other"})
    assert len(sampler.asked) == 2 and len(started.records_for("t")) == 2


async def test_a_retry_after_a_budget_cut_still_replays() -> None:
    """With 95 of a 100-token budget spent, the retry's max_tokens would be cut from 10 to 5.
    The key uses the params before the cut, so the stored reply is found."""
    sampler = FakeSampler("x" * 95)
    async with endpoint(sampler, max_context=100) as started:
        first = await ask(started, "t", max_tokens=10)
        again = await ask(started, "t", max_tokens=10, headers={"x-stainless-retry-count": "1"})
        assert first.json()["choices"] == again.json()["choices"]
        assert len(sampler.asked) == 1 and len(started.records_for("t")) == 1
        assert started.recorder.spoke == {"t": 95}, "a replay spends nothing"
        assert started.records_for("t")[0].sampling_params["max_tokens"] == 10
