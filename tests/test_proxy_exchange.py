"""One request through the app: the SDK's headers on the record, a retry or an idempotent
request replayed from the stored reply and not re-sampled, whatever the budget did to its
params in between, and the volatile cut counted once per request that left a record."""

from __future__ import annotations

from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.rendering import Rendering
from tests.proxies import FakeRenderer, FakeSampler, ask, endpoint, ids

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
    """95 of a 100-token budget spent, so the retry's max_tokens would be cut from 10 to 5:
    the key is the params as pinned, before the cut, and the stored reply is found."""
    sampler = FakeSampler("x" * 95)
    async with endpoint(sampler, max_context=100) as started:
        first = await ask(started, "t", max_tokens=10)
        again = await ask(started, "t", max_tokens=10, headers={"x-stainless-retry-count": "1"})
        assert first.json()["choices"] == again.json()["choices"]
        assert len(sampler.asked) == 1 and len(started.records_for("t")) == 1
        assert started.recorder.spoke == {"t": 95}, "a replay spends nothing"
        assert started.records_for("t")[0].sampling_params["max_tokens"] == 10


# ------------------------------------------------------------------ the volatile cut

BUDGET = r"\n*<total_tokens>[^<]*</total_tokens>"


async def test_volatile_lines_are_cut_from_system_messages_before_rendering() -> None:
    sampler = FakeSampler("ok")
    system = {"role": "system", "content": "be good\n<total_tokens>9 left</total_tokens>"}
    user = {"role": "user", "content": "go"}
    async with endpoint(sampler, volatile=(BUDGET,)) as started:
        await ask(started, "t", messages=[system, user])
        assert started.recorder.cut == {"t": 1}
        (record,) = started.records_for("t")
    assert record.prompt_token_ids == ids("be goodgo")
    assert sampler.prompts[0].to_ints() == list(ids("be goodgo"))


async def test_a_replayed_retry_counts_its_cut_once() -> None:
    sampler = FakeSampler("ok")
    system = {"role": "system", "content": "be brief\n<total_tokens>12</total_tokens>"}
    messages = [system, {"role": "user", "content": "go"}]
    async with endpoint(sampler, volatile=(BUDGET,)) as started:
        await ask(started, "t", messages=messages)
        await ask(started, "t", messages=messages, headers={"x-stainless-retry-count": "1"})
        assert len(sampler.asked) == 1 and started.recorder.cut == {"t": 1}


async def test_the_cut_is_off_unless_asked() -> None:
    sampler = FakeSampler("ok")
    system = {"role": "system", "content": "<total_tokens>9 left</total_tokens>"}
    async with endpoint(sampler) as started:
        await ask(started, "t", messages=[system, {"role": "user", "content": "go"}])
        assert started.recorder.cut == {}
    assert sampler.prompts[0].to_ints() == list(ids("<total_tokens>9 left</total_tokens>go"))


def test_the_renderer_wrapper_cuts_text_parts_too_and_leaves_other_roles_alone() -> None:
    wrapped = Rendering(FakeRenderer(), volatile=(BUDGET,))
    assert wrapped.get_stop_sequences() == []
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "a<total_tokens>1</total_tokens>"}],
        },
        {"role": "user", "content": "<total_tokens>2</total_tokens>"},
    ]
    found = Exchange()
    token = exchange.set(found)
    try:
        rendered = wrapped.build_generation_prompt(messages)
    finally:
        exchange.reset(token)
    assert found.cut == 1
    assert rendered.to_ints() == [ord("a")] + list(ids("<total_tokens>2</total_tokens>"))
