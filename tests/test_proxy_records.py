"""Tests for proxy records: the fetch route, the wire codec, and the recorder's refusals."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import tinker

from shipyard.proxy import cookbook
from shipyard.proxy import server as server_module
from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.recorder import Recorder
from shipyard.proxy.wire import (
    BUDGET,
    CONTEXT,
    IMAGE_TOKEN,
    Picture,
    Record,
    delta_encoded,
    ids_of,
    pictured,
    pictures_of,
    record_of,
    records_of,
)
from tests.proxies import FakeSampler, ask, fetch, ids, server
from tests.records import made

OK = ids("ok")
EMPTY = {"records": []}


def refused(recorder: Recorder, trial: str) -> list[str]:
    """Return the budget and context refusals recorded for `trial`, in order."""
    return [r.error for r in recorder.records_for(trial) if r.error in (BUDGET, CONTEXT)]


# ------------------------------------------------------------------- what comes back


async def test_a_trials_records_come_back_in_order_with_their_ids() -> None:
    async with server(FakeSampler("ok")) as served:
        for said in ("one", "two", "three"):
            await ask(served, "task-1", messages=[{"role": "user", "content": said}])
        answered = await fetch(served, "task-1")
    assert answered.status_code == 200
    payload = answered.json()
    records = payload["records"]
    assert [tuple(r["prompt_token_ids"]) for r in records] == [ids("one"), ids("two"), ids("three")]
    assert [tuple(r["completion_token_ids"]) for r in records] == [OK, OK, OK]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert {r["requested_model"] for r in records} == {"gpt-oss-20b"}
    assert sum(len(r["completion_token_ids"]) for r in records) == 6


async def test_the_fetch_drains_and_an_unknown_trial_is_empty_not_an_error() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-2")
        first = await fetch(served, "task-1")
        second = await fetch(served, "task-1")
        unknown = await fetch(served, "no-such-trial")
        assert served.records_for("task-1") == [] and len(served.records_for("task-2")) == 1
    assert len(first.json()["records"]) == 1
    assert second.json() == EMPTY
    assert unknown.status_code == 200 and unknown.json() == EMPTY


async def test_a_fetch_without_the_token_is_refused_and_leaves_the_records() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        assert (await fetch(served, "task-1", headers={})).status_code == 401
        wrong = {"Authorization": "Bearer not-the-token"}
        assert (await fetch(served, "task-1", headers=wrong)).status_code == 401
        assert len(served.records_for("task-1")) == 1


async def test_extending_prompts_cross_as_tails_compressed_and_come_back_whole() -> None:
    turns = [["one"], ["one", "ok", "two"], ["one", "ok", "two", "ok", "three"]]

    def said(contents: list[str]) -> list[dict[str, str]]:
        roles = ("user", "assistant")
        return [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]

    async with server(FakeSampler("ok")) as served:
        for contents in turns:
            await ask(served, "t", messages=said(contents))
        raw = await fetch(served, "t", delta="1")
    assert raw.headers.get("content-encoding") in {"gzip", "deflate", "br", "zstd"}
    items = raw.json()["records"]
    assert [item.get("prompt_from") for item in items] == [None, 0, 1]
    assert items[2]["prompt_token_ids"] == list(ids("three"))
    assert [r.prompt_token_ids for r in records_of(items)] == [
        ids("one"),
        ids("oneoktwo"),
        ids("oneoktwookthree"),
    ]


def test_a_record_that_extends_one_not_before_it_is_refused() -> None:
    whole = {"prompt_token_ids": [1], "completion_token_ids": [2]}
    extended = records_of([whole, {**whole, "prompt_from": 0, "prompt_token_ids": [3]}])
    assert extended[1].prompt_token_ids == (1, 2, 3)
    with pytest.raises(ValueError, match="extends record 1"):
        records_of([{**whole, "prompt_from": 1}])


def test_a_record_off_the_wire_with_no_ids_is_refused_and_annotations_may_be_missing() -> None:
    with pytest.raises(ValueError, match="nothing in it to train on"):
        record_of({"seq": 1, "requested_model": "m"})
    record = record_of({"prompt_token_ids": [1, 2], "completion_token_ids": [3]})
    assert (record.prompt_token_ids, record.completion_token_ids) == ((1, 2), (3,))
    assert record.requested_model is None and record.error is None and record.bridged is False


def test_every_annotation_survives_the_wire() -> None:
    """Compares every `Record` field, so the codec cannot silently drop a newly added one."""
    record = Record(
        seq=7,
        at=1234.5,
        prompt_token_ids=(1, 2, 3),
        completion_token_ids=(4, 5),
        inference_logprobs=(-0.5, -0.25),
        stop_reason="stop",
        sampling_params={"temperature": 1.0},
        requested_model="openrouter/a/b",
        sample_ms=12.5,
        served="tinker://run/sampler_weights/step-3",
        cached_tokens=3,
        request_id="req-9",
        bridged=True,
        prompt_digests=("ab", "cd"),
        reply_digest="ef",
        error="context",
    )
    back = record_of(dataclasses.asdict(record))
    for found in dataclasses.fields(Record):
        assert getattr(back, found.name) == getattr(record, found.name), found.name
    assert records_of(delta_encoded([record])) == [record]


# -------------------------------------------------------------------- a lost answer


async def test_a_retry_is_sent_the_answer_already_built() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-1")
        first = await fetch(served, "task-1")
        retried = await fetch(served, "task-1", again="1")
        plain = await fetch(served, "task-1")
    assert len(first.json()["records"]) == 2
    assert retried.json() == first.json()
    assert plain.json()["records"] == []


async def test_a_retry_whose_first_request_never_arrived_drains_as_a_first_would() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        retried = await fetch(served, "task-1", again="1")
    assert len(retried.json()["records"]) == 1


async def test_the_kept_answer_is_let_go_after_its_time(monkeypatch: pytest.MonkeyPatch) -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await fetch(served, "task-1")
        monkeypatch.setattr(server_module, "SERVED_KEPT_SECONDS", -1.0)
        retried = await fetch(served, "task-1", again="1")
    assert retried.json()["records"] == []


# ------------------------------------------------------------ refusals and failures


class Unasked:
    """Fake sampling client that records the parameters it gets and fails if called."""

    def __init__(self) -> None:
        self.asked = 0

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked += 1
        self.params = sampling_params
        raise AssertionError("should not have been asked")


async def under(trial: str, recorder: Recorder, prompt: Any, **params: Any) -> Any:
    token = exchange.set(Exchange(trial=trial))
    try:
        return await recorder.sample_async(
            prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(**params)
        )
    finally:
        exchange.reset(token)


async def test_a_turn_that_does_not_fit_is_refused_before_it_is_paid_for() -> None:
    sampler = Unasked()
    recorder = Recorder(sampler, max_tokens=1000, max_context=2048)
    prompt = tinker.ModelInput.from_ints(list(range(1500)))
    with pytest.raises(tinker.BadRequestError) as refused:
        await under("t-1", recorder, prompt)
    assert sampler.asked == 0
    assert "context window" in str(refused.value), "what the cookbook classifies an overflow by"
    assert "1500 prompt tokens" in str(refused.value) and "2048" in str(refused.value)
    (record,) = recorder.records_for("t-1")
    assert record.error == "context" and record.completion_token_ids == ()
    assert record.prompt_token_ids == tuple(range(1500)) and record.seq == 1


async def test_a_turn_that_fits_is_not_refused_and_nothing_is_refused_without_a_ceiling() -> None:
    for recorder in (
        Recorder(Unasked(), max_tokens=500, max_context=2048),
        Recorder(Unasked(), max_tokens=100_000, max_context=None),
    ):
        with pytest.raises(AssertionError, match="should not have been asked"):
            await under("t-1", recorder, tinker.ModelInput.from_ints(list(range(1500))))
        assert refused(recorder, "t-1") == []


async def test_filling_the_context_gives_a_turn_what_is_left_of_it() -> None:
    sampler = Unasked()
    recorder = Recorder(sampler, max_tokens=1000, max_context=2048, fill_context=True)
    with pytest.raises(AssertionError, match="should not have been asked"):
        await under("t-1", recorder, tinker.ModelInput.from_ints(list(range(1500))))
    assert sampler.params.max_tokens == 548 and refused(recorder, "t-1") == []
    with pytest.raises(tinker.BadRequestError):
        await under("t-1", recorder, tinker.ModelInput.from_ints(list(range(2048))))
    assert refused(recorder, "t-1") == [CONTEXT]


async def test_a_trial_past_its_budget_is_refused_with_the_cookbooks_own_400() -> None:
    sampler = FakeSampler("abcd")
    recorder = Recorder(sampler, max_tokens=100, budget=6)
    await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[0].max_tokens == 6, "cut to what is left, not overshot"
    await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[1].max_tokens == 2
    with pytest.raises(cookbook.private("_BadRequest"), match="spent its budget"):
        await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert len(sampler.asked) == 2
    assert [r.error for r in recorder.records_for("t")] == [None, None, "budget"]
    assert [r.seq for r in recorder.records_for("t")] == [1, 2, 3]
    assert recorder.spoke == {"t": 8}
    assert recorder.take("t") and recorder.take("t") == [] and recorder.spoke == {}


async def test_the_budget_refusal_is_a_400_over_the_wire_and_the_record_crosses() -> None:
    async with server(FakeSampler("abcd"), max_context=4) as served:
        assert (await ask(served, "t", max_tokens=2)).status_code == 200
        refused = await ask(served, "t", max_tokens=1)
        assert refused.status_code == 400 and "budget" in refused.text
        payload = (await fetch(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == [None, "budget"]
    assert sum(len(r["completion_token_ids"]) for r in payload["records"]) == 4


async def test_a_turn_that_does_not_fit_is_a_400_the_harness_compacts_on() -> None:
    async with server(FakeSampler("ok"), max_context=8, max_tokens=4) as served:
        refused = await ask(served, "t", messages=[{"role": "user", "content": "hello"}])
        assert refused.status_code == 400 and "prompt is too long" in refused.text
        payload = (await fetch(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == ["context"]
    assert list(payload) == ["records"]


async def test_a_failed_sample_leaves_a_record_and_re_raises() -> None:
    class Failing:
        async def sample_async(self, **_: Any) -> Any:
            raise RuntimeError("the sampler went away")

    recorder = Recorder(Failing(), served="tinker://w/step-1")
    with pytest.raises(RuntimeError, match="went away"):
        await under("t", recorder, tinker.ModelInput.from_ints([1, 2]))
    (record,) = recorder.records_for("t")
    assert record.error == "sampler: RuntimeError" and record.served == "tinker://w/step-1"
    assert record.prompt_token_ids == (1, 2) and record.completion_token_ids == ()


async def test_a_reply_that_lands_after_its_trial_was_taken_is_dropped() -> None:
    """A call cut off mid-flight by Harbor's timeout can return after the run has taken the
    trial's records. The harness still gets the reply, but the proxy keeps no record of it."""

    class Held(FakeSampler):
        def __init__(self) -> None:
            super().__init__("late")
            self.release = asyncio.Event()
            self.release.set()

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            await self.release.wait()
            return await super().sample_async(prompt, num_samples, sampling_params)

    sampler = Held()
    recorder = Recorder(sampler, budget=100)
    await under("t", recorder, tinker.ModelInput.from_ints([1]))  # answered before the take
    sampler.release.clear()
    late = asyncio.create_task(under("t", recorder, tinker.ModelInput.from_ints([1, 2])))
    await asyncio.sleep(0)
    assert len(recorder.take("t")) == 1
    sampler.release.set()
    replied = await late
    assert [list(s.tokens) for s in replied.sequences] == [list(ids("late"))]
    recorder.reply("t", 1, "digest", ended_with_stop=True)
    assert recorder.records == {} and recorder.spoke == {} and recorder.indexes == {}
    assert not any(key[0] == "t" for key in recorder._replies)
    await under("t", recorder, tinker.ModelInput.from_ints([3]))  # one asked after the take
    assert recorder.take("t") == [] and recorder.records == {}
    await under("", recorder, tinker.ModelInput.from_ints([4]))
    recorder.take("")
    await under("", recorder, tinker.ModelInput.from_ints([5]))
    assert len(recorder.records_for("")) == 1, "a call with no trial is never dropped"


async def test_a_failed_sample_is_a_500_and_a_retry_samples_fresh() -> None:
    class FailingOnce(FakeSampler):
        failed = False

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            if not self.failed:
                self.failed = True
                raise RuntimeError("once")
            return await super().sample_async(prompt, num_samples, sampling_params)

    async with server(FailingOnce("ok")) as served:
        assert (await ask(served, "t")).status_code == 500
        again = await ask(served, "t", headers={"x-stainless-retry-count": "1"})
        assert again.status_code == 200
    assert [r.error for r in served.records_for("t")] == ["sampler: RuntimeError", None]


async def test_a_replayed_retry_leaves_no_second_record() -> None:
    sampler = FakeSampler("ok")
    recorder = Recorder(sampler)
    prompt = tinker.ModelInput.from_ints([1, 2])
    token = exchange.set(Exchange(trial="t"))
    try:
        first = await recorder.sample_async(
            prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams()
        )
    finally:
        exchange.reset(token)
    token = exchange.set(Exchange(trial="t", retry=1))
    try:
        again = await recorder.sample_async(
            prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams()
        )
        other = await recorder.sample_async(
            prompt=tinker.ModelInput.from_ints([3]),
            num_samples=1,
            sampling_params=tinker.SamplingParams(),
        )
    finally:
        exchange.reset(token)
    assert again is first and other is not first
    assert len(sampler.asked) == 2 and len(recorder.records_for("t")) == 2


async def test_every_sequence_is_a_record_and_the_cache_hit_is_spent_once() -> None:
    sampler = FakeSampler("ok", cached=5)
    recorder = Recorder(sampler)
    token = exchange.set(Exchange(trial="t"))
    try:
        await recorder.sample_async(
            prompt=tinker.ModelInput.from_ints([1, 2, 3, 4, 5, 6]),
            num_samples=3,
            sampling_params=tinker.SamplingParams(),
        )
    finally:
        exchange.reset(token)
    records = recorder.records_for("t")
    assert [r.seq for r in records] == [1, 2, 3]
    assert [r.cached_tokens for r in records] == [5, 0, 0]
    assert recorder.spoke == {"t": 6}


def test_an_image_chunk_is_a_run_of_image_tokens_in_the_ids() -> None:
    prompt = SimpleNamespace(
        chunks=[
            SimpleNamespace(tokens=[1, 2]),
            SimpleNamespace(length=3),
            SimpleNamespace(tokens=[4]),
        ]
    )
    assert ids_of(prompt) == (1, 2, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 4)
    assert ids_of(tinker.ModelInput.from_ints([9])) == (9,)


def test_pictures_of_keeps_each_image_chunk_with_bytes_at_its_offset() -> None:
    import base64

    prompt = SimpleNamespace(
        chunks=[
            SimpleNamespace(tokens=[1, 2]),
            tinker.types.ImageChunk(data=b"png!", format="png", expected_tokens=3),
            SimpleNamespace(tokens=[4]),
            SimpleNamespace(length=2),  # an asset pointer: no bytes to keep
        ]
    )
    assert ids_of(prompt) == (
        1,
        2,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        4,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
    )
    assert pictures_of(prompt) == (Picture(2, 3, "png", base64.b64encode(b"png!").decode()),)
    assert pictures_of(tinker.ModelInput.from_ints([9])) == ()


async def test_a_sampled_image_prompt_keeps_its_picture_on_the_record() -> None:
    import base64

    recorder = Recorder(FakeSampler("ok"))
    prompt = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[1]),
            tinker.types.ImageChunk(data=b"png!", format="png", expected_tokens=2),
            tinker.types.EncodedTextChunk(tokens=[3]),
        ]
    )
    await under("t", recorder, prompt)
    [record] = recorder.records_for("t")
    assert record.prompt_token_ids == (1, IMAGE_TOKEN, IMAGE_TOKEN, 3)
    assert record.images == (Picture(1, 2, "png", base64.b64encode(b"png!").decode()),)
    assert pictured(record)


def test_pictures_cross_the_wire_once_with_the_record_they_belong_to() -> None:
    from dataclasses import asdict

    shot, later = Picture(1, 2, "png", "cG5nIQ=="), Picture(6, 1, "jpeg", "anBn")
    first = made((1, IMAGE_TOKEN, IMAGE_TOKEN, 4), (5, 6), images=(shot,))
    second = made(
        (1, IMAGE_TOKEN, IMAGE_TOKEN, 4, 5, 6, IMAGE_TOKEN, 8), (9,), seq=2, images=(shot, later)
    )
    rows = delta_encoded([first, second])
    assert rows[1]["prompt_from"] == 0 and rows[1]["prompt_token_ids"] == [IMAGE_TOKEN, 8]
    assert rows[1]["images"] == [asdict(later)], "the prefix's picture rides with record 0"
    assert records_of(rows) == [first, second]
    assert records_of([asdict(first)]) == [first]
    assert record_of({"prompt_token_ids": [1], "completion_token_ids": [2]}).images == ()


async def test_a_retry_of_the_turn_that_spent_the_budget_gets_its_reply() -> None:
    sampler = FakeSampler("abcd")
    recorder = Recorder(sampler, budget=4)
    prompt = tinker.ModelInput.from_ints([1])
    first = await under("t", recorder, prompt)
    assert recorder.spoke == {"t": 4}, "the budget is spent to the last token"
    token = exchange.set(Exchange(trial="t", retry=1))
    try:
        again = await recorder.sample_async(
            prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams()
        )
    finally:
        exchange.reset(token)
    assert again is first and len(sampler.asked) == 1
    assert [r.error for r in recorder.records_for("t")] == [None]


async def test_samples_asked_together_share_the_budget() -> None:
    sampler = FakeSampler("ab")
    recorder = Recorder(sampler, max_tokens=100, budget=9)
    token = exchange.set(Exchange(trial="t"))
    try:
        await recorder.sample_async(
            prompt=tinker.ModelInput.from_ints([1]),
            num_samples=4,
            sampling_params=tinker.SamplingParams(),
        )
        assert sampler.asked[0].max_tokens == 2, "nine tokens left, four samples"
        with pytest.raises(cookbook.private("_BadRequest"), match="spent its budget"):
            await recorder.sample_async(
                prompt=tinker.ModelInput.from_ints([1]),
                num_samples=4,
                sampling_params=tinker.SamplingParams(),
            )
    finally:
        exchange.reset(token)
    assert recorder.spoke == {"t": 8} and len(sampler.asked) == 1
