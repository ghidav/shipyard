"""The records: the route a run reads them from (plain, delta, again), the wire codec, and
the recorder's refusals and failures, each of which leaves an error record."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import tinker

from shipyard.proxy import cookbook
from shipyard.proxy import endpoint as endpoint_module
from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.recorder import IMAGE_TOKEN, Record, Recorder, ids_of
from shipyard.proxy.wire import delta_encoded, record_of, records_of
from tests.proxies import FakeSampler, ask, endpoint, fetch, ids

OK = ids("ok")
EMPTY = {"records": [], "turned_away": 0, "cut": 0, "spoke": 0}


def planted(**fields: Any) -> Record:
    base: dict[str, Any] = dict(
        seq=1,
        at=0.0,
        prompt_token_ids=(7, 7, 7),
        completion_token_ids=OK,
        inference_logprobs=(),
        stop_reason="stop",
        sampling_params={},
        requested_model="gpt-oss-20b",
        sample_ms=0.0,
        served=None,
        cached_tokens=0,
        request_id=None,
        bridged=False,
        prompt_digests=(),
        reply_digest=None,
        error=None,
    )
    return Record(**{**base, **fields})


# ------------------------------------------------------------------- what comes back


async def test_a_trials_records_come_back_in_order_with_their_ids() -> None:
    async with endpoint(FakeSampler("ok")) as served:
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
    assert (payload["turned_away"], payload["cut"], payload["spoke"]) == (0, 0, 6)


async def test_the_fetch_drains_and_an_unknown_trial_is_empty_not_an_error() -> None:
    async with endpoint(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-2")
        first = await fetch(served, "task-1")
        second = await fetch(served, "task-1")
        unknown = await fetch(served, "no-such-trial")
        assert served.records_for("task-1") == [] and len(served.records_for("task-2")) == 1
    assert len(first.json()["records"]) == 1
    assert second.json() == {"trial": "task-1", **EMPTY}
    assert unknown.status_code == 200 and unknown.json() == {"trial": "no-such-trial", **EMPTY}


async def test_a_fetch_without_the_token_is_refused_and_leaves_the_records() -> None:
    async with endpoint(FakeSampler("ok")) as served:
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

    async with endpoint(FakeSampler("ok")) as served:
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
    """Field by field against the dataclass, so the next field added cannot be dropped."""
    made = planted(
        seq=7,
        at=1234.5,
        prompt_token_ids=(1, 2, 3),
        completion_token_ids=(4, 5),
        inference_logprobs=(-0.5, -0.25),
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
    back = record_of(dataclasses.asdict(made))
    for found in dataclasses.fields(Record):
        assert getattr(back, found.name) == getattr(made, found.name), found.name
    assert records_of(delta_encoded([made])) == [made]


# -------------------------------------------------------------------- a lost answer


async def test_a_retry_is_sent_the_answer_already_built() -> None:
    async with endpoint(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-1")
        first = await fetch(served, "task-1")
        retried = await fetch(served, "task-1", again="1")
        plain = await fetch(served, "task-1")
    assert len(first.json()["records"]) == 2
    assert retried.json() == first.json()
    assert plain.json()["records"] == []


async def test_a_retry_whose_first_request_never_arrived_drains_as_a_first_would() -> None:
    async with endpoint(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        retried = await fetch(served, "task-1", again="1")
    assert len(retried.json()["records"]) == 1


async def test_the_kept_answer_is_let_go_after_its_time(monkeypatch: pytest.MonkeyPatch) -> None:
    async with endpoint(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await fetch(served, "task-1")
        monkeypatch.setattr(endpoint_module, "SERVED_KEPT_SECONDS", -1.0)
        retried = await fetch(served, "task-1", again="1")
    assert retried.json()["records"] == []


# ------------------------------------------------------------ refusals and failures


class Unasked:
    """A sampling client that must never be reached."""

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
    assert recorder.turned_away == {"t-1": 1}
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
        assert recorder.turned_away == {}


async def test_filling_the_context_gives_a_turn_what_is_left_of_it() -> None:
    sampler = Unasked()
    recorder = Recorder(sampler, max_tokens=1000, max_context=2048, fill_context=True)
    with pytest.raises(AssertionError, match="should not have been asked"):
        await under("t-1", recorder, tinker.ModelInput.from_ints(list(range(1500))))
    assert sampler.params.max_tokens == 548 and recorder.turned_away == {}
    with pytest.raises(tinker.BadRequestError):
        await under("t-1", recorder, tinker.ModelInput.from_ints(list(range(2048))))
    assert recorder.turned_away == {"t-1": 1}


async def test_a_trial_past_its_budget_is_refused_with_the_cookbooks_own_400() -> None:
    sampler = FakeSampler("abcd")
    recorder = Recorder(sampler, max_tokens=100, budget=6)
    await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[0].max_tokens == 6, "cut to what is left, not overshot"
    await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[1].max_tokens == 2
    with pytest.raises(cookbook.private("_BadRequest"), match="spent its budget"):
        await under("t", recorder, tinker.ModelInput.from_ints([1]))
    assert len(sampler.asked) == 2 and recorder.turned_away == {"t": 1}
    assert [r.error for r in recorder.records_for("t")] == [None, None, "budget"]
    assert [r.seq for r in recorder.records_for("t")] == [1, 2, 3]
    assert recorder.spoke == {"t": 8}
    assert recorder.take("t") and recorder.take("t") == [] and recorder.spoke == {}


async def test_the_budget_refusal_is_a_400_over_the_wire_and_the_record_crosses() -> None:
    async with endpoint(FakeSampler("abcd"), max_context=4) as served:
        assert (await ask(served, "t", max_tokens=2)).status_code == 200
        refused = await ask(served, "t", max_tokens=1)
        assert refused.status_code == 400 and "budget" in refused.text
        payload = (await fetch(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == [None, "budget"]
    assert (payload["turned_away"], payload["spoke"]) == (1, 4)


async def test_a_turn_that_does_not_fit_is_a_400_the_harness_compacts_on() -> None:
    async with endpoint(FakeSampler("ok"), max_context=8, max_tokens=4) as served:
        refused = await ask(served, "t", messages=[{"role": "user", "content": "hello"}])
        assert refused.status_code == 400 and "prompt is too long" in refused.text
        payload = (await fetch(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == ["context"]
    assert payload["turned_away"] == 1


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


async def test_a_failed_sample_is_a_500_and_a_retry_samples_fresh() -> None:
    class FailingOnce(FakeSampler):
        failed = False

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            if not self.failed:
                self.failed = True
                raise RuntimeError("once")
            return await super().sample_async(prompt, num_samples, sampling_params)

    async with endpoint(FailingOnce("ok")) as served:
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
