"""Tests for proxy records: what closing a trial hands back, the wire codec, and the trial's
refusals."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import tinker

from harness_proxy import server as server_module
from harness_proxy.chat import Caller, Refused
from harness_proxy.rendering import Prompt
from harness_proxy.trial import Sampled, Trial
from harness_proxy.wire import (
    BUDGET,
    CONTEXT,
    IMAGE_TOKEN,
    TRIALS_PATH,
    Picture,
    Record,
    delta_encoded,
    extends,
    ids_of,
    pictured,
    pictures_of,
    record_of,
    records_of,
)
from proxies import FakeSampler, ask, close, ids, made, opened, recorded, root, server

OK = ids("ok")


def refused(trial: Trial) -> list[str]:
    """Return the budget and context refusals the trial recorded, in order."""
    return [r.error for r in trial.records if r.error in (BUDGET, CONTEXT)]


# ------------------------------------------------------------------- what comes back


async def test_a_trials_records_come_back_in_order_with_their_ids() -> None:
    async with server(FakeSampler("ok")) as served:
        for said in ("one", "two", "three"):
            await ask(served, "task-1", messages=[{"role": "user", "content": said}])
        answered = await close(served, "task-1")
    assert answered.status_code == 200
    payload = answered.json()
    records = payload["records"]
    assert [tuple(r["prompt_token_ids"]) for r in records] == [ids("one"), ids("two"), ids("three")]
    assert [tuple(r["completion_token_ids"]) for r in records] == [OK, OK, OK]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert {r["requested_model"] for r in records} == {"gpt-oss-20b"}
    assert sum(len(r["completion_token_ids"]) for r in records) == 6


async def test_closing_ends_the_trial_and_one_never_opened_is_not_found() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-2")
        first = await close(served, "task-1")
        second = await close(served, "task-1")
        unknown = await close(served, "no-such-trial")
        assert served.trial("task-1") is None and len(recorded(served, "task-2")) == 1
    assert len(first.json()["records"]) == 1
    assert second.json() == first.json(), "a close asked again is answered as the first"
    assert unknown.status_code == 404


async def test_only_the_control_token_opens_or_closes_a_trial_and_a_refusal_leaves_it() -> None:
    """Every sandbox holds its trial's token, so it must not open a trial, or close one and read
    its records."""
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        token = await opened(served, "task-1")
        assert (await close(served, "task-1", headers={})).status_code == 401
        wrong = {"Authorization": "Bearer not-the-token"}
        assert (await close(served, "task-1", headers=wrong)).status_code == 401
        for harness in ({"Authorization": f"Bearer {token}"}, {"x-api-key": token}):
            assert (await close(served, "task-1", headers=harness)).status_code == 401
        async with httpx.AsyncClient() as http:
            stranger = await http.post(
                f"{root(served)}{TRIALS_PATH}",
                headers={"Authorization": f"Bearer {token}"},
                json={"trial": "mine"},
            )
        assert stranger.status_code == 401 and served.trial("mine") is None
        assert len(recorded(served, "task-1")) == 1


async def test_extending_prompts_cross_as_tails_compressed_and_come_back_whole() -> None:
    turns = [["one"], ["one", "ok", "two"], ["one", "ok", "two", "ok", "three"]]

    def said(contents: list[str]) -> list[dict[str, str]]:
        roles = ("user", "assistant")
        return [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]

    async with server(FakeSampler("ok")) as served:
        for contents in turns:
            await ask(served, "t", messages=said(contents))
        raw = await close(served, "t")
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
        weights="tinker://run/sampler_weights/step-3",
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


async def test_a_close_asked_again_is_sent_the_answer_already_built() -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await ask(served, "task-1")
        first = await close(served, "task-1")
        again = await close(served, "task-1")
    assert len(first.json()["records"]) == 2 and again.json() == first.json()


async def test_the_kept_answer_is_let_go_after_its_time(monkeypatch: pytest.MonkeyPatch) -> None:
    async with server(FakeSampler("ok")) as served:
        await ask(served, "task-1")
        await close(served, "task-1")
        monkeypatch.setattr(server_module, "CLOSED_KEPT_SECONDS", -1.0)
        again = await close(served, "task-1")
    assert again.status_code == 404


# ------------------------------------------------------------ refusals and failures


class Unasked:
    """Fake sampling client that records the parameters it gets and fails if called."""

    def __init__(self) -> None:
        self.asked = 0

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked += 1
        self.params = sampling_params
        raise AssertionError("should not have been asked")


async def under(trial: Trial, prompt: Any, **params: Any) -> Sampled:
    caller, asked = Caller(trial=trial.name), tinker.SamplingParams(**params)
    return await trial.sample(caller, Prompt(prompt), asked)


async def test_a_turn_that_does_not_fit_is_refused_before_it_is_paid_for() -> None:
    sampler = Unasked()
    trial = Trial("t", sampler, max_tokens=1000, max_context=2048)
    prompt = tinker.ModelInput.from_ints(list(range(1500)))
    with pytest.raises(Refused) as refused:
        await under(trial, prompt)
    assert sampler.asked == 0
    assert str(refused.value) == (
        "prompt is too long: 1500 prompt tokens plus 1000 max_tokens exceed the model's "
        "context window"
    ), "what a harness compacts its history on"
    (record,) = trial.records
    assert record.error == "context" and record.completion_token_ids == ()
    assert record.prompt_token_ids == tuple(range(1500)) and record.seq == 1


async def test_a_turn_that_fits_is_not_refused_and_nothing_is_refused_without_a_ceiling() -> None:
    for trial in (
        Trial("t", Unasked(), max_tokens=500, max_context=2048),
        Trial("t", Unasked(), max_tokens=100_000, max_context=None),
    ):
        with pytest.raises(AssertionError, match="should not have been asked"):
            await under(trial, tinker.ModelInput.from_ints(list(range(1500))))
        assert refused(trial) == []


async def test_filling_the_context_gives_a_turn_what_is_left_of_it() -> None:
    sampler = Unasked()
    trial = Trial("t", sampler, max_tokens=1000, max_context=2048, fill_context=True)
    with pytest.raises(AssertionError, match="should not have been asked"):
        await under(trial, tinker.ModelInput.from_ints(list(range(1500))))
    assert sampler.params.max_tokens == 548 and refused(trial) == []
    with pytest.raises(Refused, match="^prompt is too long"):
        await under(trial, tinker.ModelInput.from_ints(list(range(2048))))
    assert refused(trial) == [CONTEXT]


async def test_a_trial_past_its_budget_is_refused() -> None:
    sampler = FakeSampler("abcd")
    trial = Trial("t", sampler, max_tokens=100, budget=6)
    await under(trial, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[0].max_tokens == 6, "cut to what is left, not overshot"
    await under(trial, tinker.ModelInput.from_ints([1]))
    assert sampler.asked[1].max_tokens == 2
    with pytest.raises(Refused, match="spent its budget"):
        await under(trial, tinker.ModelInput.from_ints([1]))
    assert len(sampler.asked) == 2
    assert [r.error for r in trial.records] == [None, None, "budget"]
    assert [r.seq for r in trial.records] == [1, 2, 3]
    assert trial.spoke == 8


async def test_the_budget_refusal_is_a_400_over_the_wire_and_the_record_crosses() -> None:
    async with server(FakeSampler("abcd"), max_context=4) as served:
        assert (await ask(served, "t", max_tokens=2)).status_code == 200
        refused = await ask(served, "t", max_tokens=1)
        assert refused.status_code == 400 and "budget" in refused.text
        payload = (await close(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == [None, "budget"]
    assert sum(len(r["completion_token_ids"]) for r in payload["records"]) == 4


async def test_a_turn_that_does_not_fit_is_a_400_the_harness_compacts_on() -> None:
    async with server(FakeSampler("ok"), max_context=8, max_tokens=4) as served:
        refused = await ask(served, "t", messages=[{"role": "user", "content": "hello"}])
        assert refused.status_code == 400 and "prompt is too long" in refused.text
        payload = (await close(served, "t")).json()
    assert [r["error"] for r in payload["records"]] == ["context"]
    assert list(payload) == ["records"]


async def test_a_failed_sample_leaves_a_record_and_re_raises() -> None:
    class Failing:
        async def sample_async(self, **_: Any) -> Any:
            raise RuntimeError("the sampler went away")

    trial = Trial("t", Failing(), weights="tinker://w/step-1")
    with pytest.raises(RuntimeError, match="went away"):
        await under(trial, tinker.ModelInput.from_ints([1, 2]))
    (record,) = trial.records
    assert record.error == "sampler: RuntimeError" and record.weights == "tinker://w/step-1"
    assert record.prompt_token_ids == (1, 2) and record.completion_token_ids == ()


async def test_a_backends_refusal_faults_the_request_and_leaves_no_record() -> None:
    """A backend that refuses a request before sampling it, as Together refuses an image, is
    answered as a renderer's refusal is: a 400, and no record of a failed sampler."""

    class TextOnly:
        async def sample_async(self, **_: Any) -> Any:
            raise Refused("Together takes text only, and this input holds an image")

    trial = Trial("t", TextOnly())
    with pytest.raises(Refused, match="text only"):
        await under(trial, tinker.ModelInput.from_ints([1, 2]))
    assert trial.records == []


async def test_a_reply_that_lands_after_its_trial_closed_is_answered_and_recorded_nowhere() -> None:
    """A call cut off mid-flight by Harbor's timeout can return after the run closed its trial.
    The harness still gets the reply, but the records the close handed back stay as they were,
    and the trial's token answers nothing more."""

    class Held(FakeSampler):
        def __init__(self) -> None:
            super().__init__("late")
            self.release, self.entered = asyncio.Event(), asyncio.Event()
            self.release.set()

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            self.entered.set()
            await self.release.wait()
            return await super().sample_async(prompt, num_samples, sampling_params)

    sampler = Held()
    async with server(sampler) as served:
        token = await opened(served, "t")
        await ask(served, "t", key=token)  # answered before the close
        sampler.release.clear()
        sampler.entered.clear()
        late = asyncio.create_task(ask(served, "t", key=token))
        await sampler.entered.wait()
        closed = (await close(served, "t")).json()["records"]
        sampler.release.set()
        replied = await late
        again = (await close(served, "t")).json()["records"]
        after = await ask(served, "t", key=token)
    assert len(closed) == 1
    assert replied.status_code == 200
    assert replied.json()["choices"][0]["message"]["content"] == "late"
    assert again == closed, "the late reply changed nothing the close handed back"
    assert after.status_code == 401


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
    assert [r.error for r in recorded(served, "t")] == ["sampler: RuntimeError", None]


async def test_a_replayed_retry_leaves_no_second_record() -> None:
    sampler = FakeSampler("ok")
    trial = Trial("t", sampler)
    prompt, retried = tinker.ModelInput.from_ints([1, 2]), Caller(trial="t", retry=1)
    first = await under(trial, prompt)
    again = await trial.sample(retried, Prompt(prompt), tinker.SamplingParams())
    other = await trial.sample(
        retried, Prompt(tinker.ModelInput.from_ints([3])), tinker.SamplingParams()
    )
    assert (first.seq, again.seq, other.seq) == (1, None, 2), "a replay makes no record"
    assert again.tokens == first.tokens
    assert len(sampler.asked) == 2 and len(trial.records) == 2


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

    trial = Trial("t", FakeSampler("ok"))
    prompt = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[1]),
            tinker.types.ImageChunk(data=b"png!", format="png", expected_tokens=2),
            tinker.types.EncodedTextChunk(tokens=[3]),
        ]
    )
    await under(trial, prompt)
    [record] = trial.records
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
    trial = Trial("t", sampler, budget=4)
    prompt = tinker.ModelInput.from_ints([1])
    first = await under(trial, prompt)
    assert trial.spoke == 4, "the budget is spent to the last token"
    again = await trial.sample(Caller(trial="t", retry=1), Prompt(prompt), tinker.SamplingParams())
    assert again == dataclasses.replace(first, seq=None) and len(sampler.asked) == 1
    assert [r.error for r in trial.records] == [None]


def test_a_record_with_another_picture_in_the_prefix_is_no_extension_and_keeps_its_own() -> None:
    """Two images of one size are runs of `IMAGE_TOKEN` of one length; their pictures differ."""
    red, yellow = Picture(1, 2, "png", "cmVk"), Picture(1, 2, "png", "eWVsbG93")
    first = made((1, IMAGE_TOKEN, IMAGE_TOKEN, 4), (5, 6), images=(red,))
    other = made((1, IMAGE_TOKEN, IMAGE_TOKEN, 4, 5, 6, 7), (8,), seq=2, images=(yellow,))
    same = made((1, IMAGE_TOKEN, IMAGE_TOKEN, 4, 5, 6, 7), (8,), seq=2, images=(red,))
    assert not extends(other, first) and extends(same, first)
    rows = delta_encoded([first, other])
    assert "prompt_from" not in rows[1], "sent whole, with its own picture"
    assert records_of(rows) == [first, other]
    assert delta_encoded([first, same])[1]["prompt_from"] == 0
    assert records_of(delta_encoded([first, same])) == [first, same]
