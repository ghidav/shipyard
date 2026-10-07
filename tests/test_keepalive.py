"""Keepalives on streamed replies: a sampler slower than the grace gets SSE comment frames
ahead of the first data frame on both wires, the record is what it would have been, thinking
still reaches a kept-alive stream, an error after the frames is an error frame, and a reply
that is not streamed, or comes in time, is left alone."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from shipyard.proxy import cookbook, keepalive, thinking
from shipyard.proxy.keepalive import FRAME
from tests.proxies import (
    BridgeRenderer,
    FakeSampler,
    ScriptedSampler,
    ask,
    ask_anthropic,
    endpoint,
    ids,
)

#: Small enough to keep the tests fast: two or more frames inside the sampler's sleep.
GRACE, EVERY, SLOW = 0.12, 0.08, 0.4


@dataclass
class SlowSampler(FakeSampler):
    """A sampler that thinks for `delay` seconds before it answers, or raises after it."""

    delay: float = SLOW
    fails: bool = False

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        await asyncio.sleep(self.delay)
        if self.fails:
            raise RuntimeError("the sampler fell over")
        return await super().sample_async(prompt, num_samples, sampling_params)


def kept(sampler: Any, **overrides: Any) -> Any:
    return endpoint(sampler, keepalive_grace=GRACE, keepalive_every=EVERY, **overrides)


def frames_before(text: str, first: str) -> int:
    """How many keepalive frames came before the first real event `first` begins."""
    return text[: text.index(first)].count(FRAME.decode())


async def test_a_slow_streamed_reply_gets_frames_before_its_first_data_frame() -> None:
    async with kept(SlowSampler("hi")) as started:
        answered = await ask(started, "t-1", stream=True)
        text = answered.text
        assert answered.status_code == 200
        assert answered.headers["content-type"].startswith("text/event-stream")
        assert text.startswith(FRAME.decode()) and frames_before(text, "data:") >= 2
        assert FRAME.decode() not in text[text.index("data:") :], "none after the reply"
        chunks = [line for line in text.splitlines() if line.startswith("data: {")]
        said = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content") or "" for line in chunks
        )
        assert said == "hi" and text.rstrip().endswith("data: [DONE]")
        [made] = started.records_for("t-1")
    assert made.completion_token_ids == ids("hi") and made.error is None and made.seq == 1
    assert made.prompt_token_ids == ids("go")


async def test_the_anthropic_wire_is_kept_alive_the_same_way() -> None:
    async with kept(SlowSampler("hi")) as started:
        text = (await ask_anthropic(started, "t-1", stream=True)).text
        [made] = started.records_for("t-1")
    assert text.startswith(FRAME.decode()) and frames_before(text, "event: message_start") >= 2
    assert '"text": "hi"' in text and "event: message_stop" in text
    assert made.completion_token_ids == ids("hi")


async def test_thinking_still_reaches_a_stream_the_keepalive_opened() -> None:
    sampler = ScriptedSampler("", answers=["{look}fine"])

    async def slowly(prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        await asyncio.sleep(SLOW)
        return await ScriptedSampler.sample_async(sampler, prompt, num_samples, sampling_params)

    sampler.sample_async = slowly  # type: ignore[method-assign]
    async with kept(sampler, renderer=BridgeRenderer()) as started:
        text = (await ask(started, "t", stream=True)).text
    assert frames_before(text, "data:") >= 2
    first = json.loads(text[text.index("data:") + 6 :].split("\n", 1)[0])
    assert first["choices"][0]["delta"][thinking.REPLY_FIELD] == "look"
    assert cookbook.private("_serve_sse").shipyard_wrap == thinking.TAG
    assert cookbook.wrapped_by("_serve_sse", keepalive.TAG), "beneath thinking's wrap"


async def test_a_sampler_failing_after_the_frames_ends_the_stream_with_an_error_frame() -> None:
    async with kept(SlowSampler("hi", fails=True)) as started:
        text = (await ask(started, "t", stream=True)).text
        anthropic = (await ask_anthropic(started, "u", stream=True)).text
        [failed] = started.records_for("t")
    assert frames_before(text, "data:") >= 2
    error = json.loads(text[text.index("data:") + 6 :].split("\n", 1)[0])["error"]
    assert "the sampler fell over" in error["message"] and text.rstrip().endswith("[DONE]")
    assert "event: error" in anthropic and "the sampler fell over" in anthropic
    assert failed.error is not None and failed.error.startswith("sampler")


async def test_an_unstreamed_or_quick_reply_is_not_kept_alive() -> None:
    async with kept(SlowSampler("hi")) as started:
        plain = await ask(started, "t")
        assert plain.headers["content-type"].startswith("application/json")
        assert plain.json()["choices"][0]["message"]["content"] == "hi"
    async with kept(FakeSampler("hi")) as started:
        quick = (await ask(started, "t", stream=True)).text
    assert quick.startswith("data:") and FRAME.decode() not in quick
    async with endpoint(SlowSampler("hi"), keepalive_grace=None) as started:
        off = (await ask(started, "t", stream=True)).text
    assert off.startswith("data:") and FRAME.decode() not in off


async def test_a_refusal_before_the_grace_is_the_cookbooks_own_400() -> None:
    async with kept(SlowSampler("hi")) as started:
        refused = await ask(started, "t", stream=True, messages="not a list")
    assert refused.status_code == 400 and "error" in refused.json()


@pytest.mark.parametrize(
    ("body", "wanted"),
    [
        (b'{"stream": true, "messages": []}', True),
        (b'{"stream": false}', False),
        (b'{"messages": [{"content": "stream"}]}', False),
        (b'{"stream": tru', False),
        (b"[]", False),
    ],
)
async def test_only_a_body_asking_to_stream_is_kept_alive(body: bytes, wanted: bool) -> None:
    class Request:
        async def read(self) -> bytes:
            return body

    assert await keepalive.streaming(Request()) is wanted


def test_the_defaults_are_a_thirty_second_grace_and_a_fifteen_second_beat() -> None:
    assert (keepalive.KEEPALIVE_GRACE, keepalive.KEEPALIVE_EVERY) == (30.0, 15.0)
    assert FRAME == b": keepalive\n\n"
    made = endpoint()
    assert (made.keepalive_grace, made.keepalive_every) == (30.0, 15.0)


async def test_a_client_gone_mid_wait_still_records_the_sample() -> None:
    async with kept(SlowSampler("hi", delay=0.4)) as started:
        async with httpx.AsyncClient() as http:
            with pytest.raises(httpx.ReadTimeout):
                await http.post(
                    f"{started.address_for('t')}/chat/completions",
                    headers={"Authorization": f"Bearer {started.token}"},
                    json={
                        "model": "m",
                        "stream": True,
                        "messages": [{"role": "user", "content": "x"}],
                    },
                    timeout=httpx.Timeout(5.0, read=0.05),
                )
        await asyncio.sleep(0.5)
        [made] = started.records_for("t")
    assert made.completion_token_ids == ids("hi"), "the sample finishes and is recorded"
