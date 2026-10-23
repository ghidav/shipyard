"""Tests for the Together backend: a proxy that samples the run's own Together session, whose
generator answers as Tinker's sampling client does."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import tinker

from harness_proxy.backends import together
from harness_proxy.chat import Refused
from harness_proxy.client import Proxy
from harness_proxy.server import Lane, Server
from proxies import MODEL, FakeRenderer, ask, ids, opened, recorded, root


@dataclass
class FakeGenerator:
    """Fake Together generator: answers `answer`, its token ids as strings, as int64 tokens
    arrive on Together's wire."""

    answer: str
    asked: list[tuple[Any, int, Any]] = field(default_factory=list)

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked.append((prompt, num_samples, sampling_params))
        tokens = [str(ord(character)) for character in self.answer]
        sequence = SimpleNamespace(
            tokens=tokens,
            logprobs=[-0.5] * len(tokens),
            stop_reason="STOP_REASON_STOP",
            prompt_cache_hit_tokens=2,
        )
        return SimpleNamespace(sequences=[sequence] * num_samples, policy_segments=[])


@dataclass
class FakeSession:
    """Fake attached Together session: its generator, and whether the proxy let it go."""

    generator: FakeGenerator
    detached: bool = False

    async def detach_async(self) -> None:
        self.detached = True


def on_together(session: FakeSession) -> Server:
    lane = Lane(
        name="agent",
        model=MODEL,
        renderer=FakeRenderer(),
        backend="together",
        max_tokens=64,
        max_context=4096,
        _service=session,
    )
    return Server(lanes=[lane], host="127.0.0.1", bind="127.0.0.1", control_token="ctl")


async def test_a_together_proxy_samples_the_runs_session_and_records_as_on_tinker() -> None:
    """The run publishes new weights to its session itself, so a trial opened on them only
    names them: each record says which weights answered it."""
    session = FakeSession(FakeGenerator("hi"))
    async with on_together(session) as started:
        reply = await ask(started, "t1")
        assert reply.status_code == 200
        assert reply.json()["choices"][0]["message"]["content"] == "hi"
        (prompt, samples, params) = session.generator.asked[0]
        assert prompt == {"chunks": [{"encoded_text": {"tokens": list(ids("go"))}}]}
        assert samples == 1 and "stop" not in params
        token = await opened(started, "t2", weights="together://s1/sample-1")
        with pytest.raises(RuntimeError, match="is not a together:// path"):
            proxy = Proxy(origin=root(started), control_token=started.control_token)
            await proxy.open("t3", weights="tinker://run/weights")
        await ask(started, "t2", key=token)
        (first,), (second,) = recorded(started, "t1"), recorded(started, "t2")
    assert (first.weights, second.weights) == (None, "together://s1/sample-1")
    assert first.completion_token_ids == ids("hi") and first.inference_logprobs == (-0.5, -0.5)
    assert first.cached_tokens == 2 and first.stop_reason == "stop"
    assert session.detached, "the proxy lets the session go and leaves it to the run"


def test_together_stops_on_strings_and_leaves_token_ids_to_the_end_of_sequence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="harness_proxy.backends.together"):
        found = together.wire_params(tinker.SamplingParams(max_tokens=8, seed=3, stop=[90210]))
    assert found["max_tokens"] == 8 and found["seed"] == 3 and "stop" not in found
    assert "token-id stops [90210]" in caplog.text
    assert together.wire_params(tinker.SamplingParams(stop=["</done>"]))["stop"] == ["</done>"]


async def test_together_refuses_an_image_before_it_samples() -> None:
    generator = FakeGenerator("hi")
    pictured = SimpleNamespace(
        chunks=[tinker.types.EncodedTextChunk(tokens=[1, 2]), SimpleNamespace(type="image")]
    )
    with pytest.raises(Refused, match="text only"):
        await together.Sampler(generator).sample_async(pictured, 1, tinker.SamplingParams())
    assert generator.asked == []


def test_together_answers_in_tinkers_shape() -> None:
    cut = SimpleNamespace(tokens=["5", "6"], logprobs=None, stop_reason="STOP_REASON_LENGTH")
    cut.prompt_cache_hit_tokens = 3
    found = together.response(SimpleNamespace(sequences=[cut]))
    (one,) = found.sequences
    assert list(one.tokens) == [5, 6] and one.logprobs is None and one.stop_reason == "length"
    assert found.prompt_cache_hit_tokens == 3


async def test_a_together_proxy_needs_the_runs_session_and_a_backend_it_knows() -> None:
    with pytest.raises(ValueError, match="pass its id as `session`"):
        await together.connect(metadata=None, session=None)
    with pytest.raises(ValueError, match="no backend 'modal'"):
        Lane(name="agent", model=MODEL, backend="modal")
