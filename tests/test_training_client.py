"""Test the training client over a fake SDK client."""

from __future__ import annotations

from typing import Any

import pytest
import tinker

from shipyard.pack import datum
from shipyard.proxy.wire import Picture
from shipyard.training.client import DEFAULT_LORA_RANK, TinkerClient
from shipyard.types import SERVE_TTL, Adam, Forward, Sequence
from tests.proxies import MODEL
from tests.trainers import FakeService, FakeTrainingClient, capable


def _datum(tokens: int = 4, *, mu: float = -0.5, advantage: float = 1.0) -> tinker.Datum:
    sequence = Sequence(tuple(range(1, tokens + 1)), ((tokens - 2, tokens),), None)
    return datum(sequence, logprobs=[mu, mu], advantages=[advantage, advantage])


def _client(**fake: Any) -> tuple[TinkerClient, FakeTrainingClient]:
    client = FakeTrainingClient(**fake)
    return TinkerClient(client), client


# --------------------------------------------------------------------- opening


async def test_open_makes_a_lora_at_the_rank_named_or_tinkers_default() -> None:
    service = FakeService()
    opened = await TinkerClient.open(service, MODEL, lora_rank=16)
    assert service.calls == [("lora", MODEL, 16)] and service.metadata == [None]
    assert opened.client is service.client and opened.service is service
    service = FakeService()
    await TinkerClient.open(service, MODEL, metadata={"shipyard_run": "r"})
    assert service.calls == [("lora", MODEL, DEFAULT_LORA_RANK)] and DEFAULT_LORA_RANK == 32
    assert service.metadata == [{"shipyard_run": "r"}], "the training run is tagged too"


async def test_open_from_a_state_path_restores_the_moments_only_if_asked() -> None:
    path = "tinker://run/weights/step-3"
    service = FakeService()
    await TinkerClient.open(service, MODEL, from_checkpoint=path, lora_rank=64)
    assert service.calls == [("from_state", path)]
    service = FakeService()
    await TinkerClient.open(
        service, MODEL, from_checkpoint=path, restore_optimizer=True, metadata={"a": "b"}
    )
    assert service.calls == [("from_state_with_optimizer", path)]
    assert service.metadata == [{"a": "b"}]
    service = FakeService()
    await TinkerClient.open(service, MODEL, restore_optimizer=True)
    assert service.calls == [("lora", MODEL, 32)]


async def test_open_refuses_a_model_the_backend_lacks_or_cannot_train_before_spending() -> None:
    service = FakeService(capabilities=capable(("thinkingmachines/Inkling-Small", True)))
    with pytest.raises(ValueError, match="does not serve 'thinkingmachines/Inkling-Smal'"):
        await TinkerClient.open(service, "thinkingmachines/Inkling-Smal")
    assert service.calls == [] and len(service.closed) == 1
    status, detail = service.closed[0]
    assert status == "errored" and str(detail).startswith("ValueError: the backend does not serve")
    service = FakeService(capabilities=capable((MODEL, True), (f"{MODEL}:sampling-nvfp4", False)))
    with pytest.raises(ValueError, match="sampled but not trained"):
        await TinkerClient.open(service, f"{MODEL}:sampling-nvfp4")
    service = FakeService(capabilities=None)
    opened = await TinkerClient.open(service, "anything/at-all")
    assert opened.client is service.client, "a capabilities read that fails stops nothing"
    service = FakeService(refusing=RuntimeError("no capacity for a new model"))
    with pytest.raises(RuntimeError, match="no capacity"):
        await TinkerClient.open(service, MODEL)
    assert service.closed == [("errored", "RuntimeError: no capacity for a new model")]


# -------------------------------------------------------------------- the step


async def test_a_step_call_is_submitted_on_the_first_await_and_read_on_the_second() -> None:
    client, fake = _client(trained=-0.3)
    adam = Adam(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0)
    first = await client.forward_backward([_datum(), _datum(6)], "ppo", {"clip_low_threshold": 0.8})
    stepped = await client.optim_step(adam, 2e-5)
    second = await client.forward_backward([_datum()], "cispo", None)
    assert [call[0] for call in fake.calls] == [
        "forward_backward",
        "optim_step",
        "forward_backward",
    ]
    found = await first
    assert isinstance(found, Forward)
    assert found.logprobs == [pytest.approx([-0.3] * 3), pytest.approx([-0.3] * 5)]
    assert await stepped is None and (await second).logprobs == [pytest.approx([-0.3] * 3)]
    assert [call[1] for call in fake.calls if call[0] == "consumed"] == ["fb", "optim", "fb"]
    ppo, cispo = [call for call in fake.calls if call[0] == "forward_backward"]
    assert ppo[1:] == ([3, 5], "ppo", {"clip_low_threshold": 0.8}) and cispo[1:] == (
        [3],
        "cispo",
        {},
    )
    (params,) = [call[1] for call in fake.calls if call[0] == "optim_step"]
    assert isinstance(params, tinker.AdamParams)
    assert (params.learning_rate, params.beta1, params.beta2, params.eps) == (2e-5, 0.9, 0.95, 1e-8)
    assert (params.weight_decay, params.grad_clip_norm) == (0.1, 1.0)
    for sent in fake.sent:
        assert set(sent.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}, "no mask"


async def test_compute_logprobs_is_one_batched_forward_aligned_by_the_shift() -> None:
    client, fake = _client(positional=True)
    chained = Sequence((1, 2, 3, 4, 5, 6, 7, 8), ((2, 3), (4, 6), (7, 8)), None)
    single = Sequence((1, 2, 9), ((2, 3),), None)
    assert await client.compute_logprobs([chained, single]) == [[-3.0, -5.0, -6.0, -8.0], [-9.0]]
    assert [call[0] for call in fake.calls] == ["forward"], "one batched pass"
    first, _ = fake.sent
    assert first.model_input.to_ints() == [1, 2, 3, 4, 5, 6, 7]
    assert first.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert first.loss_fn_inputs["weights"].tolist() == [1.0] * 7
    assert await client.compute_logprobs([]) == [] and len(fake.sent) == 2
    pictured = Sequence(
        (1, -1, -1, 5, 6, 7), ((4, 6),), (-0.6, -0.7), (Picture(1, 2, "png", "cG5nIQ=="),)
    )
    assert await client.compute_logprobs([pictured]) == [[-6.0, -7.0]]
    sent = fake.sent[-1]
    kinds = [(type(chunk).__name__, chunk.length) for chunk in sent.model_input.chunks]
    assert kinds == [("EncodedTextChunk", 1), ("ImageChunk", 2), ("EncodedTextChunk", 2)]
    assert sent.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 5, 6, 7]


# --------------------------------------------------------------- saves and loads


async def test_saves_carry_their_ttl_and_return_the_path() -> None:
    client, fake = _client()
    path = await client.save_weights_for_sampler("sample-0", ttl_seconds=SERVE_TTL)
    assert path == "tinker://fake/sampler_weights/sample-0" and SERVE_TTL == 12 * 3600
    assert await client.save_state("step-1", ttl_seconds=604800) == "tinker://fake/weights/step-1"
    await client.save_weights_for_sampler("final", ttl_seconds=None)
    assert fake.calls == [
        ("save_weights_for_sampler", "sample-0", SERVE_TTL),
        ("save_state", "step-1", 604800),
        ("save_weights_for_sampler", "final", None),
    ]


async def test_load_state_restores_the_optimizer_too() -> None:
    client, fake = _client()
    await client.load_state("tinker://run/weights/step-3")
    assert fake.calls == [
        ("load_state_with_optimizer", "tinker://run/weights/step-3"),
        ("consumed", "load"),
    ]


async def test_close_finishes_the_session_once_with_its_status() -> None:
    service = FakeService()
    opened = await TinkerClient.open(service, MODEL)
    await opened.close()
    await opened.close()
    assert service.closed == [("success", None)] and opened.service is None
    service = FakeService()
    opened = await TinkerClient.open(service, MODEL)
    await opened.close("errored", "RuntimeError: the loop broke")
    assert service.closed == [("errored", "RuntimeError: the loop broke")]
    handed, _ = _client()
    await handed.close()  # a client handed in has no session of ours to finish
