"""The trainer against a fake training client: how it opens, what it publishes and saves
at which TTL, and the step itself: pipelined substeps, the mask kept back, the metrics
read off the step's own logprobs."""

from __future__ import annotations

from typing import Any

import pytest
import tinker
import torch

from shipyard.pack import Sequence, datum
from shipyard.recipes.train import row
from shipyard.trainer import DEFAULT_LORA_RANK, SERVE_TTL, Trainer, Update, _observed, split
from tests.trainers import FakeService, FakeTrainingClient, batch, capable, preset

MODEL = "Qwen/Qwen3-8B"


def _datum(tokens: int = 4, *, mu: float = -0.5, advantage: float = 1.0) -> tinker.Datum:
    """A sequence of `tokens` whose last two are the policy's, as `pack.datum` builds it."""
    sequence = Sequence(tuple(range(1, tokens + 1)), ((tokens - 2, tokens),), None)
    return datum(sequence, logprobs=[mu, mu], advantages=[advantage, advantage])


def _trainer(**client: Any) -> tuple[Trainer, FakeTrainingClient]:
    fake = FakeTrainingClient(**client)
    return Trainer(fake), fake


def _kinds(client: FakeTrainingClient) -> list[str]:
    return [str(call[0]) for call in client.calls]


# --------------------------------------------------------------------- opening


async def test_create_opens_a_lora_at_the_rank_named_or_tinkers_default() -> None:
    service = FakeService()
    trained = await Trainer.create(service, MODEL, lora_rank=16)
    assert service.calls == [("lora", MODEL, 16)] and service.metadata == [None]
    assert trained.client is service.client and trained.updates == 0
    assert trained.service is service
    service = FakeService()
    await Trainer.create(service, MODEL, metadata={"shipyard_run": "r"})
    assert service.calls == [("lora", MODEL, DEFAULT_LORA_RANK)] and DEFAULT_LORA_RANK == 32
    assert service.metadata == [{"shipyard_run": "r"}], "the training run is tagged too"


async def test_create_from_a_state_path_restores_the_weights_and_the_moments_only_if_asked() -> (
    None
):
    path = "tinker://run/weights/step-3"
    service = FakeService()
    trained = await Trainer.create(service, MODEL, from_checkpoint=path, lora_rank=64)
    assert service.calls == [("from_state", path)] and trained.client is service.client
    service = FakeService()
    await Trainer.create(
        service, MODEL, from_checkpoint=path, restore_optimizer=True, metadata={"a": "b"}
    )
    assert service.calls == [("from_state_with_optimizer", path)]
    assert service.metadata == [{"a": "b"}]
    service = FakeService()
    await Trainer.create(service, MODEL, from_checkpoint=None, restore_optimizer=True)
    assert service.calls == [("lora", MODEL, 32)], "no state to restore is a fresh LoRA"


async def test_create_refuses_a_model_the_backend_lacks_or_cannot_train_before_spending() -> None:
    service = FakeService(capabilities=capable(("thinkingmachines/Inkling-Small", True)))
    with pytest.raises(ValueError, match="does not serve 'thinkingmachines/Inkling-Smal'"):
        await Trainer.create(service, "thinkingmachines/Inkling-Smal")
    assert service.calls == [] and len(service.closed) == 1
    status, detail = service.closed[0]
    assert status == "errored" and str(detail).startswith("ValueError: the backend does not serve")
    service = FakeService(capabilities=capable((MODEL, True), (f"{MODEL}:sampling-nvfp4", False)))
    with pytest.raises(ValueError, match="sampled but not trained"):
        await Trainer.create(service, f"{MODEL}:sampling-nvfp4")
    assert service.calls == []
    service = FakeService(capabilities=capable((MODEL, True)))
    trained = await Trainer.create(service, MODEL)
    assert trained.client is service.client and service.closed == []


async def test_a_capabilities_read_that_fails_does_not_stop_the_run() -> None:
    service = FakeService(capabilities=None)
    trained = await Trainer.create(service, "anything/at-all")
    assert trained.client is service.client and service.calls == [("lora", "anything/at-all", 32)]


async def test_a_session_whose_factory_refused_is_closed_as_errored() -> None:
    service = FakeService(refusing=RuntimeError("no capacity for a new model"))
    with pytest.raises(RuntimeError, match="no capacity"):
        await Trainer.create(service, MODEL)
    assert service.closed == [("errored", "RuntimeError: no capacity for a new model")]


# ------------------------------------------------------------ publish and save


async def test_publish_saves_sampler_weights_under_the_serve_ttl_and_returns_the_path() -> None:
    trained, client = _trainer()
    assert await trained.publish("sample-0") == "tinker://fake/sampler_weights/sample-0"
    assert client.calls == [("save_weights_for_sampler", "sample-0", SERVE_TTL)]
    assert SERVE_TTL == 12 * 3600
    await trained.publish("sample-1", ttl_seconds=7200)
    assert client.calls[-1] == ("save_weights_for_sampler", "sample-1", 7200)


async def test_save_writes_state_and_sampler_weights_under_one_ttl() -> None:
    trained, client = _trainer()
    saved = await trained.save("step-1", ttl_hours=168.0)
    assert (saved.tag, saved.ttl_hours) == ("step-1", 168.0)
    assert saved.state_path == "tinker://fake/weights/step-1"
    assert saved.sampler_path == "tinker://fake/sampler_weights/step-1"
    assert client.calls == [
        ("save_state", "step-1", 604800),
        ("save_weights_for_sampler", "step-1", 604800),
    ]
    permanent = await trained.save("final")
    assert permanent.ttl_hours is None
    assert client.calls[-2:] == [
        ("save_state", "final", None),
        ("save_weights_for_sampler", "final", None),
    ]


async def test_save_refuses_a_ttl_under_the_sdks_hour() -> None:
    trained, client = _trainer()
    with pytest.raises(ValueError, match="ttl_hours = 0.5"):
        await trained.save("doomed", ttl_hours=0.5)
    assert client.calls == []


# ------------------------------------------------------------------- the step


async def test_apply_pipelines_each_substep_ahead_of_the_one_it_consumes() -> None:
    trained, client = _trainer()
    found = batch(*(_datum(4) for _ in range(4)))
    update = await trained.apply(found, preset(substeps=2))
    assert _kinds(client) == [
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
        "consumed",
        "consumed",
        "consumed",
        "consumed",
    ], "the second substep is enqueued before the first result is consumed"
    assert [call[1] for call in client.calls if call[0] == "consumed"] == [
        "fb",
        "optim",
        "fb",
        "optim",
    ]
    assert update.substeps == 2 and update.train_tokens == 12 and trained.updates == 1
    assert [call[1] for call in client.calls if call[0] == "forward_backward"] == [[3, 3], [3, 3]]


async def test_apply_hands_the_loss_and_its_config_and_the_adam_knobs_through() -> None:
    trained, client = _trainer()
    update = await trained.apply(
        batch(_datum()),
        preset(
            name="cispo",
            loss_fn="cispo",
            loss_config={"clip_low_threshold": 0.0, "clip_high_threshold": 1.2},
        ),
    )
    (sent,) = [call for call in client.calls if call[0] == "forward_backward"]
    assert sent[2:] == ("cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 1.2})
    (adam,) = [call[1] for call in client.calls if call[0] == "optim_step"]
    assert isinstance(adam, tinker.AdamParams)
    assert (adam.learning_rate, adam.beta1, adam.beta2, adam.eps) == (2e-5, 0.9, 0.95, 1e-8)
    assert update.loss_fn == "cispo" and update.learning_rate == 2e-5 and update.seconds >= 0.0


async def test_apply_sends_the_datums_without_their_mask() -> None:
    trained, client = _trainer()
    await trained.apply(batch(_datum(5)), preset())
    (sent,) = client.sent
    assert set(sent.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert sent.model_input.to_ints() == [1, 2, 3, 4]
    assert sent.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4, 5]
    assert sent.loss_fn_inputs["advantages"].tolist() == [0.0, 0.0, 1.0, 1.0]


async def test_more_substeps_than_datums_runs_one_per_datum() -> None:
    trained, client = _trainer()
    update = await trained.apply(batch(_datum(), _datum()), preset(substeps=5))
    assert update.substeps == 2 and _kinds(client).count("optim_step") == 2


async def test_substeps_draw_on_the_whole_batch_the_same_way_every_time() -> None:
    async def halves() -> list[list[int]]:
        trained, client = _trainer()
        found = batch(*(_datum(tokens) for tokens in range(3, 19)))
        await trained.apply(found, preset(substeps=2))
        return [call[1] for call in client.calls if call[0] == "forward_backward"]

    first, second = await halves()
    assert sorted(first + second) == list(range(2, 18)), "every datum once"
    assert first != list(range(2, 10)), "not the batch's own order"
    assert min(first) < 10 <= max(first), "each substep draws on both halves"
    assert await halves() == [first, second]
    trained, client = _trainer()
    await trained.apply(batch(*(_datum(tokens) for tokens in range(3, 11))), preset())
    assert [call[1] for call in client.calls if call[0] == "forward_backward"] == [
        list(range(2, 10))
    ]
    # The cookbook's cut: the shorter parts first.
    assert split([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4, 5]] and split([1, 2], 2) == [[1], [2]]
    assert split(list(range(7)), 3) == [[0, 1], [2, 3], [4, 5, 6]]
    assert split(list(range(10)), 4) == [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]


async def test_apply_reports_what_the_steps_own_logprobs_say_against_mu() -> None:
    """mu is -0.5 on the two acted tokens, the forward pass says -0.3 there: the policy
    moved -0.2 nats a token, and the entropy is the mean of -mu over the same tokens."""
    trained, _ = _trainer(trained=-0.3)
    update = await trained.apply(batch(_datum(), _datum(6)), preset())
    assert update.kl_v1 == pytest.approx(-0.2, abs=1e-5)
    assert update.kl_v2 == pytest.approx(0.5 * 0.04, abs=1e-5)
    assert update.entropy == pytest.approx(0.5, abs=1e-5)
    assert update.train_tokens == 3 + 5 and update.loss_fn == "ppo"
    logged = row(2, batch(_datum(), _datum(6)), update)
    assert logged["step"] == 2 and logged["trained"] is True
    assert [key for key in logged if key not in ("step", "trained")] == [
        "groups",
        "rollouts",
        "graded",
        "masked",
        "degenerate",
        "sequences",
        "sequences_per_rollout",
        "train_tokens",
        "reward_mean",
        "reward_spread",
        "length_mean",
        "kl_v1",
        "kl_v2",
        "entropy",
        "learning_rate",
        "substeps",
        "loss_fn",
        "seconds",
    ]
    assert logged["train_tokens"] == 8 and logged["kl_v1"] == pytest.approx(-0.2, abs=1e-5)


def test_observed_reads_nothing_rather_than_raising_where_it_cannot_compare() -> None:
    one = _datum()
    assert _observed([one], [torch.tensor([-0.9, -0.3, -0.3])]) == pytest.approx(
        {"kl_v1": -0.2, "kl_v2": 0.02, "entropy": 0.5}, abs=1e-5
    ), "the prompt's position is not acted; only the two the policy wrote count"
    supervised = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0] * 3},
    )
    assert _observed([supervised], [torch.tensor([-0.5] * 3)]) == {}
    unacted = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "logprobs": [-0.5] * 3,
            "mask": tinker.TensorData.from_torch(torch.zeros(3)),
        },
    )
    assert _observed([unacted], [torch.tensor([-0.5] * 3)]) == {}
    assert _observed([one, one], [torch.tensor([-0.3] * 3)]) == {}, "misaligned lists"
    assert _observed([one], [torch.tensor([-0.3] * 7)]) == {}, "a tensor of the wrong length"
    none = row(0, batch(_datum()), Update(3, 1, 1e-5, "ppo", None, None, None, 0.0))
    assert "kl_v1" not in none and "entropy" not in none and none["train_tokens"] == 3


async def test_an_empty_batch_is_refused_and_a_failed_step_still_counts_as_moved() -> None:
    trained, client = _trainer()
    empty = batch(rollouts=2, graded=2, degenerate=1, reward_mean=1.0, reward_spread=0.0)
    with pytest.raises(ValueError, match="no datums"):
        await trained.apply(empty, preset())
    assert client.calls == [] and trained.updates == 0
    trained, client = _trainer(failing=RuntimeError("substep 2 of 4 failed"))
    with pytest.raises(RuntimeError, match="substep 2"):
        await trained.apply(batch(*(_datum() for _ in range(4))), preset(substeps=4))
    assert trained.updates == 1, "optimizer steps may have landed; credit must refuse"


# -------------------------------------------------------------------- closing


async def test_close_finishes_the_session_once_and_carries_the_status() -> None:
    service = FakeService()
    trained = await Trainer.create(service, MODEL)
    await trained.close()
    await trained.close()
    assert service.closed == [("success", None)] and trained.service is None
    service = FakeService()
    trained = await Trainer.create(service, MODEL)
    await trained.close("errored", "RuntimeError: the loop broke")
    assert service.closed == [("errored", "RuntimeError: the loop broke")]
    handed, _ = _trainer()
    await handed.close()  # a client handed in has no session of ours to finish
