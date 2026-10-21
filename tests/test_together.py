"""Tests for an agent on Together post-training: the run provisions a model resource, trains a
session on it, and its proxy samples that session's generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker

from proxies import MODEL
from shipyard import Blueprint, Model, Trainer
from shipyard.models.clients.together import TogetherClient, adam_params, loss, sample
from shipyard.run import files
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import ONE_STEP, blueprint, train
from tests.service import FakeService

#: The blueprint's agent, on Together.
ON_TOGETHER = {'harness = "pi@0.85.1"': 'provider = "together"\nharness = "pi@0.85.1"'}


@dataclass
class FakeTogetherTrainer:
    """Fake Together trainer: each operation it ran, and logprobs of -0.3 for every slot."""

    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def forward_backward_async(
        self, *, samples: Any, loss: Any, return_loss_fn_outputs: bool
    ) -> Any:
        self.calls.append(("forward_backward", (list(samples), loss, return_loss_fn_outputs)))
        return SimpleNamespace(loss_fn_outputs=[_scored(one) for one in samples])

    async def forward_async(self, *, samples: Any, loss: Any, return_loss_fn_outputs: bool) -> Any:
        self.calls.append(("forward", (list(samples), loss)))
        return SimpleNamespace(loss_fn_outputs=[_scored(one) for one in samples])

    async def optim_step_async(self, *, adam_params: Any) -> None:
        self.calls.append(("optim_step", adam_params))

    async def weights_sync_async(self) -> None:
        self.calls.append(("weights_sync", None))


def _scored(one: Any) -> Any:
    slots = sum(len(chunk["encoded_text"]["tokens"]) for chunk in one["model_input"]["chunks"])
    return SimpleNamespace(tensors={"logprobs": SimpleNamespace(data=[-0.3] * slots)})


@dataclass
class FakeTogetherSession:
    session_id: str
    trainer: FakeTogetherTrainer
    saved: list[str] = field(default_factory=list)
    stopped: bool = False

    async def create_training_checkpoint_async(self) -> Any:
        self.saved.append("training")
        return SimpleNamespace(id=f"training-{len(self.saved)}")

    async def create_inference_checkpoint_async(self) -> Any:
        self.saved.append("inference")
        return SimpleNamespace(id=f"inference-{len(self.saved)}")

    async def stop_async(self) -> None:
        self.stopped = True


@dataclass
class FakeResources:
    """Fake Together model resource: the sessions opened on it, with what each asked for."""

    model: str
    trainer: FakeTogetherTrainer = field(default_factory=FakeTogetherTrainer)
    sessions: list[tuple[dict[str, Any], FakeTogetherSession]] = field(default_factory=list)
    stopped: bool = False

    async def create_session_async(self, **asked: Any) -> FakeTogetherSession:
        session = FakeTogetherSession(f"s{len(self.sessions) + 1}", self.trainer)
        self.sessions.append((asked, session))
        return session

    async def stop_async(self) -> None:
        self.stopped = True


@pytest.fixture
def provisioned(monkeypatch: pytest.MonkeyPatch) -> list[FakeResources]:
    """Provision fake Together resources, and return them in the order they were made."""
    made: list[FakeResources] = []

    async def provision(model: str) -> FakeResources:
        made.append(FakeResources(model))
        return made[-1]

    monkeypatch.setattr("shipyard.models.clients.together.provisioned", provision)
    return made


async def test_a_run_on_together_trains_its_session_and_its_proxy_samples_it(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],
) -> None:
    """The proxy samples the session the run opened. An update trains it, syncs the generator
    and names the new weights; the checkpoints are Together's, and both GPUs bills stop."""
    proxy, service = fakes
    harbor(Alternating())
    home = blueprint(tmp_path, **ONE_STEP, **ON_TOGETHER)
    directory = await train(home, tmp_path / "runs")
    (resources,) = provisioned
    ((asked, session),) = resources.sessions
    assert resources.model == MODEL and service.calls == [], "Tinker is never asked"
    assert asked == {"display_name": directory.name, "lora_config": {"rank": 32}}
    ((_, settings, _),) = proxy.placed
    (lane,) = settings["lanes"]
    assert (lane["backend"], lane["session"]) == ("together", "s1")
    assert proxy.swapped == ["together://s1/start", "together://s1/sample-1"]
    assert [name for name, _ in resources.trainer.calls] == [
        "forward_backward",
        "optim_step",
        "weights_sync",
    ]
    samples, chosen, outputs = resources.trainer.calls[0][1]
    assert chosen == {"type": "LOSS_TYPE_IMPORTANCE_SAMPLING"} and outputs is True
    inputs = {key for one in samples for key in one["loss_fn_inputs"]}
    assert inputs == {"target_tokens", "logprobs", "advantages"}, "the mask stays local"
    adam = resources.trainer.calls[1][1]
    assert adam["learning_rate"] == 2e-5 and adam["grad_clip_norm"] == 0.0
    (job,) = files.read(directory / files.JOBS)
    assert job["backend"] == "together"
    rows = list(files.read(directory / files.CHECKPOINTS))
    assert [(row["tag"], row["sampler_weights"]) for row in rows] == [
        ("step-1", {}),
        ("final", {"agent": "inference-3"}),
    ]
    assert all(row["training_state"]["agent"].startswith("training-") for row in rows)
    assert [row["ttl_hours"] for row in rows] == [None, None], "Together keeps them until deleted"
    assert session.stopped and resources.stopped, "the session and its GPUs stop with the run"
    spent = files.read_json(directory / files.COSTS)["tokens"]["together"][MODEL]
    assert spent["train"] > 0 and spent["sample"] > 0


async def test_a_model_on_tinker_opens_a_session_of_its_own_beside_a_together_agent(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],
) -> None:
    _, service = fakes

    def taught(teacher: str) -> type[Trainer]:
        class Taught(Trainer):
            def __init__(self, config: Blueprint, directory: Path) -> None:
                super().__init__(config, directory)
                self.teacher = Model(teacher)

        return Taught

    harbor(Alternating())
    home = blueprint(tmp_path, **ONE_STEP, **ON_TOGETHER)
    await train(home, tmp_path / "runs", taught("Qwen/Qwen3-32B"))
    assert ("sampling", "Qwen/Qwen3-32B", None) in service.calls and len(provisioned) == 1
    assert [status for status, _ in service.closed] == ["success"], "its session ends with the run"
    with pytest.raises(ValueError, match="^teacher = Model.*does not serve 'Qwen/Qwen3-33B'"):
        await train(home, tmp_path / "runs", taught("Qwen/Qwen3-33B"))
    assert len(provisioned) == 1, "no GPU is provisioned for a run its teacher refuses"


async def test_a_resumed_together_run_opens_its_session_on_the_checkpoint() -> None:
    """A session cannot load a checkpoint once open, so the resumed one is the first opened."""
    resources = FakeResources(MODEL)

    async def provide(model: str) -> FakeResources:
        return resources

    client = TogetherClient(MODEL, run="r", provision=provide)
    await client.open()
    await client.load_state("training-7")
    assert await client.serving() == {"backend": "together", "session": "s1"}
    assert [asked for asked, _ in resources.sessions] == [
        {"display_name": "r", "resume_from_checkpoint_id": "training-7", "load_optimizer": True}
    ]


def test_together_takes_tinkers_losses_and_adam_settings_as_its_own() -> None:
    clipped = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
    assert loss("ppo", clipped) == {"type": "LOSS_TYPE_PPO", "ppo_params": clipped}
    assert loss("importance_sampling", None) == {"type": "LOSS_TYPE_IMPORTANCE_SAMPLING"}
    assert loss("cross_entropy", None) == {"type": "LOSS_TYPE_CROSS_ENTROPY"}
    adam = adam_params(tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8), 1e-4)
    assert adam["learning_rate"] == 1e-4 and adam["grad_clip_norm"] == 0.0
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3], dtype="int64"),
            "logprobs": tinker.TensorData(data=[0.0, -0.5], dtype="float32"),
            "advantages": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
            "mask": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
        },
    )
    weighed = {"data": [0.0, 1.0], "dtype": "float32"}
    assert sample(datum, "cross_entropy") == {
        "model_input": {"chunks": [{"encoded_text": {"tokens": [1, 2]}}]},
        "loss_fn_inputs": {"target_tokens": {"data": [2, 3], "dtype": "int64"}, "weights": weighed},
    }, "each target weighed by its advantage, as on Tinker; the mask stays local"
    assert set(sample(datum, "ppo")["loss_fn_inputs"]) == {
        "target_tokens",
        "logprobs",
        "advantages",
    }
