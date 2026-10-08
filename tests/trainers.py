"""The Tinker training side, faked: a service with the factories `Trainer.create` chooses
between and the sampling client an anchor is, a training client that notes every call
and every result consumed, futures answering at once; a Preset and a Batch built by name."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tinker
import torch
from tinker import TensorData
from tinker.types import GetServerCapabilitiesResponse, SupportedModel

from shipyard.credit import Batch
from shipyard.recipes.train import Preset


def preset(**named: Any) -> Preset:
    """dapo's loss and clipping with every field set, any of them overridden by name: one
    substep, token losses summed and no refill unless a test asks for them."""
    defaults = dict(
        name="dapo",
        normalize=True,
        loss_fn="ppo",
        loss_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
        clipping="clip 0.2 / 0.28",
        aggregation="sum",
        length_penalty=0.0,
        length_floor=0,
        kl_coef=0.0,
        reference="trainer",
        learning_rate=2e-5,
        substeps=1,
        refill=0,
    )
    return Preset(**{**defaults, **named})


def batch(*datums: tinker.Datum, **named: Any) -> Batch:
    """A Batch over `datums`, one credited rollout, sequence and group each, any count
    overridden."""
    count = len(datums)
    defaults = dict(
        datums=tuple(datums),
        owners=tuple(range(count)),
        groups=1,
        rollouts=count,
        graded=count,
        masked=0,
        degenerate=0,
        credited=count,
        sequences=count,
        reward_mean=0.5,
        reward_spread=0.5,
        length_mean=2.0,
        anchor_kl=None,
        reference_tokens=0,
        anchor_tokens=0,
    )
    return Batch(**{**defaults, **named})


@dataclass
class FakeFuture:
    """What the SDK hands back from a save, a forward or an optimizer step: awaited by
    `result_async`, which the client notes so a pipeline's order can be read off `calls`."""

    value: Any
    client: FakeTrainingClient | None = None
    kind: str = ""

    async def result_async(self) -> Any:
        if self.client is not None:
            self.client.calls.append(("consumed", self.kind))
        return self.value


@dataclass
class FakeSaved:
    path: str


@dataclass
class FakeOptimStep:
    metrics: dict[str, float] | None = None


@dataclass
class FakeForwardBackward:
    """`loss_fn_outputs[i]["logprobs"]`, a TensorData per datum, is all the trainer reads."""

    loss_fn_outputs: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeTrainingClient:
    """`tinker.TrainingClient` with the network out: `forward` scores every position
    `logprob` (the reference mu), or under `positional` minus the token it predicts;
    `forward_backward` scores `trained` (pi), or raises `failing`; each `optim_step` moves
    `trained` by `drift`, as an update moves the weights. Every call lands on `calls` in
    order, the datums sent on `sent`."""

    logprob: float = -0.5
    trained: float = -0.5
    drift: float = 0.0
    positional: bool = False
    failing: Exception | None = None
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    sent: list[Any] = field(default_factory=list)

    def _scored(self, data: list[Any], value: float) -> list[dict[str, Any]]:
        return [
            {"logprobs": TensorData.from_torch(torch.full((one.model_input.length,), value))}
            for one in data
        ]

    def _predicted(self, data: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "logprobs": TensorData.from_torch(
                    torch.tensor([-float(t) for t in one.loss_fn_inputs["target_tokens"].tolist()])
                )
            }
            for one in data
        ]

    async def forward_async(self, data: list[Any], loss_fn: str, loss_fn_config: Any = None) -> Any:
        self.calls.append(("forward", [one.model_input.length for one in data], loss_fn))
        self.sent.extend(data)
        rows = self._predicted(data) if self.positional else self._scored(data, self.logprob)
        return FakeFuture(FakeForwardBackward(rows))

    async def forward_backward_async(
        self, data: list[Any], loss_fn: str, loss_fn_config: Any = None
    ) -> Any:
        if self.failing is not None:
            raise self.failing
        lengths = [one.model_input.length for one in data]
        self.calls.append(("forward_backward", lengths, loss_fn, dict(loss_fn_config or {})))
        self.sent.extend(data)
        return FakeFuture(FakeForwardBackward(self._scored(data, self.trained)), self, "fb")

    async def optim_step_async(self, adam_params: Any) -> Any:
        self.calls.append(("optim_step", adam_params))
        self.trained += self.drift
        return FakeFuture(FakeOptimStep({"lr": adam_params.learning_rate}), self, "optim")

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> Any:
        self.calls.append(("save_state", name, ttl_seconds))
        return FakeFuture(FakeSaved(f"tinker://fake/weights/{name}"))

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> Any:
        self.calls.append(("save_weights_for_sampler", name, ttl_seconds))
        return FakeFuture(FakeSaved(f"tinker://fake/sampler_weights/{name}"))


class FlatAnchor:
    """A sampling client on the starting weights, scoring every token `logprob`; the first
    position is unscored, as `compute_logprobs` leaves it."""

    def __init__(self, logprob: float = -0.1) -> None:
        self.logprob = logprob
        self.asked: list[list[int]] = []

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        self.asked.append(list(prompt.to_ints()))
        return [None] + [self.logprob] * (prompt.length - 1)


def capable(*models: tuple[str, bool]) -> GetServerCapabilitiesResponse:
    """`get_server_capabilities` answering these `(name, trainable)` pairs."""
    return GetServerCapabilitiesResponse(
        supported_models=[SupportedModel(model_name=n, trainable=t) for n, t in models]
    )


@dataclass
class FakeService:
    """`tinker.ServiceClient` in what a run asks of it: the three training factories (the
    `user_metadata` each was given on `metadata`), the anchor's sampling client, the
    capabilities (`None` raises, which must not stop a run), and `close` with its status."""

    client: FakeTrainingClient = field(default_factory=FakeTrainingClient)
    capabilities: GetServerCapabilitiesResponse | None = None
    anchor: FlatAnchor = field(default_factory=FlatAnchor)
    refusing: Exception | None = None
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    metadata: list[dict[str, str] | None] = field(default_factory=list)
    closed: list[tuple[str, str | None]] = field(default_factory=list)

    async def get_server_capabilities_async(self) -> GetServerCapabilitiesResponse:
        if self.capabilities is None:
            raise RuntimeError("no capabilities here")
        return self.capabilities

    def _opened(self, user_metadata: dict[str, str] | None) -> FakeTrainingClient:
        if self.refusing is not None:
            raise self.refusing
        self.metadata.append(user_metadata)
        return self.client

    async def create_lora_training_client_async(
        self, base_model: str, rank: int = 32, user_metadata: dict[str, str] | None = None
    ) -> FakeTrainingClient:
        self.calls.append(("lora", base_model, rank))
        return self._opened(user_metadata)

    async def create_training_client_from_state_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> FakeTrainingClient:
        self.calls.append(("from_state", path))
        return self._opened(user_metadata)

    async def create_training_client_from_state_with_optimizer_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> FakeTrainingClient:
        self.calls.append(("from_state_with_optimizer", path))
        return self._opened(user_metadata)

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> FlatAnchor:
        self.calls.append(("sampling", base_model, model_path))
        return self.anchor

    async def close(self, status: str, detail: str | None = None) -> None:
        self.closed.append((status, detail))
