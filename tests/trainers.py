"""Fakes for the Tinker training side: a service client, a training client, and futures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tinker import TensorData
from tinker.types import GetServerCapabilitiesResponse, SupportedModel


@dataclass
class FakeFuture:
    """Fake SDK future for a save, a forward pass, or an optimizer step. `result_async`
    appends a "consumed" entry to the client's `calls`, so tests can check pipelining order."""

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
    """Fake forward or forward-backward result. The trainer reads only
    `loss_fn_outputs[i]["logprobs"]`."""

    loss_fn_outputs: list[dict[str, Any]]


@dataclass
class FakeTrainingClient:
    """Fake `tinker.TrainingClient`. `forward` scores every position `logprob` (the reference
    logprobs), or the negated target token id when `positional` is set; `forward_backward` scores
    `trained` (pi) or raises `failing`. Calls are appended to `calls` and datums to `sent`."""

    logprob: float = -0.5
    trained: float = -0.5
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
        return FakeFuture(FakeOptimStep({"lr": adam_params.learning_rate}), self, "optim")

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> Any:
        self.calls.append(("save_state", name, ttl_seconds))
        return FakeFuture(FakeSaved(f"tinker://fake/weights/{name}"))

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> Any:
        self.calls.append(("save_weights_for_sampler", name, ttl_seconds))
        return FakeFuture(FakeSaved(f"tinker://fake/sampler_weights/{name}"))

    async def load_state_with_optimizer_async(self, path: str) -> Any:
        self.calls.append(("load_state_with_optimizer", path))
        return FakeFuture(None, self, "load")


class FlatAnchor:
    """Fake sampling client on the starting weights that scores every token `logprob`. The
    first position is unscored, as in `compute_logprobs`."""

    def __init__(self, logprob: float = -0.1) -> None:
        self.logprob = logprob
        self.asked: list[list[int]] = []

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        self.asked.append(chunk_ids(prompt))
        return [None] + [self.logprob] * (prompt.length - 1)


def chunk_ids(prompt: Any) -> list[int]:
    """Return a prompt's token ids with each image chunk as a run of -1 of its length. This
    avoids `to_ints`, which rejects images."""
    out: list[int] = []
    for chunk in prompt.chunks:
        tokens = getattr(chunk, "tokens", None)
        out.extend(list(tokens) if tokens is not None else [-1] * int(chunk.length))
    return out


def capable(*models: tuple[str, bool]) -> GetServerCapabilitiesResponse:
    """Build a `get_server_capabilities` response listing these `(name, trainable)` pairs."""
    return GetServerCapabilitiesResponse(
        supported_models=[SupportedModel(model_name=n, trainable=t) for n, t in models]
    )


@dataclass
class FakeService:
    """Fake `tinker.ServiceClient` covering what a run uses: the three training client
    factories (each `user_metadata` is kept in `metadata`), the anchor sampling client, the
    server capabilities (None makes the call raise, which a run must survive), and `close`,
    which records its status."""

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
