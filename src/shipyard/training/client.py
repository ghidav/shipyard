"""Tinker's training client, reduced to what a recipe needs."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Mapping
from collections.abc import Sequence as Values
from typing import Any

import tinker
import torch
from tinker import TensorData

from shipyard.pack import shifted
from shipyard.session import quietly_closed, server_has
from shipyard.types import Adam, Datum, Forward, Sequence, TrainingClient

logger = logging.getLogger(__name__)

#: Tinker's default rank, used when the blueprint names none.
DEFAULT_LORA_RANK = 32


class TinkerClient(TrainingClient):
    """Over an SDK training client; `restored` means the optimizer state came with the weights."""

    def __init__(self, client: Any, *, service: Any = None, restored: bool = False) -> None:
        self.client = client
        self.service = service
        self.restored = restored

    @classmethod
    async def open(
        cls,
        service: Any,
        base_model: str,
        *,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> TinkerClient:
        """A LoRA on `base_model`, or the weights at `from_checkpoint`; the server is checked."""
        try:
            await server_has(service, base_model)
            if from_checkpoint:
                resume = (
                    service.create_training_client_from_state_with_optimizer_async
                    if restore_optimizer
                    else service.create_training_client_from_state_async
                )
                client = await resume(from_checkpoint, user_metadata=metadata)
            else:
                rank = DEFAULT_LORA_RANK if lora_rank is None else int(lora_rank)
                client = await service.create_lora_training_client_async(
                    base_model=base_model, rank=rank, user_metadata=metadata
                )
        except BaseException as failed:
            await quietly_closed(service, failed)
            raise
        return cls(client, service=service, restored=bool(from_checkpoint) and restore_optimizer)

    async def forward_backward(
        self, datums: Values[Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[Forward]:
        future = await self.client.forward_backward_async(
            [_unmasked(one) for one in datums],
            loss_fn=loss_fn,
            loss_fn_config=dict(loss_config or {}) or None,
        )
        return _forward(future)

    async def optim_step(self, adam: Adam, rate: float) -> Awaitable[None]:
        params = tinker.AdamParams(
            learning_rate=rate,
            beta1=adam.beta1,
            beta2=adam.beta2,
            eps=adam.eps,
            weight_decay=adam.weight_decay,
            grad_clip_norm=adam.grad_clip_norm,
        )
        return _done(await self.client.optim_step_async(params))

    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """One `cross_entropy` forward; target `p` reads position `p - 1`, 0.0 where unscored."""
        if not sequences:
            return []
        data: list[tinker.Datum] = []
        for one in sequences:
            read, target_tokens = shifted(one.tokens, one.images)
            weights = TensorData.from_torch(torch.ones(read.length))
            inputs = {"target_tokens": target_tokens, "weights": weights}
            data.append(tinker.Datum(model_input=read, loss_fn_inputs=inputs))
        future = await self.client.forward_async(data, loss_fn="cross_entropy")
        result = await future.result_async()
        out: list[list[float]] = []
        for one, row in zip(sequences, result.loss_fn_outputs, strict=True):
            scored = _floats(row["logprobs"])
            out.append([scored[at - 1] if at > 0 else 0.0 for at in one.targets])
        return out

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str:
        future = await self.client.save_weights_for_sampler_async(name, ttl_seconds=ttl_seconds)
        return str((await future.result_async()).path)

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str:
        future = await self.client.save_state_async(name, ttl_seconds=ttl_seconds)
        return str((await future.result_async()).path)

    async def load_state(self, path: str) -> None:
        """The weights and the optimizer state at `path`; the client is then `restored`."""
        future = await self.client.load_state_with_optimizer_async(path)
        await future.result_async()
        self.restored = True

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Finish the session; idempotent, and quiet when the server already finished it."""
        service, self.service = self.service, None
        if service is None:
            return
        try:
            await service.close(status, detail)
        except Exception:  # noqa: BLE001 - a session that will not finish is not a run failure
            logger.warning("could not finish the Tinker session", exc_info=True)


async def _forward(future: Any) -> Forward:
    result = await future.result_async()
    logprobs = [_floats(output["logprobs"]) for output in result.loss_fn_outputs]
    return Forward(logprobs, dict(result.metrics or {}))


async def _done(future: Any) -> None:
    await future.result_async()


def _floats(scored: Any) -> list[float]:
    found = scored.tolist() if hasattr(scored, "tolist") else list(scored)
    return [0.0 if one is None else float(one) for one in found]


def _unmasked(datum: Datum) -> tinker.Datum:
    """The datum without `mask`, which only this side uses."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
    )
