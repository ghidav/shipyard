"""Tinker's service and training clients, reduced to what a run needs."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Mapping
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any

import tinker
import torch
from tinker import TensorData

from shipyard.pack import shifted
from shipyard.types import SERVE_TTL, Adam, Datum, Forward, Sequence, TrainingClient

if TYPE_CHECKING:
    from shipyard.config import Model
    from shipyard.run import Run

logger = logging.getLogger(__name__)

#: Tinker's default rank, used when the blueprint names none.
DEFAULT_LORA_RANK = 32


def service_client(metadata: dict[str, str] | None = None) -> Any:
    """A `tinker.ServiceClient`, which reads `TINKER_API_KEY` and `TINKER_BASE_URL` from the
    environment. `metadata` becomes the session's `user_metadata`, naming the run on
    Tinker's side."""
    # The transport is the SDK's own. tinker 0.32 takes no `http_client`: `ServiceClient`
    # warns and drops unknown kwargs (lib/public_interfaces/service_client.py), and
    # `InternalClientHolder` builds every `AsyncTinker` from its own kwargs on its own
    # thread (lib/internal_client_holder.py). pyqwest is switched off only by the server's
    # `ClientConfigResponse.use_pyqwest_transport`.
    return tinker.ServiceClient(user_metadata=metadata)


class TinkerClient(TrainingClient):
    """Over a service's SDK training client, made on `open` from the model config."""

    def __init__(self, service: Any, model: Model, metadata: dict[str, str] | None = None) -> None:
        self.service = service
        self.model = model
        self.metadata = metadata
        self.client: Any = None

    @classmethod
    def from_run(cls, run: Run) -> TinkerClient:
        """On the run's model, with the session named after the run."""
        cfg = run.config
        metadata = {"shipyard_run": run.id, "shipyard_recipe": cfg.recipe.kind}
        return cls(service_client(metadata), cfg.model, metadata)

    async def open(self) -> None:
        """A LoRA on the model, or the weights at its checkpoint; the server is checked."""
        model, metadata, service = self.model, self.metadata, self.service
        try:
            await _server_has(service, model.name)
            if model.from_checkpoint:
                resume = (
                    service.create_training_client_from_state_with_optimizer_async
                    if model.restore_optimizer
                    else service.create_training_client_from_state_async
                )
                self.client = await resume(model.from_checkpoint, user_metadata=metadata)
            else:
                rank = DEFAULT_LORA_RANK if model.lora_rank is None else int(model.lora_rank)
                self.client = await service.create_lora_training_client_async(
                    base_model=model.name, rank=rank, user_metadata=metadata
                )
        except BaseException as failed:
            self.service = None
            await _quietly_closed(service, failed)
            raise

    async def anchor(self) -> Any:
        """A sampling client on the weights as opened, published first when they are a state."""
        model = self.model
        start = (
            await self.save_weights_for_sampler("anchor", ttl_seconds=SERVE_TTL)
            if model.from_checkpoint
            else None
        )
        return await self.service.create_sampling_client_async(
            base_model=model.name, model_path=start
        )

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
        """The weights and the optimizer state at `path`."""
        future = await self.client.load_state_with_optimizer_async(path)
        await future.result_async()

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
    return Forward(logprobs)


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


async def _server_has(service: Any, base_model: str) -> None:
    """Raise `ValueError` for a model the backend does not list or cannot train, before a
    session spends. Returns silently when the capabilities cannot be read."""
    try:
        caps = await service.get_server_capabilities_async()
        models = list(caps.supported_models or [])
    except Exception:  # noqa: BLE001 - a courtesy read, never worth failing a run over
        logger.debug("could not read the server capabilities", exc_info=True)
        return
    if not models:
        return
    found = next((one for one in models if one.model_name == base_model), None)
    if found is None:
        names = sorted(str(one.model_name) for one in models)
        raise ValueError(f"the backend does not serve {base_model!r}; it serves {', '.join(names)}")
    if found.trainable is False:
        raise ValueError(
            f"{base_model!r} can be sampled but not trained; name the trainable variant "
            "under [model] name"
        )


async def _quietly_closed(service: Any, why: BaseException) -> None:
    """Finish a session that never became a trainer. A failure to close is logged, not
    raised, so it cannot hide the caller's own failure."""
    try:
        await service.close("errored", f"{type(why).__name__}: {why}")
    except Exception:  # noqa: BLE001 - the failure being reported is the caller's
        logger.warning("could not finish the Tinker session after a failed open", exc_info=True)
