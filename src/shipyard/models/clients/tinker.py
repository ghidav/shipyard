from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Mapping
from collections.abc import Sequence as Values
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import tinker
import torch
from tinker import TensorData
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)

from harness_proxy.wire import Picture, model_input

if TYPE_CHECKING:
    from shipyard.groups import Sequence

logger = logging.getLogger(__name__)

#: The provider of a model Tinker serves.
TINKER = "tinker"
#: Seconds served sampler weights live: long enough to outlast the batch sampled from them,
#: short enough that a crashed run's leftovers expire on their own.
SERVE_TTL = 12 * 3600
#: LoRA rank used when `[recipe] lora_rank` sets none (Tinker's default).
DEFAULT_LORA_RANK = 32


def backend() -> str:
    """Return the provider Tinker's tokens are billed to: `tinker`, or `tinker@<host>`."""
    host = urlsplit(os.environ.get("TINKER_BASE_URL") or "").hostname
    return f"{TINKER}@{host}" if host else TINKER


def tagged(run: str) -> dict[str, str]:
    """Return the tags of the run's Tinker sessions, so Tinker shows which run each is."""
    return {"shipyard_run": run}


def service_client(metadata: dict[str, str] | None = None) -> Any:
    """Return a `tinker.ServiceClient` configured from the environment, tagged `metadata`."""
    return tinker.ServiceClient(user_metadata=metadata)


class TinkerClient:
    """The agent's Tinker training client, created on `open`."""

    def __init__(
        self,
        model: str,
        *,
        run: str | None = None,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        anchor_ttl: int = SERVE_TTL,
        service: Any = None,
    ) -> None:
        self.model = model
        #: Seconds the saved `INITIAL` weights of a run started from a checkpoint are kept.
        self.anchor_ttl = anchor_ttl
        self.provider = backend()
        self.from_checkpoint = from_checkpoint
        self.lora_rank = lora_rank
        self.restore_optimizer = restore_optimizer
        self.metadata = tagged(run) if run else None
        self.service = service if service is not None else service_client(self.metadata)
        self.client: Any = None

    async def open(self) -> None:
        """Create the training client: a new LoRA on the model, or the checkpoint's weights."""
        metadata, service = self.metadata, self.service
        try:
            await _server_has(service, self.model)
            if self.from_checkpoint:
                resume = (
                    service.create_training_client_from_state_with_optimizer_async
                    if self.restore_optimizer
                    else service.create_training_client_from_state_async
                )
                self.client = await resume(self.from_checkpoint, user_metadata=metadata)
            else:
                rank = DEFAULT_LORA_RANK if self.lora_rank is None else int(self.lora_rank)
                self.client = await service.create_lora_training_client_async(
                    base_model=self.model, rank=rank, user_metadata=metadata
                )
        except BaseException as failed:
            self.service = None
            await _quietly_closed(service, failed)
            raise

    async def anchor(self) -> Any:
        """Return a sampling client on the weights as opened: the run's `INITIAL` weights."""
        start = (
            await self.save_weights_for_sampler("anchor", ttl_seconds=self.anchor_ttl)
            if self.from_checkpoint
            else None
        )
        return await self.service.create_sampling_client_async(
            base_model=self.model, model_path=start
        )

    async def sampler(self, model: str) -> Any:
        """Return a sampling client in this session on a base model or a `tinker://` path."""
        if model.startswith("tinker://"):
            return await self.service.create_sampling_client_async(model_path=model)
        return await self.service.create_sampling_client_async(base_model=model)

    async def forward_backward(
        self, datums: Values[tinker.Datum], loss_fn: str, loss_config: Mapping[str, float] | None
    ) -> Awaitable[list[list[float]]]:
        """Submit one training pass, and return what awaits its logprobs."""
        future = await self.client.forward_backward_async(
            [_unmasked(one) for one in datums],
            loss_fn=loss_fn,
            loss_fn_config=dict(loss_config or {}) or None,
        )
        return _forward(future)

    async def optim_step(self, adam: tinker.AdamParams, rate: float) -> Awaitable[None]:
        """Submit an optimizer step with `adam`'s settings at learning rate `rate`."""
        params = adam.model_copy(update={"learning_rate": rate})
        return _done(await self.client.optim_step_async(params))

    async def compute_logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
        """Return each sequence's logprobs at its targets, from one `cross_entropy` pass."""
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

    async def serving(self) -> dict[str, str]:
        """Return what the proxy samples through beyond its default: nothing, as it is Tinker."""
        return {}

    async def publish(self, tag: str) -> str:
        """Save the weights for sampling as `tag`, kept `SERVE_TTL`, and return their path."""
        return await self.save_weights_for_sampler(tag, ttl_seconds=SERVE_TTL)

    async def save_weights_for_sampler(self, name: str, *, ttl_seconds: int | None) -> str:
        future = await self.client.save_weights_for_sampler_async(name, ttl_seconds=ttl_seconds)
        return str((await future.result_async()).path)

    async def save_state(self, name: str, *, ttl_seconds: int | None) -> str:
        future = await self.client.save_state_async(name, ttl_seconds=ttl_seconds)
        return str((await future.result_async()).path)

    async def load_state(self, path: str) -> None:
        """Load the weights and the optimizer state saved at `path`."""
        future = await self.client.load_state_with_optimizer_async(path)
        await future.result_async()

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Finish the Tinker session with `status`; later calls do nothing."""
        service, self.service = self.service, None
        if service is None:
            return
        try:
            await service.close(status, detail)
        except Exception:  # noqa: BLE001 - a session that will not finish is not a run failure
            logger.warning("could not finish the Tinker session", exc_info=True)


def shifted(
    tokens: Values[int], images: Values[Picture] = ()
) -> tuple[tinker.ModelInput, TensorData]:
    """Shift a sequence by one into the model's input and its targets, as the cookbook does."""
    read, targets = create_rightshifted_model_input_and_leftshifted_targets(
        list(model_input(tokens, images).chunks)
    )
    return read, TensorData.from_torch(torch.tensor(targets))


def datum(
    sequence: Sequence, *, logprobs: Values[float], advantages: Values[float]
) -> tinker.Datum:
    """Build a Tinker datum from a sequence, with an advantage per target token; reproduced from
    tinker-cookbook's rl/data_processing.py (Apache-2.0)."""
    read, target_tokens = shifted(sequence.tokens, sequence.images)
    positions = sequence.targets
    if len(logprobs) != len(positions) or len(advantages) != len(positions):
        raise ValueError(
            f"{len(positions)} target(s) take {len(positions)} logprobs and advantages; got "
            f"{len(logprobs)} and {len(advantages)}"
        )
    width = read.length
    mask, logprob_at, advantage_at = [0.0] * width, [0.0] * width, [0.0] * width
    for at, position in enumerate(positions):
        if position == 0:
            continue  # the first token is read but not predicted; the cookbook drops it too
        mask[position - 1] = 1.0
        logprob_at[position - 1] = float(logprobs[at])
        advantage_at[position - 1] = float(advantages[at])
    return tinker.Datum(
        model_input=read,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": TensorData.from_torch(torch.tensor(logprob_at)),
            "advantages": TensorData.from_torch(torch.tensor(advantage_at)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        },
    )


async def _forward(future: Any) -> list[list[float]]:
    result = await future.result_async()
    return [_floats(output["logprobs"]) for output in result.loss_fn_outputs]


async def _done(future: Any) -> None:
    await future.result_async()


def _floats(scored: Any) -> list[float]:
    found = scored.tolist() if hasattr(scored, "tolist") else list(scored)
    return [0.0 if one is None else float(one) for one in found]


def _unmasked(datum: tinker.Datum) -> tinker.Datum:
    """Return the datum without `mask`, which is used locally and not sent to Tinker."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
    )


async def _server_has(service: Any, base_model: str) -> None:
    """Raise `ValueError` if the backend does not list `base_model` or cannot train it."""
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
            "under [agent] model"
        )


async def _quietly_closed(service: Any, why: BaseException) -> None:
    """Close a session as errored after its training client failed to open."""
    try:
        await service.close("errored", f"{type(why).__name__}: {why}")
    except Exception:  # noqa: BLE001 - the failure being reported is the caller's
        logger.warning("could not finish the Tinker session after a failed open", exc_info=True)
