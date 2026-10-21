from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass
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
#: What names weights saved for sampling in a `tinker://` path: a run samples and scores them,
#: and trains only from a training state (`tinker://.../weights/...`).
SAMPLER_WEIGHTS = "/sampler_weights/"
#: Most scoring requests sent to one sampling client at once.
LOGPROB_CONCURRENCY = 16


@dataclass(frozen=True)
class Loss:
    """One of Tinker's losses: the inputs it reads beside `target_tokens`, and its settings."""

    reads: tuple[str, ...]
    settings: frozenset[str] = frozenset()


#: The ratio against the sampler's logprobs, and each target's advantage.
_RATIO = ("logprobs", "advantages")
_CLIPPED = frozenset({"clip_low_threshold", "clip_high_threshold"})
#: Tinker's losses, which Together trains with too. `cross_entropy` weighs each target token by
#: its advantage; the clip thresholds bound the ratio itself, as 0.8 and 1.2 do.
LOSSES: dict[str, Loss] = {
    "importance_sampling": Loss(_RATIO),
    "ppo": Loss(_RATIO, _CLIPPED),
    "cispo": Loss(_RATIO, _CLIPPED),
    "dro": Loss(_RATIO, frozenset({"beta"})),
    "cross_entropy": Loss(("weights",)),
}


def check_loss(loss: str, settings: Mapping[str, float] | None) -> None:
    """Raise `ValueError` unless `loss` is one of Tinker's and takes each of `settings`."""
    known = LOSSES.get(loss)
    if known is None:
        raise ValueError(f"no loss {loss!r}: the losses are {', '.join(LOSSES)}")
    unknown = sorted(set(settings or {}) - known.settings)
    if unknown:
        takes = ", ".join(sorted(known.settings)) or "no settings"
        raise ValueError(f"the {loss} loss takes {takes}; got {', '.join(unknown)}")


def starts_from(checkpoint: str | None, *, trains: bool) -> None:
    """Raise `ValueError` unless `[agent] from_checkpoint` fits the run: a training state for a
    run that trains its agent, weights saved for sampling for one that only serves it."""
    if checkpoint is None:
        return
    sampled = SAMPLER_WEIGHTS in checkpoint
    if trains and sampled:
        raise ValueError(
            f"[agent] from_checkpoint: {checkpoint} holds weights saved for sampling, and "
            "training goes on from a training state: name the checkpoint's `weights` path"
        )
    if not trains and not sampled:
        raise ValueError(
            f"[agent] from_checkpoint: a run that does not train its agent serves it from "
            f"weights saved for sampling, and {checkpoint} is none: name a kept checkpoint's "
            "`sampler` path"
        )


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
    """The agent's Tinker session: its training client, created on `open` when the run trains
    the agent, or else a sampling client on its weights, which scores it."""

    #: The backend it reaches, as a job's row names it.
    kind = TINKER
    #: Whether what it saves lives only the TTL it is given.
    expires = True

    def __init__(
        self,
        model: str,
        *,
        run: str | None = None,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        trains: bool = True,
        service: Any = None,
    ) -> None:
        """Raise `ValueError` for a `from_checkpoint` the run cannot start from."""
        starts_from(from_checkpoint, trains=trains)
        self.model = model
        #: Who its tokens are billed to: `tinker`, or `tinker@<host>`.
        self.provider = backend()
        self.from_checkpoint = from_checkpoint
        self.lora_rank = lora_rank
        self.restore_optimizer = restore_optimizer
        #: Whether the run trains the model, so `open` creates a training client.
        self.trains = trains
        self.metadata = tagged(run) if run else None
        self.service = service if service is not None else service_client(self.metadata)
        self.client: Any = None
        #: What scores the model when no training client holds it: a sampling client on its
        #: weights, `from_checkpoint`'s or the base model's.
        self.scorer: Any = None

    async def open(self) -> None:
        """Create the training client: a new LoRA on the model, or the checkpoint's weights.
        A run that does not train the model opens a sampling client on its weights instead."""
        metadata, service = self.metadata, self.service
        try:
            await _server_has(service, self.model, trains=self.trains)
            if not self.trains:
                self.scorer = await _sampler(service, self.from_checkpoint or self.model)
                return
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

    async def fit(
        self,
        parts: Values[Values[tinker.Datum]],
        adam: tinker.AdamParams,
        rate: float,
        loss_fn: str,
        loss_config: Mapping[str, float] | None,
    ) -> list[list[float]]:
        """Run a training pass and an optimizer step on each part in turn, at learning rate
        `rate`, and return each datum's logprobs; each part is submitted before the one before
        it is read, so Tinker runs them back to back."""
        params = adam.model_copy(update={"learning_rate": rate})
        config = dict(loss_config or {}) or None

        async def submit(part: Values[tinker.Datum]) -> tuple[Any, Any]:
            forward = await self.client.forward_backward_async(
                [sent(one, loss_fn) for one in part], loss_fn=loss_fn, loss_fn_config=config
            )
            return forward, await self.client.optim_step_async(params)

        trained: list[list[float]] = []
        pending = await submit(parts[0])
        for index in range(len(parts)):
            following = await submit(parts[index + 1]) if index + 1 < len(parts) else None
            forward, optim = pending
            trained.extend(await _forward(forward))
            await optim.result_async()
            if following is not None:
                pending = following
        return trained

    async def compute_logprobs(
        self, sequences: Values[Sequence]
    ) -> tuple[list[list[float]], dict[str, int]]:
        """Return each sequence's logprobs at its targets, and the tokens that took by kind:
        one `cross_entropy` pass of the training client, or the scorer's when there is none."""
        if self.scorer is not None:
            return await sampled(self.scorer, sequences)
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
        return out, {"train": sum(int(one.model_input.length) for one in data)}

    async def serving(self) -> dict[str, str]:
        """Return what the proxy samples through beyond its default: nothing, as it is Tinker."""
        return {}

    async def publish(self, tag: str) -> str | None:
        """Save the weights for sampling as `tag`, kept `SERVE_TTL`, and return their path. With
        no training client, return the weights served as they are: `from_checkpoint`'s, or None
        for the base model."""
        if self.client is None:
            return self.from_checkpoint
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
        if service is not None:
            await finished(service, status, detail)


async def sampling(load: str, *, run: str | None = None) -> tuple[Any, Any, str, str]:
    """Open a Tinker session of its own for `load`, a model's name or a `tinker://` path, and
    return the session, a sampling client on it, the base model its tokens are priced as, and
    who they are billed to. Raise `ValueError` for a model the backend does not serve; a failed
    open finishes the session as errored."""
    service = service_client(tagged(run) if run else None)
    try:
        if load.startswith("tinker://"):
            sampler = await _sampler(service, load)
            model = str(await sampler.get_base_model_async())
        else:
            await _server_has(service, load, trains=False)
            sampler, model = await _sampler(service, load), load
    except BaseException as failed:
        await _quietly_closed(service, failed)
        raise
    return service, sampler, model, backend()


async def finished(service: Any, status: str = "success", detail: str | None = None) -> None:
    """Finish a Tinker session with `status`: one that will not finish is not a run failure."""
    try:
        await service.close(status, detail)
    except Exception:  # noqa: BLE001 - a session that will not finish is not a run failure
        logger.warning("could not finish the Tinker session", exc_info=True)


async def sampled(
    sampler: Any, sequences: Values[Sequence]
) -> tuple[list[list[float]], dict[str, int]]:
    """Return a sampling client's logprobs at each sequence's targets, and the tokens it took."""
    gate = asyncio.Semaphore(LOGPROB_CONCURRENCY)
    took = {"prefill": 0, "cache": 0, "sample": 0}

    async def one(sequence: Sequence) -> list[float]:
        prompt = model_input(sequence.tokens, sequence.images)
        async with gate:
            reply = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
            )
        cache = min(int(reply.prompt_cache_hit_tokens or 0), prompt.length)
        took["prefill"] += prompt.length - cache
        took["cache"] += cache
        took["sample"] += sum(len(drawn.tokens) for drawn in reply.sequences)
        scored = list(reply.prompt_logprobs or [])
        return [float(scored[at] or 0.0) for at in sequence.targets]

    found = list(await asyncio.gather(*(one(sequence) for sequence in sequences)))
    return found, took


async def _sampler(service: Any, load: str) -> Any:
    """Return a sampling client in `service`'s session on a base model or a `tinker://` path."""
    if load.startswith("tinker://"):
        return await service.create_sampling_client_async(model_path=load)
    return await service.create_sampling_client_async(base_model=load)


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


def _floats(scored: Any) -> list[float]:
    found = scored.tolist() if hasattr(scored, "tolist") else list(scored)
    return [0.0 if one is None else float(one) for one in found]


def sent(datum: tinker.Datum, loss: str) -> tinker.Datum:
    """Return the datum as `loss` reads it: its targets and the loss's own inputs, each target's
    weight its advantage. The rest, such as `mask`, is used here and not sent."""
    held = {**datum.loss_fn_inputs, "weights": datum.loss_fn_inputs["advantages"]}
    inputs = {key: held[key] for key in ("target_tokens", *LOSSES[loss].reads)}
    return tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs)


async def _server_has(service: Any, base_model: str, *, trains: bool = True) -> None:
    """Raise `ValueError` if the backend does not list `base_model`, or, when the run `trains`
    it, cannot train it."""
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
    if trains and found.trainable is False:
        raise ValueError(
            f"{base_model!r} can be sampled but not trained; name the trainable variant "
            "under [agent] model"
        )


async def _quietly_closed(service: Any, why: BaseException) -> None:
    """Close a session as errored after what it was opened for failed to open."""
    try:
        await service.close("errored", f"{type(why).__name__}: {why}")
    except Exception:  # noqa: BLE001 - the failure being reported is the caller's
        logger.warning("could not finish the Tinker session after a failed open", exc_info=True)
