"""The training client over the Tinker SDK: opened on a base model or a saved state, the
weights published for the proxy, a checkpoint saved, the step taken, the session closed."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence as Values
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker
import torch

from shipyard.config import MIN_TTL_HOURS
from shipyard.session import quietly_closed, server_has

if TYPE_CHECKING:
    from shipyard.credit import Batch
    from shipyard.recipes.train import Preset

logger = logging.getLogger(__name__)

#: Tinker's own default rank, for a blueprint that names none.
DEFAULT_LORA_RANK = 32
#: How long a step's published sampler weights are kept: they outlive the batch sampled
#: from them, and a crashed run's leftovers expire on their own.
SERVE_TTL = 12 * 3600
SECONDS_PER_HOUR = 3600
#: Adam as the cookbook's `train_step` sets it; the SDK's own `eps` default is 1e-12.
BETA1, BETA2, EPS = 0.9, 0.95, 1e-8


@dataclass(frozen=True)
class Update:
    """What one `apply` did: the tokens through `forward_backward`, the substeps run, the
    knobs, and what the step's own logprobs said against mu (None where unreadable)."""

    train_tokens: int
    substeps: int
    learning_rate: float
    loss_fn: str
    kl_v1: float | None
    kl_v2: float | None
    entropy: float | None
    seconds: float


@dataclass(frozen=True)
class Checkpoint:
    """A durable save: the state path a training client is rebuilt from, the sampler path
    rollouts are served from, and the TTL in hours, None being kept until deleted."""

    tag: str
    state_path: str
    sampler_path: str
    ttl_hours: float | None


@dataclass
class Trainer:
    """A Tinker training client and what a run asks of it. `updates` counts the gradients
    applied: how credit tells whether the weights a batch was sampled at have moved."""

    client: Any
    updates: int = 0
    service: Any = None

    @classmethod
    async def create(
        cls,
        service: Any,
        base_model: str,
        from_checkpoint: str | None = None,
        lora_rank: int | None = None,
        restore_optimizer: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> Trainer:
        """A LoRA on `base_model`, or the weights at `from_checkpoint` (the SDK's `from_state`
        loads no optimizer state; `restore_optimizer` asks for the one that does), tagged
        `metadata` on Tinker's side; the capabilities are read before a session spends."""
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
        return cls(client, service=service)

    async def publish(self, name: str, *, ttl_seconds: int = SERVE_TTL) -> str:
        """The current weights at a `tinker://` path a proxy anywhere resolves, kept for
        `ttl_seconds`: `save_weights_for_sampler_async`, whose own default is forever."""
        future = await self.client.save_weights_for_sampler_async(
            name, ttl_seconds=int(ttl_seconds)
        )
        saved = await future.result_async()
        return str(saved.path)

    async def save(self, tag: str, *, ttl_hours: float | None = None) -> Checkpoint:
        """State and sampler weights under `tag` with one TTL; None is the SDK's keep-forever."""
        if ttl_hours is not None and ttl_hours < MIN_TTL_HOURS:
            raise ValueError(
                f"ttl_hours = {ttl_hours}: Tinker keeps a checkpoint for an hour at least"
            )
        ttl_seconds = None if ttl_hours is None else int(ttl_hours * SECONDS_PER_HOUR)
        state_future = await self.client.save_state_async(tag, ttl_seconds=ttl_seconds)
        state = await state_future.result_async()
        weights_future = await self.client.save_weights_for_sampler_async(
            tag, ttl_seconds=ttl_seconds
        )
        weights = await weights_future.result_async()
        return Checkpoint(tag, str(state.path), str(weights.path), ttl_hours)

    async def apply(self, batch: Batch, preset: Preset) -> Update:
        """The gradient over the batch's datums in `substeps` parts, refused empty: a step
        that trained on nothing must not read like one that trained."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0): each substep's forward_backward and optim_step are enqueued before
        # the substep before it is consumed; the per-datum training logprobs are read off
        # `loss_fn_outputs[i]["logprobs"]` as its `_training_logprobs_from_fwd_bwd` does,
        # and `mask` is stripped from what is sent as its `_remove_mask` does.
        began = time.monotonic()
        datums = list(batch.datums)
        if not datums:
            raise ValueError("the batch holds no datums; the loop logs such a step untrained")
        train_tokens = sum(int(one.model_input.length) for one in datums)
        parts = min(preset.substeps, len(datums))
        if parts > 1:
            # A batch arrives task by task; shuffled, every substep draws on every task,
            # and the same batch splits the same way in any process.
            random.Random(f"{len(datums)}:{train_tokens}").shuffle(datums)
        adam = tinker.AdamParams(
            learning_rate=preset.learning_rate, beta1=BETA1, beta2=BETA2, eps=EPS
        )
        # Before the call and never rolled back: a substep that fails after the first
        # leaves optimizer steps landed, and this count is what guards credit's mu.
        self.updates += 1
        trained: list[torch.Tensor] = []
        split_parts = split(datums, parts)
        forward, optim = await self._enqueue(split_parts[0], preset, adam)
        for index in range(parts):
            following = (
                await self._enqueue(split_parts[index + 1], preset, adam)
                if index + 1 < parts
                else None
            )
            result = await forward.result_async()
            trained.extend(output["logprobs"].to_torch() for output in result.loss_fn_outputs)
            await optim.result_async()
            if following is not None:
                forward, optim = following
        observed = _observed(datums, trained)
        return Update(
            train_tokens=train_tokens,
            substeps=parts,
            learning_rate=preset.learning_rate,
            loss_fn=preset.loss_fn,
            kl_v1=observed.get("kl_v1"),
            kl_v2=observed.get("kl_v2"),
            entropy=observed.get("entropy"),
            seconds=round(time.monotonic() - began, 3),
        )

    async def _enqueue(self, part: list[Any], preset: Preset, adam: Any) -> tuple[Any, Any]:
        """One substep submitted: its forward_backward, then its optim_step."""
        forward = await self.client.forward_backward_async(
            [_unmasked(one) for one in part],
            loss_fn=preset.loss_fn,
            loss_fn_config=dict(preset.loss_config) or None,
        )
        return forward, await self.client.optim_step_async(adam)

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Finish the session, whose heartbeat otherwise runs as long as the process does;
        idempotent, and quiet about a session the server has finished already."""
        service, self.service = self.service, None
        if service is None:
            return
        try:
            await service.close(status, detail)
        except Exception:  # noqa: BLE001 - a session that will not finish is not a run failure
            logger.warning("could not finish the Tinker session", exc_info=True)


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """`parts` slices in order, their sizes differing by at most one: the cut of the
    cookbook's `split_list` (`np.linspace(0, len, parts + 1).astype(int)`), its float
    arithmetic kept so a batch splits as it did under the cookbook."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


def _unmasked(datum: Any) -> tinker.Datum:
    """The datum without `mask`, which is this side's bookkeeping and no loss input."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
    )


def _observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """kl_v1 = mean(mu - pi), kl_v2 = half the mean square, entropy = mean(-mu) over the
    acted tokens (the cookbook's `compute_kl_sample_train`); {} wherever a datum carries
    no mu or mask or the logprobs do not line up, never a failure of the step."""
    if len(trained) != len(datums):
        return {}
    try:
        drifts: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        for datum, scored in zip(datums, trained, strict=True):
            inputs = datum.loss_fn_inputs
            if "logprobs" not in inputs or "mask" not in inputs or scored is None:
                return {}
            mu = inputs["logprobs"].to_torch()
            acted = inputs["mask"].to_torch() > 0
            if not bool(acted.any()):
                continue
            drifts.append(mu[acted] - scored[acted])
            chosen.append(mu[acted])
        if not drifts:
            return {}
        moved, sampled = torch.cat(drifts), torch.cat(chosen)
        return {
            "kl_v1": float(moved.mean()),
            "kl_v2": float(0.5 * (moved**2).mean()),
            "entropy": float(-sampled.mean()),
        }
    except Exception:  # noqa: BLE001 - a number beside the step, not the step
        logger.warning("could not read the step's own logprobs; the row goes without them")
        return {}
