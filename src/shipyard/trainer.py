"""The training client over the Tinker SDK. It opens on a base model or a saved state,
publishes weights for the proxy, saves checkpoints, takes a step, and closes the session."""

from __future__ import annotations

import logging
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

#: Tinker's default rank, used when the blueprint names none.
DEFAULT_LORA_RANK = 32
#: How long a step's published sampler weights are kept. They outlive the batch sampled
#: from them, and a crashed run's leftovers expire on their own.
SERVE_TTL = 12 * 3600
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class Adam:
    """AdamW as a recipe's paper sets it: the betas, eps, the decoupled weight decay, the
    global gradient norm each step is clipped to (Tinker's `grad_clip_norm`; 0 is none), and
    the `warmup` steps over which the learning rate rises linearly to its value (0 is none)."""

    beta1: float
    beta2: float
    eps: float
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    warmup: int = 0

    def rate(self, learning_rate: float, done: int) -> float:
        """The learning rate for the update after `done` updates: `(done + 1) / warmup` of
        `learning_rate` during the warm-up, so the first update moves the weights, and all
        of `learning_rate` afterwards."""
        if done >= self.warmup:
            return learning_rate
        return learning_rate * (done + 1) / self.warmup


#: Adam as the cookbook's `train_step` sets it (no weight decay, gradient clipping or
#: warm-up), for whatever a paper leaves unstated. The SDK's `eps` default is 1e-12.
COOKBOOK = Adam(beta1=0.9, beta2=0.95, eps=1e-8)


@dataclass(frozen=True)
class Update:
    """What one `apply` did: the tokens sent through `forward_backward`, the substeps run,
    the learning rate and loss function, and the KL and entropy of the step's own logprobs against
    mu (None where unreadable)."""

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
    rollouts are served from, and the TTL in hours (None: kept until deleted)."""

    tag: str
    state_path: str
    sampler_path: str
    ttl_hours: float | None


@dataclass
class Trainer:
    """A Tinker training client. `updates` counts the gradients applied. Credit reads it to
    tell whether the weights a batch was sampled at have moved, and `Adam.rate` reads it to
    tell how far the run's warm-up has gone. `restored` is set when the optimizer state was
    loaded with the weights. The run then continues an earlier run's optimizer state and has
    no warm-up."""

    client: Any
    updates: int = 0
    service: Any = None
    restored: bool = False

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
        """Open a LoRA on `base_model`, or the weights at `from_checkpoint`, tagged with
        `metadata` on Tinker's side. The SDK's `from_state` loads no optimizer state;
        `restore_optimizer` uses the variant that does, and the trainer is then `restored`.
        The server's capabilities are checked before the session spends anything."""
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

    async def publish(self, name: str, *, ttl_seconds: int = SERVE_TTL) -> str:
        """Save the current weights as sampler weights and return their `tinker://` path,
        which a proxy anywhere can resolve. They are kept for `ttl_seconds`.
        `save_weights_for_sampler_async` keeps them forever by default."""
        future = await self.client.save_weights_for_sampler_async(
            name, ttl_seconds=int(ttl_seconds)
        )
        saved = await future.result_async()
        return str(saved.path)

    async def save(self, tag: str, *, ttl_hours: float | None = None) -> Checkpoint:
        """Save state and sampler weights under `tag` with one TTL. None keeps them forever
        (the SDK default)."""
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
        """Take the gradient over the batch's datums in `substeps` parts of whole groups, with
        at most one part per group. Raise on an empty batch, so a step that trained on
        nothing does not read like one that trained. Every part carries credit's mu, the
        logprobs of the weights the batch was sampled at, so the ratio moves from the second
        part on."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0). Each substep's forward_backward and optim_step are enqueued before the
        # previous substep is consumed. Per-datum training logprobs are read from
        # `loss_fn_outputs[i]["logprobs"]`, as its `_training_logprobs_from_fwd_bwd` does, and
        # `mask` is stripped from what is sent, as its `_remove_mask` does.
        began = time.monotonic()
        if not batch.datums:
            raise ValueError("the batch holds no datums; the loop logs such a step untrained")
        groups = grouped(batch.datums, batch.owners)
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        parts = min(preset.substeps, len(groups))
        split_parts = [[one for group in part for one in group] for part in split(groups, parts)]
        datums = [one for part in split_parts for one in part]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        optimizer = preset.adam
        # The optimizer state loaded with the weights continues the earlier run's, so the
        # rate does not warm up again.
        rate = (
            preset.learning_rate
            if self.restored
            else optimizer.rate(preset.learning_rate, self.updates)
        )
        adam = tinker.AdamParams(
            learning_rate=rate,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            eps=optimizer.eps,
            weight_decay=optimizer.weight_decay,
            grad_clip_norm=optimizer.grad_clip_norm,
        )
        # Incremented before the call and not rolled back. A substep that fails after the
        # first leaves optimizer steps landed, and credit's mu guard reads this count.
        self.updates += 1
        trained: list[torch.Tensor] = []
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
            learning_rate=rate,
            loss_fn=preset.loss_fn,
            kl_v1=observed.get("kl_v1"),
            kl_v2=observed.get("kl_v2"),
            entropy=observed.get("entropy"),
            seconds=round(time.monotonic() - began, 3),
        )

    async def _enqueue(self, part: list[Any], preset: Preset, adam: Any) -> tuple[Any, Any]:
        """Submit one substep: forward_backward, then optim_step."""
        forward = await self.client.forward_backward_async(
            [_unmasked(one) for one in part],
            loss_fn=preset.loss_fn,
            loss_fn_config=dict(preset.loss_config) or None,
        )
        return forward, await self.client.optim_step_async(adam)

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        """Finish the session, whose heartbeat otherwise runs for the life of the process.
        Idempotent, and quiet when the server has already finished the session."""
        service, self.service = self.service, None
        if service is None:
            return
        try:
            await service.close(status, detail)
        except Exception:  # noqa: BLE001 - a session that will not finish is not a run failure
            logger.warning("could not finish the Tinker session", exc_info=True)


def grouped(datums: Values[Any], owners: Values[int]) -> list[list[Any]]:
    """The datums gathered by the group they came from, groups in the batch's order."""
    if len(owners) != len(datums):
        raise ValueError(f"the batch names a group for {len(owners)} of its {len(datums)} datum(s)")
    found: dict[int, list[Any]] = {}
    for owner, one in zip(owners, datums, strict=True):
        found.setdefault(owner, []).append(one)
    return list(found.values())


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """`parts` slices in order, with sizes differing by at most one. This is the cut of the
    cookbook's `split_list` (`np.linspace(0, len, parts + 1).astype(int)`), with its float
    arithmetic kept so a batch splits as it did under the cookbook."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


def _unmasked(datum: Any) -> tinker.Datum:
    """The datum without `mask`, which only this side uses."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
    )


def _observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """kl_v1 = mean(mu - pi), kl_v2 = half the mean square, entropy = mean(-mu) over the
    acted tokens (the cookbook's `compute_kl_sample_train`). Returns {} when a datum carries
    no mu or mask or the logprobs do not line up."""
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
    except Exception:  # noqa: BLE001 - these numbers are extras and must not fail the step
        logger.warning("could not read the step's own logprobs; the row goes without them")
        return {}
