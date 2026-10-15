"""The loop's handle on the training side until the trainer takes it, over the `TinkerClient`."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from shipyard.config import MIN_TTL_HOURS
from shipyard.training.client import DEFAULT_LORA_RANK, SERVE_TTL, TinkerClient
from shipyard.types import Adam, Forward, Update

if TYPE_CHECKING:
    from shipyard.credit import Batch
    from shipyard.gradient import Preset

__all__ = [
    "COOKBOOK",
    "DEFAULT_LORA_RANK",
    "SERVE_TTL",
    "Adam",
    "Checkpoint",
    "Trainer",
    "Update",
    "grouped",
    "split",
]

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600

#: Adam as the cookbook's `train_step` sets it (no weight decay, gradient clipping or
#: warm-up), for whatever a paper leaves unstated. The SDK's `eps` default is 1e-12.
COOKBOOK = Adam(beta1=0.9, beta2=0.95, eps=1e-8)


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
    """`updates` counts the gradients applied; `restored` means the optimizer state was loaded."""

    client: Any
    updates: int = 0
    service: Any = None
    restored: bool = False
    tinker: TinkerClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tinker = TinkerClient(self.client, service=self.service, restored=self.restored)

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
        opened = await TinkerClient.open(
            service,
            base_model,
            from_checkpoint=from_checkpoint,
            lora_rank=lora_rank,
            restore_optimizer=restore_optimizer,
            metadata=metadata,
        )
        return cls(opened.client, service=service, restored=opened.restored)

    async def publish(self, name: str, *, ttl_seconds: int = SERVE_TTL) -> str:
        """Save the current weights as sampler weights for `ttl_seconds`; returns their path."""
        return await self.tinker.save_weights_for_sampler(name, ttl_seconds=int(ttl_seconds))

    async def save(self, tag: str, *, ttl_hours: float | None = None) -> Checkpoint:
        """Save state and sampler weights under `tag` with one TTL. None keeps them forever
        (the SDK default)."""
        if ttl_hours is not None and ttl_hours < MIN_TTL_HOURS:
            raise ValueError(
                f"ttl_hours = {ttl_hours}: Tinker keeps a checkpoint for an hour at least"
            )
        ttl_seconds = None if ttl_hours is None else int(ttl_hours * SECONDS_PER_HOUR)
        state = await self.tinker.save_state(tag, ttl_seconds=ttl_seconds)
        weights = await self.tinker.save_weights_for_sampler(tag, ttl_seconds=ttl_seconds)
        return Checkpoint(tag, state, weights, ttl_hours)

    async def apply(self, batch: Batch, preset: Preset) -> Update:
        """Take the gradient over the batch's datums in `substeps` parts of whole groups, with
        at most one part per group. Raise on an empty batch, so a step that trained on
        nothing does not read like one that trained. Every part carries credit's mu, the
        logprobs of the weights the batch was sampled at, so the ratio moves from the second
        part on."""
        # Reproduces `train_step` of tinker_cookbook/rl/train.py (Thinking Machines Lab,
        # Apache-2.0). Each substep's forward_backward and optim_step are submitted before
        # the previous substep is read.
        began = time.monotonic()
        if not batch.datums:
            raise ValueError("the batch holds no datums; the loop logs such a step untrained")
        groups = grouped(batch.datums, batch.owners)
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        parts = min(preset.substeps, len(groups))
        split_parts = [[one for group in part for one in group] for part in split(groups, parts)]
        datums = [one for part in split_parts for one in part]
        train_tokens = sum(int(one.model_input.length) for one in datums)
        # The optimizer state loaded with the weights continues the earlier run's, so the
        # rate does not warm up again.
        rate = (
            preset.learning_rate
            if self.restored
            else preset.adam.rate(preset.learning_rate, self.updates)
        )
        # Incremented before the call and not rolled back. A substep that fails after the
        # first leaves optimizer steps landed, and credit's mu guard reads this count.
        self.updates += 1
        trained: list[torch.Tensor] = []
        forward, optim = await self._submit(split_parts[0], preset, rate)
        for index in range(parts):
            following = (
                await self._submit(split_parts[index + 1], preset, rate)
                if index + 1 < parts
                else None
            )
            result = await forward
            trained.extend(torch.tensor(row) for row in result.logprobs)
            await optim
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

    async def _submit(
        self, part: list[Any], preset: Preset, rate: float
    ) -> tuple[Awaitable[Forward], Awaitable[None]]:
        forward = await self.tinker.forward_backward(part, preset.loss_fn, preset.loss_config)
        return forward, await self.tinker.optim_step(preset.adam, rate)

    async def close(self, status: str = "success", detail: str | None = None) -> None:
        await self.tinker.close(status, detail)
        self.service = None


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
