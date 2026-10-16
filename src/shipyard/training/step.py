"""The mechanics of one training step: the warmed-up learning rate, the split into substeps,
awaiting several losses, and the metrics read from the training pass."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Sequence as Values
from typing import Any

import torch

from shipyard.types import Forward

logger = logging.getLogger(__name__)


def warmed(learning_rate: float, warmup: int, done: int) -> float:
    """Return the learning rate after `done` updates, warmed up linearly over `warmup` updates."""
    if done >= warmup:
        return learning_rate
    return learning_rate * (done + 1) / warmup


def split(items: list[Any], parts: int) -> list[list[Any]]:
    """Split `items` into `parts` slices whose sizes differ by at most one.

    The slices are cut as the cookbook's `split_list` cuts them."""
    step = len(items) / parts
    edges = [int(index * step) for index in range(parts)] + [len(items)]
    return [items[start:end] for start, end in zip(edges[:-1], edges[1:], strict=True)]


async def together(*pending: Awaitable[Forward]) -> Forward:
    """Await several submitted forward-backward passes and return the last one's result.

    For a `forward_backward` that sends more than one loss: Tinker adds their gradients up
    before the optimizer step, and the last result's logprobs feed the step's metrics."""
    found = [await one for one in pending]
    return found[-1]


def observed(datums: Values[Any], trained: Values[Any]) -> dict[str, float]:
    """Return kl_v1, kl_v2 and entropy over the target tokens, computed as the cookbook does.

    Return {} if the logprobs cannot be read."""
    if len(trained) != len(datums):
        return {}
    try:
        drifts: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        for one, scored in zip(datums, trained, strict=True):
            inputs = one.loss_fn_inputs
            if "logprobs" not in inputs or "mask" not in inputs or scored is None:
                return {}
            reference = inputs["logprobs"].to_torch()
            acted = inputs["mask"].to_torch() > 0
            if not bool(acted.any()):
                continue
            drifts.append(reference[acted] - scored[acted])
            chosen.append(reference[acted])
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
