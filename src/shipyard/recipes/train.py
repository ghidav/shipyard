"""The gradient loop the three recipes share, over the Tinker SDK: a Preset each name
resolves to, `step` (publish, point, sample and refill, credit, apply, log, checkpoint by
`every`) and `train`, the session of steps around it."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from shipyard import session
from shipyard.config import LENGTH_CAP, PAPER_OVERLONG_BUFFER
from shipyard.credit import Batch, carrying, credit
from shipyard.data import batches
from shipyard.pack import Group, group
from shipyard.trainer import SERVE_TTL, Adam, Trainer, Update

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shipyard.rollout import Rollouts
    from shipyard.run import Run
    from shipyard.serving import Serving

logger = logging.getLogger(__name__)

#: How a step samples its batch: the tasks and the step's index in, the rollouts out.
Sampler = Callable[["Sequence[Path]", int], Awaitable["Rollouts"]]

__all__ = [
    "ROW",
    "Preset",
    "Refill",
    "clipped",
    "optimizer",
    "refilled",
    "resolution",
    "row",
    "shared",
    "step",
    "train",
    "training",
]

#: The step's row, in this order; a measure with nothing behind it is left out, never zero.
ROW = (
    "step",
    "trained",
    "groups",
    "rollouts",
    "refills",
    "refill_rollouts",
    "graded",
    "masked",
    "degenerate",
    "surplus",
    "sequences",
    "sequences_per_rollout",
    "train_tokens",
    "reward_mean",
    "reward_spread",
    "length_mean",
    "kl_v1",
    "kl_v2",
    "entropy",
    "anchor_kl",
    "overlong",
    "learning_rate",
    "substeps",
    "loss_fn",
    "seconds",
)


@dataclass(frozen=True, kw_only=True)
class Preset:
    """What a recipe name and its knobs resolve to; never user-facing. `loss_config` is
    Tinker's `loss_fn_config` for the step (see `clipped`); `clipping` says it for `check`.
    `aggregation` is "prompt", each prompt's token losses averaged so every prompt weighs
    the same, or "sum", Tinker's own sum over tokens; `refill` caps the extra sampling
    rounds a step may take to fill its batch with groups that carry a gradient. The length
    rule docks solved answers (`credit.shaped`); the overlong term docks a rollout that
    sampled into the last `overlong_buffer` of its token budget (`credit.overlong`).
    `adam` is the optimizer as the recipe's paper sets it."""

    name: str
    normalize: bool
    loss_fn: str
    loss_config: dict[str, float]
    clipping: str
    aggregation: Literal["prompt", "sum"]
    length_penalty: float = 0.0
    length_floor: int = 0
    length_cap: float = LENGTH_CAP
    overlong_penalty: float = 0.0
    overlong_buffer: float = PAPER_OVERLONG_BUFFER
    adam: Adam
    kl_coef: float
    reference: str
    learning_rate: float
    substeps: int
    refill: int


@dataclass(frozen=True)
class Refill:
    """What a step's extra sampling rounds took: how many, and their rollouts."""

    refills: int
    refill_rollouts: int


def shared(recipe: Any) -> dict[str, Any]:
    """The knobs every gradient recipe carries, as `Preset` takes them."""
    return {
        "kl_coef": float(recipe.kl_coef),
        "reference": str(recipe.reference),
        "learning_rate": float(recipe.learning_rate),
        "substeps": int(recipe.substeps),
        "refill": int(recipe.refill),
    }


def clipped(low: float, high: float) -> dict[str, float]:
    """`loss_fn_config` for a ratio clipped to `[1 - low, 1 + high]`: Tinker's `ppo` and
    `cispo` take the bounds themselves, where DAPO's `clip_low` / `clip_high` are epsilons."""
    # The route and the key names are the installed SDK's: tinker/types/
    # forward_backward_input.py:21 documents `loss_fn_config` as "Optional configuration
    # parameters for the loss function (e.g., PPO clip thresholds, DPO beta)", and
    # tinker/types/datum.py:32-33 names `clip_low_threshold` / `clip_high_threshold`.
    # That they are bounds is the losses page's `torch.clamp(prob_ratio,
    # clip_low_threshold, clip_high_threshold)` with the example `{"clip_low_threshold":
    # 0.9, "clip_high_threshold": 1.1}` (https://tinker-docs.thinkingmachines.ai/losses);
    # the installed SDK spells out neither that nor a default for `cispo`.
    return {"clip_low_threshold": round(1.0 - low, 6), "clip_high_threshold": round(1.0 + high, 6)}


def resolution(preset: Preset) -> str:
    """The comment line `check` prints after the resolved config: how the advantage is
    formed, which loss with which clipping and aggregation, the shaping if any, the
    substeps, the optimizer, what becomes of degenerate groups, and the KL if any."""
    spread = "divided by spread" if preset.normalize else "not divided by spread"
    summed = "averaged per prompt" if preset.aggregation == "prompt" else "summed over tokens"
    parts = [
        f"advantage = group mean, {spread}",
        f"loss = {preset.loss_fn}, {preset.clipping}, {summed}",
    ]
    if preset.length_penalty > 0:
        capped = (
            "uncapped" if preset.length_cap == float("inf") else f"capped at {preset.length_cap}"
        )
        parts.append(
            f"length penalty {preset.length_penalty} over {preset.length_floor} tokens "
            f"among solved answers, {capped}"
        )
    if preset.overlong_penalty > 0:
        parts.append(
            f"overlong penalty up to {preset.overlong_penalty} over the last "
            f"{preset.overlong_buffer * 100:g}% of the token budget"
        )
    parts.append("1 substep" if preset.substeps == 1 else f"{preset.substeps} substeps by prompt")
    parts.append(optimizer(preset.adam))
    if preset.refill > 0:
        rounds = "round" if preset.refill == 1 else "rounds"
        parts.append(
            f"degenerate groups dropped and refilled from the plan, up to {preset.refill} "
            f"more {rounds}"
        )
    else:
        parts.append("degenerate groups dropped")
    if preset.kl_coef > 0:
        parts.append(f"kl {preset.kl_coef} to the starting weights")
    return f"# {preset.name}: " + "; ".join(parts)


def optimizer(adam: Adam) -> str:
    """The optimizer as the resolution line states it: betas and eps, then the weight decay,
    the gradient clipping and the warm-up when the recipe has them."""
    said = f"adamw betas {adam.beta1} / {adam.beta2}, eps {adam.eps:g}"
    if adam.weight_decay > 0:
        said += f", weight decay {adam.weight_decay}"
    if adam.grad_clip_norm > 0:
        said += f", gradient norm clipped at {adam.grad_clip_norm}"
    if adam.warmup > 0:
        steps = "step" if adam.warmup == 1 else "steps"
        said += f", learning rate warmed up over {adam.warmup} {steps}"
    return said


async def train(run: Run, preset: Preset) -> None:
    """The session around the steps: a `step` per batch of the plan inside `training`, each
    step free to draw further batches from the same plan to fill itself."""
    cfg = run.config
    data = cfg.data
    async with training(run, preset) as (trainer, anchor):
        planned = batches(cfg.datasets, size=data.batch_size, seed=data.seed, epochs=data.epochs)
        for index, tasks in enumerate(planned):
            await step(run, trainer, anchor, preset, tasks, index, plan=planned)


@asynccontextmanager
async def training(run: Run, preset: Preset) -> AsyncIterator[tuple[Trainer, Any]]:
    """The trainer and the KL anchor opened, the proxy started (a warning when the preset's
    overlong term has no token budget to dock against); on a clean exit `final`
    checkpointed, and the session closed either way with how it ended."""
    serving = serving_of(run, preset)
    model = run.config.model
    metadata = {"shipyard_run": run.id, "shipyard_recipe": preset.name}
    service = session.service_client(metadata)
    trainer = await Trainer.create(
        service,
        model.name,
        from_checkpoint=model.from_checkpoint,
        lora_rank=model.lora_rank,
        restore_optimizer=model.restore_optimizer,
        metadata=metadata,
    )
    run.trainer = trainer
    try:
        anchor = None
        if preset.kl_coef > 0:
            # The starting weights: the base model, or the loaded state published as
            # sampler weights, since Tinker samples only a sampler path.
            start = await trainer.publish("anchor") if model.from_checkpoint else None
            anchor = await service.create_sampling_client_async(
                base_model=model.name, model_path=start
            )
        await serving.start()
        if preset.overlong_penalty > 0 and serving.budget is None:
            logger.warning(
                "%s's overlong term is off for this run: the proxy reports no token budget, "
                "so overlong_penalty = %s docks no rollout",
                preset.name,
                preset.overlong_penalty,
            )
        yield trainer, anchor
        await run.checkpoint(trainer, "final", keep=True)
    except BaseException as failed:
        status = "errored" if isinstance(failed, Exception) else "interrupted"
        await trainer.close(status, f"{type(failed).__name__}: {failed}")
        raise
    await trainer.close()


async def step(
    run: Run,
    trainer: Trainer,
    anchor: Any,
    preset: Preset,
    tasks: Sequence[Path],
    index: int,
    *,
    sample: Sampler | None = None,
    about: Mapping[str, Any] | None = None,
    plan: Iterator[Sequence[Path]] | None = None,
) -> None:
    """One step over `tasks`: the weights published and the proxy pointed at them, the
    batch sampled (by `sample` when given), refilled from `plan` under the preset's
    `refill`, credited, the gradient applied unless nothing was credited, the row logged
    with `about`, and a checkpoint when `index + 1` divides by `[checkpoints] every`."""
    serving = serving_of(run, preset)
    data, every = run.config.data, run.config.checkpoints.every
    # Published as `sample-<index>`, not the spec's `step-<index>`: the checkpoint after
    # step n saves sampler weights as `step-<n + 1>`, the name step n + 1 would publish
    # under, and `save_weights_for_sampler` does not overwrite a name by default.
    path = await trainer.publish(f"sample-{index}", ttl_seconds=SERVE_TTL)
    await serving.point(path)

    async def draw(chosen: Sequence[Path], purpose: str = "rollout") -> Rollouts:
        if sample is None:
            return await run.sample(chosen, rollouts=data.group_size, index=index, purpose=purpose)
        return await sample(chosen, index)

    groups = group(await draw(tasks), data.group_size)
    refill: Refill | None = None
    # A lone rollout is never compared, so groups of one could only spend the plan.
    if preset.refill > 0 and plan is not None and data.group_size > 1:
        groups, refill = await refilled(
            groups, preset, data.batch_size, data.group_size, plan, draw
        )
    limit = data.batch_size if refill is not None else None
    batch = await credit(groups, preset, trainer, anchor, limit=limit, budget=serving.budget)
    run.spent(
        serving.party, reference_tokens=batch.reference_tokens, anchor_tokens=batch.anchor_tokens
    )
    extra = dict(about or {})
    if batch.empty:
        run.log(**row(index, batch, None, refill), **extra)
        return
    update = await trainer.apply(batch, preset)
    run.spent(serving.party, train_tokens=update.train_tokens)
    run.log(**row(index, batch, update, refill), **extra)
    if (index + 1) % every == 0:
        await run.checkpoint(trainer, f"step-{index + 1}")


async def refilled(
    groups: Sequence[Group],
    preset: Preset,
    size: int,
    group_size: int,
    plan: Iterator[Sequence[Path]],
    draw: Callable[[Sequence[Path], str], Awaitable[Rollouts]],
) -> tuple[list[Group], Refill]:
    """DAPO's dynamic sampling (Alg. 1, lines 6 to 8): while fewer than `size` groups carry
    a gradient, the plan's next batch is sampled at the same weights and its groups join
    the step, consumed as DAPO consumes its dataloader; at most `preset.refill` rounds, and
    none once the plan runs out."""
    found = list(groups)
    rounds = rollouts = 0
    while rounds < preset.refill and carrying(found, preset) < size:
        more = next(plan, None)
        if more is None:
            break
        rolled = await draw(more, "refill")
        rounds += 1
        rollouts += len(rolled)
        found += group(rolled, group_size)
    return found, Refill(refills=rounds, refill_rollouts=rollouts)


def serving_of(run: Run, preset: Preset) -> Serving:
    """The proxy this run serves its model through, refused when the model is served
    elsewhere: the records a gradient trains on are the proxy's."""
    if run.serving is None:
        raise ValueError(
            f"[recipe] kind = {preset.name!r} trains the weights this run serves, and "
            f"[model] provider = {run.config.model.provider!r} serves them elsewhere"
        )
    return run.serving


def row(
    index: int, batch: Batch, update: Update | None, refill: Refill | None = None
) -> dict[str, Any]:
    """The step's metrics row in `ROW` order, `trained` saying whether a gradient was
    taken; what was not measured is absent, the refill's counts with no refill among it."""
    found: dict[str, Any] = {"step": index, "trained": update is not None}
    for source in (batch, update, refill):
        found.update({key: getattr(source, key) for key in ROW if hasattr(source, key)})
    return {key: value for key in ROW if (value := found.get(key)) is not None}
