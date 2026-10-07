"""The gradient loop the three recipes share, over the Tinker SDK: a Preset each name
resolves to, `step` (publish, point, sample, credit, apply, log, checkpoint by `every`)
and `train`, the session of steps around it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shipyard import session
from shipyard.credit import Batch, credit
from shipyard.data import batches
from shipyard.pack import group
from shipyard.trainer import SERVE_TTL, Trainer, Update

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shipyard.run import Run
    from shipyard.serving import Serving

__all__ = ["ROW", "Preset", "clipped", "resolution", "row", "shared", "step", "train"]

#: The step's row, in this order; a measure with nothing behind it is left out, never zero.
ROW = (
    "step",
    "trained",
    "groups",
    "rollouts",
    "graded",
    "masked",
    "degenerate",
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
    "learning_rate",
    "substeps",
    "loss_fn",
    "seconds",
)


@dataclass(frozen=True, kw_only=True)
class Preset:
    """What a recipe name and its knobs resolve to; never user-facing. `loss_config` is
    Tinker's `loss_fn_config` for the step (see `clipped`); `clipping` says it for `check`."""

    name: str
    normalize: bool
    loss_fn: str
    loss_config: dict[str, float]
    clipping: str
    length_penalty: float = 0.0
    length_floor: int = 0
    kl_coef: float
    reference: str
    learning_rate: float
    substeps: int


def shared(recipe: Any) -> dict[str, Any]:
    """The knobs every gradient recipe carries, as `Preset` takes them."""
    return {
        "kl_coef": float(recipe.kl_coef),
        "reference": str(recipe.reference),
        "learning_rate": float(recipe.learning_rate),
        "substeps": int(recipe.substeps),
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
    formed, which loss with which clipping, the shaping if any, and the KL if any."""
    spread = "divided by spread" if preset.normalize else "not divided by spread"
    parts = [f"advantage = group mean, {spread}", f"loss = {preset.loss_fn}, {preset.clipping}"]
    if preset.length_penalty > 0:
        parts.append(
            f"length penalty {preset.length_penalty} over {preset.length_floor} tokens "
            "among solved answers"
        )
    parts.append("degenerate groups dropped")
    if preset.kl_coef > 0:
        parts.append(f"kl {preset.kl_coef} to the starting weights")
    return f"# {preset.name}: " + "; ".join(parts)


async def train(run: Run, preset: Preset) -> None:
    """The session around the steps: the trainer and the anchor opened, the proxy started,
    a `step` per batch of the plan, `final` checkpointed always, the session closed."""
    serving = serving_of(run, preset)
    cfg = run.config
    model, data = cfg.model, cfg.data
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
            # `from_checkpoint` is a sampler path here and for the proxy, a state path for
            # `Trainer.create`: a blueprint names one both factories accept.
            anchor = await service.create_sampling_client_async(
                base_model=model.name, model_path=model.from_checkpoint or None
            )
        await serving.start()
        planned = batches(cfg.datasets, size=data.batch_size, seed=data.seed, epochs=data.epochs)
        for index, tasks in enumerate(planned):
            await step(run, trainer, anchor, preset, tasks, index)
        await run.checkpoint(trainer, "final")
    except BaseException as failed:
        status = "errored" if isinstance(failed, Exception) else "interrupted"
        await trainer.close(status, f"{type(failed).__name__}: {failed}")
        raise
    await trainer.close()


async def step(
    run: Run, trainer: Trainer, anchor: Any, preset: Preset, tasks: Sequence[Path], index: int
) -> None:
    """One step over `tasks`: the weights published and the proxy pointed at them, the
    batch sampled and credited, the gradient applied unless nothing was credited, the row
    logged, and a checkpoint when `index + 1` divides by `[checkpoints] every`."""
    serving = serving_of(run, preset)
    data, every = run.config.data, run.config.checkpoints.every
    # Published as `sample-<index>`, not the spec's `step-<index>`: the checkpoint after
    # step n saves sampler weights as `step-<n + 1>`, the name step n + 1 would publish
    # under, and `save_weights_for_sampler` does not overwrite a name by default.
    path = await trainer.publish(f"sample-{index}", ttl_seconds=SERVE_TTL)
    await serving.point(path)
    rollouts = await run.sample(tasks, rollouts=data.group_size, index=index)
    batch = await credit(group(rollouts, data.group_size), preset, trainer, anchor)
    run.spent(
        serving.party, reference_tokens=batch.reference_tokens, anchor_tokens=batch.anchor_tokens
    )
    if batch.empty:
        run.log(**row(index, batch, None))
        return
    update = await trainer.apply(batch, preset)
    run.spent(serving.party, train_tokens=update.train_tokens)
    run.log(**row(index, batch, update))
    if (index + 1) % every == 0:
        await run.checkpoint(trainer, f"step-{index + 1}")


def serving_of(run: Run, preset: Preset) -> Serving:
    """The proxy this run serves its model through, refused when the model is served
    elsewhere: the records a gradient trains on are the proxy's."""
    if run.serving is None:
        raise ValueError(
            f"[recipe] kind = {preset.name!r} trains the weights this run serves, and "
            f"[model] provider = {run.config.model.provider!r} serves them elsewhere"
        )
    return run.serving


def row(index: int, batch: Batch, update: Update | None) -> dict[str, Any]:
    """The step's metrics row in `ROW` order, `trained` saying whether a gradient was
    taken; what was not measured is absent."""
    found: dict[str, Any] = {"step": index, "trained": update is not None}
    for source in (batch, update):
        found.update({key: getattr(source, key) for key in ROW if hasattr(source, key)})
    return {key: value for key in ROW if (value := found.get(key)) is not None}
