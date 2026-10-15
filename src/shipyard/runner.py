"""What a blueprint builds; the runner loops over epochs and never sees weights or groups."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard import session
from shipyard.data import Generator
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer
from shipyard.types import SERVE_TTL, Agent, Policy, harness

if TYPE_CHECKING:
    from shipyard.config import Model
    from shipyard.run import Run

__all__ = ["TRAINERS", "Runner"]

#: The trainer class of each `[recipe] kind` the runner runs.
TRAINERS: dict[str, type[Trainer]] = {
    "dapo": DAPOTrainer,
    "dr-grpo": DrGRPOTrainer,
    "cispo": CISPOTrainer,
}


class Runner:
    def __init__(self, run: Run, generator: Generator, trainer: Trainer, *, epochs: int) -> None:
        self.opened = run
        self.generator = generator
        self.trainer = trainer
        self.epochs = epochs

    @classmethod
    def of(cls, run: Run, *, trainer: type[Trainer] | None = None) -> Runner:
        """The runner for an open run; `trainer` replaces the class its kind names."""
        cfg = run.config
        if not cfg.model.served:
            raise ValueError(
                f"[recipe] kind = {cfg.recipe.kind!r} trains the weights this run serves, and "
                f"[model] provider = {cfg.model.provider!r} serves them elsewhere"
            )
        procedure = (TRAINERS[cfg.recipe.kind] if trainer is None else trainer)(run)
        run.trainer = procedure
        return cls(run, Generator.of(cfg), procedure, epochs=cfg.data.epochs)

    async def run(self) -> Agent:
        """Open the agent on a training client, train an epoch at a time, checkpoint `final`."""
        run, trainer = self.opened, self.trainer
        cfg = run.config
        model, recipe = cfg.model, cfg.recipe
        metadata = {"shipyard_run": run.id, "shipyard_recipe": recipe.kind}
        service = session.service_client(metadata)
        client = await TinkerClient.open(
            service,
            model.name,
            from_checkpoint=model.from_checkpoint,
            lora_rank=model.lora_rank,
            restore_optimizer=model.restore_optimizer,
            metadata=metadata,
        )
        agent = Agent(Policy(run.endpoint, client), [harness(cfg.rollout.harness)], run.carried)
        async with agent.policy:
            if recipe.kl_coef > 0:
                trainer.anchor = await anchor(service, client, model)
            for epoch in range(self.epochs):
                trainer.epoch = epoch
                agent = await trainer.train(agent, self.generator.batches(epoch))
            await trainer.checkpoint(agent, "final", keep=True)
        return agent


async def anchor(service: Any, client: TinkerClient, model: Model) -> Any:
    """The KL anchor: a sampling client on the start, published first when it is a loaded state."""
    start = (
        await client.save_weights_for_sampler("anchor", ttl_seconds=SERVE_TTL)
        if model.from_checkpoint
        else None
    )
    return await service.create_sampling_client_async(base_model=model.name, model_path=start)
