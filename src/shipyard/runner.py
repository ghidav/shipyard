"""What a blueprint builds; the runner loops over epochs and never sees weights or groups."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shipyard import session
from shipyard.data import Generator
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import Trainer
from shipyard.types import SERVE_TTL, Agent, Policy, harness

if TYPE_CHECKING:
    from shipyard.config import Model
    from shipyard.run import Run

__all__ = ["Runner"]


class Runner:
    def __init__(
        self,
        run: Run,
        generator: Generator,
        trainer: Trainer,
        *,
        epochs: int,
        evaluate_every: int = 0,
    ) -> None:
        self.opened = run
        self.generator = generator
        self.trainer = trainer
        self.epochs = epochs
        self.evaluate_every = evaluate_every

    @classmethod
    def of(cls, run: Run, trainer: type[Trainer]) -> Runner:
        """The runner for an open run under the trainer class given."""
        cfg = run.config
        if cfg.trains and not cfg.model.served:
            raise ValueError(
                f"[recipe] kind = {cfg.recipe.kind!r} trains the weights this run serves, and "
                f"[model] provider = {cfg.model.provider!r} serves them elsewhere"
            )
        procedure = trainer(run)
        run.trainer = procedure
        return cls(
            run,
            Generator.of(cfg),
            procedure,
            epochs=0 if cfg.recipe.kind == "evaluate" else cfg.data.epochs,
            evaluate_every=cfg.data.evaluate_every,
        )

    async def run(self) -> Agent:
        """Open the agent, train an epoch at a time, evaluate on the cadence, checkpoint `final`."""
        run, trainer, generator = self.opened, self.trainer, self.generator
        cfg = run.config
        model, recipe = cfg.model, cfg.recipe
        service = client = None
        if cfg.trains:
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
            if client is not None and recipe.kl_coef > 0:
                trainer.anchor = await anchor(service, client, model)
            evaluated = 0
            for epoch in range(self.epochs):
                trainer.epoch = epoch
                agent = await trainer.train(agent, generator.batches(epoch))
                if self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                    await trainer.evaluate(agent, generator.test)
                    evaluated = epoch + 1
            if generator.test and (self.epochs == 0 or evaluated != self.epochs):
                await trainer.evaluate(agent, generator.test)
            if client is not None:
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
