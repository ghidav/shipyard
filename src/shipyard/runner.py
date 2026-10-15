"""What a blueprint builds; the runner loops over epochs and never sees weights or groups."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from collections.abc import Sequence as Values
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shipyard import record, session
from shipyard.data import Generator
from shipyard.recipes import trainer_for
from shipyard.run import Run
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import Trainer
from shipyard.types import SERVE_TTL, Agent, Evaluation, Modules, Policy, harness

if TYPE_CHECKING:
    from shipyard.config import Model

__all__ = ["Runner", "runner_in"]


class Runner:
    #: The trainer class a subclass names; None means the kind's. An instance's `trainer` is
    #: the one built.
    trainer: Any = None

    def __init__(
        self,
        run: Run,
        generator: Generator,
        trainer: Trainer,
        *,
        epochs: int,
        evaluate_every: int = 0,
        modules: Modules | None = None,
    ) -> None:
        self.opened = run
        self.generator = generator
        self.trainer = trainer
        self.epochs = epochs
        self.evaluate_every = evaluate_every
        #: What the agent carries: the run's own modules, or the head of a restored search.
        self.modules = modules
        #: On a resume, the checkpoint row and how many batches of its epoch were stepped.
        self.position: tuple[Mapping[str, Any], int] | None = None

    @classmethod
    def of(cls, run: Run, trainer: type[Trainer] | None = None) -> Runner:
        """The runner for an open run: under `trainer`, else the class's own, else the kind's."""
        cfg = run.config
        if cfg.trains and not cfg.model.served:
            raise ValueError(
                f"[recipe] kind = {cfg.recipe.kind!r} trains the weights this run serves, and "
                f"[model] provider = {cfg.model.provider!r} serves them elsewhere"
            )
        chosen = trainer or cls.trainer or trainer_for(cfg.recipe)
        return cls(
            run,
            Generator.of(cfg),
            chosen(run),
            epochs=0 if cfg.recipe.kind == "evaluate" else cfg.data.epochs,
            evaluate_every=cfg.data.evaluate_every,
            modules=run.carried,
        )

    @classmethod
    def from_run(cls, directory: Path, *, checkpoint: str = "last") -> Runner:
        """The runner over a run reopened at one of its checkpoints, standing where it stood;
        a run opened under a runner file reopens under its copy."""
        directory = Path(directory)
        if not (directory / record.CONFIG).is_file():
            raise ValueError(f"no run at {directory}")
        kept = directory / record.RUNNER
        if cls is Runner and kept.is_file():
            return runner_in(kept).from_run(directory, checkpoint=checkpoint)
        rows = list(record.read(directory / record.CHECKPOINTS))
        if not rows:
            raise ValueError(f"{directory.name} has no checkpoint to go on from")
        row = (
            rows[-1]
            if checkpoint == "last"
            else next((r for r in rows if r["tag"] == checkpoint), None)
        )
        if row is None:
            tags = ", ".join(r["tag"] for r in rows)
            raise ValueError(
                f"{directory.name} has no checkpoint tagged {checkpoint!r}; it has {tags}"
            )
        run = Run.reopen(directory, resumed=row["tag"])
        runner = cls.of(run)
        consumed = runner.trainer.consumed(list(record.read(directory / record.JOBS)), row)
        runner.modules = runner.trainer.restore(row, consumed)
        runner.position = (row, consumed)
        return runner

    async def run(self) -> Agent:
        """Open the agent, train an epoch at a time, evaluate on the cadence, checkpoint `final`."""
        run, trainer, generator = self.opened, self.trainer, self.generator
        cfg = run.config
        model, recipe = cfg.model, cfg.recipe
        row, consumed = self.position or (None, 0)
        service = client = None
        if cfg.trains:
            metadata = {"shipyard_run": run.id, "shipyard_recipe": recipe.kind}
            service = session.service_client(metadata)
            client = await TinkerClient.open(
                service,
                model.name,
                from_checkpoint=model.from_checkpoint if row is None else row["state_path"],
                lora_rank=model.lora_rank,
                restore_optimizer=model.restore_optimizer if row is None else True,
                metadata=metadata,
            )
        policy = Policy(run.endpoint, client, None if row is None else row["sampler_path"])
        agent = Agent(policy, [harness(cfg.rollout.harness)], self.modules)
        async with agent.policy:
            if client is not None and recipe.kl_coef > 0:
                trainer.anchor = await anchor(service, client, model)
            evaluated = 0
            for epoch in range(trainer.epoch, self.epochs):
                trainer.epoch = epoch
                batches = generator.batches(epoch)
                if row is not None and epoch == row["epoch"]:
                    batches = islice(batches, consumed, None)
                agent = await trainer.train(agent, batches)
                if self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                    await trainer.evaluate(agent, generator.test)
                    evaluated = epoch + 1
            if generator.test and (self.epochs == 0 or evaluated != self.epochs):
                await trainer.evaluate(agent, generator.test)
            if client is not None:
                await trainer.checkpoint(agent, "final", keep=True)
        return agent

    async def evaluate(self, tasks: Values[Path]) -> Evaluation | None:
        """One evaluation of the checkpoint the runner was reopened at, on `tasks`."""
        if self.position is None:
            raise ValueError("an evaluation needs a run reopened at a checkpoint")
        row, _ = self.position
        run = self.opened
        policy = Policy(run.endpoint, None, row["sampler_path"])
        agent = Agent(policy, [harness(run.config.rollout.harness)], self.modules)
        async with agent.policy:
            return await self.trainer.evaluate(agent, list(tasks), checkpoint=row["tag"])


async def anchor(service: Any, client: TinkerClient, model: Model) -> Any:
    """The KL anchor: a sampling client on the start, published first when it is a loaded state."""
    start = (
        await client.save_weights_for_sampler("anchor", ttl_seconds=SERVE_TTL)
        if model.from_checkpoint
        else None
    )
    return await service.create_sampling_client_async(base_model=model.name, model_path=start)


def runner_in(path: Path) -> type[Runner]:
    """The one `Runner` subclass the Python file defines; a path is loaded once."""
    path = Path(path).resolve()
    name = f"shipyard_runner_{sha256(str(path).encode()).hexdigest()[:12]}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{path} is not a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    found = [
        one
        for one in vars(module).values()
        if isinstance(one, type) and issubclass(one, Runner) and one.__module__ == name
    ]
    if len(found) != 1:
        raise ValueError(
            f"{path} defines {len(found)} Runner subclasses; a runner file defines one"
        )
    return found[0]
