"""What a blueprint builds; the runner loops over epochs and never sees weights or groups."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from collections.abc import Sequence as Values
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any

from shipyard import record
from shipyard.data import Generator
from shipyard.modules import Candidate
from shipyard.recipes import trainer_for
from shipyard.run import Run
from shipyard.sampling.endpoint import endpoint_for
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import Trainer
from shipyard.types import Agent, Batches, Evaluation, Policy, harness

__all__ = ["Runner", "runner_in"]


class Runner:
    #: The trainer class a subclass names; None means the kind's. An instance's `trainer` is
    #: the one built.
    trainer: Any = None

    def __init__(
        self,
        generator: Generator,
        trainer: Trainer,
        *,
        epochs: int,
        evaluate_every: int = 0,
        modules: Candidate | None = None,
    ) -> None:
        self.generator = generator
        self.trainer = trainer
        self.epochs = epochs
        self.evaluate_every = evaluate_every
        #: What the agent carries: the run's own modules, or the head of a restored search.
        self.modules = modules
        #: On a resume, the checkpoint row the run reopened at and how many batches of its
        #: epoch were already stepped.
        self.checkpoint: Mapping[str, Any] | None = None
        self.consumed = 0

    @classmethod
    def from_run(cls, run: Run, trainer: type[Trainer] | None = None) -> Runner:
        """The runner for an open run: under `trainer`, else the class's own, else the kind's."""
        cfg = run.config
        chosen = trainer or cls.trainer or trainer_for(cfg.recipe)
        return cls(
            Generator.of(cfg),
            chosen(run),
            epochs=0 if cfg.recipe.kind == "evaluate" else cfg.data.epochs,
            evaluate_every=cfg.data.evaluate_every,
            modules=run.carried,
        )

    @classmethod
    def from_checkpoint(cls, directory: Path, *, tag: str = "last") -> Runner:
        """The runner over a run reopened at one of its checkpoints, standing where it stood;
        a run opened under a runner file reopens under its copy."""
        directory = Path(directory)
        if not (directory / record.CONFIG).is_file():
            raise ValueError(f"no run at {directory}")
        kept = directory / record.RUNNER
        if cls is Runner and kept.is_file():
            return runner_in(kept).from_checkpoint(directory, tag=tag)
        rows = list(record.read(directory / record.CHECKPOINTS))
        if not rows:
            raise ValueError(f"{directory.name} has no checkpoint to go on from")
        row = rows[-1] if tag == "last" else next((r for r in rows if r["tag"] == tag), None)
        if row is None:
            tags = ", ".join(r["tag"] for r in rows)
            raise ValueError(f"{directory.name} has no checkpoint tagged {tag!r}; it has {tags}")
        run = Run.reopen(directory, resumed=row["tag"])
        runner = cls.from_run(run)
        runner.consumed = runner.trainer.consumed(list(record.read(directory / record.JOBS)), row)
        runner.modules = runner.trainer.restore(row, runner.consumed)
        runner.checkpoint = row
        return runner

    async def run(self) -> Agent:
        """Open the agent, train an epoch at a time, evaluate on the cadence, checkpoint `final`."""
        run = self.trainer.run
        cfg = run.config
        row = self.checkpoint or {}
        client = TinkerClient.from_run(run) if cfg.trains else None
        policy = Policy(
            endpoint_for(cfg, run.directory),
            client,
            start=row.get("sampler_path"),
            state=row.get("state_path"),
            anchored=cfg.trains and cfg.recipe.kl_coef > 0,
        )
        agent = Agent(policy, [harness(cfg.rollout.harness)], self.modules)
        async with agent.policy:
            evaluated = 0
            for epoch in range(self.trainer.epoch, self.epochs):
                self.trainer.epoch = epoch
                agent = await self.trainer.train(agent, self.batches(epoch))
                if self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                    await self.trainer.evaluate(agent, self.generator.test)
                    evaluated = epoch + 1
            if self.generator.test and (self.epochs == 0 or evaluated != self.epochs):
                await self.trainer.evaluate(agent, self.generator.test)
            if client is not None:
                await self.trainer.checkpoint(agent, "final", keep=True)
        return agent

    def batches(self, epoch: int) -> Batches:
        """The epoch's batches, less the ones stepped before the checkpoint the run reopened at."""
        found = self.generator.batches(epoch)
        if self.checkpoint is not None and epoch == self.checkpoint["epoch"]:
            return islice(found, self.consumed, None)
        return found

    async def evaluate(self, tasks: Values[Path]) -> Evaluation | None:
        """One evaluation of the checkpoint the runner was reopened at, on `tasks`."""
        if self.checkpoint is None:
            raise ValueError("an evaluation needs a run reopened at a checkpoint")
        run = self.trainer.run
        policy = Policy(
            endpoint_for(run.config, run.directory), start=self.checkpoint["sampler_path"]
        )
        agent = Agent(policy, [harness(run.config.rollout.harness)], self.modules)
        async with agent.policy:
            return await self.trainer.evaluate(
                agent, list(tasks), checkpoint=self.checkpoint["tag"]
            )


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
