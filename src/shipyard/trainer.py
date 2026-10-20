from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import TracebackType
from typing import Any, Self

import tinker
import torch
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.utils.misc_utils import split_list

from shipyard.config import Blueprint, Recipe
from shipyard.groups import Group, by_task, grouped
from shipyard.jobs import harnesses
from shipyard.jobs.engine import HarborEngine, Job
from shipyard.jobs.modules import Candidate, keep, seed
from shipyard.jobs.secrets import Pool, Secrets
from shipyard.models.agent import Frozen, Provider, Session
from shipyard.models.client import TINKER, datum
from shipyard.policy import Policy
from shipyard.run import record
from shipyard.run.data import Generator
from shipyard.run.logs import Logs

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600
#: The checkpoint a run saves when it ends: the weights its updates moved to, kept for good.
FINAL = "final"
#: Where the run keeps the modules every job carries, under `<run>/modules/`.
CARRIED = "carried"


@dataclass(frozen=True)
class Update:
    """What one `update` trained, and what its training pass read."""

    sequences: int
    train_tokens: int
    substeps: int
    learning_rate: float
    loss_fn: str
    seconds: float
    kl_v1: float
    kl_v2: float
    entropy: float


@dataclass(frozen=True)
class Evaluation:
    """What one `evaluate` rolled out, and the mean reward it measured."""

    step: int
    tasks: int
    rollouts: int
    graded: int
    masked: int
    #: Mean reward over the graded rollouts; None when none were graded.
    reward_mean: float | None


class Trainer:
    """A run: the four primitives, and the `step` a recipe writes from them."""

    #: Adam settings from the cookbook's `train_step`. Its `learning_rate` is not read: each
    #: update takes `[recipe] learning_rate`, warmed up over `[recipe] warmup` updates.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
    #: The `[recipe]` keys the trainer reads. A recipe with keys of its own declares them in a
    #: subclass, `class Recipe(Trainer.Recipe)`, and its run.toml sets every one.
    Recipe = Recipe

    def __init__(self, config: Blueprint, directory: Path) -> None:
        self.config = config
        self.recipe = config.recipe
        #: Run directory: its rows, checkpoints, costs and modules. A trainer on a directory an
        #: earlier one wrote continues from it.
        self.directory = Path(directory)
        self.generator = Generator.of(config)
        self.modules = carried(config, self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        table = config.rollout.secrets
        #: One engine, so every job shares one `[rollout] concurrency` limit.
        self.engine = HarborEngine(
            config.rollout,
            env=config.agent.env,
            kwargs=config.agent.kwargs,
            secrets=Pool.of(table) if table else Secrets(),
        )
        #: The run's record: job and step rows, checkpoints and costs.
        self.logs = Logs(config, self.directory)
        self._agent = Policy(config.agent.model, provider=config.agent.provider)
        #: The agent on Tinker, made and opened when `train` starts; None for a provider's.
        self._session: Session | None = None
        #: The recipe's own numbers for this step's row, such as a KL it measured.
        self.metrics: dict[str, Any] = {}
        self.steps = self.updates = self.drawn = 0
        #: Checkpoint row the run resumed from; None on a fresh run.
        self.resumed: Mapping[str, Any] | None = None
        #: Step after which the test set was last evaluated.
        self.evaluated: int | None = None
        self._stream: Iterator[list[Path]] | None = None
        self._seen: list[Group] = []
        self._update: Update | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        failed: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.engine.close()
        self.logs.close(failed)

    @property
    def policy(self) -> Policy:
        """The agent: the run's own policy, made from `[agent]`."""
        return self._agent

    # ------------------------------------------------------------------ what a recipe writes

    async def step(self) -> None:
        """Draw a batch, roll the policy out on it, score the rollouts, and update."""
        groups = await self.rollout(self.draw())
        await self.score(groups)
        await self.update(groups)

    async def score(self, groups: Values[Group]) -> None:
        """Give each rollout its reward minus its group's mean, skipping groups of equal rewards."""
        for group in groups:
            if group.reward_variance == 0:
                continue
            for trial, advantage in zip(group.measured, group.centered_rewards, strict=True):
                trial.set_advantage(advantage)

    # ------------------------------------------------------------------ the primitives

    def draw(self) -> list[Path]:
        """Return the next batch of `[data] batch_size` training tasks."""
        if self._stream is None:
            self._stream = self.generator.batches(self.drawn)
        batch = next(self._stream)
        self.drawn += 1
        return batch

    async def rollout(
        self,
        tasks: Values[Path],
        *,
        attempts: int | None = None,
        modules: Candidate | None = None,
        harness: str | None = None,
        purpose: str = "training",
    ) -> list[Group]:
        """Run the agent `attempts` times on each task, and return its trials grouped by task."""
        cfg, tasks = self.config, list(tasks)
        carried = self.modules if modules is None else modules
        jobs = split(
            tasks,
            attempts or cfg.data.group_size,
            [harness] if harness else cfg.agent.harness,
            mix=cfg.agent.mix_harnesses,
            turn=self.steps,
        )
        parts = await asyncio.gather(
            *(self._job(one, part, count, carried, purpose) for one, part, count in jobs)
        )
        first = {task: at for at, task in reversed(list(enumerate(tasks)))}
        found = sorted(by_task([g for part in parts for g in part]), key=lambda g: first[g.task])
        if purpose == "training":
            self._seen += found
        return found

    async def _job(
        self,
        harness: str,
        tasks: list[Path],
        attempts: int,
        modules: Candidate | None,
        purpose: str,
    ) -> list[Group]:
        """Run one judged job of `harness` on `tasks`, write its row, and return its groups."""
        endpoint = self._agent.endpoint
        job = Job(
            self.logs.reserve(),
            purpose=purpose,
            step=self.steps,
            modules=modules,
            harness=harnesses.harness(harness),
            grade_timeouts=self.recipe.grade_timeouts,
        )
        await self.engine.run(job, tasks, attempts, endpoint=endpoint)
        self.logs.job(job, endpoint)
        endpoint.check(job.name, job.trials)
        return grouped(job.trials)

    async def update(
        self,
        groups: Values[Group],
        *,
        loss: str = "importance_sampling",
        loss_config: Mapping[str, float] | None = None,
    ) -> Update | None:
        """Train the agent on every sequence with advantages, and serve its new weights."""
        policy, session = self._agent, self._session
        if session is None:
            raise ValueError(
                f"update trains the agent, and {policy.provider} serves {policy.model}: there "
                "are no weights to train"
            )
        client = session.client
        learning_rate = self.recipe.learning_rate
        if learning_rate is None:
            raise ValueError("[recipe] learning_rate: update needs it, and the blueprint sets none")
        kept = [group.scored for group in groups if group.is_scored]
        if not kept:
            return None
        began, recipe = time.monotonic(), self.recipe
        flat = [one for part in kept for one in part]
        unrecorded = [one for one in flat if one.sampled_logprobs is None]
        scored = dict(zip(unrecorded, await policy.logprobs(unrecorded), strict=True))
        parts = [
            [
                datum(
                    one,
                    logprobs=(
                        one.sampled_logprobs
                        if one.sampled_logprobs is not None
                        else scored[one].tolist()  # type: ignore[union-attr]
                    ),
                    advantages=one.advantages.tolist(),  # type: ignore[union-attr]
                )
                for one in part
            ]
            for part in kept
        ]
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        count = min(recipe.substeps, len(parts))
        pieces = [[one for part in piece for one in part] for piece in split_list(parts, count)]
        datums = [one for piece in pieces for one in piece]
        tokens = sum(int(one.model_input.length) for one in datums)
        rate = self._rate(learning_rate)
        trained: list[torch.Tensor] = []
        forward, optim = await self._submit(client, pieces[0], rate, loss, loss_config)
        for index in range(count):
            following = (
                await self._submit(client, pieces[index + 1], rate, loss, loss_config)
                if index + 1 < count
                else None
            )
            trained.extend(torch.tensor(row) for row in await forward)
            await optim
            if following is not None:
                forward, optim = following
        session.count(train=tokens)
        self.updates += 1
        await session.serve(f"sample-{self.updates}")
        drift = compute_kl_sample_train(datums, trained)
        self._update = Update(
            sequences=len(flat),
            train_tokens=tokens,
            substeps=count,
            learning_rate=rate,
            loss_fn=loss,
            seconds=round(time.monotonic() - began, 3),
            kl_v1=drift["optim/kl_sample_train_v1"],
            kl_v2=drift["optim/kl_sample_train_v2"],
            entropy=drift["optim/entropy"],
        )
        return self._update

    def _rate(self, learning_rate: float) -> float:
        """Return the next update's learning rate, warmed up over `[recipe] warmup` updates."""
        recipe = self.recipe
        restored = self.config.agent.from_checkpoint and recipe.restore_optimizer
        if restored or self.updates >= recipe.warmup:
            return learning_rate
        return learning_rate * (self.updates + 1) / recipe.warmup

    async def _submit(
        self,
        client: Any,
        part: list[tinker.Datum],
        rate: float,
        loss: str,
        loss_config: Mapping[str, float] | None,
    ) -> tuple[Awaitable[list[list[float]]], Awaitable[None]]:
        """Submit one substep's pass and optimizer step, before the one before is read."""
        forward = await client.forward_backward(part, loss, loss_config)
        return forward, await client.optim_step(self.ADAM, rate)

    # ------------------------------------------------------------------ the run

    async def train(self) -> None:
        """Open the policies, take `[recipe] steps` steps, then evaluate and save `final`."""
        if self.resumed is not None and self.resumed.get("tag") == FINAL:
            logger.info(
                "run %s finished at step %d: nothing is left to run",
                self.directory.name,
                self.steps,
            )
            return
        self._bind()
        async with self._session if self._session is not None else nullcontext():
            while self.steps < self.recipe.steps:
                await self._advance()
            if self.generator.test and self.evaluated != self.steps:
                await self.evaluate()
            if self.updates:
                await self.checkpoint(FINAL, keep=True)

    async def _advance(self) -> None:
        self._seen, self._update, self.metrics = [], None, {}
        await self.step()
        self.logs.step(self.steps, self._seen, self._update, self.metrics)
        self.steps += 1
        every = self.config.checkpoints.every
        if self._update is not None and self.updates % every == 0:
            await self.checkpoint(f"step-{self.steps}")
        every = self.recipe.evaluate_every
        if every and self.steps % every == 0:
            await self.evaluate()

    async def evaluate(self) -> Evaluation | None:
        """Roll the agent out on the test set, and log an `eval` row with its mean reward."""
        tasks = list(self.generator.test)
        if not tasks:
            return None
        data = self.config.data
        found: list[Group] = []
        for start in range(0, len(tasks), data.batch_size):
            batch = tasks[start : start + data.batch_size]
            attempts = data.test_group_size or data.group_size
            found += await self.rollout(batch, attempts=attempts, purpose="evaluation")
        measured = [one for group in found for one in group.measured]
        rollouts = sum(len(group.trials) for group in found)
        result = Evaluation(
            step=self.steps,
            tasks=len(found),
            rollouts=rollouts,
            graded=len(measured),
            masked=rollouts - len(measured),
            reward_mean=fmean(one.reward or 0.0 for one in measured) if measured else None,
        )
        self.logs.log("eval", **vars(result))
        self.evaluated = self.steps
        return result

    async def checkpoint(self, tag: str, *, keep: bool = False) -> None:
        """Save the agent's training state as `tag`, which a resume loads, and log its row."""
        session = self._session
        if session is None:
            raise ValueError(
                f"{self._agent.provider} serves the agent: there are no weights to save"
            )
        hours = None if keep else self.config.checkpoints.ttl_hours
        ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
        state, sampler = await session.save(tag, ttl_seconds=ttl, sampler=keep)
        self.logs.checkpoint(
            {
                "tag": tag,
                "state_path": state,
                "sampler_path": sampler,
                "ttl_hours": hours,
                "step": self.steps,
                "updates": self.updates,
                "drawn": self.drawn,
                "modules": self.modules.digest if self.modules is not None else None,
                "harness": [harnesses.harness(one).name for one in self.config.agent.harness],
            }
        )
        logger.info("checkpoint %s: %s", tag, state)

    def keep(self, modules: Candidate, name: str) -> Path:
        """Keep `modules` in the run as `modules/<name>/`, for a later run to carry."""
        into = self.directory / record.MODULES / name
        keep(modules, into)
        return into

    def restore(self, row: Mapping[str, Any]) -> None:
        """Go on from a checkpoint row; `train` loads its weights and optimizer state."""
        self.resumed = row
        self.steps, self.updates, self.drawn = row["step"], row["updates"], row["drawn"]
        digest = row.get("modules")
        if digest and (self.modules is None or self.modules.digest != digest):
            self.modules = self._kept(digest)

    def _kept(self, digest: str) -> Candidate:
        """Return the modules the run kept whose digest is `digest`."""
        for path in sorted(p for p in (self.directory / record.MODULES).rglob("*") if p.is_dir()):
            try:
                found = seed(path)
            except ValueError:
                continue  # not a modules directory, such as one module's own folder
            if found.digest == digest:
                return found
        raise ValueError(
            f"the checkpoint carried modules {digest}, which the run did not keep: a recipe "
            "that carries new modules keeps them with `keep`, so that a resume finds them"
        )

    # ------------------------------------------------------------------ plumbing

    def _bind(self) -> None:
        """Give each policy the trainer holds its backend: `Session`, `Frozen` or `Provider`."""
        agent, costs = self._agent, self.logs.costs
        session = (
            Session(self.config, self.directory, costs, resumed=self.resumed)
            if agent.provider == TINKER
            else None
        )
        backend: Session | Frozen | Provider
        for one in self._held():
            if one is agent and session is not None:
                backend = session
            elif one.provider != TINKER:
                backend = Provider(one.load, one.provider, costs)
            elif session is None:
                raise ValueError(
                    f"Policy({one.load!r}) opens in the agent's Tinker session, and the agent "
                    f"is {agent.provider}'s model, which has none"
                )
            else:
                backend = Frozen(one.load, one.model, costs)
                session.frozen.append(backend)
            one.bind(backend, reader=self.config.agent.model, limit=lambda: agent.budget)
        self._session = session

    def _held(self) -> list[Policy]:
        """Return every policy the trainer holds, the agent first."""
        found = [self._agent]
        for value in vars(self).values():
            inner = value.values() if isinstance(value, dict) else value
            for one in inner if isinstance(value, dict | list | tuple) else (value,):
                if isinstance(one, Policy) and all(one is not seen for seen in found):
                    found.append(one)
        return found


def split(
    tasks: list[Path], attempts: int, names: list[str], *, mix: bool, turn: int
) -> list[tuple[str, list[Path], int]]:
    """Return the jobs a rollout runs, one per harness, as (harness, tasks, attempts)."""
    count = len(names)
    if mix:
        shares = [attempts // count + (at < attempts % count) for at in range(count)]
        jobs = [(one, tasks, shares[at]) for at, one in enumerate(names)]
    else:
        turns = [(place + turn) % count for place in range(len(tasks))]
        jobs = [
            (one, [task for task, to in zip(tasks, turns, strict=True) if to == at], attempts)
            for at, one in enumerate(names)
        ]
    return [job for job in jobs if job[1] and job[2]]


def carried(config: Blueprint, directory: Path) -> Candidate | None:
    """Return the modules every job carries, from the run's copy or `[agent] modules`."""
    kept = directory / record.MODULES / CARRIED
    if kept.is_dir():
        return seed(kept)
    if not config.agent.modules:
        return None
    found = seed(config.home / config.agent.modules)
    keep(found, kept)
    return found
