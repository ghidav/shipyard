from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Iterator, Mapping
from collections.abc import Sequence as Values
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from statistics import fmean
from types import TracebackType
from typing import Any, Self

import tinker
import torch
from pydantic import ValidationError
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.utils.misc_utils import split_list

from shipyard.config import Blueprint, Recipe
from shipyard.groups import Group, grouped, merged
from shipyard.jobs import harnesses
from shipyard.jobs.engine import HarborEngine, Job
from shipyard.jobs.modules import Modules, keep
from shipyard.jobs.secrets import Pool, Secrets
from shipyard.models.agent import Frozen, Provider, Session
from shipyard.models.clients import TRAINED
from shipyard.models.clients.tinker import TINKER, datum
from shipyard.models.clients.together import FROZEN, TOGETHER
from shipyard.policy import Policy
from shipyard.run import record
from shipyard.run.data import Generator
from shipyard.run.logs import AGENT, Logs, breakdown
from shipyard.state import State

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
    loss: str
    #: How long the update took.
    update_seconds: float
    kl_v1: float
    kl_v2: float
    entropy: float


@dataclass(frozen=True)
class Evaluation:
    """What one `evaluate` rolled out and measured; its step's row holds it as `eval_` fields."""

    step: int
    tasks: int
    trials: int
    graded: int
    masked: int
    #: Mean reward over the graded trials; None when none were graded.
    reward_mean: float | None
    #: Each harness's and each renderer's mean reward and sampled tokens, when there are several.
    breakdown: dict[str, float] = field(default_factory=dict)


class Trainer:
    """A run: the four primitives, and the `step` a recipe writes from them."""

    #: Adam settings from the cookbook's `train_step`. Its `learning_rate` is not read: each
    #: update takes `[recipe] learning_rate`, warmed up over `[recipe] warmup` updates.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
    #: The `[recipe]` keys the trainer reads. A recipe with keys of its own declares them in a
    #: subclass, `class Recipe(Trainer.Recipe)`, and its run.toml sets every one.
    Recipe = Recipe
    #: What the run has done so far, which a checkpoint saves. A recipe with state of its own
    #: declares it in a subclass, `class State(Trainer.State)`, each field with a default.
    State = State

    def __init__(self, config: Blueprint, directory: Path) -> None:
        if config.rollout.renderer and config.agent.provider not in TRAINED:
            raise ValueError(
                "[rollout] renderer: the proxy renders the calls of a model the run trains, and "
                f"{config.agent.provider} serves {config.agent.model} to its harnesses itself"
            )
        self.config = config
        self.recipe = config.recipe
        #: Run directory: its rows, checkpoints, costs and modules, written by this run alone.
        self.directory = Path(directory)
        self.generator = Generator.of(config)
        #: What the run has done so far, which a checkpoint saves and a resume goes on from.
        self.state = self.State(modules=carried(config, self.directory))
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
        self.logs = Logs(config, self.directory, step=lambda: self._current)
        self._agent = Policy(config.agent.model, provider=config.agent.provider)
        #: The agent on Tinker, made and opened when `train` starts; None for a provider's.
        self._session: Session | None = None
        #: The recipe's own numbers for this step's row, such as a KL it measured.
        self.metrics: dict[str, Any] = {}
        #: The step the rows written now belong to, counted from 1: the step under way, and
        #: after it, until the next starts, the step just taken.
        self._current = 0
        #: The training state of the checkpoint the run went on from, which the session opens.
        self._weights: str | None = None
        self._stream: Iterator[list[Path]] | None = None
        self._seen: list[Group] = []
        self._update: Update | None = None
        #: The evaluation that belongs to the step under way, for its row.
        self._evaluation: Evaluation | None = None

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
            self._stream = self.generator.batches(self.state.drawn)
        batch = next(self._stream)
        self.state.drawn += 1
        return batch

    async def rollout(
        self,
        tasks: Values[Path],
        *,
        attempts: int | None = None,
        modules: Modules | None = None,
        harness: str | None = None,
        renderer: str | None = None,
        purpose: str = "training",
    ) -> list[Group]:
        """Run each task in groups of `attempts`, per renderer and, unless mixed, per harness."""
        cfg, tasks = self.config, list(tasks)
        carried = self.state.modules if modules is None else modules
        named = cfg.rollout.renderer
        if renderer is not None and renderer not in named:
            raise ValueError(
                f"rollout(renderer={renderer!r}): the proxy renders with "
                f"{', '.join(named) or 'the model default alone'}, from [rollout] renderer"
            )
        jobs = split(
            tasks,
            attempts or cfg.data.group_size,
            [harness] if harness else cfg.agent.harness,
            [renderer] if renderer else list(named) or [None],
            mix=cfg.agent.mix_harnesses,
        )
        parts = await asyncio.gather(
            *(
                self._job(one, rendered, part, count, carried, purpose)
                for one, rendered, part, count in jobs
            )
        )
        first = {task: at for at, task in reversed(list(enumerate(tasks)))}
        mixed = merged([g for part in parts for g in part], mix=cfg.agent.mix_harnesses)
        found = sorted(mixed, key=lambda g: first[g.task])
        if purpose == "training":
            self._seen += found
        return found

    async def _job(
        self,
        harness: str,
        renderer: str | None,
        tasks: list[Path],
        attempts: int,
        modules: Modules | None,
        purpose: str,
    ) -> list[Group]:
        """Run and log a judged job of `harness` on `tasks` at `renderer`; return its groups."""
        endpoint = self._agent.endpoint
        job = Job(
            self.logs.reserve(),
            purpose=purpose,
            step=self._current,
            modules=modules,
            harness=harnesses.harness(harness),
            renderer=renderer,
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
        self.state.updates += 1
        await session.serve(f"sample-{self.state.updates}")
        drift = compute_kl_sample_train(datums, trained)
        self._update = Update(
            sequences=len(flat),
            train_tokens=tokens,
            substeps=count,
            learning_rate=rate,
            loss=loss,
            update_seconds=round(time.monotonic() - began, 3),
            kl_v1=drift["optim/kl_sample_train_v1"],
            kl_v2=drift["optim/kl_sample_train_v2"],
            entropy=drift["optim/entropy"],
        )
        return self._update

    def _rate(self, learning_rate: float) -> float:
        """Return the next update's learning rate, warmed up over `[recipe] warmup` updates."""
        recipe = self.recipe
        restored = self.config.agent.from_checkpoint and recipe.restore_optimizer
        if restored or self.state.updates >= recipe.warmup:
            return learning_rate
        return learning_rate * (self.state.updates + 1) / recipe.warmup

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
        """Open the policies, take `[recipe] steps` steps, the last evaluated, then save `final`."""
        self._current = self.state.steps
        self._bind()
        async with self._session if self._session is not None else nullcontext():
            while self.state.steps < self.recipe.steps:
                await self._advance()
            await self.checkpoint(FINAL, keep=True)

    async def _advance(self) -> None:
        """Take a step, evaluate and checkpoint it on schedule, and write its row."""
        self._seen, self._update, self._evaluation, self.metrics = [], None, None, {}
        self._current = self.state.steps + 1
        await self.step()
        self.state.steps += 1
        every, steps = self.recipe.evaluate_every, self.state.steps
        due = (every and steps % every == 0) or steps == self.recipe.steps
        if due and self.generator.test and self.state.evaluated != self._current:
            await self.evaluate()
        self.logs.step(self._current, self._seen, self._update, self._evaluation, self.metrics)
        if steps % self.config.checkpoints.every == 0:
            await self.checkpoint(f"step-{steps}")

    async def evaluate(self) -> Evaluation | None:
        """Roll the agent out on every test task at each harness and renderer, unmixed."""
        tasks = list(self.generator.test)
        if not tasks:
            return None
        data = self.config.data
        attempts = data.test_group_size or data.group_size
        # Each harness and renderer pinned, so no harnesses are mixed; named only when there are
        # several, so that a recipe's own `rollout`, which may take neither, is called as before.
        named = self.config.agent.harness
        pinned = [
            {**({"harness": one} if len(named) > 1 else {}), **({"renderer": it} if it else {})}
            for one in named
            for it in self.config.rollout.renderer or [None]
        ]
        found: list[Group] = []
        for start in range(0, len(tasks), data.batch_size):
            batch = tasks[start : start + data.batch_size]
            parts = await asyncio.gather(
                *(
                    self.rollout(batch, attempts=attempts, purpose="evaluation", **one)
                    for one in pinned
                )
            )
            found += [group for part in parts for group in part]
        measured = [one for group in found for one in group.measured]
        trials = sum(len(group.trials) for group in found)
        self._evaluation = Evaluation(
            step=self._current,
            tasks=len({group.task for group in found}),
            trials=trials,
            graded=len(measured),
            masked=trials - len(measured),
            reward_mean=fmean(one.reward or 0.0 for one in measured) if measured else None,
            breakdown=breakdown(found),
        )
        self.state.evaluated = self._current
        return self._evaluation

    async def checkpoint(self, tag: str, *, keep: bool = False) -> None:
        """Save the run's state as `tag`, which a resume goes on from; `keep` keeps the weights."""
        session, weights, saved = self._session, None, {"sampler": None}
        if session is not None and self.state.updates:
            hours = None if keep else self.config.checkpoints.ttl_hours
            ttl = None if hours is None else int(hours * SECONDS_PER_HOUR)
            weights, sampler = await session.save(tag, ttl_seconds=ttl, sampler=keep)
            saved = {"sampler": sampler, "ttl_hours": hours}
        state = f"{record.STATES}/step-{self.state.steps}.json"
        record.write_json(self.directory / state, self.state.model_dump(mode="json"))
        self.logs.checkpoint(
            {
                "tag": tag,
                "step": self.state.steps,
                "state": state,
                "weights": weights,
                **saved,
            }
        )
        logger.info("checkpoint %s: %s", tag, state)

    def keep(self, modules: Modules, name: str) -> Path:
        """Keep `modules` in the run as `modules/<name>/`, for a later run to carry."""
        into = self.directory / record.MODULES / name
        keep(modules, into)
        return into

    def restore(self, origin: Path, row: Mapping[str, Any]) -> None:
        """Restore checkpoint `row` of `origin`: its state now, its weights once `train` opens."""
        tag, state = row["tag"], row.get("state")
        self._weights = row.get("weights")
        saved = record.read_json(Path(origin) / state) if state else {}
        if not saved:
            raise ValueError(f"checkpoint {tag}: its state is missing or unreadable ({state})")
        named = f"{type(self).__name__}.State"
        dropped = sorted(set(saved) - set(self.State.model_fields))
        if dropped:
            logger.warning("checkpoint %s: %s no longer holds %s", tag, named, ", ".join(dropped))
        try:
            self.state = self.State.model_validate(saved)
        except ValidationError as refused:
            raise ValueError(
                f"checkpoint {tag}: its state does not fit {named}: {refused}"
            ) from None

    # ------------------------------------------------------------------ plumbing

    def _bind(self) -> None:
        """Give each policy the trainer holds its backend: `Session`, `Frozen` or `Provider`."""
        agent, spend = self._agent, self.logs.spend
        session = (
            Session(self.config, self.directory, partial(spend, AGENT), weights=self._weights)
            if agent.provider in TRAINED
            else None
        )
        backend: Session | Frozen | Provider
        for role, one in self._held():
            if one is agent and session is not None:
                backend = session
            elif one.provider != TINKER:
                backend = Provider(one.load, one.provider, partial(spend, role))
            elif session is None:
                raise ValueError(
                    f"Policy({one.load!r}) opens in the agent's Tinker session, and the agent "
                    f"is {agent.provider}'s model, which has none"
                )
            elif agent.provider == TOGETHER:
                raise ValueError(f"Policy({one.load!r}): {FROZEN}")
            else:
                backend = Frozen(one.load, one.model, partial(spend, role))
                session.frozen.append(backend)
            one.bind(backend, reader=self.config.agent.model, limit=lambda: agent.budget)
        self._session = session

    def _held(self) -> list[tuple[str, Policy]]:
        """Return every policy the trainer holds, the agent first, each with its spend's role."""
        found = [(AGENT, self._agent)]
        for name, value in vars(self).items():
            if isinstance(value, dict):
                inner = [(f"{name}[{key}]", one) for key, one in value.items()]
            elif isinstance(value, list | tuple):
                inner = [(f"{name}[{at}]", one) for at, one in enumerate(value)]
            else:
                inner = [(name, value)]
            for role, one in inner:
                if isinstance(one, Policy) and all(one is not seen for _, seen in found):
                    found.append((role, one))
        return found


def split(
    tasks: list[Path],
    attempts: int,
    harnesses: list[str],
    renderers: list[str | None],
    *,
    mix: bool,
) -> list[tuple[str, str | None, list[Path], int]]:
    """Return one job per harness and renderer: (harness, renderer, tasks, attempts)."""
    count = len(harnesses)
    shares = (
        [attempts // count + (at < attempts % count) for at in range(count)]
        if mix
        else [attempts] * count
    )
    return [
        (one, renderer, tasks, share)
        for one, share in zip(harnesses, shares, strict=True)
        if share and tasks
        for renderer in renderers
    ]


def carried(config: Blueprint, directory: Path) -> Modules | None:
    """Return the modules every job carries, from the run's copy or `[agent] modules`."""
    kept = directory / record.MODULES / CARRIED
    if kept.is_dir():
        return Modules.read(kept)
    if not config.agent.modules:
        return None
    found = Modules.read(config.home / config.agent.modules)
    keep(found, kept)
    return found
