from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from collections.abc import Sequence as Values
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import tinker
from pydantic import ValidationError
from tinker_cookbook.utils.misc_utils import split_list

from shipyard.config import Blueprint, Recipe
from shipyard.groups import Group, Sequence
from shipyard.jobs.engine import EVALUATION, HarborEngine
from shipyard.jobs.harnesses import resolved
from shipyard.jobs.modules import Modules, carried, keep
from shipyard.jobs.secrets import Pool, Secrets
from shipyard.model import Model
from shipyard.models.backends import Session, connect
from shipyard.models.clients.tinker import check_loss
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.data import Generator
from shipyard.run.logs import AGENT, Logs, breakdown, tally
from shipyard.state import State

logger = logging.getLogger(__name__)

#: The checkpoint a run saves when it ends: the weights its updates moved to, kept for good.
FINAL = "final"


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
    #: update takes the agent's, `[recipe] learning_rate`, warmed up over `[recipe] warmup`.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
    #: The `[recipe]` keys the trainer reads. A recipe with keys of its own declares them in a
    #: subclass, `class Recipe(Trainer.Recipe)`, and its run.toml sets every one.
    Recipe = Recipe
    #: What the run has done so far, which a checkpoint saves. A recipe with state of its own
    #: declares it in a subclass, `class State(Trainer.State)`, each field with a default.
    State = State

    def __init__(self, config: Blueprint, directory: Path) -> None:
        # The agent trains unless a recipe that never updates sets `self.agent.training`
        # False in its __init__; the run checks it can when it starts.
        agent = Model(config.agent.model, provider=config.agent.provider, train=True)
        # The proxy serves a model on Tinker or Together: one a run could train.
        if config.rollout.renderer and not agent.trainable:
            raise ValueError(
                "[rollout] renderer: the proxy renders the calls of a model on Tinker or "
                f"Together, and {config.agent.provider} serves {config.agent.model} to its "
                "harnesses itself"
            )
        self.config = config
        self.recipe = config.recipe
        #: Run directory: its rows, checkpoints, costs and modules, written by this run alone.
        self.directory = Path(directory)
        self.generator = Generator.of(config)
        #: What the run has done so far, which a checkpoint saves and a resume goes on from.
        self.state = self.State(modules=carried(config, self.directory))
        self.directory.mkdir(parents=True, exist_ok=True)
        #: The run's logs: its job, step and request rows, its usage ledger and its costs.
        self.logs = Logs(config, self.directory, step=lambda: self._current)
        table = config.rollout.secrets
        #: One engine, so every job shares one `[rollout] concurrency` limit.
        self.engine = HarborEngine(
            config.rollout,
            logs=self.logs,
            env=config.agent.env,
            kwargs=config.agent.kwargs,
            secrets=Pool.of(table) if table else Secrets(),
            mix=config.agent.mix_harnesses,
            grade_timeouts=self.recipe.grade_timeouts,
        )
        self._agent = agent
        #: The agent on Tinker or Together, made and opened when `run` starts; None for a
        #: provider's.
        self._session: Session | None = None
        #: The recipe's own numbers for this step's row, such as a KL it measured.
        self.metrics: dict[str, Any] = {}
        #: The step the rows written now belong to, counted from 1: the step under way, and
        #: after it, until the next starts, the step just taken.
        self._current = 0
        #: The training state of the checkpoint the run went on from, which the session opens.
        self._training_state: str | None = None
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
    def agent(self) -> Model:
        """The agent: the run's own model, made from `[agent]`. The run trains it, unless a
        recipe that never updates sets `self.agent.training = False` in its __init__."""
        return self._agent

    # ------------------------------------------------------------------ what a recipe writes

    async def step(self) -> None:
        """Draw a batch, roll the agent out on it, score its groups, and update."""
        groups = await self.rollout(self.draw())
        await self.score(groups)
        await self.update(groups)

    async def score(self, groups: Values[Group]) -> None:
        """Give each trial its reward minus its group's mean, skipping groups of equal rewards."""
        for group in groups:
            if group.reward_variance == 0:
                continue
            for trial, advantage in zip(group.graded, group.centered_rewards, strict=True):
                trial.set_advantage(advantage)

    # ------------------------------------------------------------------ the other primitives

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
        model: Model | None = None,
        env: Mapping[str, str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> list[Group]:
        """Run each task in groups of `attempts`, per renderer and, unless mixed, per harness.
        The agent runs by default, carrying the run's modules, through `[agent]`'s harnesses;
        `model` runs another model the trainer holds, through `harness` with `env` and
        `kwargs`, carrying `modules` alone. Only the agent's training rollouts count in the
        step's row."""
        if model is not None and model is not self._agent:
            return await self._rollout(
                model, list(tasks), attempts, modules, harness, renderer, purpose, env, kwargs
            )
        named = self.config.rollout.renderer
        if renderer is not None and renderer not in named:
            raise ValueError(
                f"rollout(renderer={renderer!r}): the proxy renders with "
                f"{', '.join(named) or 'the model default alone'}, from [rollout] renderer"
            )
        harnesses = self.config.agent.harness
        if harness is not None:
            try:
                harnesses = [resolved(harness, harnesses)]
            except ValueError as refused:
                raise ValueError(f"rollout(harness=...): {refused}") from None
        found = await self.engine.rollout(
            list(tasks),
            attempts=attempts or self.config.data.group_size,
            harnesses=harnesses,
            renderers=[renderer] if renderer else list(named) or [None],
            endpoint=self._agent.endpoint,
            modules=self.state.modules if modules is None else modules,
            purpose=purpose,
            step=self._current,
            env=env,
            kwargs=kwargs,
        )
        if purpose == "training":
            self._seen += found
        return found

    async def _rollout(
        self,
        model: Model,
        tasks: list[Path],
        attempts: int | None,
        modules: Modules | None,
        harness: str | None,
        renderer: str | None,
        purpose: str,
        env: Mapping[str, str] | None,
        kwargs: Mapping[str, Any] | None,
    ) -> list[Group]:
        """Run `model`, a model the trainer holds besides the agent, as `rollout` says."""
        role = next((named for named, one in self._held() if one is model), None)
        if role is None:
            raise ValueError(
                f"rollout(model={model!r}): keep the model on the trainer, as self.<name>"
            )
        if harness is None:
            raise ValueError(f"rollout(model={role}): name the harness it runs through")
        if renderer is not None:
            raise ValueError(f"rollout(model={role}): the proxy renders the agent's calls alone")
        if model.endpoint is None:
            raise ValueError(
                f"rollout(model={role}): {model!r} has nothing a harness could dial: it is "
                "not connected yet"
            )
        return await self.engine.rollout(
            tasks,
            attempts=attempts or self.config.data.group_size,
            harnesses=[harness],
            renderers=[None],
            endpoint=model.endpoint,
            modules=modules,
            purpose=purpose,
            step=self._current,
            role=role,
            env=dict(env or {}),
            kwargs=dict(kwargs or {}),
        )

    async def update(
        self,
        groups: Values[Group],
        *,
        loss: str = "importance_sampling",
        loss_config: Mapping[str, float] | None = None,
    ) -> Update | None:
        """Train the agent on every sequence with advantages, and serve its new weights. Raise
        `ValueError` for a loss Tinker does not have, or a setting it does not take, before
        anything is scored or sent."""
        check_loss(loss, loss_config)
        agent, session, learning_rate = self._agent, self._session, self.recipe.learning_rate
        if not agent.trainable:
            raise ValueError(
                f"update trains the agent, and {agent.provider} serves {agent.name}: there "
                "are no weights to train"
            )
        if not agent.training:
            raise ValueError(
                "update trains the agent, and this recipe set `self.agent.training = False`"
            )
        if session is None or learning_rate is None:
            raise RuntimeError("update trains the agent, which opens when the run starts")
        kept = [group.scored for group in groups if group.is_scored]
        if not kept:
            return None
        began, recipe = time.monotonic(), self.recipe
        flat = [one for part in kept for one in part]
        unrecorded = [one for one in flat if one.sampled_logprobs is None]
        # Each ratio is taken against the sampler's logprobs, else the current weights' own.
        logprobs: dict[Sequence, Values[float]] = {
            one: one.sampled_logprobs for one in flat if one.sampled_logprobs is not None
        }
        for one, scored in zip(unrecorded, await agent.logprobs(unrecorded), strict=True):
            logprobs[one] = scored.tolist()  # type: ignore[union-attr]
        # The papers' mini-batches are sets of prompts, so a group is not split across parts.
        count = min(recipe.substeps, len(kept))
        parts = [[one for part in piece for one in part] for piece in split_list(kept, count)]
        rate = self._rate(learning_rate)
        trained = await session.fit(
            parts, logprobs=logprobs, adam=self.ADAM, rate=rate, loss=loss, loss_config=loss_config
        )
        made = self.state.updates[AGENT] = self.state.updates.get(AGENT, 0) + 1
        await session.serve(f"sample-{made}")
        self._update = Update(
            sequences=len(flat),
            train_tokens=trained.tokens,
            substeps=count,
            learning_rate=rate,
            loss=loss,
            update_seconds=round(time.monotonic() - began, 3),
            kl_v1=trained.kl_v1,
            kl_v2=trained.kl_v2,
            entropy=trained.entropy,
        )
        return self._update

    def _rate(self, learning_rate: float) -> float:
        """Return the agent's next learning rate, warmed up over `[recipe] warmup` updates."""
        recipe, made = self.recipe, self.state.updates.get(AGENT, 0)
        restored = self.config.agent.from_checkpoint and recipe.restore_optimizer
        if restored or made >= recipe.warmup:
            return learning_rate
        return learning_rate * (made + 1) / recipe.warmup

    # ------------------------------------------------------------------ the run

    async def run(self) -> None:
        """Open the models, take `[recipe] steps` steps, the last evaluated, then save `final`.
        Raise `ValueError`, before anything opens, for a model the run cannot train or open."""
        if self._agent.training and self.recipe.learning_rate is None:
            raise ValueError(
                "[recipe] learning_rate: the agent trains, and the blueprint sets no rate: set "
                "it, or, in a recipe that never updates, set `self.agent.training = False`"
            )
        self._current = self.state.steps
        connected = connect(
            self.config,
            self.directory,
            self._held(),
            self.logs.spend,
            training_state=self._training_state,
        )
        self._session = connected.session
        async with connected:
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
        found: list[Group] = []
        for start in range(0, len(tasks), data.batch_size):
            batch = tasks[start : start + data.batch_size]
            found += await self.rollout(batch, attempts=attempts, purpose=EVALUATION)
        self._evaluation = Evaluation(
            step=self._current,
            tasks=len({group.task for group in found}),
            **tally(found),
            breakdown=breakdown(found),
        )
        self.state.evaluated = self._current
        return self._evaluation

    async def checkpoint(self, tag: str, *, keep: bool = False) -> None:
        """Save the run's state as `tag`, which a resume goes on from; `keep` keeps the weights.
        Its row names each trained model's training state by role, once an update has moved it."""
        session = self._session
        training_state: dict[str, str] = {}
        sampler_weights: dict[str, str] = {}
        hours: float | None = None
        if session is not None and self.state.updates.get(AGENT):
            made = await session.save(tag, keep=keep)
            training_state = {AGENT: made.training_state}
            if made.sampler_weights is not None:
                sampler_weights = {AGENT: made.sampler_weights}
            hours = made.ttl_hours
        saved = Checkpoint.save(
            self.directory,
            tag,
            self.state.steps,
            self.state.model_dump(mode="json"),
            training_state=training_state,
            sampler_weights=sampler_weights,
            ttl_hours=hours,
        )
        logger.info("checkpoint %s: %s", tag, saved.state)

    def keep(self, modules: Modules, name: str) -> Path:
        """Keep `modules` in the run as `modules/<name>/`, for a later run to carry."""
        into = self.directory / files.MODULES / name
        keep(modules, into)
        return into

    def restore(self, origin: Path, checkpoint: Checkpoint) -> None:
        """Restore `checkpoint` of `origin`: its state now, its training state once `run` opens."""
        tag = checkpoint.tag
        self._training_state = checkpoint.training_state.get(AGENT)
        saved = checkpoint.saved_state(origin)
        if not saved:
            raise ValueError(
                f"checkpoint {tag}: its state is missing or unreadable ({checkpoint.state})"
            )
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

    def _held(self) -> list[tuple[str, Model]]:
        """Return every model the trainer holds, the agent first, each with its spend's role."""
        found = [(AGENT, self._agent)]
        for name, value in vars(self).items():
            if isinstance(value, dict):
                inner = [(f"{name}[{key}]", one) for key, one in value.items()]
            elif isinstance(value, list | tuple):
                inner = [(f"{name}[{at}]", one) for at, one in enumerate(value)]
            else:
                inner = [(name, value)]
            for role, one in inner:
                if isinstance(one, Model) and all(one is not seen for _, seen in found):
                    found.append((role, one))
        return found
