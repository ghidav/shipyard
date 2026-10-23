from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator, Mapping
from collections.abc import Sequence as Values
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import tinker
from pydantic import ValidationError
from tinker_cookbook.utils.misc_utils import split_list

from shipyard.config import Config, Recipe
from shipyard.groups import Group, Sequence
from shipyard.jobs.engine import HarborEngine
from shipyard.jobs.harnesses import Harness
from shipyard.jobs.hooks import hooks_of
from shipyard.jobs.modules import Modules, carried, keep
from shipyard.jobs.secrets import Pool, Secrets
from shipyard.model import Model, Models
from shipyard.models.backends import Connected, Session, connect
from shipyard.models.clients.tinker import check_loss
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.data import Generator
from shipyard.run.logs import Logs, tally
from shipyard.state import State

logger = logging.getLogger(__name__)

#: The checkpoint a run saves when it ends: the weights its updates moved to, kept for good.
FINAL = "final"
#: What a rollout is for unless it says: a step's training, whose groups the step's row counts.
TRAINING = "training"
#: What the rollouts `evaluate` makes are for: the test set, whose row is its own.
EVALUATION = "evaluation"
#: The step whose weights the evaluation under way measures, set in the task the evaluation runs
#: in, which runs beside the next step's update: its rollouts are the test set's and filed under
#: that step, while the step under way trains. None outside an evaluation.
EVALUATING: ContextVar[int | None] = ContextVar("evaluating", default=None)


@dataclass(frozen=True)
class Update:
    """What one `update` trained, and what its training pass read."""

    #: The role of the model it trained.
    role: str
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
    """What one evaluation rolled out and measured, written as a row of its own: `eval_` fields
    at the step whose weights it measured."""

    #: The step after which the weights it measured were served; 0 for the model the run starts
    #: from.
    step: int
    tasks: int
    trials: int
    graded: int
    masked: int
    #: Mean reward over the graded trials; None when none were graded.
    reward_mean: float | None


class Trainer:
    """A run: the primitives a recipe writes its `step` from, and the loop that takes the steps,
    evaluates them and checkpoints them."""

    #: Adam settings from the cookbook's `train_step`. Its `learning_rate` is not read: each
    #: update takes the one the recipe gives it, else its model's `learning_rate`.
    ADAM = tinker.AdamParams(beta1=0.9, beta2=0.95, eps=1e-8)
    #: The `[recipe]` keys the trainer reads. A recipe declares its own in a subclass,
    #: `class Recipe(Trainer.Recipe)`, such as its harnesses and group size, and its run.toml
    #: sets every one.
    Recipe = Recipe
    #: What the run has done so far, which a checkpoint saves. A recipe with state of its own
    #: declares it in a subclass, `class State(Trainer.State)`, each field with a default.
    State = State

    def __init__(self, config: Config, directory: Path) -> None:
        """Raise `ValueError` for a model the run cannot hold, such as renderers of a model a
        provider serves."""
        self.config = config
        self.recipe = config.recipe
        #: Run directory: its rows, checkpoints, costs and modules, written by this run alone.
        self.directory = Path(directory)
        #: The models run.toml names, by role. A recipe reads the ones it uses in its
        #: `__init__`, as `self.models["agent"]`, and sets `training = True` on the ones it
        #: trains; the run refuses a model it never read.
        self.models = Models(config.models)
        self.generator = Generator.of(config)
        for task in (*self.generator.train, *self.generator.test):
            hooks_of(task)  # a task whose hooks.py is wrong stops the run before anything opens
        #: What the run has done so far, which a checkpoint saves and a resume goes on from.
        self.state = self.State(modules=carried(config, self.directory))
        self.directory.mkdir(parents=True, exist_ok=True)
        #: The run's logs: its job, step and request rows, its usage ledger and its costs.
        self.logs = Logs(config, self.directory, step=lambda: self._current)
        table = config.rollout.secrets
        #: One engine, so every rollout shares one `[rollout] concurrency` limit.
        self.engine = HarborEngine(
            config.rollout,
            logs=self.logs,
            secrets=Pool.of(table) if table else Secrets(),
            grade_timeouts=self.recipe.grade_timeouts,
        )
        #: The recipe's own numbers for this step's row, such as a KL it measured.
        self.metrics: dict[str, Any] = {}
        #: The step the rows written now belong to, counted from 1: the step under way, and
        #: after it, until the next starts, the step just taken.
        self._current = 0
        #: The training states of the checkpoint the run went on from, by role, which each
        #: trained model's session opens on.
        self._training_states: dict[str, str] = {}
        self._stream: Iterator[list[Path]] | None = None
        self._seen: list[Group] = []
        #: The updates of the step under way, by the role of the model each trained.
        self._updates: dict[str, Update] = {}
        #: The step whose weights wait to be evaluated. The samplers serve them until the next
        #: update publishes, so that update starts the evaluation as it trains and publishes once
        #: it has finished. None when none waits.
        self._pending: int | None = None
        #: The evaluation under way beside an update.
        self._evaluating: asyncio.Task[Evaluation] | None = None

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

    # ------------------------------------------------------------------ what a recipe writes

    async def step(self) -> None:
        """Take one step of the recipe's algorithm, from `draw`, `rollout` and `update`."""
        raise NotImplementedError(f"{type(self).__name__} writes no step")

    async def evaluate(self, tasks: list[Path]) -> list[Group]:
        """Roll the test set's `tasks` out as the recipe measures them, and return their groups.
        The run calls it on schedule and puts what the groups came to in the step's row."""
        raise NotImplementedError(f"{type(self).__name__} writes no evaluate")

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
        model: Model,
        harness: Harness | str,
        attempts: int,
        modules: Modules | None,
        renderer: str | None = None,
        purpose: str | None = None,
    ) -> list[Group]:
        """Run `model` on each task `attempts` times through `harness`, carrying `modules`
        (`self.state.modules` for the run's, None for none), as one job, and return one group
        per task, in the order of `tasks`. Its calls are rendered with `renderer`, one of the
        model's renderers, which a model with several must be given. A rollout `evaluate` makes
        is the test set's, filed under the step whose weights it measures, and any other's is
        `training` unless `purpose` names another; only training rollouts count in the step's
        row. Raise `ValueError` before any trial runs for a rollout no trial could run."""
        if all(one is not model for one in self.models.all):
            raise ValueError(f"rollout(model={model!r}): roll out a model of self.models")
        if model.endpoint is None:
            raise ValueError(
                f"rollout(model={model.role}): {model!r} has nothing a harness could dial: it "
                "is not connected yet"
            )
        evaluating = EVALUATING.get()
        purpose = purpose or (TRAINING if evaluating is None else EVALUATION)
        found = await self.engine.rollout(
            list(tasks),
            attempts=attempts,
            harness=Harness.model_validate(harness),
            endpoint=model.endpoint,
            renderer=_rendered(model, renderer),
            modules=modules,
            purpose=purpose,
            step=self._current if evaluating is None else evaluating,
            role=model.role,
        )
        if purpose == TRAINING:
            self._seen += found
        return found

    async def update(
        self,
        model: Model,
        groups: Values[Group],
        *,
        substeps: int = 1,
        learning_rate: float | None = None,
        loss: str = "importance_sampling",
        loss_config: Mapping[str, float] | None = None,
    ) -> Update | None:
        """Train `model` on every sequence with advantages in `groups`, in `substeps` optimizer
        steps without splitting a group, at `learning_rate`, else its table's, and serve its new
        weights. Each sequence's ratio is taken against the logprobs its sampler gave, when
        `model` sampled it, and else against `model`'s own at its current weights. An evaluation
        due for the weights served until now runs while it trains, and the new weights are
        served once it has finished. Return None when no group is scored. Raise `ValueError` for
        a model the recipe does not train, a loss Tinker does not have, or a setting it does not
        take, before anything is scored or sent."""
        check_loss(loss, loss_config)
        role, session = model.role, model.backend
        if not model.training:
            raise ValueError(
                f"update(model={role}): the recipe does not train it: set "
                f"`self.models[{role!r}].training = True` in its __init__"
            )
        rate = model.table.learning_rate if learning_rate is None else learning_rate
        if not isinstance(session, Session) or rate is None:
            raise RuntimeError(f"update trains {model!r}, which opens when the run starts")
        kept = [group.scored for group in groups if group.is_scored]
        if not kept:
            return None
        began = time.monotonic()
        self._start_evaluation()
        try:
            flat = [one for part in kept for one in part]
            # Each ratio is taken against the sampler's logprobs, else the current weights' own.
            logprobs: dict[Sequence, Values[float]] = {
                one: one.sampled_logprobs
                for one in flat
                if one.trial.role == role and one.sampled_logprobs is not None
            }
            rest = [one for one in flat if one not in logprobs]
            for one, scored in zip(rest, await model.logprobs(rest), strict=True):
                logprobs[one] = scored.tolist()
            # The papers' mini-batches are sets of prompts, so a group is not split across parts.
            count = min(substeps, len(kept))
            parts = [[one for part in piece for one in part] for piece in split_list(kept, count)]
            trained = await session.fit(
                parts,
                logprobs=logprobs,
                adam=self.ADAM,
                rate=rate,
                loss=loss,
                loss_config=loss_config,
            )
        except BaseException:
            self._cancel_evaluation()
            raise
        took = time.monotonic() - began
        await self._finish_evaluation()
        made = self.state.updates[role] = self.state.updates.get(role, 0) + 1
        serving = time.monotonic()
        await session.serve(f"sample-{made}")
        update = self._updates[role] = Update(
            role=role,
            sequences=len(flat),
            train_tokens=trained.tokens,
            substeps=count,
            learning_rate=rate,
            loss=loss,
            # Training and serving, not the wait for an evaluation between them.
            update_seconds=round(took + time.monotonic() - serving, 3),
            kl_v1=trained.kl_v1,
            kl_v2=trained.kl_v2,
            entropy=trained.entropy,
        )
        return update

    # ------------------------------------------------------------------ the run

    async def run(self) -> None:
        """Open the models, take `[recipe] steps` steps, evaluating on schedule, then save
        `final`. Raise `ValueError`, before anything opens, for a recipe the run cannot take, or
        a model it cannot train or open."""
        connected = self.check()
        self._current = self.state.steps
        async with connected:
            try:
                while self.state.steps < self.recipe.steps:
                    await self._advance()
            finally:
                self._cancel_evaluation()
            await self.checkpoint(FINAL, keep=True)

    def check(self) -> Connected:
        """Raise `ValueError` for what the run refuses before it opens anything: a recipe that
        writes no step, a test set it writes no evaluation for, a model it never reads, a model
        it trains with no rate, a saved training state of a model it no longer trains, or a
        model it cannot train or open. Return the models' backends and the proxy, built and not
        opened: `run` opens them, and a dry run drops them."""
        made = type(self)
        if made.step is Trainer.step:
            raise ValueError(f"{made.__name__} writes no step: a recipe writes `async def step`")
        if self.generator.test and made.evaluate is Trainer.evaluate:
            raise ValueError(
                f"[data] test: {made.__name__} writes no `evaluate`, so the test set cannot be "
                "rolled out: write `async def evaluate(self, tasks)`, or leave [data] test out"
            )
        unread = self.models.unread
        if unread:
            named = ", ".join(f"[models.{one}]" for one in unread)
            raise ValueError(
                f"{named}: {made.__name__} never reads it: a recipe reads the models it uses in "
                "its __init__, as self.models[role]; leave out the ones it does not"
            )
        trained = {one.role for one in self.models.all if one.training}
        for one in self.models.all:
            if one.training and one.table.learning_rate is None:
                raise ValueError(
                    f"[models.{one.role}] learning_rate: {made.__name__} trains it, and its "
                    "table sets no rate"
                )
        lost = sorted(set(self._training_states) - trained)
        if lost:
            raise ValueError(
                f"the checkpoint the run goes on from holds the training state of "
                f"{', '.join(lost)}, which {made.__name__} does not train"
            )
        return connect(
            self.config,
            self.directory,
            self.models.all,
            self.logs.spend,
            training_states=self._training_states,
        )

    async def _advance(self) -> None:
        """Take a step, checkpoint it on schedule, and write its row. The weights it starts from,
        when due, are evaluated beside its update, or after it when it makes none; the last
        step's, after it."""
        self._seen, self._updates, self.metrics = [], {}, {}
        self._wait(self.state.steps)
        self._current = self.state.steps + 1
        await self.step()
        await self._finish_evaluation()  # a step that served no new weights is evaluated now
        self.state.steps += 1
        steps = self.state.steps
        self.logs.step(self._current, self._seen, self._updates, self.metrics)
        if steps == self.recipe.steps:
            self._wait(steps)
            await self._finish_evaluation()
        if steps % self.config.checkpoints.every == 0:
            await self.checkpoint(f"step-{steps}")

    def _wait(self, step: int) -> None:
        """Have the weights served after `step` evaluated, when they are due and not evaluated
        yet: every `evaluate_every` steps from the start, the model the run starts from
        included, and after the last step."""
        every = self.recipe.evaluate_every
        due = (every > 0 and step % every == 0) or step == self.recipe.steps
        done = self.state.evaluated is not None and self.state.evaluated >= step
        if self.generator.test and due and not done:
            self._pending = step

    def _start_evaluation(self) -> None:
        """Start evaluating the waiting weights, beside what the caller goes on to do."""
        if self._pending is not None and self._evaluating is None:
            self._evaluating = asyncio.create_task(self._evaluate(self._pending))

    async def _finish_evaluation(self) -> None:
        """Have the waiting weights evaluated, starting the evaluation if it has not started,
        and write its row."""
        self._start_evaluation()
        if self._evaluating is None:
            return
        try:
            evaluation = await self._evaluating
        finally:
            self._evaluating = None
        self._pending = None
        self.state.evaluated = evaluation.step
        self.logs.evaluation(evaluation)

    def _cancel_evaluation(self) -> None:
        """Stop the evaluation under way, which a resume runs again."""
        if self._evaluating is not None:
            self._evaluating.cancel()
            self._evaluating = None

    async def _evaluate(self, step: int) -> Evaluation:
        """Roll the test set out with the recipe's `evaluate` at the weights served after
        `step`, and return what it came to. It runs as a task of its own, so `EVALUATING` is
        set for its rollouts alone."""
        EVALUATING.set(step)
        found = await self.evaluate(list(self.generator.test))
        return Evaluation(step=step, tasks=len({group.task for group in found}), **tally(found))

    async def checkpoint(self, tag: str, *, keep: bool = False) -> None:
        """Save the run's state as `tag`, which a resume goes on from; `keep` keeps the weights.
        Its row names each trained model's training state by role, once an update has moved it."""
        training_state: dict[str, str] = {}
        sampler_weights: dict[str, str] = {}
        hours: float | None = None
        for one in self.models.all:
            session = one.backend
            if not isinstance(session, Session) or not self.state.updates.get(one.role):
                continue
            made = await session.save(tag, keep=keep)
            training_state[one.role] = made.training_state
            if made.sampler_weights is not None:
                sampler_weights[one.role] = made.sampler_weights
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
        """Keep `modules` in the run as `modules/<name>/`, for a later run to start from."""
        into = self.directory / files.MODULES / name
        keep(modules, into)
        return into

    def restore(self, origin: Path, checkpoint: Checkpoint) -> None:
        """Restore `checkpoint` of `origin`: its state now, its training states once `run`
        opens."""
        tag = checkpoint.tag
        self._training_states = dict(checkpoint.training_state)
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


def _rendered(model: Model, renderer: str | None) -> str | None:
    """Return the renderer a rollout of `model` renders its calls with: `renderer`, one of the
    model's, or its only one; None for the model's own. Raise `ValueError` for a renderer it
    does not hold, or none named when it holds several."""
    held = model.renderers
    if renderer is None:
        if len(held) > 1:
            raise ValueError(
                f"rollout(model={model.role}): name its renderer, one of {', '.join(held)}"
            )
        return held[0] if held else None
    if renderer not in held:
        has = ", ".join(held) or "none: its calls are rendered as the model's own"
        raise ValueError(
            f"rollout(renderer={renderer!r}): [models.{model.role}] renderers holds {has}"
        )
    return renderer
