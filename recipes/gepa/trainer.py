"""GEPA (Agrawal et al., arXiv 2507.19457), through the `gepa` library: a search over the
modules the agent carries, its prompt and skills, that takes no gradient step.

`gepa.optimize` runs the search. Each time it scores a candidate on some tasks, the agent is
rolled out on them as one job carrying the candidate's modules, and a task scores its mean
reward. A reflector, a provider's model, reads the trials of a few tasks and rewrites one
module at a time. The best candidate is kept in the run as `modules/best/`, in the layout
`[agent] modules` reads, and every job after the search carries it, the test set's included.

It searches for any agent: one Tinker serves, which this run never trains, or a provider's,
with `[agent] provider`. It needs the `gepa` extra, and `[recipe] steps = 1`."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import gepa
from gepa import EvaluationBatch

from shipyard import Blueprint, Group, Inadmissible, Modules, Policy, Trainer

#: Characters of a trial's transcript and of its tests' output a reflection reads, from the end,
#: where the attempt finished and the tests said what failed.
TRANSCRIPT, TESTS = 3000, 2000
#: Tokens the reflector may write: a rewritten module, with its reasoning.
REFLECTION_TOKENS = 8192

#: Waits on the run's event loop for a coroutine started from GEPA's thread.
Wait = Callable[[Coroutine[Any, Any, Any]], Any]


class GEPATrainer(Trainer):
    class Recipe(Trainer.Recipe):
        #: The model that rewrites the modules, and the provider that serves it.
        reflector: str
        reflector_provider: str
        #: The most task evaluations the search may run, each `[data] group_size` rollouts.
        budget: int
        #: How many tasks a reflection reads.
        minibatch: int
        #: How many training tasks to hold out and score every kept candidate on; unset, two
        #: thirds of them, as in the paper.
        pareto: int | None = None

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        if self.state.modules is None:
            raise ValueError("GEPA searches the modules the agent carries: set [agent] modules")
        self.reflector = Policy(self.recipe.reflector, provider=self.recipe.reflector_provider)

    async def step(self) -> None:
        loop, seed = asyncio.get_running_loop(), self.state.modules
        assert seed is not None

        # GEPA calls its adapter and its reflector synchronously, from the thread it runs in,
        # while the rollouts and the reflector's calls run on the run's own event loop.
        def wait(work: Coroutine[Any, Any, Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(work, loop).result()

        def reflect(prompt: str | list[dict[str, Any]]) -> str:
            text = (
                prompt
                if isinstance(prompt, str)
                else "\n\n".join(str(m["content"]) for m in prompt)
            )
            [reply] = wait(self.reflector.sample([text], max_tokens=REFLECTION_TOKENS))
            return reply

        feedback, pareto = split(
            self.generator.train, self.recipe.pareto, seed=self.config.data.seed
        )
        result = await asyncio.to_thread(
            gepa.optimize,
            seed_candidate=seed.texts,
            trainset=feedback,
            valset=pareto,
            adapter=Rollouts(self, seed, wait),
            reflection_lm=reflect,
            reflection_minibatch_size=self.recipe.minibatch,
            max_metric_calls=self.recipe.budget,
            run_dir=str(self.directory / "gepa"),
            logger=Logged(),
            seed=self.config.data.seed,
        )
        best = seed.with_texts(result.best_candidate)  # type: ignore[arg-type]
        self.keep(best, "best")
        self.state.modules = best
        self.metrics.update(
            candidates=len(result.candidates),
            best_score=result.val_aggregate_scores[result.best_idx],
            task_evaluations=result.total_metric_calls,
        )


class Rollouts:
    """GEPA's adapter. A candidate's score on some tasks is a job of the agent's rollouts
    carrying its modules; what a reflection reads is each trial's task, transcript, reward and
    test output."""

    #: None: GEPA proposes new texts itself, through the reflector.
    propose_new_texts = None

    def __init__(self, trainer: Trainer, seed: Modules, wait: Wait) -> None:
        self.trainer, self.seed, self.wait = trainer, seed, wait

    def evaluate(
        self, batch: list[Path], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        # GEPA pads a minibatch with tasks it already holds: each is rolled out once, and
        # scored wherever it is.
        tasks = list(dict.fromkeys(batch))
        try:
            modules = self.seed.with_texts(candidate)
        except Inadmissible as refused:
            # Not rolled out: it would fail every trial. Scoring nothing, it is never kept.
            logging.getLogger("gepa").warning("a proposal scores 0: %s", refused)
            groups = [Group(task, []) for task in tasks]
        else:
            groups = self.wait(self.trainer.rollout(tasks, modules=modules, purpose="search"))
        # A task's groups, one per renderer and harness, scored as one.
        found: dict[Path, Group] = {}
        for group in groups:
            found.setdefault(group.task, Group(group.task, [])).trials.extend(group.trials)
        each = [found[task] for task in batch]
        return EvaluationBatch(
            outputs=[group.rewards for group in each],
            scores=[fmean(group.rewards) if group.rewards else 0.0 for group in each],
            trajectories=each if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records = [
            {
                "Inputs": {"task": trial.instruction},
                "Generated Outputs": trial.transcript[-TRANSCRIPT:],
                "Feedback": (
                    f"Reward {trial.reward:.3f}. What the tests printed:\n"
                    f"{trial.test_output[-TESTS:]}"
                ),
            }
            for group in eval_batch.trajectories or []
            for trial in group.measured
        ]
        return {name: records for name in components_to_update}


class Logged:
    """GEPA's logger: its lines go to Python's logging rather than over the progress bars."""

    def log(self, message: str) -> None:
        logging.getLogger("gepa").info(message)


def split(tasks: Sequence[Path], pareto: int | None, *, seed: int) -> tuple[list[Path], list[Path]]:
    """Split the training tasks into those GEPA reflects on and those it scores every kept
    candidate on: `pareto` of them, two thirds by default, drawn with `seed`."""
    listed = sorted(tasks)
    count = len(listed) * 2 // 3 if pareto is None else pareto
    if not 0 < count < len(listed):
        raise ValueError(
            f"[recipe] pareto: hold out between 1 and {len(listed) - 1} of {len(listed)} tasks"
        )
    held = set(random.Random(seed).sample(listed, count))
    return [one for one in listed if one not in held], [one for one in listed if one in held]
