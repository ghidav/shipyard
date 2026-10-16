"""Conversion of finished trials into search outcomes.

Each task becomes one `Outcome`, judged by the same `admit.verdicts` that training uses. It
carries the grader's output, the task's instruction and the tail of the worst rollout's
transcript, so the reflector can see why a rollout scored as it did."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.sampling.attempt import inputs, observed, transcript
from shipyard.training.recipes.gepa.fitness import Outcome
from shipyard.types import Verdict

if TYPE_CHECKING:
    from shipyard.types import Group


def outcomes(groups: Sequence[Group]) -> list[Outcome]:
    """Return one outcome per group, built from its members' verdicts."""
    return [
        _outcome(group.task, [(one.trial.dir, one.verdict) for one in group.members])
        for group in groups
    ]


def _outcome(task: Path, judged: Sequence[tuple[Path, Verdict]]) -> Outcome:
    """Build a task's outcome: the mean over the measured verdicts, with the feedback and
    transcript of the worst-scoring rollout."""
    measured = [(trial, found.reward) for trial, found in judged if found.reward is not None]
    if not measured:
        return Outcome(
            task=str(task),
            reward=None,
            feedback=_why_nothing([found for _, found in judged]),
            inputs=inputs(task),
            count=0,
        )
    worst = min(measured, key=lambda pair: pair[1])[0]
    rewards = [reward for _, reward in measured]
    return Outcome(
        task=str(task),
        reward=sum(rewards) / len(rewards),
        feedback=observed(worst),
        inputs=inputs(task),
        count=len(rewards),
        transcript=transcript(worst),
    )


def _why_nothing(judged: Sequence[Verdict]) -> str:
    """Return the feedback for a task with no measured rollout. It names the masks and how
    Harbor reported each trial ended, so the reflector can recognize an environment error."""
    masks = sorted({found.mask for found in judged if found.mask})
    ended = sorted({found.ended for found in judged if found.ended})
    said = ", ".join(masks + ended)
    if not said:
        return "no rollout of this task produced a reward"
    return f"no rollout of this task was measurable ({said})"
