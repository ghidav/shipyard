"""Conversion of finished trials into search outcomes.

Each task becomes one `Outcome`, judged by the same `admit.verdicts` that training uses. It
carries the grader's output, the task's instruction and the tail of the worst rollout's
transcript, so the reflector can see why a rollout scored as it did."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.training.recipes.gepa.fitness import Outcome
from shipyard.types import Verdict

if TYPE_CHECKING:
    from shipyard.types import Group

#: Characters of verifier output kept per task for the reflector. The tail is kept, since a
#: test script prints a long preamble and reports what went wrong at the end.
MAX_FEEDBACK = 2000
#: Characters of the instruction kept per task. The head is kept, since a task states its
#: problem first and where to write the answer last. Smaller than `MAX_FEEDBACK` because the
#: instruction is the most redundant input.
MAX_INPUTS = 1500
#: Characters of the worst rollout's transcript kept per task, taken from the tail, where the
#: attempt ended.
MAX_TRANSCRIPT = 3000
#: Path of the verifier's printed output within a Harbor trial, read as plain text.
VERIFIER_STDOUT = ("verifier", "test-stdout.txt")
#: Directory of the agent's transcript in a Harbor trial (`agent/<harness>.txt`).
AGENT_LOGS = "agent"
#: Text files in `AGENT_LOGS` that are not the transcript; some harnesses copy the
#: instruction there.
NOT_TRANSCRIPTS = ("instruction.txt",)
INSTRUCTION = "instruction.md"


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
            inputs=_inputs(task),
            count=0,
        )
    worst = min(measured, key=lambda pair: pair[1])[0]
    rewards = [reward for _, reward in measured]
    return Outcome(
        task=str(task),
        reward=sum(rewards) / len(rewards),
        feedback=_tail(worst.joinpath(*VERIFIER_STDOUT), MAX_FEEDBACK),
        inputs=_inputs(task),
        count=len(rewards),
        transcript=_transcript(worst),
    )


def _inputs(task: Path) -> str:
    """Return the head of the task's `instruction.md`, or "" when it cannot be read."""
    try:
        text = (task / INSTRUCTION).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= MAX_INPUTS:
        return text
    return text[:MAX_INPUTS] + "\n...(rest of the problem cut)..."


def _transcript(trial: Path) -> str:
    """Return the tail of the agent's transcript: the first text file under `agent/` that is
    not a copy of the instruction, or "" when there is none."""
    home = trial / AGENT_LOGS
    if not home.is_dir():
        return ""
    found = sorted(
        path for path in home.glob("*.txt") if path.is_file() and path.name not in NOT_TRANSCRIPTS
    )
    return _tail(found[0], MAX_TRANSCRIPT) if found else ""


def _tail(path: Path, limit: int) -> str:
    """Return a file's text, or "" when it cannot be read. Text longer than `limit` is cut to
    its last `limit` characters, behind a marker."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return "...(earlier output cut)...\n" + text[-limit:]


def _why_nothing(judged: Sequence[Verdict]) -> str:
    """Return the feedback for a task with no measured rollout. It names the masks and how
    Harbor reported each trial ended, so the reflector can recognize an environment error."""
    masks = sorted({found.mask for found in judged if found.mask})
    ended = sorted({found.ended for found in judged if found.ended})
    said = ", ".join(masks + ended)
    if not said:
        return "no rollout of this task produced a reward"
    return f"no rollout of this task was measurable ({said})"
