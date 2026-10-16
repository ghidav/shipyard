"""Finished trials as the search reads them: one `Outcome` per task, judged by the same
`admit.verdicts` that training uses. Each carries what the grader printed, what the task
asked and the tail of the worst rollout's transcript, so a reflector sees why a rollout
scored as it did."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from shipyard.training.recipes.gepa.fitness import Outcome
from shipyard.types import Verdict

if TYPE_CHECKING:
    from shipyard.types import Group

#: Characters of verifier output kept per task for the reflector: the tail, since a test
#: script prints a long preamble and says what went wrong at the end.
MAX_FEEDBACK = 2000
#: Characters of the instruction kept per task: the head, since a task states its problem
#: first and where to write the answer last. Smaller than the feedback limit because the
#: instruction is the most redundant.
MAX_INPUTS = 1500
#: Characters of the worst rollout's transcript kept per task: the tail, where the attempt
#: ended.
MAX_TRANSCRIPT = 3000
#: Where Harbor's verifier leaves what it printed. Read as plain text.
VERIFIER_STDOUT = ("verifier", "test-stdout.txt")
#: Where Harbor's agents keep their transcript: `agent/<harness>.txt` under the trial.
AGENT_LOGS = "agent"
#: Files in that directory that are not the transcript: some harnesses write the
#: instruction there again.
NOT_TRANSCRIPTS = ("instruction.txt",)
INSTRUCTION = "instruction.md"


def outcomes(groups: Sequence[Group]) -> list[Outcome]:
    """One outcome per group, judged by its members' verdicts."""
    return [
        _outcome(group.task, [(one.trial.dir, one.verdict) for one in group.members])
        for group in groups
    ]


def _outcome(task: Path, judged: Sequence[tuple[Path, Verdict]]) -> Outcome:
    """The mean over the measured verdicts; feedback and transcript off the worst."""
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
    """The head of the task's `instruction.md`; "" when it cannot be read, so the reflector
    sees less."""
    try:
        text = (task / INSTRUCTION).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= MAX_INPUTS:
        return text
    return text[:MAX_INPUTS] + "\n...(rest of the problem cut)..."


def _transcript(trial: Path) -> str:
    """The tail of the agent's transcript, `agent/<harness>.txt`: the first text file there
    that is not the instruction. "" when there is none."""
    home = trial / AGENT_LOGS
    if not home.is_dir():
        return ""
    found = sorted(
        path for path in home.glob("*.txt") if path.is_file() and path.name not in NOT_TRANSCRIPTS
    )
    return _tail(found[0], MAX_TRANSCRIPT) if found else ""


def _tail(path: Path, limit: int) -> str:
    """A file's text; when longer than `limit`, its last `limit` characters preceded by a marker."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return "...(earlier output cut)...\n" + text[-limit:]


def _why_nothing(judged: Sequence[Verdict]) -> str:
    """The feedback for a task with no measurement: the masks' names and Harbor's ending
    names, so the reflector can see an environment error."""
    masks = sorted({found.mask for found in judged if found.mask})
    ended = sorted({found.ended for found in judged if found.ended})
    said = ", ".join(masks + ended)
    if not said:
        return "no rollout of this task produced a reward"
    return f"no rollout of this task was measurable ({said})"
