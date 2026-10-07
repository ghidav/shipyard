"""Finished trials as the search reads them: one `Outcome` per task, judged by the same
`admit.verdicts` training uses, with what the grader printed, what the task asked and
the tail of the worst rollout's transcript, so a reflector is told why and not only how much."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from shipyard.admit import Verdict, verdicts
from shipyard.gepa.fitness import Outcome
from shipyard.rollout import Rollouts

#: Per task, in the feedback handed to a reflector: the tail of what the verifier printed,
#: since a test script announces itself for lines and says what went wrong at the end.
MAX_FEEDBACK = 2000
#: Per task, of the instruction: the head, since a task states its problem first and
#: where to write the answer last; smaller than the feedback, being the most redundant.
MAX_INPUTS = 1500
#: Per task, of the worst rollout's transcript: the tail, where the attempt ended.
MAX_TRANSCRIPT = 3000
#: Where Harbor's verifier leaves what it printed, read as text and never parsed.
VERIFIER_STDOUT = ("verifier", "test-stdout.txt")
#: Where Harbor's agents keep their transcript: `agent/<harness>.txt` under the trial.
AGENT_LOGS = "agent"
#: What some harnesses leave beside it that is not the transcript: the instruction again.
NOT_TRANSCRIPTS = ("instruction.txt",)
INSTRUCTION = "instruction.md"


def outcomes(rollouts: Rollouts, tasks: Sequence[Path], group_size: int = 1) -> list[Outcome]:
    """One outcome per task from a job's trials, which arrive in plan order, `group_size`
    per task: the mean over the measured verdicts (a budget-cut rollout is a 0 here as
    in training), the feedback and transcript off the worst, the inputs off the task."""
    if group_size < 1:
        raise ValueError(f"group_size must be at least 1; got {group_size}")
    trials = [Path(trial) for trial in rollouts.trials]
    if len(trials) != len(tasks) * group_size:
        raise ValueError(
            f"{len(trials)} trial(s) do not divide into {len(tasks)} task(s) at {group_size} "
            "rollout(s) each: a crashed trial keeps its slot, so this job was not rolled out "
            "against this task list"
        )
    judged = list(zip(trials, verdicts(rollouts), strict=True))
    built: list[Outcome] = []
    for index, task in enumerate(tasks):
        task = Path(task)
        members = judged[index * group_size : (index + 1) * group_size]
        measured = [(trial, found.reward) for trial, found in members if found.reward is not None]
        if not measured:
            built.append(
                Outcome(
                    task=str(task),
                    reward=None,
                    feedback=_why_nothing([found for _, found in members]),
                    inputs=_inputs(task),
                    count=0,
                )
            )
            continue
        worst = min(measured, key=lambda pair: pair[1])[0]
        rewards = [reward for _, reward in measured]
        built.append(
            Outcome(
                task=str(task),
                reward=sum(rewards) / len(rewards),
                feedback=_tail(worst.joinpath(*VERIFIER_STDOUT), MAX_FEEDBACK),
                inputs=_inputs(task),
                count=len(rewards),
                transcript=_transcript(worst),
            )
        )
    return built


def _inputs(task: Path) -> str:
    """The task's `instruction.md`, its head; "" when it cannot be read, which is a worse
    reflection rather than a failed search."""
    try:
        text = (task / INSTRUCTION).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= MAX_INPUTS:
        return text
    return text[:MAX_INPUTS] + "\n...(rest of the problem cut)..."


def _transcript(trial: Path) -> str:
    """The tail of the transcript Harbor's agent kept, `agent/<harness>.txt`, the first
    text file there that is not the instruction written back; "" without one."""
    home = trial / AGENT_LOGS
    if not home.is_dir():
        return ""
    found = sorted(
        path for path in home.glob("*.txt") if path.is_file() and path.name not in NOT_TRANSCRIPTS
    )
    return _tail(found[0], MAX_TRANSCRIPT) if found else ""


def _tail(path: Path, limit: int) -> str:
    """A file's text, its tail past `limit`, marked so a reader knows it begins mid-way."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return "...(earlier output cut)...\n" + text[-limit:]


def _why_nothing(judged: Sequence[Verdict]) -> str:
    """Why a task produced no measurement, in the masks' names and Harbor's for the
    endings, handed to the reflector as feedback: an environment error is worth knowing."""
    masks = sorted({found.mask for found in judged if found.mask})
    ended = sorted({found.ended for found in judged if found.ended})
    said = ", ".join(masks + ended)
    if not said:
        return "no rollout of this task produced a reward"
    return f"no rollout of this task was measurable ({said})"
