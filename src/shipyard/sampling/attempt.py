"""What a finished trial left for a reader: the task's instruction, the tail of the agent's
transcript and the tail of the tests' output.

GEPA's reflector reads rollouts through these, and so can a trainer's `privileged` hook."""

from __future__ import annotations

from pathlib import Path

#: Characters of verifier output kept. The tail is kept, since a test script prints a long
#: preamble and reports what went wrong at the end.
MAX_FEEDBACK = 2000
#: Characters of the instruction kept. The head is kept, since a task states its problem
#: first and where to write the answer last. Smaller than `MAX_FEEDBACK` because the
#: instruction is the most redundant input.
MAX_INPUTS = 1500
#: Characters of a transcript kept, taken from the tail, where the attempt ended.
MAX_TRANSCRIPT = 3000
#: Path of the verifier's printed output within a Harbor trial, read as plain text.
VERIFIER_STDOUT = ("verifier", "test-stdout.txt")
#: Directory of the agent's transcript in a Harbor trial (`agent/<harness>.txt`).
AGENT_LOGS = "agent"
#: Text files in `AGENT_LOGS` that are not the transcript; some harnesses copy the
#: instruction there.
NOT_TRANSCRIPTS = ("instruction.txt",)
INSTRUCTION = "instruction.md"


def attempt(task: Path, trial: Path, reward: float | None) -> str:
    """Render one rollout as the reflector reads it: its score, what it was asked, the tail of
    its transcript and what the tests printed."""
    score = "unmeasured" if reward is None else f"{reward:.3f}"
    body = f"score: {score}\n\n"
    if asked := inputs(task):
        body += f"--- asked ---\n{asked}\n\n"
    if said := transcript(trial):
        body += f"--- transcript ---\n{said}\n\n"
    return body + f"--- observed ---\n{observed(trial)}\n"


def inputs(task: Path) -> str:
    """Return the head of the task's `instruction.md`, or "" when it cannot be read."""
    try:
        text = (task / INSTRUCTION).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= MAX_INPUTS:
        return text
    return text[:MAX_INPUTS] + "\n...(rest of the problem cut)..."


def transcript(trial: Path) -> str:
    """Return the tail of the agent's transcript: the first text file under `agent/` that is
    not a copy of the instruction, or "" when there is none."""
    home = trial / AGENT_LOGS
    if not home.is_dir():
        return ""
    found = sorted(
        path for path in home.glob("*.txt") if path.is_file() and path.name not in NOT_TRANSCRIPTS
    )
    return tail(found[0], MAX_TRANSCRIPT) if found else ""


def observed(trial: Path) -> str:
    """Return the tail of what the verifier printed, or "" when it printed nothing."""
    return tail(trial.joinpath(*VERIFIER_STDOUT), MAX_FEEDBACK)


def tail(path: Path, limit: int) -> str:
    """Return a file's text, or "" when it cannot be read. Text longer than `limit` is cut to
    its last `limit` characters, behind a marker."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return "...(earlier output cut)...\n" + text[-limit:]
