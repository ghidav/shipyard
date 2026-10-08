"""Finished trials as the search reads them: one outcome per task, judged by admission,
with the grader's words off the worst rollout, the task's own instruction, and the tail
of the transcript Harbor kept."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard.admit import BUDGET
from shipyard.gepa.outcomes import MAX_FEEDBACK, MAX_INPUTS, MAX_TRANSCRIPT, outcomes
from shipyard.rollout import Rollouts
from tests.records import made
from tests.trials import write_result


def _trial(home: Path, *, reward: float | None = None, stdout: str = "", **about) -> Path:
    write_result(home, reward=reward, **about)
    if stdout:
        (home / "verifier").mkdir(exist_ok=True)
        (home / "verifier" / "test-stdout.txt").write_text(stdout)
    return home


def _rollouts(trials: list[Path], **more) -> Rollouts:
    return Rollouts(job="job-0000", trials=trials, plan=[], **more)


def _task(tmp_path: Path, name: str, instruction: str | None = None) -> Path:
    home = tmp_path / "tasks" / "fixture" / name
    home.mkdir(parents=True, exist_ok=True)
    if instruction is not None:
        (home / "instruction.md").write_text(instruction)
    return home


def test_one_outcome_per_task_with_the_measured_mean(tmp_path: Path) -> None:
    rewards = [1.0, 0.0, 1.0, 1.0]
    trials = [_trial(tmp_path / f"t{i}", reward=r) for i, r in enumerate(rewards)]
    tasks = [_task(tmp_path, "a"), _task(tmp_path, "b")]
    got = outcomes(_rollouts(trials), tasks, 2)
    assert [o.task for o in got] == [str(task) for task in tasks]
    assert [o.reward for o in got] == [0.5, 1.0] and [o.count for o in got] == [2, 2]


def test_a_masked_rollout_is_absent_from_the_mean_and_a_task_nobody_measured_says_why(
    tmp_path: Path,
) -> None:
    trials = [_trial(tmp_path / "ok", reward=1.0), tmp_path / "gone"]
    (got,) = outcomes(_rollouts(trials), [_task(tmp_path, "a")], 2)
    assert got.reward == 1.0 and got.count == 1
    dead = [
        tmp_path / "never",
        _trial(tmp_path / "late", exception="AgentTimeoutError"),
        _trial(tmp_path / "ungraded"),
    ]
    (nothing,) = outcomes(_rollouts(dead), [_task(tmp_path, "b")], 3)
    assert nothing.reward is None and not nothing.measured and nothing.count == 0
    assert nothing.feedback == (
        "no rollout of this task was measurable "
        "(env_error, grading_error, timeout, AgentTimeoutError)"
    )


def test_a_budget_cut_rollout_scores_zero_through_admission(tmp_path: Path) -> None:
    """The same reader training uses: a served rollout the proxy cut on its budget is a
    real 0 that overrides the verifier, and one with no records is masked."""
    cut = _trial(tmp_path / "cut__1", reward=1.0, stdout="passed")
    silent = _trial(tmp_path / "silent__1", reward=1.0)
    records = {cut.name: [made(seq=1), made(seq=2, error=BUDGET)], silent.name: []}
    (got,) = outcomes(
        _rollouts([cut, silent], records=records), [_task(tmp_path, "a")], group_size=2
    )
    assert got.reward == 0.0 and got.count == 1
    assert got.feedback == "passed", "off the budget-cut rollout, the worst measured"


def test_feedback_and_transcript_come_off_the_worst_rollout_as_tails(tmp_path: Path) -> None:
    won = _trial(tmp_path / "won", reward=1.0, stdout="Correct answer")
    lost = _trial(tmp_path / "lost", reward=0.0, stdout="Expected: 385")
    (lost / "agent").mkdir()
    (lost / "agent" / "pi.txt").write_text("thinking...\nI will answer 12")
    (lost / "agent" / "trajectory.json").write_text("{}")
    (won / "agent").mkdir()
    (won / "agent" / "pi.txt").write_text("the winning transcript")
    (got,) = outcomes(_rollouts([won, lost]), [_task(tmp_path, "a")], 2)
    assert got.feedback == "Expected: 385"
    assert got.transcript == "thinking...\nI will answer 12"
    (lost / "agent" / "instruction.txt").write_text("the task, as some harnesses write it back")
    (got,) = outcomes(_rollouts([won, lost]), [_task(tmp_path, "a")], 2)
    assert got.transcript == "thinking...\nI will answer 12", (
        "sorts first, and is not the transcript"
    )
    long_out = "start\n" + ("x" * MAX_FEEDBACK) + "\nwhat actually failed"
    (lost / "verifier" / "test-stdout.txt").write_text(long_out)
    (lost / "agent" / "pi.txt").write_text("y" * (MAX_TRANSCRIPT + 5) + "\nthe last turn")
    (got,) = outcomes(_rollouts([won, lost]), [_task(tmp_path, "a")], 2)
    assert got.feedback.startswith("...(earlier output cut)...")
    assert got.feedback.endswith("what actually failed")
    assert got.transcript.startswith("...(earlier output cut)...")
    assert got.transcript.endswith("the last turn") and len(got.transcript) < MAX_TRANSCRIPT + 40
    bare = _trial(tmp_path / "bare", reward=0.0)
    (got,) = outcomes(_rollouts([bare]), [_task(tmp_path, "a")], 1)
    assert got.feedback == "" and got.transcript == ""


def test_the_inputs_are_the_task_directorys_instruction_head(tmp_path: Path) -> None:
    task = _task(tmp_path, "a", "Every morning Aya goes for a 9-kilometer walk")
    (got,) = outcomes(_rollouts([_trial(tmp_path / "t", reward=0.0)]), [task], 1)
    assert got.inputs == "Every morning Aya goes for a 9-kilometer walk"
    task = _task(tmp_path, "b", "HEAD" + "x" * (MAX_INPUTS * 2) + "TAIL")
    (got,) = outcomes(_rollouts([_trial(tmp_path / "t", reward=0.0)]), [task], 1)
    assert got.inputs.startswith("HEAD") and "TAIL" not in got.inputs
    assert got.inputs.endswith("...(rest of the problem cut)...")
    (got,) = outcomes(_rollouts([_trial(tmp_path / "t", reward=0.0)]), [_task(tmp_path, "c")], 1)
    assert got.inputs == "" and got.reward == 0.0, (
        "no instruction: a worse reflection, not a failure"
    )


def test_trials_that_do_not_divide_into_the_task_list_are_refused(tmp_path: Path) -> None:
    trials = [_trial(tmp_path / f"t{i}", reward=1.0) for i in range(3)]
    with pytest.raises(ValueError, match="do not divide"):
        outcomes(_rollouts(trials), [_task(tmp_path, "a"), _task(tmp_path, "b")], 2)
    with pytest.raises(ValueError, match="group_size"):
        outcomes(_rollouts(trials), [_task(tmp_path, "a")], 0)


def test_a_task_named_twice_keeps_its_own_rollouts(tmp_path: Path) -> None:
    trials = [_trial(tmp_path / f"t{i}", reward=r) for i, r in enumerate([0.0, 0.0, 1.0, 1.0])]
    task = _task(tmp_path, "a")
    assert [o.reward for o in outcomes(_rollouts(trials), [task, task], 2)] == [0.0, 1.0]
