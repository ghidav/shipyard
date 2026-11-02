"""Tests for the shipped rft-async recipe: workers roll a task out `group_size` times as one job,
from the teacher when run.toml names one and else from the agent; each step takes the groups in
the order they finished until a batch's worth hold a correct trial, at most a pass over the
tasks, and trains the agent on the correct trials by cross-entropy, every kept token weighing the
same. The fake proxy records two sampled tokens a trial."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard.groups import Group, Trial
from shipyard.jobs.admit import Verdict
from shipyard.jobs.harnesses import Harness
from shipyard.run import files
from shipyard.run.data import Lender, tasks
from shipyard.run.directory import open_run, trainer_class
from tests.fakes import FakeProxy, Harbor, Scripted
from tests.runs import ONE_STEP, rows, run_toml, train
from tests.service import FakeService

TRAINER = Path(__file__).parents[1] / "recipes" / "rft-async" / "trainer.py"
TEACHER = "Qwen/Qwen3-32B"
#: Has a teacher sample for the agent.
TAUGHT = {"[data]": f'[models.teacher]\nmodel = "{TEACHER}"\n\n[data]'}


def drawn(tmp_path: Path, n: int) -> list[str]:
    """The first `n` tasks the run's lender gives out, one at a time, as its workers draw them."""
    lender = Lender(tasks("four", root=tmp_path / "tasks"), 2, 0)
    return [lender.draw(1)[0].name for _ in range(n)]


def weights(service: FakeService) -> list[float]:
    """The weights the updates sent, at every token that has one, in order."""
    return sorted(
        value
        for one in service.client.sent
        if "weights" in one.loss_fn_inputs
        for value in one.loss_fn_inputs["weights"].tolist()
        if value
    )


async def test_a_step_trains_the_agent_on_its_correct_trials_every_kept_token_alike(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """Two groups, one with a correct trial and one with two: three trials of two tokens each are
    kept, every token weighing a sixth, and the failed trial none."""
    _, service = fakes
    home = run_toml(tmp_path, "in_flight = 1", **ONE_STEP)
    first, second = drawn(tmp_path, 2)
    harbor(Scripted({first: [1.0, 0.0], second: [1.0, 1.0]}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    assert weights(service) == pytest.approx([1 / 6] * 6)
    (row,) = rows(directory)
    assert row["trained"] and row["loss"] == "cross_entropy" and row["sequences"] == 3
    assert (row["groups"], row["kept"], row["kept_tokens"]) == (2, 3, 6)


async def test_a_step_takes_groups_until_a_batchs_worth_hold_a_correct_trial(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """The first group to finish has no correct trial: the step takes it and two more."""
    home = run_toml(tmp_path, "in_flight = 1", **ONE_STEP)
    first, second, third = drawn(tmp_path, 3)
    harbor(Scripted({first: [0.0], second: [1.0, 0.0], third: [0.0, 1.0]}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    (row,) = rows(directory)
    assert (row["groups"], row["uniform"], row["kept"]) == (3, 1, 2)


async def test_keep_per_group_caps_the_correct_trials_a_group_gives(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    home = run_toml(tmp_path, "in_flight = 1", "keep_per_group = 1", **ONE_STEP)
    harbor(Scripted({}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    (row,) = rows(directory)
    assert (row["groups"], row["kept"], row["kept_tokens"]) == (2, 2, 4)
    assert weights(service) == pytest.approx([1 / 4] * 4)


async def test_a_step_takes_at_most_a_pass_over_the_tasks(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """No trial is correct: the step stops after the four tasks' groups, and trains nothing."""
    home = run_toml(tmp_path, "in_flight = 1", **ONE_STEP)
    harbor(Scripted({task: [0.0] for task in "abcd"}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    (row,) = rows(directory)
    assert (row["groups"], row["kept"]) == (4, 0) and not row["trained"]


async def test_with_a_teacher_the_agent_learns_the_teachers_correct_trials_unscored(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """The teacher rolls every task out; the agent trains on its correct trials without a pass to
    score them first, since cross-entropy takes no ratio."""
    _, service = fakes
    home = run_toml(tmp_path, "in_flight = 1", **ONE_STEP, **TAUGHT)
    first, second = drawn(tmp_path, 2)
    harbor(Scripted({first: [1.0, 0.0], second: [0.0, 1.0]}))
    directory = await train(home, tmp_path / "runs", TRAINER)
    training = [job for job in files.read(directory / files.JOBS) if job["purpose"] == "training"]
    assert training and {job["role"] for job in training} == {"teacher"}
    steps = [c[0] for c in service.client.calls if c[0] in ("forward", "forward_backward")]
    assert steps == ["forward_backward"], "the teacher's sequences are trained on as they are"
    (row,) = rows(directory)
    assert row["trained"] and row["kept"] == 2


def test_correct_trials_with_no_tokens_fail_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler its provider serves leaves its trials no sequences: the step says so rather than
    train on nothing."""
    monkeypatch.chdir(tmp_path)  # where the run finds tasks/four
    home = run_toml(tmp_path, **TAUGHT)
    directory, loaded = open_run(home, root=tmp_path / "runs", trainer=TRAINER)
    with trainer_class(directory)(loaded, directory) as trainer:
        (task,) = trainer.lender.draw(1)
        solved = Trial(
            "j", task, Harness(name="pi"), tmp_path / "j" / "t", verdict=Verdict(1.0, None, None)
        )
        with pytest.raises(RuntimeError, match=r"\[models.teacher\]'s correct trials hold no"):
            trainer.score([Group(task, [solved])])
