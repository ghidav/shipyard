"""Tests of the lender: the run's source of training tasks. It gives them out in a seeded order,
gives the ones put back out before new ones, gives retired ones out no more, and keeps which are
out until their groups complete, so that a resume gives those out first."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from shipyard import Trainer
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.data import Lender, tasks
from shipyard.run.directory import open_run
from shipyard.state import Lending
from tests.fakes import dataset
from tests.runs import GRPO, run_toml


def lender_of(tmp_path: Path, *names: str, batch: int = 2) -> Lender:
    """A lender of a dataset of tasks under `names`, `batch` to a batch, seed 0."""
    if not (tmp_path / "tasks" / "set").is_dir():
        dataset(tmp_path / "tasks", "set", *names)
    return Lender(tasks("set", root=tmp_path / "tasks"), batch, 0)


def names(found: list[Path]) -> list[str]:
    return [one.name for one in found]


def test_each_pass_is_given_out_in_its_own_seeded_order_and_ends_on_a_short_batch(
    tmp_path: Path,
) -> None:
    lender = lender_of(tmp_path, *"abcde")
    first, second = list(lender.train), list(lender.train)
    random.Random(0).shuffle(first)
    random.Random(1).shuffle(second)
    drawn = [names(lender.draw()) for _ in range(4)]
    assert drawn == [names(first[:2]), names(first[2:4]), names(first[4:]), names(second[:2])], (
        "the third batch is short: a batch never holds a task twice"
    )
    assert lender.per_pass == 3


def test_drawing_one_at_a_time_gives_the_same_order_as_drawing_batches(tmp_path: Path) -> None:
    batches, singles = lender_of(tmp_path, *"abcde"), lender_of(tmp_path, *"abcde")
    by_batch = [one for _ in range(6) for one in batches.draw()]
    one_by_one = [one for _ in range(10) for one in singles.draw(1)]
    assert by_batch == one_by_one and len(set(by_batch[:5])) == 5, "two whole passes"


def test_a_task_put_back_is_given_out_before_new_ones_and_is_out_no_more(tmp_path: Path) -> None:
    lender = lender_of(tmp_path, *"abcd")
    first = lender.draw()
    lender.complete(first[0])
    lender.back(first[1])  # put back before its group completed: a recipe found it stale
    assert lender.lending.out == []
    again = lender.draw()
    assert again[0] == first[1] and again[1] not in first
    assert lender.lending.out == [str(one) for one in again]


def test_a_retired_task_is_given_out_no_more_until_it_is_put_back(tmp_path: Path) -> None:
    lender = lender_of(tmp_path, *"abc", batch=1)
    solved = lender.train[0]
    lender.retire(solved)
    drawn = [one for _ in range(6) for one in lender.draw()]
    assert solved not in drawn and len(drawn) == 6, "three passes over the two left"
    lender.back(solved)
    assert lender.draw() == [solved] and lender.lending.retired == []


def test_drawing_raises_once_every_task_is_retired(tmp_path: Path) -> None:
    lender = lender_of(tmp_path, *"ab", batch=1)
    for one in lender.train:
        lender.retire(one)
    with pytest.raises(RuntimeError, match="every training task is retired"):
        lender.draw()


def test_the_lender_takes_back_and_retires_only_the_tasks_it_gives_out(tmp_path: Path) -> None:
    lender = lender_of(tmp_path, *"ab")
    with pytest.raises(ValueError, match="gives out training tasks"):
        lender.back(tmp_path / "a test task")
    with pytest.raises(ValueError, match="gives out training tasks"):
        lender.retire(tmp_path / "a test task")


def test_a_completed_group_takes_its_task_out_once_and_a_test_task_is_let_be(
    tmp_path: Path,
) -> None:
    lender = lender_of(tmp_path, *"ab", batch=1)
    task = lender.draw()[0]
    lender.back(task)
    assert lender.draw() == [task]
    lender.lending.out.append(str(task))  # given out twice, as a short dataset can
    lender.complete(tmp_path / "a test task")
    lender.complete(task)
    assert lender.lending.out == [str(task)], "out once fewer"


def test_a_resume_gives_the_tasks_out_at_its_checkpoint_out_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lender = lender_of(tmp_path, *"abcd")
    out = lender.draw()  # the run stopped before their groups completed
    saved = Lending.model_validate(lender.lending.model_dump())
    saved.out.append(str(tmp_path / "tasks" / "set" / "gone"))
    later = lender_of(tmp_path, *"abcd")
    with caplog.at_level("WARNING"):
        later.resume(saved)
    assert later.draw() == out and "gone" in caplog.text
    assert later.lending.place == 2, "then the seeded order goes on where it was"


def test_batches_drawn_before_the_lender_become_its_place(tmp_path: Path) -> None:
    lender = lender_of(tmp_path, *"abcde")  # three batches a pass: 2, 2 and 1
    assert [lender.place_after(n) for n in range(5)] == [0, 2, 4, 5, 7]


def test_a_resumed_trainer_draws_the_tasks_its_checkpoint_had_out_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, config = open_run(run_toml(tmp_path), root=tmp_path / "runs", trainer=GRPO)
    stopped = Trainer(config, directory)
    out = stopped.draw()
    saved = f"{files.STATES}/step-1.json"
    files.write_json(directory / saved, stopped.state.model_dump(mode="json"))
    resumed = Trainer(config, directory)
    resumed.restore(directory, Checkpoint("step-1", 1, saved))
    assert resumed.draw() == out, "their trials ended with the run that drew them"
    assert not set(resumed.draw()) & set(out), "then the seeded order, where it was"
