"""Tests for datasets of Harbor tasks and the shuffled batches drawn from them."""

from __future__ import annotations

import random
import shutil
from itertools import islice
from pathlib import Path

import pytest

from shipyard.config import load
from shipyard.data import Generator, NoSuchDataset, held_out, home_of, tasks
from tests.trials import FIXTURES, dataset, fixture_tasks, write_blueprint


def _names(found: list[list[Path]]) -> list[str]:
    return [task.name for batch in found for task in batch]


def _pass(made: Generator, turn: int = 0) -> list[list[Path]]:
    """Return the batches of pass `turn` over the train tasks."""
    return list(islice(made.batches(turn * made.per_pass), made.per_pass))


def test_tasks_are_the_valid_directories_sorted_by_name(tmp_path: Path) -> None:
    home = dataset(tmp_path, "d", "zeta", "alpha", "mid")
    (home / "notes").mkdir()  # a directory with no task.toml
    (home / "notes" / "README.md").write_text("not a task")
    (home / ".hidden").mkdir()
    (home / "stray.txt").write_text("not a directory")
    broken = shutil.copytree(FIXTURES / "fixture" / "beta", home / "broken")
    (broken / "instruction.md").unlink()  # Harbor refuses a task without one
    found = tasks("d", root=tmp_path)
    assert [task.name for task in found] == ["alpha", "mid", "zeta"]
    assert all(task.parent == home for task in found)


def test_the_fixture_dataset_reads_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert [task.name for task in tasks("fixture")] == ["alpha", "beta"]
    assert home_of("fixture") == Path("tasks") / "fixture"


def test_a_missing_or_empty_dataset_is_refused_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(NoSuchDataset) as missing:
        tasks("nothing", root=tmp_path)
    assert str(tmp_path / "nothing") in str(missing.value)
    assert "harbor datasets download" in str(missing.value)
    (tmp_path / "empty" / "stray").mkdir(parents=True)
    with pytest.raises(NoSuchDataset) as empty:
        tasks("empty", root=tmp_path)
    assert str(tmp_path / "empty") in str(empty.value)


def test_a_batch_size_under_one_is_refused_not_clamped(tmp_path: Path) -> None:
    dataset(tmp_path, "d", "x")
    with pytest.raises(ValueError, match="batch_size"):
        next(Generator(tasks("d", root=tmp_path), 0, 0).batches())


def test_the_train_set_is_the_union_in_list_order_and_a_batch_is_shuffled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset(tmp_path / "tasks", "a", "a1", "a2", "a3")
    dataset(tmp_path / "tasks", "b", "b1", "b2")
    text = (
        '[agent]\nmodel = "Qwen/Qwen3-8B"\nharness = "pi@0.85.1"\n'
        '[data]\ntrain = ["a", "b"]\nbatch_size = 5\n'
        '[recipe]\nkind = "dapo"\nlearning_rate = 2e-5\nsteps = 1\n'
    )
    made = Generator.of(load(write_blueprint(tmp_path, text)))
    assert [task.name for task in made.train] == ["a1", "a2", "a3", "b1", "b2"]
    expected = ["a1", "a2", "a3", "b1", "b2"]
    random.Random(0).shuffle(expected)
    assert _names(_pass(made)) == expected
    other = Generator.of(
        load(write_blueprint(tmp_path / "other", text.replace('["a", "b"]', '["b", "a"]')))
    )
    assert [task.name for task in other.train] == ["b1", "b2", "a1", "a2", "a3"]
    assert _names(_pass(other)) != expected


def test_same_seed_same_order_and_the_short_last_batch_is_kept(tmp_path: Path) -> None:
    dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    first = _pass(Generator(tasks("d", root=tmp_path), 3, 7))
    assert [len(batch) for batch in first] == [3, 3, 2]
    assert sorted(_names(first)) == [f"t{n:02d}" for n in range(8)]
    assert _pass(Generator(tasks("d", root=tmp_path), 3, 7)) == first
    other = _pass(Generator(tasks("d", root=tmp_path), 3, 8))
    assert _names(other) != _names(first)


def test_another_pass_is_another_order_of_the_same_tasks(tmp_path: Path) -> None:
    dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    made = Generator(tasks("d", root=tmp_path), 3, 7)
    one, two = _pass(made, 0), _pass(made, 1)
    assert _names(two) != _names(one) and sorted(_names(two)) == sorted(_names(one))


def test_held_out_draws_a_seeded_count_and_keeps_list_order() -> None:
    listed = [Path(f"t{n}") for n in range(9)]
    kept, held = held_out(listed, 6, seed=0)
    assert len(kept) == 3 and len(held) == 6 and sorted(kept + held) == listed
    assert kept == sorted(kept) and held == sorted(held), "each in list order"
    assert held_out(listed, 6, seed=0) == (kept, held), "the same draw for the same seed"
    assert held_out(listed, 6, seed=1) != (kept, held)
    assert held_out(listed, 0, seed=0) == (listed, [])
    with pytest.raises(ValueError, match="cannot hold out 10 of 9"):
        held_out(listed, 10, seed=0)


def test_a_generator_streams_pass_after_pass_from_any_batch(tmp_path: Path) -> None:
    dataset(tmp_path, "d", "a", "b", "c", "d", "e")
    made = Generator(tasks("d", root=tmp_path), 2, 7)
    assert made.per_pass == 3 and [len(batch) for batch in _pass(made)] == [2, 2, 1]
    assert _pass(made, 0) != _pass(made, 1)
    streamed = list(islice(made.batches(), 7))
    assert streamed == _pass(made, 0) + _pass(made, 1) + _pass(made, 2)[:1]
    assert list(islice(made.batches(4), 3)) == streamed[4:7], "resumed at batch 4"
    assert list(Generator([], 2, 7).batches()) == [], "no tasks, no batches"


def test_a_generator_is_built_from_the_blueprints_data_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture_tasks(tmp_path)
    home = write_blueprint(
        tmp_path,
        '[agent]\nmodel = "Qwen/Qwen3-8B"\nharness = "pi@0.85.1"\n'
        '[data]\ntrain = "fixture"\ntest = "fixture"\n'
        "batch_size = 1\nseed = 3\n"
        '[recipe]\nkind = "dapo"\nlearning_rate = 2e-5\nsteps = 1\n',
    )
    made = Generator.of(load(home))
    assert [task.name for task in made.train] == ["alpha", "beta"]
    assert [task.name for task in made.test] == ["alpha", "beta"]
    assert (made.batch_size, made.seed) == (1, 3)
    assert [[task.name for task in batch] for batch in _pass(made)] == [["beta"], ["alpha"]]
