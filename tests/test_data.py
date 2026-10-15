"""Datasets are directories of Harbor tasks; batches cover them, shuffled once per epoch."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from shipyard.config import load
from shipyard.data import Generator, NoSuchDataset, held_out, home_of, tasks
from tests.trials import FIXTURES, fixture_tasks, write_blueprint


def _dataset(root: Path, name: str, *names: str) -> Path:
    """A dataset of copies of the fixture `alpha` task, under the names given."""
    home = root / name
    for task in names:
        shutil.copytree(FIXTURES / "fixture" / "alpha", home / task)
    return home


def _names(found: list[list[Path]]) -> list[str]:
    return [task.name for batch in found for task in batch]


def test_tasks_are_the_valid_directories_sorted_by_name(tmp_path: Path) -> None:
    home = _dataset(tmp_path, "d", "zeta", "alpha", "mid")
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
    _dataset(tmp_path, "d", "x")
    with pytest.raises(ValueError, match="batch_size"):
        list(Generator(tasks("d", root=tmp_path), 0, 0).batches(0))


def test_the_train_set_is_the_union_in_list_order_and_a_batch_is_shuffled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _dataset(tmp_path / "tasks", "a", "a1", "a2", "a3")
    _dataset(tmp_path / "tasks", "b", "b1", "b2")
    text = (
        '[model]\nname = "Qwen/Qwen3-8B"\n[data]\ntrain = ["a", "b"]\nbatch_size = 5\n'
        '[rollout]\nharness = "pi@0.85.1"\n[recipe]\nkind = "dapo"\nlearning_rate = 2e-5\n'
    )
    made = Generator.of(load(write_blueprint(tmp_path, text)))
    assert [task.name for task in made.train] == ["a1", "a2", "a3", "b1", "b2"]
    expected = ["a1", "a2", "a3", "b1", "b2"]
    random.Random(0).shuffle(expected)
    assert _names(list(made.batches(0))) == expected
    other = Generator.of(
        load(write_blueprint(tmp_path / "other", text.replace('["a", "b"]', '["b", "a"]')))
    )
    assert [task.name for task in other.train] == ["b1", "b2", "a1", "a2", "a3"]
    assert _names(list(other.batches(0))) != expected


def test_same_seed_same_order_and_the_short_last_batch_is_kept(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    first = list(Generator(tasks("d", root=tmp_path), 3, 7).batches(0))
    assert [len(batch) for batch in first] == [3, 3, 2]
    assert sorted(_names(first)) == [f"t{n:02d}" for n in range(8)]
    assert list(Generator(tasks("d", root=tmp_path), 3, 7).batches(0)) == first
    other = list(Generator(tasks("d", root=tmp_path), 3, 8).batches(0))
    assert _names(other) != _names(first)


def test_another_epoch_is_another_order_of_the_same_tasks(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    made = Generator(tasks("d", root=tmp_path), 3, 7)
    one, two = list(made.batches(0)), list(made.batches(1))
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


def test_a_generator_cuts_one_epoch_at_a_time(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", "a", "b", "c", "d", "e")
    made = Generator(tasks("d", root=tmp_path), 2, 7)
    assert [len(batch) for batch in made.batches(0)] == [2, 2, 1]
    assert list(made.batches(0)) != list(made.batches(1))
    assert list(made.batches(0)) == list(made.batches(0)), "a fresh iterator each call"


def test_a_generator_is_built_from_the_blueprints_data_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture_tasks(tmp_path)
    home = write_blueprint(
        tmp_path,
        '[model]\nname = "Qwen/Qwen3-8B"\n[data]\ntrain = "fixture"\ntest = "fixture"\n'
        "batch_size = 1\nseed = 3\n"
        '[rollout]\nharness = "pi@0.85.1"\n[recipe]\nkind = "dapo"\nlearning_rate = 2e-5\n',
    )
    made = Generator.of(load(home))
    assert [task.name for task in made.train] == ["alpha", "beta"]
    assert [task.name for task in made.test] == ["alpha", "beta"]
    assert (made.batch_size, made.seed) == (1, 3)
    assert [[task.name for task in batch] for batch in made.batches(0)] == [["beta"], ["alpha"]]
