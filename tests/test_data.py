"""Datasets are directories of Harbor tasks; batches cover them, shuffled once per epoch."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from shipyard.config import load
from shipyard.data import Generator, NoSuchDataset, batches, held_out, home_of, tasks
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
    with pytest.raises(NoSuchDataset):
        batches(["empty"], size=2, seed=0, epochs=1, root=tmp_path)


def test_a_batch_size_under_one_is_refused_not_clamped(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", "x")
    with pytest.raises(ValueError, match="size"):
        batches(["d"], size=0, seed=0, epochs=1, root=tmp_path)


def test_batches_take_the_union_in_list_order_then_shuffle(tmp_path: Path) -> None:
    _dataset(tmp_path, "a", "a1", "a2", "a3")
    _dataset(tmp_path, "b", "b1", "b2")
    found = list(batches(["a", "b"], size=5, seed=0, epochs=1, root=tmp_path))
    expected = ["a1", "a2", "a3", "b1", "b2"]
    random.Random(0).shuffle(expected)
    assert _names(found) == expected
    reversed_ = list(batches(["b", "a"], size=5, seed=0, epochs=1, root=tmp_path))
    assert sorted(_names(reversed_)) == sorted(expected) and _names(reversed_) != expected


def test_same_seed_same_order_and_the_short_last_batch_is_kept(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    first = list(batches(["d"], size=3, seed=7, epochs=1, root=tmp_path))
    assert [len(batch) for batch in first] == [3, 3, 2]
    assert sorted(_names(first)) == [f"t{n:02d}" for n in range(8)]
    assert list(batches(["d"], size=3, seed=7, epochs=1, root=tmp_path)) == first
    other = list(batches(["d"], size=3, seed=8, epochs=1, root=tmp_path))
    assert _names(other) != _names(first)


def test_another_epoch_is_another_order_of_the_same_tasks(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", *[f"t{n:02d}" for n in range(8)])
    one = list(batches(["d"], size=3, seed=7, epochs=1, root=tmp_path))
    two = list(batches(["d"], size=3, seed=7, epochs=2, root=tmp_path))
    assert two[:3] == one and len(two) == 6
    assert _names(two[3:]) != _names(one) and sorted(_names(two[3:])) == sorted(_names(one))


def test_each_call_is_a_fresh_generator(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", "x", "y")
    once = batches(["d"], size=1, seed=0, epochs=1, root=tmp_path)
    assert len(list(once)) == 2 and list(once) == []
    assert len(list(batches(["d"], size=1, seed=0, epochs=1, root=tmp_path))) == 2


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


def test_a_generator_cuts_one_epoch_at_a_time_as_batches_does(tmp_path: Path) -> None:
    _dataset(tmp_path, "d", "a", "b", "c", "d", "e")
    made = Generator(tasks("d", root=tmp_path), 2, 7)
    whole = list(batches(["d"], size=2, seed=7, epochs=2, root=tmp_path))
    assert list(made.batches(0)) + list(made.batches(1)) == whole
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
