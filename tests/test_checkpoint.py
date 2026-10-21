"""Tests for `shipyard.run.checkpoint`: a checkpoint saves the trainer's state beside its row,
reads rows of every age, and is found by its tag; and for `resumed`, which goes on from one."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint, found
from shipyard.run.directory import open_run, resumed
from tests.runs import blueprint


def test_a_checkpoint_saves_the_state_beside_its_row(tmp_path: Path) -> None:
    made = Checkpoint.save(tmp_path, "step-2", 2, {"steps": 2})
    assert made == Checkpoint("step-2", 2, f"{files.STATES}/step-2.json")
    assert made.saved_state(tmp_path) == {"steps": 2}
    kept = Checkpoint.save(
        tmp_path,
        "final",
        3,
        {"steps": 3},
        training_state={"agent": "tinker://w/final"},
        sampler_weights={"agent": "tinker://s/final"},
    )
    rows = list(files.read(tmp_path / files.CHECKPOINTS))
    assert all(row.pop("at") for row in rows), "each row says when it was written"
    assert rows == [
        {
            "tag": "step-2",
            "step": 2,
            "state": made.state,
            "training_state": {},
            "sampler_weights": {},
        },
        {
            "tag": "final",
            "step": 3,
            "state": kept.state,
            "training_state": {"agent": "tinker://w/final"},
            "sampler_weights": {"agent": "tinker://s/final"},
            "ttl_hours": None,
        },
    ], "how long saves live, once there are saves"
    assert [Checkpoint.of(row) for row in rows] == [made, kept]


def test_a_row_from_before_saves_were_named_by_role_reads_as_the_agents() -> None:
    old = {"tag": "step-1", "step": 1, "weights": "tinker://w/1", "sampler": None}
    read = Checkpoint.of(old)
    assert read.state is None and read.sampler_weights == {}
    assert read.training_state == {"agent": "tinker://w/1"}
    assert read.saved_state(Path("anywhere")) == {}, "no state to read"


def test_a_checkpoint_is_found_by_its_tag_or_as_the_last(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no checkpoint 'last'; it has none"):
        found(tmp_path)
    for step in (1, 2):
        Checkpoint.save(tmp_path, f"step-{step}", step, {})
    assert found(tmp_path).tag == "step-2" and found(tmp_path, "step-1").step == 1
    with pytest.raises(ValueError, match="has no checkpoint 'step-9'; it has step-1, step-2"):
        found(tmp_path, "step-9")


def test_a_run_goes_on_from_a_checkpoint_until_its_last_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    origin, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    assert config.recipe.steps == 2
    Checkpoint.save(origin, "step-1", 1, {"steps": 1})
    going = resumed(origin)
    assert going is not None
    directory, again, checkpoint = going
    assert checkpoint.tag == "step-1" and again.recipe.steps == 2
    noted = files.read_json(directory / files.RUN)["origin"]
    assert noted == {"run": str(origin.resolve()), "tag": "step-1", "first": origin.name}
    Checkpoint.save(origin, "final", 2, {"steps": 2})
    assert resumed(origin) is None, "at its last step, nothing is left to run"
    assert resumed(origin, tag="step-1") is not None, "an earlier checkpoint still goes on"
