"""Opening a run writes its definition and process note; closing it says how it ended."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from shipyard import record
from shipyard.config import KINDS, ConfigError
from shipyard.data import NoSuchDataset
from shipyard.recipes import run_recipe
from shipyard.run import Run, RunRefused, runs_root, version
from shipyard.show import alive, state_of

BLUEPRINTS = Path(__file__).parent / "blueprints"
DAPO = BLUEPRINTS / "dapo"


def test_open_creates_the_files(tmp_path: Path) -> None:
    run = Run.open(DAPO, root=tmp_path)
    assert run.directory.parent == tmp_path and run.id == run.directory.name
    assert re.fullmatch(r"dapo__[0-9A-Za-z]{7}", run.id)
    assert sorted(p.name for p in run.directory.iterdir()) == ["process.json", "run.toml"]
    assert (run.directory / "run.toml").read_bytes() == (DAPO / "run.toml").read_bytes()
    note = Run.read(run.directory)
    assert note["pid"] == os.getpid() and note["started_at"]
    assert note["version"] == version() != ""
    assert note["blueprint"] == str(DAPO.resolve())
    assert note["kind"] == "dapo"
    assert run.config.recipe.kind == "dapo" and run.config.datasets == ["aime-train"]


def test_open_from_the_file_names_its_directory(tmp_path: Path) -> None:
    run = Run.open(DAPO / "run.toml", root=tmp_path)
    assert run.id.startswith("dapo__")
    assert Run.read(run.directory)["blueprint"] == str(DAPO.resolve())


def test_log_numbers_rows_from_one(tmp_path: Path) -> None:
    run = Run.open(DAPO, root=tmp_path)
    assert run.log(reward=0.5)["seq"] == 1
    assert run.log(reward=0.7)["seq"] == 2
    rows = list(record.read(run.directory / record.METRICS))
    assert [row["seq"] for row in rows] == [1, 2]
    assert all("at" in row for row in rows)


def test_exit_records_finished(tmp_path: Path) -> None:
    with Run.open(DAPO, root=tmp_path) as run:
        run.note(hello="world")
    note = Run.read(run.directory)
    assert note["finished_at"] and note["failed"] is False and note["error"] is None
    assert note["hello"] == "world" and note["kind"] == "dapo"
    assert state_of(note) == "finished"


def test_exit_records_failed_with_the_exception_text(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), Run.open(DAPO, root=tmp_path) as run:
        raise RuntimeError("the loop broke")
    note = Run.read(run.directory)
    assert note["failed"] is True and note["error"] == "RuntimeError: the loop broke"
    assert note["finished_at"] and state_of(note) == "failed"


def test_exit_records_a_signal_as_stopped(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt), Run.open(DAPO, root=tmp_path) as run:
        raise KeyboardInterrupt("SIGTERM")
    note = Run.read(run.directory)
    assert note["failed"] is True and note["error"] == "stopped: SIGTERM"


def test_exit_records_a_bare_interrupt_as_sigint(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt), Run.open(DAPO, root=tmp_path) as run:
        raise KeyboardInterrupt()
    assert Run.read(run.directory)["error"] == "stopped: SIGINT"


async def test_run_recipe_dispatches_to_the_kind(tmp_path: Path, monkeypatch) -> None:
    """gepa reaches its recipe, which seeds and then wants the datasets; the evaluate
    recipe runs in `test_evaluate`, gepa in `test_gepa_recipe`, the gradient recipes in
    `test_loop`."""
    assert set(KINDS) == {"dapo", "dr-grpo", "cispo", "gepa", "fst", "evaluate"}
    monkeypatch.chdir(tmp_path)
    opened = Run.open(BLUEPRINTS / "gepa", root=tmp_path / "runs")
    with pytest.raises(NoSuchDataset, match="aime-train"):
        await run_recipe(opened)
    assert (opened.directory / record.MODULES / "seed" / "solving" / "SKILL.md").is_file()


def test_open_refuses_an_existing_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("shipyard.run._run_id", lambda name: f"{name}__fixed00")
    Run.open(DAPO, root=tmp_path)
    with pytest.raises(RunRefused):
        Run.open(DAPO, root=tmp_path)


def test_open_refuses_a_broken_blueprint_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Run.open(tmp_path / "nothing", root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_state_of_all_four_states() -> None:
    mine = os.getpid()
    assert state_of({"failed": True, "finished_at": "now", "pid": mine}) == "failed"
    assert state_of({"finished_at": "now", "failed": False, "pid": mine}) == "finished"
    assert state_of({"pid": mine}) == "running"
    gone = 2**22 + 12345
    assert not alive(gone)
    assert state_of({"pid": gone}) == "died"
    assert state_of({}) == "died"


def test_runs_root_precedence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHIPYARD_RUNS", raising=False)
    assert runs_root(None) == tmp_path / "runs"
    monkeypatch.setenv("SHIPYARD_RUNS", str(tmp_path / "env"))
    assert runs_root(None) == tmp_path / "env"
    assert runs_root(tmp_path / "flag") == tmp_path / "flag"
