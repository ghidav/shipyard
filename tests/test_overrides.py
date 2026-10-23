"""Tests for setting a config's keys on the command line, as `shipyard run ... --data.train X`:
how the flags are read, what they set, and the run's copy of its config, which a resume reads."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from shipyard import Harness
from shipyard.cli import app, overrides_in
from shipyard.config import ConfigError, load
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.directory import continued, open_run
from tests.runs import GRPO, GRPO_FILE, run_toml

#: A color code, which rich writes around what it highlights once a terminal is forced, as on CI.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: A trainer file of one's own, which takes the `[recipe]` keys of `run_toml`'s config.
MINE = """from shipyard import Harness, Trainer


class Mine(Trainer):
    class Recipe(Trainer.Recipe):
        harness: Harness
        group_size: int

    async def step(self) -> None:
        pass
"""


def _trained(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stand in for training, and return the config each run would have trained with."""
    trained: list[Any] = []
    monkeypatch.setattr("shipyard.cli._train", lambda directory, config: trained.append(config))
    return trained


def test_dotted_flags_are_taken_out_and_the_rest_is_left_to_typer() -> None:
    found, rest = overrides_in(
        ["recipes/dapo", "--data.train", "swe-bench", "--recipe.steps=2", "--root", "runs.v2"]
    )
    assert found == {"data.train": "swe-bench", "recipe.steps": "2"}
    assert rest == ["recipes/dapo", "--root", "runs.v2"], "a dot in a value makes no flag"
    assert overrides_in(["--", "--data.train", "x"]) == ({}, ["--", "--data.train", "x"])


@pytest.mark.parametrize("args", [["--data.train"], ["--data.train", "--root", "runs"]])
def test_a_dotted_flag_needs_a_value(args: list[str]) -> None:
    with pytest.raises(ValueError, match="--data.train needs a value"):
        overrides_in(args)


def test_a_value_is_read_as_toml_and_otherwise_as_text(tmp_path: Path) -> None:
    config = load(
        run_toml(tmp_path),
        GRPO.Recipe,
        overrides={
            "data.train": '["four", "swe-bench"]',
            "data.test": "swe-bench",
            "recipe.steps": "3",
            "models.agent.learning_rate": "1e-5",
        },
    )
    assert config.data.train == ["four", "swe-bench"] and config.data.test == "swe-bench"
    assert config.recipe.steps == 3 and config.models["agent"].learning_rate == 1e-5


def test_a_mistyped_key_is_refused_as_the_config_would_be(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refused:
        load(run_toml(tmp_path), GRPO.Recipe, overrides={"data.trian": "swe-bench"})
    assert refused.value.problems == ["[data] trian: unknown key"]
    with pytest.raises(ConfigError, match="--data.train.name: train is a key, not a table"):
        load(run_toml(tmp_path), GRPO.Recipe, overrides={"data.train.name": "x"})
    with pytest.raises(ConfigError, match=r"\[agent\]: unknown table: it moved, to \[models"):
        load(run_toml(tmp_path), GRPO.Recipe, overrides={"agent.model": "Qwen/Qwen3-32B"})


def test_the_run_keeps_its_overrides_in_its_config_and_a_resume_reads_them(
    tmp_path: Path,
) -> None:
    home = run_toml(tmp_path)
    given = {"data.train": "swe-bench", "recipe.steps": "3"}
    directory, config = open_run(home, root=tmp_path / "runs", trainer=GRPO_FILE, overrides=given)
    assert config.data.train == "swe-bench"
    written = (directory / "run.toml").read_text(encoding="utf-8")
    first = f"# {home / 'run.toml'} with --data.train swe-bench --recipe.steps 3"
    assert written.splitlines()[0] == first
    assert tomllib.loads(written)["data"]["train"] == "swe-bench"
    resumed, again = continued(directory, Checkpoint("step-1", 1))
    assert (again.data.train, again.recipe.steps) == ("swe-bench", 3)
    assert (resumed / "run.toml").read_text(encoding="utf-8") == written


def test_without_overrides_the_config_is_copied_as_it_is(tmp_path: Path) -> None:
    home = run_toml(tmp_path)
    directory, _ = open_run(home, root=tmp_path / "runs", trainer=GRPO)
    assert (directory / "run.toml").read_bytes() == (home / "run.toml").read_bytes()


def test_the_cli_takes_overrides_beside_its_trainer_and_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = run_toml(tmp_path)
    (home / "mine.py").write_text(MINE, encoding="utf-8")
    trained = _trained(monkeypatch)
    runs = tmp_path / "runs"
    argv = ["run", str(home), str(home / "mine.py"), "--data.train", "swe-bench", "--root"]
    result = CliRunner().invoke(app, [*argv, str(runs)])
    assert result.exit_code == 0, result.output
    assert "trained by Mine" in result.output
    (config,) = trained
    assert config.data.train == "swe-bench" and next(runs.iterdir()).is_dir()


def test_a_models_keys_and_the_recipes_own_are_set_as_any_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, runs = run_toml(tmp_path), tmp_path / "runs"
    trained = _trained(monkeypatch)
    given = ["--models.agent.model", "Qwen/Qwen3-32B", "--recipe.harness", "pi@0.86.0"]
    result = CliRunner().invoke(
        app, ["run", str(home), str(GRPO_FILE), *given, "--root", str(runs)]
    )
    assert result.exit_code == 0, result.output
    (config,) = trained
    agent = config.models["agent"]
    assert agent.model == "Qwen/Qwen3-32B" and agent.learning_rate == 2e-5, "the rest is kept"
    assert config.recipe.harness == Harness(name="pi@0.86.0"), "a name, read as text"


def test_the_cli_refuses_a_mistyped_key_before_opening_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, runs = run_toml(tmp_path), tmp_path / "runs"
    trained = _trained(monkeypatch)
    argv = ["run", str(home), str(GRPO_FILE), "--data.trian=x", "--root", str(runs)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "[data] trian: unknown key" in result.output
    assert not trained and not runs.exists()
    result = CliRunner().invoke(app, ["run", str(home), "--data.train"])
    assert result.exit_code == 2 and "--data.train needs a value" in ANSI.sub("", result.output)
