"""Tests for `shipyard run --dry`: the config printed with the values the run takes, defaults
included and the whole loadable as TOML, and the checks the run makes before it opens anything,
with no run made. The `fakes` fixture's Tinker service says the model's context; nothing else is
asked."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from tinker.types import GetServerCapabilitiesResponse, SupportedModel
from typer.testing import CliRunner

from proxies import MODEL
from shipyard.cli import app
from shipyard.config import Config, Said, load, shown
from tests.runs import GRPO, GRPO_FILE, run_toml


def _dry(home: Path, *given: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, ["run", str(home), str(GRPO_FILE), *given, "--dry"])
    return result.exit_code, result.stdout, result.stderr


@pytest.fixture
def context(fakes: Any) -> None:
    """The fake Tinker service says its model's context: 32768 tokens."""
    _, service = fakes
    found = SupportedModel(model_name=MODEL, trainable=True, max_context_length=32768)
    service.capabilities = GetServerCapabilitiesResponse(supported_models=[found])


def test_the_config_shown_is_every_key_and_loads_back_as_the_same_config(tmp_path: Path) -> None:
    home = run_toml(tmp_path, "test_group_size = 3")
    config = load(home, GRPO.Recipe)
    text = shown(config)
    assert 'model = "Qwen/Qwen3-8B"\n' in text and "*" not in text, "no key is marked"
    assert 'provider = "tinker"\n' in text and "max_tokens = 8192\n" in text
    assert "# from_checkpoint = none\n" in text, "TOML has no null"
    assert "test_group_size = 3\n" in text and "[checkpoints]\nevery = 1\n" in text
    assert 'harness = { name = "pi@0.85.1", env = {}, kwargs = {} }\n' in text
    again = Config[GRPO.Recipe].model_validate(tomllib.loads(text))  # type: ignore[valid-type]
    assert again.model_dump() == config.model_dump()
    values = {("data", "seed"): 7, ("models", "agent", "from_checkpoint"): Said("the base")}
    taken = shown(config, values)
    assert "seed = 7\n" in taken and "# from_checkpoint = the base\n" in taken


def test_dry_prints_the_config_with_what_the_run_makes_of_it_and_makes_no_run(
    tmp_path: Path, context: None
) -> None:
    home = run_toml(tmp_path)
    code, out, err = _dry(home, "--data.seed", "3")
    assert code == 0, err
    assert out.startswith(f"# {home / 'run.toml'} with --data.seed 3, as GRPOTrainer reads it")
    assert "seed = 3\n" in out, "an override is the run's own setting, not a default"
    assert 'renderers = ["qwen3"]\n' in out, "the renderer a lane with none takes"
    assert "max_context = 32768\n" in out, "the model's context, as its backend says it"
    assert "# timeout = none\n" in out, "the fixture task sets no limit"
    assert "setup_timeout = 360.0\n" in out, "pi's own"
    assert "checked: no run was made" in err and "checked" not in out
    assert not (tmp_path / "runs").exists() and not (tmp_path / "jobs").exists()


def test_the_rank_a_trained_model_takes_is_shown_unless_the_file_sets_it(
    tmp_path: Path, context: None
) -> None:
    code, out, err = _dry(run_toml(tmp_path, **{"lora_rank = 32\n": ""}))
    assert code == 0, err
    assert "lora_rank = 32\n" in out


def test_a_context_the_backend_cannot_say_stops_the_dry_run_with_its_reason(
    tmp_path: Path, fakes: Any
) -> None:
    _, service = fakes
    service.capabilities = None  # the fake service raises "no capabilities here"
    code, out, err = _dry(run_toml(tmp_path))
    assert code == 1
    assert "max_context = 0\n" in out, "as the file has it, since it could not be read"
    said = "[models.agent] max_context: tinker did not say the context of Qwen/Qwen3-8B: "
    assert said + "RuntimeError: no capabilities here\n" in err
    assert "the run would stop as it opens its models; no run was made" in err
    assert "could not read the context length" not in err, "the warning is the note's, once"


def test_dry_refuses_what_the_run_would_and_still_shows_the_config(
    tmp_path: Path, context: None
) -> None:
    home = run_toml(tmp_path, '[models.critic]\nmodel = "Qwen/Qwen3-32B"')
    code, out, err = _dry(home)
    assert code == 1 and "[models.critic]: GRPOTrainer never reads it" in err
    assert "[models.critic]\n" in out, "shown before the check that refused it"
    code, out, err = _dry(run_toml(tmp_path), "--data.train", "missing")
    assert code == 1 and "no dataset at tasks/missing" in err
    code, out, err = _dry(home, "--data.trian", "x")
    assert code == 1 and "[data] trian: unknown key" in err and not out


def test_dry_starts_no_wandb_run(
    tmp_path: Path, context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("wandb")

    def refused(**_: object) -> None:
        raise AssertionError("a dry run started a wandb run")

    monkeypatch.setattr("wandb.init", refused)
    home = run_toml(tmp_path, '[logging.wandb]\nproject = "p"')
    code, out, err = _dry(home)
    assert code == 0, err
    assert '[logging.wandb]\nproject = "p"\n# entity = none\n' in out


def test_the_tasks_own_timeouts_are_said_as_their_range(tmp_path: Path) -> None:
    from shipyard.run.dry import timeout

    tasks = []
    for name, agent in (
        ("short", "[agent]\ntimeout_sec = 120.0\n"),
        ("long", "[agent]\ntimeout_sec = 3000.0\n"),
        ("open", ""),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.toml").write_text(f'schema_version = "1.4"\n{agent}')
        tasks.append(tmp_path / name)
    assert timeout(tasks) == Said("each task's own: 120 to 3000 s, no limit for 1 task")
    assert timeout(tasks[:1]) == 120.0 and timeout(tasks[2:]) is None, "one value; no limit"
    assert timeout([]) == Said("each task's own")
