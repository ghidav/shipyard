"""The commands, through typer's runner: check, run, runs, show, and where runs go."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shipyard import record
from shipyard.cli import app
from shipyard.run import Run
from shipyard.show import state_of
from tests.trials import FakeTrials, fixture_tasks

BLUEPRINTS = Path(__file__).parent / "blueprints"
DAPO = BLUEPRINTS / "dapo"
EVALUATE = BLUEPRINTS / "evaluate"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs from a directory holding only the fixture tasks, with no root from
    the shell and a terminal wide enough that rich never folds a run id across lines."""
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHIPYARD_RUNS", raising=False)
    monkeypatch.setenv("COLUMNS", "200")


async def _finishes(run: Run) -> None:
    run.log(reward=1.0)
    run.note(hello="world")


def _only_run(root: Path) -> Path:
    [directory] = [found for found in root.iterdir() if found.is_dir()]
    return directory


def _hosted_dapo(tmp_path: Path) -> Path:
    """The dapo fixture over the fixture dataset with a provider-served model, so `check`
    passes and what `run` records is the loop's own refusal, before any session opens."""
    home = tmp_path / "blueprints" / "dapo"
    home.mkdir(parents=True)
    text = (DAPO / "run.toml").read_text(encoding="utf-8")
    text = text.replace("lora_rank = 32", 'lora_rank = 32\nprovider = "openrouter"')
    text = text.replace('dataset = "aime-train"', 'dataset = "fixture"')
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


SECTIONS = ("process", "metrics", "jobs", "checkpoints", "costs")
COSTS = {
    "parties": {"m": {"trials": 2, "input_tokens": 15, "cache_tokens": 0, "output_tokens": 3}},
    "sandbox": {"docker": {"trials": 2, "seconds": 1.5}},
}


def _headings(output: str) -> list[str]:
    """The section names printed at column 0, in order."""
    return [
        line.split()[0]
        for line in output.splitlines()
        if line.split()[:1] and line[0] != " " and line.split()[0] in SECTIONS
    ]


def _recorded(root: Path) -> Path:
    """A finished run with two metrics, twelve jobs, two checkpoints and a costs document."""
    with Run.open(DAPO, root=root) as run:
        run.log(reward=0.5)
        run.log(reward=0.7, step=2)
        for n in range(1, 13):
            record.append(run.directory / record.JOBS, {"name": f"job-{n:02d}", "reward": n / 12})
        record.append(run.directory / record.JOBS, {"name": "job-13", "reward": 1.0, "extra": [1]})
        for n in (1, 2):
            record.append(run.directory / record.CHECKPOINTS, {"name": f"ckpt-{n}", "step": n})
        record.write_json(run.directory / record.COSTS, COSTS)
    return run.directory


def test_check_ok() -> None:
    """The resolved config, then nothing: the ok lines are hidden unless asked for."""
    result = runner.invoke(app, ["check", str(EVALUATE)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("[model]\n")
    assert 'kind = "evaluate"' in result.output and "ttl_hours = 168.0" in result.output
    assert "ok  " not in result.output and "blocked" not in result.output
    assert "recipe evaluate" not in result.output
    verbose = runner.invoke(app, ["check", str(EVALUATE), "--verbose"])
    assert verbose.exit_code == 0, verbose.output
    assert verbose.output.startswith("[model]\n")
    assert "ok  recipe evaluate" in verbose.output
    assert "ok  dataset fixture: 2 tasks under tasks/fixture" in verbose.output


def test_check_prints_the_resolution_line_for_a_gradient_recipe(tmp_path: Path) -> None:
    """The acceptance check: the dapo fixture over the fixture dataset, served on this
    machine, prints the resolved config with its comment line, no ok lines, and exits 0
    with the probe's warning as its only finding."""
    home = tmp_path / "dapo"
    home.mkdir()
    text = (DAPO / "run.toml").read_text(encoding="utf-8")
    (home / "run.toml").write_text(text.replace('"aime-train"', '"fixture"'), encoding="utf-8")
    result = runner.invoke(app, ["check", str(home)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "[model]"
    comment = next(line for line in lines if line.startswith("#"))
    assert comment == (
        "# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28; "
        "degenerate groups dropped"
    )
    assert lines.index(comment) == lines.index("ttl_hours = 168.0") + 1
    findings = [line for line in lines if line.startswith(("ok  ", "warning  ", "blocked  "))]
    assert len(findings) == 1 and findings[0].startswith("warning  could not ask tinker")
    assert "ok  " not in result.output
    verbose = runner.invoke(app, ["check", str(home), "--verbose"])
    assert verbose.exit_code == 0
    assert "ok  recipe dapo" in verbose.output
    assert "ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox" in verbose.output


def test_check_blocks_a_missing_dataset_directory(tmp_path: Path) -> None:
    shutil.rmtree(tmp_path / "tasks")
    result = runner.invoke(app, ["check", str(EVALUATE)])
    assert result.exit_code == 1
    assert "blocked" in result.output and "no dataset at tasks/fixture" in result.output
    assert "harbor datasets download" in result.output


def test_check_reports_a_served_model_on_loopback_for_a_docker_sandbox(tmp_path: Path) -> None:
    """The acceptance check: a served evaluate blueprint over the fixture dataset with a
    docker sandbox is ok but for the probe, which cannot ask without a key."""
    home = tmp_path / "served"
    home.mkdir()
    text = (EVALUATE / "run.toml").read_text(encoding="utf-8")
    text = text.replace('provider = "openrouter"', 'provider = "tinker"')
    text = text.replace('sandbox = "modal"', 'sandbox = "docker"')
    (home / "run.toml").write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["check", str(home), "--verbose"])
    assert result.exit_code == 0, result.output
    assert "serving some-hosted-model on this machine for a docker sandbox" in result.output
    assert "warning  could not ask tinker whether it serves some-hosted-model" in result.output
    assert "blocked" not in result.output
    # The dapo fixture is served the same way; only its missing dataset blocks it.
    result = runner.invoke(app, ["check", str(DAPO), "--verbose"])
    assert result.exit_code == 1
    assert "serving Qwen/Qwen3-8B on this machine for a docker sandbox" in result.output
    assert result.output.count("blocked") == 1 and "no dataset at" in result.output
    hidden = runner.invoke(app, ["check", str(DAPO)])
    assert hidden.exit_code == 1 and "serving Qwen" not in hidden.output
    assert hidden.output.count("blocked") == 1 and 'kind = "dapo"' in hidden.output
    result = runner.invoke(app, ["run", str(DAPO)])
    assert result.exit_code == 1 and "nothing was started" in result.output


def test_check_blocked(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "run.toml").write_text('[model]\nname = "m"\n[recipe]\nkind = "dapo"\n')
    result = runner.invoke(app, ["check", str(broken)])
    assert result.exit_code == 1
    assert "blocked" in result.output and "[model]" not in result.output
    assert "[data]: table required" in result.output
    assert "[recipe] learning_rate: field required" in result.output
    result = runner.invoke(app, ["check", str(tmp_path / "missing")])
    assert result.exit_code == 1 and "blocked" in result.output


def test_check_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["check", str(EVALUATE), "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert list(report) == ["config", "findings"]
    assert list(report["config"]) == ["model", "data", "rollout", "recipe", "checkpoints"]
    assert report["config"]["recipe"] == {"kind": "evaluate", "modules": None}
    assert report["config"]["model"]["from_checkpoint"] is None
    assert report["config"]["checkpoints"] == {"every": 1, "ttl_hours": 168.0}
    findings = report["findings"]
    assert findings[0] == {"level": "ok", "text": "recipe evaluate"}
    assert all(found["level"] == "ok" for found in findings)
    result = runner.invoke(app, ["check", str(tmp_path / "missing"), "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["config"] is None
    assert [found["level"] for found in report["findings"]] == ["blocked"]


def test_run_refuses_a_blocked_blueprint(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "missing"), "--root", str(tmp_path / "r")])
    assert result.exit_code == 1
    assert "nothing was started" in result.output
    assert not (tmp_path / "r").exists()


def test_run_on_a_fixture_ends_failed_with_the_recipes_refusal(tmp_path: Path) -> None:
    """A gradient recipe over a provider's model: refused by the loop before a Tinker
    session opens, and the run records that refusal as its failure."""
    root = tmp_path / "r"
    result = runner.invoke(app, ["run", str(_hosted_dapo(tmp_path)), "--root", str(root)])
    assert result.exit_code == 1
    directory = _only_run(root)
    note = Run.read(directory)
    assert state_of(note) == "failed"
    assert note["error"].startswith("ValueError: [recipe] kind = 'dapo' trains the weights")
    assert "provider = 'openrouter'" in note["error"]
    assert directory.name in result.output and "failed" in result.output
    assert "ValueError" in result.output and "trains the weights" in result.output


def test_run_with_a_recipe_that_returns_ends_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shipyard.recipes.dapo.run", _finishes)
    root, home = tmp_path / "r", _hosted_dapo(tmp_path)
    result = runner.invoke(app, ["run", str(home), "--root", str(root)])
    assert result.exit_code == 0, result.output
    directory = _only_run(root)
    note = Run.read(directory)
    assert state_of(note) == "finished" and note["error"] is None and note["hello"] == "world"
    assert [row["seq"] for row in record.read(directory / record.METRICS)] == [1]
    assert "finished" in result.output

    listed = runner.invoke(app, ["runs", "--root", str(root)])
    assert listed.exit_code == 0
    assert directory.name in listed.output and "finished" in listed.output
    assert "dapo" in listed.output

    shown = runner.invoke(app, ["show", directory.name, "--root", str(root)])
    assert shown.exit_code == 0, shown.output
    assert "finished" in shown.output and _headings(shown.output) == ["process", "metrics"]
    assert str(home.resolve()) in shown.output


def test_run_stopped_by_sigterm_records_the_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stopped(run: Run) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(5)

    monkeypatch.setattr("shipyard.recipes.dapo.run", stopped)
    root = tmp_path / "r"
    before = signal.getsignal(signal.SIGTERM)
    result = runner.invoke(app, ["run", str(_hosted_dapo(tmp_path)), "--root", str(root)])
    assert signal.getsignal(signal.SIGTERM) is before
    assert result.exit_code == 130, result.output
    note = Run.read(_only_run(root))
    assert state_of(note) == "failed" and note["error"] == "stopped: SIGTERM"
    assert "stopped: SIGTERM" in result.output


def test_runs_and_show_with_nothing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "--root", str(tmp_path / "none")])
    assert result.exit_code == 0 and "no runs" in result.output
    result = runner.invoke(app, ["show", "nope__0000000", "--root", str(tmp_path / "none")])
    assert result.exit_code == 1 and "no run named" in result.output


def test_root_flag_env_and_default_are_honored_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shipyard.recipes.evaluate.run", _finishes)
    flagged, env = tmp_path / "flag", tmp_path / "env"

    assert runner.invoke(app, ["run", str(EVALUATE), "--root", str(flagged)]).exit_code == 0
    flag_id = _only_run(flagged).name

    monkeypatch.setenv("SHIPYARD_RUNS", str(env))
    assert runner.invoke(app, ["run", str(EVALUATE)]).exit_code == 0
    env_id = _only_run(env).name
    assert runner.invoke(app, ["run", str(EVALUATE), "--root", str(flagged)]).exit_code == 0
    assert len(list(flagged.iterdir())) == 2 and len(list(env.iterdir())) == 1

    listed = runner.invoke(app, ["runs"]).output
    assert env_id in listed and flag_id not in listed
    listed = runner.invoke(app, ["runs", "--root", str(flagged)]).output
    assert flag_id in listed and env_id not in listed

    assert runner.invoke(app, ["show", env_id]).exit_code == 0
    assert runner.invoke(app, ["show", flag_id]).exit_code == 1
    assert runner.invoke(app, ["show", flag_id, "--root", str(flagged)]).exit_code == 0
    assert runner.invoke(app, ["show", env_id, "--root", str(flagged)]).exit_code == 1

    monkeypatch.delenv("SHIPYARD_RUNS")
    assert runner.invoke(app, ["run", str(EVALUATE)]).exit_code == 0
    default_id = _only_run(tmp_path / "runs").name
    assert default_id in runner.invoke(app, ["runs"]).output
    assert runner.invoke(app, ["show", default_id]).exit_code == 0


def test_show_process_and_metrics_and_skips_absent_sections(tmp_path: Path) -> None:
    with Run.open(DAPO, root=tmp_path / "r") as run:
        run.log(reward=0.5)
        run.log(reward=0.7, step=2)
    shown = runner.invoke(app, ["show", run.id, "--root", str(tmp_path / "r")])
    assert shown.exit_code == 0, shown.output
    assert _headings(shown.output) == ["process", "metrics"]
    note = Run.read(run.directory)
    for text in ("finished", str(note["pid"]), note["started_at"], note["finished_at"]):
        assert text in shown.output
    assert note["version"] in shown.output and "dapo" in shown.output
    assert str(DAPO.resolve()) in shown.output and str(run.directory) in shown.output
    assert "2 rows" in shown.output and "0.7" in shown.output and "step" in shown.output
    # The last row only; any timestamp shown can spell "0.5" at the wrong second.
    scrubbed = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.+]+", "", shown.output)
    assert "0.5" not in scrubbed
    assert "error" not in shown.output

    bare = tmp_path / "bare"
    with Run.open(DAPO, root=bare):
        pass
    shown = runner.invoke(app, ["show", _only_run(bare).name, "--root", str(bare)])
    assert shown.exit_code == 0 and _headings(shown.output) == ["process"]


def test_show_prints_the_error_of_a_failed_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), Run.open(DAPO, root=tmp_path) as run:
        raise RuntimeError("the loop broke")
    shown = runner.invoke(app, ["show", run.id, "--root", str(tmp_path)])
    assert shown.exit_code == 0 and "failed" in shown.output
    assert "RuntimeError: the loop broke" in shown.output


def test_show_jobs_cuts_to_the_last_ten_unless_full(tmp_path: Path) -> None:
    directory = _recorded(tmp_path)
    shown = runner.invoke(app, ["show", directory.name, "--root", str(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert _headings(shown.output) == list(SECTIONS)
    assert "13 rows (last 10; --full for all)" in shown.output
    assert "job-04" in shown.output and "job-13" in shown.output
    assert "job-03" not in shown.output and "job-01" not in shown.output
    header = next(line for line in shown.output.splitlines() if "name" in line and "at" in line)
    assert (
        header.index("name") < header.index("at") < header.index("reward") < header.index("extra")
    )

    full = runner.invoke(app, ["show", directory.name, "--root", str(tmp_path), "--full"])
    assert full.exit_code == 0 and "13 rows" in full.output and "last 10" not in full.output
    assert all(f"job-{n:02d}" in full.output for n in range(1, 14))


def test_show_lists_every_checkpoint_and_pretty_prints_costs(tmp_path: Path) -> None:
    directory = _recorded(tmp_path)
    shown = runner.invoke(app, ["show", directory.name, "--root", str(tmp_path)])
    assert "2 rows" in shown.output and "ckpt-1" in shown.output and "ckpt-2" in shown.output
    assert '      "input_tokens": 15' in shown.output and '"seconds": 1.5' in shown.output


def test_show_json_has_the_five_sections_with_every_row(tmp_path: Path) -> None:
    directory = _recorded(tmp_path)
    for flags in ([], ["--full"]):
        shown = runner.invoke(
            app, ["show", directory.name, "--root", str(tmp_path), "--json", *flags]
        )
        assert shown.exit_code == 0, shown.output
        found = json.loads(shown.stdout)
        assert list(found) == list(SECTIONS)
        process = found["process"]
        assert process["state"] == "finished" and process["kind"] == "dapo"
        assert process["blueprint"] == str(DAPO.resolve())
        assert process["directory"] == str(directory) and process["error"] is None
        assert found["metrics"]["count"] == 2 and found["metrics"]["last"]["reward"] == 0.7
        assert found["jobs"]["count"] == 13 and len(found["jobs"]["rows"]) == 13
        assert found["jobs"]["rows"][0]["name"] == "job-01" and "at" in found["jobs"]["rows"][0]
        assert [row["name"] for row in found["checkpoints"]["rows"]] == ["ckpt-1", "ckpt-2"]
        assert found["checkpoints"]["count"] == 2 and found["costs"] == COSTS

    bare = tmp_path / "bare"
    with Run.open(DAPO, root=bare):
        pass
    shown = runner.invoke(app, ["show", _only_run(bare).name, "--root", str(bare), "--json"])
    found = json.loads(shown.stdout)
    assert found["process"]["state"] == "finished"
    assert [found[name] for name in SECTIONS[1:]] == [None, None, None, None]


def test_an_evaluate_run_shows_its_jobs_and_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run` on the evaluate fixture with Harbor's trial replaced: `show` renders the job
    rows with the new columns and the costs document as it was written."""
    monkeypatch.setenv("COLUMNS", "400")  # sixteen columns, none cut to an ellipsis
    monkeypatch.setattr(
        "shipyard.rollout._run_trial",
        FakeTrials(
            outcomes={
                "alpha": {"reward": 1.0, "tokens": (10, 0, 2), "provider": "openrouter"},
                "beta": {"reward": 0.0, "exception": "AgentTimeoutError"},
            }
        ),
    )
    root = tmp_path / "r"
    result = runner.invoke(app, ["run", str(EVALUATE), "--root", str(root)])
    assert result.exit_code == 0, result.output
    directory = _only_run(root)
    assert sorted(p.name for p in (tmp_path / "jobs").iterdir()) == [
        f"{directory.name}-0000",
        f"{directory.name}-0001",
    ]
    shown = runner.invoke(app, ["show", directory.name, "--root", str(root)])
    assert shown.exit_code == 0, shown.output
    assert _headings(shown.output) == ["process", "metrics", "jobs", "costs"]
    assert "jobs  2 rows" in shown.output
    header = next(line for line in shown.output.splitlines() if "purpose" in line)
    for column in ("job", "purpose", "batch", "tasks", "trials", "graded", "masked"):
        assert column in header
    for column in ("party", "input_tokens", "output_tokens", "sandbox", "sandbox_seconds"):
        assert column in header
    assert f"{directory.name}-0000" in shown.output and "rollout" in shown.output
    assert '{"timeout": 2}' in shown.output and "openrouter" in shown.output
    assert '"parties"' in shown.output and '"modal"' in shown.output
    assert "evaluation" in shown.output and "mean" in shown.output

    as_json = json.loads(
        runner.invoke(app, ["show", directory.name, "--root", str(root), "--json"]).stdout
    )
    rows = as_json["jobs"]["rows"]
    assert as_json["jobs"]["count"] == 2 and [row["batch"] for row in rows] == [0, 1]
    assert sum(row["graded"] for row in rows) == 2 and sum(row["trials"] for row in rows) == 4
    assert as_json["costs"]["parties"]["openrouter"]["input_tokens"] == 20
    assert as_json["costs"]["sandbox"]["modal"]["trials"] == 4
    final = as_json["metrics"]["last"]
    assert as_json["metrics"]["count"] == 3 and final["evaluation"] is True
    assert final["graded"] == 2 and final["masked"] == 2 and final["mean"] == 1.0
    assert "total" not in json.dumps(as_json["costs"])
