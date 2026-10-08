"""`shipyard run --smoke` on fakes: the overrides (one task, the first of the first batch,
one rollout, one job, no gradient, checkpoint or reflection), the report's lines, the run
directory it leaves, and the exit codes. The exit code is 0 only when a trial came back with
a record (served) or a verdict (provider-served)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from shipyard import record, serving, session
from shipyard.cli import app
from shipyard.config import load
from shipyard.data import batches
from shipyard.modules import seed
from shipyard.run import Run
from shipyard.smoke import Smoked, Trial, carried, planned, wiring
from tests.records import FakeProxy, made
from tests.test_gepa_recipe import BLUEPRINT as GEPA
from tests.test_gepa_recipe import VALID
from tests.trials import FIXTURES, FakeTrials, write_blueprint

BLUEPRINTS = Path(__file__).parent / "blueprints"
runner = CliRunner()


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A four-task dataset, no Tinker session, a wide terminal."""
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHIPYARD_RUNS", raising=False)
    monkeypatch.setenv("COLUMNS", "200")

    def no_session(metadata: Any = None) -> Any:
        raise AssertionError("a smoke run opens no training session")

    monkeypatch.setattr(session, "service_client", no_session)
    return tmp_path


def blueprint(here: Path, kind: str, **replacements: str) -> Path:
    text = (BLUEPRINTS / kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"aime-train"', '"four"').replace('"fixture"', '"four"')
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = here / "bp" / kind
    home.mkdir(parents=True, exist_ok=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    fake = FakeProxy()
    monkeypatch.setattr(serving, "Proxy", fake)
    return fake


@pytest.fixture
def trials(monkeypatch: pytest.MonkeyPatch) -> FakeTrials:
    fake = FakeTrials(default={"reward": 1.0, "seconds": 12.5})
    monkeypatch.setattr("shipyard.rollout._run_trial", fake)
    return fake


def only_run(here: Path) -> Path:
    [directory] = list((here / "runs").iterdir())
    return directory


def report_of(output: str) -> list[str]:
    lines = output.splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("smoke  "))
    return lines[start : start + 5]


# --------------------------------------------------------------------- the overrides


def test_the_smoke_task_is_the_first_of_the_first_batch(here: Path) -> None:
    cfg = load(blueprint(here, "dapo"))
    first = next(batches(cfg.datasets, size=cfg.data.batch_size, seed=0, epochs=1))
    assert planned(cfg) == first[:1] and len(first) == 2
    assert carried(cfg) is None


def test_a_gepa_smoke_carries_the_seed_the_search_would_score_first(here: Path) -> None:
    home = write_blueprint(here, GEPA.replace('"fixture"', '"four"'))
    shutil.copytree(VALID, home / "modules")
    found = carried(load(home))
    assert found is not None and found.digest == seed(home / "modules").digest


# ------------------------------------------------------------------- served runs


def test_a_served_smoke_samples_one_trial_and_reports_it(
    here: Path, proxy: FakeProxy, trials: FakeTrials
) -> None:
    proxy.default = [made("prompt", "said", seq=1, cached=2), made("prompt+", "more", seq=2)]
    result = runner.invoke(app, ["run", str(blueprint(here, "dapo")), "--smoke"])
    assert result.exit_code == 0, result.output
    directory = only_run(here)
    [config] = trials.configs
    task = Path(str(config.task.path)).name
    trial = f"{task}__{config.trial_name.split('__')[1]}"
    assert report_of(result.output) == [
        f"smoke  {directory.name}",
        "  sandbox     docker       proxy  http://host.docker.internal:8000  (local)",
        "  harness     pi@0.85.1    profile pi (model_api=openai-completions)",
        f"  trial       {trial}   records 2   served Qwen/Qwen3-8B   verdict 1.0",
        "  sandbox seconds 12.5   tokens prompt 15 cached 2 sampled 8",
    ]
    assert Run.read(directory)["smoke"] is True and Run.read(directory)["failed"] is False
    [job] = list(record.read(directory / record.JOBS))
    assert job["purpose"] == "smoke" and job["trials"] == 1 and job["tasks"] == 1
    assert (
        not (directory / record.METRICS).exists() and not (directory / "checkpoints.jsonl").exists()
    )
    assert proxy.swapped == [] and proxy.closed == 1
    assert config.agent.env["OPENAI_BASE_URL"].endswith(f"/r/trial/{config.trial_name}/v1")


def test_a_served_smoke_nothing_reached_reports_and_fails_with_the_dead_endpoint(
    here: Path, proxy: FakeProxy, trials: FakeTrials
) -> None:
    proxy.default = []
    result = runner.invoke(app, ["run", str(blueprint(here, "dapo")), "--smoke"])
    assert result.exit_code == 1
    lines = report_of(result.output)
    assert lines[1].endswith("proxy  http://host.docker.internal:8000  (local)")
    assert lines[3].endswith("records 0   served nothing   verdict masked env_error")
    assert lines[4] == "  sandbox seconds 12.5   tokens prompt 0 cached 0 sampled 0"
    note = Run.read(only_run(here))
    assert note["error"].startswith("NothingServed: job ") and "no trial reached" in note["error"]
    assert "no trial reached the proxy" in result.output and "Traceback" not in result.output


def test_a_tunnelled_smoke_names_the_tunnel_and_claude_codes_wiring(
    here: Path, proxy: FakeProxy, trials: FakeTrials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shipyard.proxy.tunnel.found", lambda: ("/opt/bin/cloudflared", []))
    home = blueprint(
        here, "dapo", **{'harness = "pi@0.85.1"': 'harness = "claude-code"\nsandbox = "modal"'}
    )
    result = runner.invoke(app, ["run", str(home), "--smoke"])
    assert result.exit_code == 0, result.output
    lines = report_of(result.output)
    assert lines[1] == (
        "  sandbox     modal        proxy  https://fake-words.trycloudflare.com  (tunnel)"
    )
    assert lines[2] == "  harness     claude-code  profile claude-code (dialect=anthropic)"
    assert Run.read(only_run(here))["proxy"] == {
        "placement": "tunnel",
        "origin": "https://fake-words.trycloudflare.com",
    }


# ----------------------------------------------------------- provider-served runs


def test_a_provider_served_smoke_passes_on_a_verdict(here: Path, trials: FakeTrials) -> None:
    trials.default = {"reward": 0.0, "tokens": (100, 40, 7), "provider": "openrouter"}
    result = runner.invoke(app, ["run", str(blueprint(here, "evaluate")), "--smoke"])
    assert result.exit_code == 0, result.output
    lines = report_of(result.output)
    assert lines[1] == (
        "  sandbox     modal        proxy  none (the harness calls openrouter directly)"
    )
    assert lines[2] == "  harness     pi@0.85.1    profile none (provider-served)"
    assert lines[3].endswith("records —   served openrouter   verdict 0.0")
    assert lines[4] == "  sandbox seconds 0.0   tokens prompt 100 cached 40 sampled 7"
    assert len(trials.configs) == 1, "one rollout, though group_size is 2"


def test_a_provider_served_smoke_with_no_verdict_reports_and_fails(
    here: Path, trials: FakeTrials
) -> None:
    trials.breaks = ("a", "b", "c", "d")
    result = runner.invoke(app, ["run", str(blueprint(here, "evaluate")), "--smoke"])
    assert result.exit_code == 1
    lines = report_of(result.output)
    assert lines[3].endswith("verdict masked env_error")
    note = Run.read(only_run(here))
    assert note["error"] == "SmokeFailed: no trial of the smoke job came back with a verdict"
    assert "Traceback" not in result.output


def test_a_gepa_smoke_scores_the_seed_and_reflects_on_nothing(
    here: Path, trials: FakeTrials
) -> None:
    home = write_blueprint(here, GEPA.replace('"fixture"', '"four"'))
    shutil.copytree(VALID, home / "modules")
    result = runner.invoke(app, ["run", str(home), "--smoke"])
    assert result.exit_code == 0, result.output
    [job] = list(record.read(only_run(here) / record.JOBS))
    assert job["purpose"] == "smoke" and job["modules"] == seed(home / "modules").digest
    assert [config.agent.name for config in trials.configs] == ["pi"]


# --------------------------------------------------------------------- the report


def test_the_report_and_what_passes() -> None:
    graded = Trial("t__1", None, "openrouter", "1.0", True)
    masked = Trial("t__2", 0, "nothing", "masked env_error", False)
    smoked = Smoked("r", "docker", "p", "h", "w", proxied=False, trials=[graded])
    assert smoked.passed and smoked.lines()[3] == (
        "  trial       t__1   records —   served openrouter   verdict 1.0"
    )
    assert not Smoked("r", "docker", "p", "h", "w", proxied=True, trials=[masked]).passed
    served = Trial("t__3", 3, "base", "masked timeout", False)
    assert Smoked("r", "docker", "p", "h", "w", proxied=True, trials=[served]).passed


def test_the_wiring_names_the_profile_and_what_it_adds(here: Path) -> None:
    def cfg(harness: str) -> Any:
        return load(blueprint(here, "dapo", **{'"pi@0.85.1"': f'"{harness}"'}))

    assert wiring(cfg("pi@0.85.1")) == "pi (model_api=openai-completions)"
    assert wiring(cfg("terminus-2")) == "terminus-2"
    assert wiring(cfg("swe-agent@1")) == "generic"
    assert wiring(cfg("opencode@1.18.35")) == "opencode (provider=shipyard, opencode_config)"
    assert wiring(cfg("claude-code")) == "claude-code (dialect=anthropic)"
