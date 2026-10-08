"""What `rollout` hands Harbor and what it leaves on disk: configs in plan order, one
agent per batch, a slot kept for a trial that could not run, the semaphore honoured."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import TrialConfig

from shipyard import ui
from shipyard.proxy.profiles import PROFILES, Profile
from shipyard.rollout import (
    MODAL_IMAGE_BUILDER,
    MODAL_IMAGE_BUILDER_VERSION,
    Served,
    allowlisted,
    laid_over,
    quiet_litellm,
    rollout,
    served_config,
)
from tests.records import FakeProxy
from tests.trials import FakeTrials


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    root = tmp_path / "tasks"
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return [root / name for name in names]


async def _rollout(tmp_path: Path, batch: list[Path], harbor: object, **options: object):
    asked: dict[str, object] = {
        "harness": "pi",
        "model": "m",
        "sandbox": "docker",
        "concurrency": 4,
        "rollouts": 1,
        "jobs_dir": tmp_path / "jobs",
        "run_trial": harbor,
    }
    asked.update(options)
    return await rollout(batch, "run-0000", **asked)  # type: ignore[arg-type]


async def test_configs_are_built_in_plan_order_with_harbors_fields(tmp_path: Path) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha", "beta")
    found = await _rollout(
        tmp_path,
        batch,
        harbor,
        harness="pi@0.85.1",
        model="anthropic/some-model",
        rollouts=3,
        env={"API_KEY": "k"},
        kwargs={"thinking": "high"},
        setup_timeout=360.0,
        timeout=1800.0,
    )
    assert found.job == "run-0000" and len(found) == 6
    assert found.plan == [(batch[0], 0), (batch[0], 1), (batch[0], 2)] + [
        (batch[1], n) for n in range(3)
    ]
    assert (
        sorted(config.task.path.name for config in harbor.configs) == ["alpha"] * 3 + ["beta"] * 3
    )
    for config in harbor.configs:
        assert isinstance(config, TrialConfig)
        assert config.trials_dir == tmp_path / "jobs" / "run-0000"
        assert config.agent.name == "pi" and config.agent.import_path is None
        assert config.agent.model_name == "anthropic/some-model"
        assert config.agent.kwargs == {"version": "0.85.1", "thinking": "high"}
        assert config.agent.env == {"API_KEY": "k"}
        assert config.agent.override_setup_timeout_sec == 360.0
        assert config.agent.override_timeout_sec == 1800.0
        assert config.environment.type == EnvironmentType.DOCKER
        assert config.environment.delete is True
        assert config.verifier.disable is False
    # The trials come back in plan order under the job, named as Harbor names them.
    assert [trial.parent for trial in found.trials] == [tmp_path / "jobs" / "run-0000"] * 6
    assert [trial.name.split("__")[0] for trial in found.trials] == ["alpha"] * 3 + ["beta"] * 3
    assert all(len(trial.name.split("__")[1]) == 7 for trial in found.trials)
    assert {trial.name for trial in found.trials} == {c.trial_name for c in harbor.configs}


async def test_the_harness_pin_is_split_and_a_colon_name_is_left_whole(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "alpha")
    harbor = FakeTrials()
    await _rollout(tmp_path, batch, harbor, harness="pi")
    assert harbor.configs[0].agent.name == "pi" and harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    await _rollout(tmp_path, batch, harbor, harness="acp:some.module:Agent@1")
    assert harbor.configs[0].agent.name == "acp:some.module:Agent@1"
    assert harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    await _rollout(tmp_path, batch, harbor, harness="pi@0.85.1", kwargs={"version": "0.90"})
    assert harbor.configs[0].agent.kwargs == {"version": "0.90"}


async def test_verification_can_be_switched_off_and_modal_named(tmp_path: Path) -> None:
    harbor = FakeTrials()
    await _rollout(tmp_path, _batch(tmp_path, "alpha"), harbor, sandbox="modal", verify=False)
    assert harbor.configs[0].environment.type == EnvironmentType.MODAL
    assert harbor.configs[0].verifier.disable is True


async def test_a_trial_that_could_not_run_keeps_its_slot(tmp_path: Path) -> None:
    harbor = FakeTrials(breaks=("beta",))
    batch = _batch(tmp_path, "alpha", "beta", "gamma")
    found = await _rollout(tmp_path, batch, harbor, rollouts=2)
    assert len(found) == 6
    assert [trial.name.split("__")[0] for trial in found.trials] == (
        ["alpha"] * 2 + ["beta"] * 2 + ["gamma"] * 2
    )
    assert [trial.is_dir() for trial in found.trials] == [True, True, False, False, True, True]
    # Nothing of ours in the job directory: only what Harbor's stand-in wrote.
    job_dir = tmp_path / "jobs" / "run-0000"
    assert sorted(p.name for p in job_dir.iterdir()) == sorted(
        trial.name for trial in found.trials if trial.is_dir()
    )
    assert sorted(p.name for p in job_dir.rglob("*") if p.is_file()) == ["result.json"] * 4


async def test_concurrency_never_exceeds_the_semaphore(tmp_path: Path) -> None:
    batch = _batch(tmp_path, *[f"t{n}" for n in range(6)])
    harbor = FakeTrials(dwell=0.01)
    await _rollout(tmp_path, batch, harbor, concurrency=2)
    assert harbor.peak == 2 and len(harbor.configs) == 6
    harbor = FakeTrials(dwell=0.005)
    await _rollout(tmp_path, batch, harbor, concurrency=1)
    assert harbor.peak == 1


@pytest.mark.parametrize("harness", ["", "   "])
async def test_an_empty_harness_is_refused_before_anything_is_written(
    tmp_path: Path, harness: str
) -> None:
    harbor = FakeTrials()
    with pytest.raises(ValueError, match="oracle"):
        await _rollout(tmp_path, _batch(tmp_path, "alpha"), harbor, harness=harness)
    assert harbor.configs == [] and not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("knob", ["rollouts", "concurrency"])
async def test_a_batch_that_could_not_happen_is_refused(tmp_path: Path, knob: str) -> None:
    with pytest.raises(ValueError, match=knob):
        await _rollout(tmp_path, _batch(tmp_path, "alpha"), FakeTrials(), **{knob: 0})


async def test_the_modal_builder_is_set_unless_the_shell_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(tmp_path, "alpha")
    await _rollout(tmp_path, batch, FakeTrials(), sandbox="docker")
    assert MODAL_IMAGE_BUILDER not in os.environ
    await _rollout(tmp_path, batch, FakeTrials(), sandbox="modal")
    assert os.environ[MODAL_IMAGE_BUILDER] == MODAL_IMAGE_BUILDER_VERSION
    monkeypatch.setenv(MODAL_IMAGE_BUILDER, "legacy")
    await _rollout(tmp_path, batch, FakeTrials(), sandbox="modal")
    assert os.environ[MODAL_IMAGE_BUILDER] == "legacy"


async def test_every_finished_trial_is_reported_to_the_progress_with_its_mask(
    tmp_path: Path,
) -> None:
    class Counting(ui.Reporter):
        def __init__(self) -> None:
            self.seen: dict[Path, bool] = {}

        def finished(self, trial: Path, *, masked: bool) -> None:
            self.seen[trial] = masked

    progress = Counting()
    batch = _batch(tmp_path, "alpha", "beta")
    found = await _rollout(tmp_path, batch, FakeTrials(breaks=("beta",)), progress=progress)
    assert sorted(progress.seen) == sorted(found.trials)
    assert [progress.seen[trial] for trial in found.trials] == [False, True]


async def test_a_stray_cancellation_is_one_trials_mask_not_the_batchs(tmp_path: Path) -> None:
    async def stray(config: TrialConfig) -> None:
        if config.task.path.name == "beta":
            raise asyncio.CancelledError()
        (Path(config.trials_dir) / config.trial_name).mkdir(parents=True)

    found = await _rollout(tmp_path, _batch(tmp_path, "alpha", "beta", "gamma"), stray)
    assert [trial.is_dir() for trial in found.trials] == [True, False, True]


async def test_a_real_cancellation_still_stops_the_batch(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def slow(config: TrialConfig) -> None:
        started.set()
        await asyncio.sleep(10)

    running = asyncio.ensure_future(_rollout(tmp_path, _batch(tmp_path, "alpha"), slow))
    await started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


def test_litellm_is_told_not_to_print_its_provider_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(suppress_debug_info=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    quiet_litellm()
    assert fake.suppress_debug_info is True


# ------------------------------------------------------------------- a served model


def _task_toml(tmp_path: Path, name: str, text: str) -> Path:
    task = tmp_path / "tasks" / name
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(text, encoding="utf-8")
    return task


async def test_every_trial_is_pointed_at_the_address_that_names_it(tmp_path: Path) -> None:
    """One address per trial, over the batch's env; the shared agent config is copied,
    so no trial is handed another's address."""
    harbor = FakeTrials()
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="secret")
    found = await _rollout(
        tmp_path,
        _batch(tmp_path, "alpha", "beta"),
        harbor,
        harness="pi@0.85.1",
        model="openai/Qwen/Qwen3-8B",
        rollouts=2,
        env={"OPENAI_BASE_URL": "https://api.openai.com/v1", "HTTPS_PROXY": "corp"},
        served=Served(proxy, PROFILES["pi"]),
    )
    assert len(harbor.configs) == 4
    addresses = [config.agent.env["OPENAI_BASE_URL"] for config in harbor.configs]
    assert len(set(addresses)) == 4 and sorted(proxy.addressed) == sorted(
        c.trial_name for c in harbor.configs
    )
    for config, address in zip(harbor.configs, addresses, strict=True):
        assert address == f"http://host.docker.internal:8000/r/trial/{config.trial_name}/v1"
        assert config.agent.env["OPENAI_API_KEY"] == "secret"
        assert config.agent.env["HTTPS_PROXY"] == "corp", "the batch's other env survives"
        assert config.agent.kwargs == {"version": "0.85.1", "model_api": "openai-completions"}
        assert config.agent.model_name == "openai/Qwen/Qwen3-8B"
        assert config.agent.extra_allowed_hosts == []
    assert {trial.name for trial in found.trials} == {c.trial_name for c in harbor.configs}


def test_claude_code_is_handed_the_anthropic_wire_without_the_v1(tmp_path: Path) -> None:
    harbor = FakeTrials()
    proxy = FakeProxy(origin="https://x.trycloudflare.com", token="k")
    config = _config(tmp_path, "alpha", harness="claude-code", env={"KEEP": "1"})
    pointed = served_config(config, Served(proxy, PROFILES["claude-code"], fill_context=True))
    assert pointed.agent.env == {
        "KEEP": "1",
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": f"https://x.trycloudflare.com/r/trial/{config.trial_name}",
        "ANTHROPIC_API_KEY": "k",
    }
    assert pointed.agent.kwargs == {} and pointed.agent.name == "claude-code"
    plain = served_config(config, Served(proxy, PROFILES["claude-code"]))
    assert "DISABLE_COMPACT" not in plain.agent.env, "the profile's env only when filling"
    assert config.agent.env == {"KEEP": "1"}, "the batch's config was not written into"
    assert harbor.configs == []


def test_opencode_gets_its_providers_env_and_keeps_a_blueprints_own_config(
    tmp_path: Path,
) -> None:
    """The env the profile's opencode config names is the env the trial is handed; a
    blueprint's own `opencode_config` keeps its keys, and the profile's win where both
    set one."""
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="k")
    mine = {
        "provider": {"shipyard": {"models": {"Qwen/Qwen3-8B": {"limit": {"context": 32768}}}}},
        "agent": {"title": {"disable": False}, "build": {"steps": 40}},
    }
    config = _config(tmp_path, "alpha", harness="opencode", kwargs={"opencode_config": mine})
    pointed = served_config(config, Served(proxy, PROFILES["opencode"]))
    address = f"http://host.docker.internal:8000/r/trial/{config.trial_name}/v1"
    assert pointed.agent.env == {"SHIPYARD_BASE_URL": address, "SHIPYARD_API_KEY": "k"}
    merged = pointed.agent.kwargs["opencode_config"]
    assert merged["provider"]["shipyard"] == {
        "npm": "@ai-sdk/openai-compatible",
        "env": ["SHIPYARD_API_KEY"],
        "options": {"baseURL": "${SHIPYARD_BASE_URL}"},
        "models": {"Qwen/Qwen3-8B": {"limit": {"context": 32768}}},
    }
    assert merged["agent"] == {"title": {"disable": True}, "build": {"steps": 40}}
    assert "npm" not in str(config.agent.kwargs) and "npm" not in str(mine), "nothing written into"
    merged["agent"]["title"]["disable"] = False
    assert PROFILES["opencode"].kwargs["opencode_config"]["agent"]["title"]["disable"] is True
    profile_env = PROFILES["opencode"].kwargs["opencode_config"]["provider"]["shipyard"]["env"]
    assert merged["provider"]["shipyard"]["env"] is not profile_env, "a list is copied too"


def test_tables_are_laid_over_key_by_key_and_anything_else_is_replaced() -> None:
    under = {"a": {"x": 1, "y": {"p": 1}}, "b": [1], "c": {"k": 1}, "d": 1}
    over = {"a": {"y": {"q": 2}, "z": 3}, "b": [2], "c": 5, "d": {"k": 2}}
    assert laid_over(under, over) == {
        "a": {"x": 1, "y": {"p": 1, "q": 2}, "z": 3},
        "b": [2],
        "c": 5,
        "d": {"k": 2},
    }
    assert under == {"a": {"x": 1, "y": {"p": 1}}, "b": [1], "c": {"k": 1}, "d": 1}
    assert laid_over({}, {}) == {} and laid_over({"a": 1}, {}) == {"a": 1}


def test_a_task_in_allowlist_mode_gets_the_proxys_host(tmp_path: Path) -> None:
    allow = _task_toml(
        tmp_path,
        "allow",
        '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    )
    assert allowlisted(allow)
    phase = _task_toml(tmp_path, "phase", '[agent]\nnetwork_mode = "allowlist"\n')
    assert allowlisted(phase)
    public = _task_toml(tmp_path, "public", '[environment]\nnetwork_mode = "public"\n')
    assert not allowlisted(public)
    overridden = _task_toml(
        tmp_path,
        "overridden",
        '[environment]\nnetwork_mode = "allowlist"\n[agent]\nnetwork_mode = "public"\n',
    )
    assert not allowlisted(overridden)
    assert not allowlisted(tmp_path / "tasks" / "nothing")
    assert not allowlisted(_task_toml(tmp_path, "broken", "[environment\n"))

    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="k")
    pointed = served_config(_config(tmp_path, "allow"), Served(proxy, Profile()))
    assert pointed.agent.extra_allowed_hosts == ["host.docker.internal"]
    had = _config(tmp_path, "allow").model_copy()
    had.agent.extra_allowed_hosts = ["host.docker.internal", "pypi.org"]
    assert served_config(had, Served(proxy, Profile())).agent.extra_allowed_hosts == [
        "host.docker.internal",
        "pypi.org",
    ]
    assert (
        served_config(
            _config(tmp_path, "public"), Served(proxy, Profile())
        ).agent.extra_allowed_hosts
        == []
    )


def _config(tmp_path: Path, task: str, harness: str = "pi", **agent: object) -> TrialConfig:
    from harbor.models.trial.config import AgentConfig

    from shipyard.rollout import _harness_at_version, _trial_config

    name, version = _harness_at_version(harness)
    made = AgentConfig(name=name, model_name="m", **agent)
    return _trial_config(
        tmp_path / "tasks" / task,
        trials_dir=tmp_path / "jobs" / "j",
        agent=made,
        sandbox="docker",
        verify=True,
    )
