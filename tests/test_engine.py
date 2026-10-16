"""Test the engine: what reaches Harbor, a failed trial's slot, one concurrency limit per run."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, TrialConfig

from shipyard.config import Rollout
from shipyard.harnesses import PROFILES
from shipyard.progress import Reporter
from shipyard.sampling.admit import result_of
from shipyard.sampling.endpoint import ProviderEndpoint, ProxyEndpoint
from shipyard.sampling.engine import (
    MODAL_IMAGE_BUILDER,
    MODAL_IMAGE_BUILDER_VERSION,
    HarborEngine,
    _harness_at_version,
    _trial_config,
    allowlisted,
    connected,
    laid_over,
    quiet_litellm,
)
from shipyard.types import Agent, Connection, Endpoint, Policy, Settings, Trial, harness
from tests.records import FakeProxy, served
from tests.trials import FakeTrials


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    root = tmp_path / "tasks"
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return [root / name for name in names]


def _engine(tmp_path: Path, harbor: Any, **rollout: Any) -> HarborEngine:
    """An engine over `[rollout]`; `env` and `kwargs` are the harness's, from `[agent]`."""
    env, kwargs = rollout.pop("env", None), rollout.pop("kwargs", None)
    table = {"sandbox": "docker", "concurrency": 4, "jobs_dir": str(tmp_path / "jobs")}
    return HarborEngine(Rollout(**{**table, **rollout}), env=env, kwargs=kwargs, run_trial=harbor)


def _agent(*names: str, endpoint: Endpoint | None = None) -> Agent:
    policy = Policy(ProviderEndpoint("m") if endpoint is None else endpoint)
    return Agent(policy, [harness(name) for name in names])


async def test_configs_are_built_in_plan_order_with_harbors_fields(tmp_path: Path) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha", "beta")
    engine = _engine(
        tmp_path,
        harbor,
        env={"API_KEY": "k"},
        kwargs={"thinking": "high"},
        setup_timeout=360.0,
        timeout=1800.0,
    )
    agent = _agent("pi@0.85.1", endpoint=ProviderEndpoint("anthropic/some-model"))
    found = await engine.run("run-0000", batch, 3, agent=agent)
    assert len(found) == 6 and {one.job for one in found} == {"run-0000"}
    assert [(one.task, one.attempt) for one in found] == [(batch[0], n) for n in range(3)] + [
        (batch[1], n) for n in range(3)
    ]
    assert all(one.harness is agent.harness[0] for one in found)
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
    assert [one.dir.parent for one in found] == [tmp_path / "jobs" / "run-0000"] * 6
    assert [one.name.split("__")[0] for one in found] == ["alpha"] * 3 + ["beta"] * 3
    assert all(len(one.name.split("__")[1]) == 7 for one in found)
    assert {one.name for one in found} == {c.trial_name for c in harbor.configs}


async def test_a_call_runs_under_its_own_settings_or_the_tables(tmp_path: Path) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha")
    engine = _engine(
        tmp_path, harbor, sandbox="modal", env={"A": "1"}, kwargs={"turns": 10}, timeout=900.0
    )
    assert engine.settings == Settings(env={"A": "1"}, kwargs={"turns": 10}, timeout=900.0)
    own = Settings(verify=False, setup_timeout=1800.0)
    await engine.run("j0", batch, 1, agent=_agent("pi@0.85.1"), settings=own)
    await engine.run("j1", batch, 1, agent=_agent("pi@0.85.1"))
    theirs, tables = harbor.configs
    assert theirs.verifier.disable is True and tables.verifier.disable is False
    assert theirs.agent.env == {} and tables.agent.env == {"A": "1"}
    assert theirs.agent.kwargs == {"version": "0.85.1"}
    assert tables.agent.kwargs == {"version": "0.85.1", "turns": 10}
    assert (theirs.agent.override_setup_timeout_sec, theirs.agent.override_timeout_sec) == (
        1800.0,
        None,
    )
    assert (tables.agent.override_setup_timeout_sec, tables.agent.override_timeout_sec) == (
        None,
        900.0,
    )
    assert {c.environment.type for c in harbor.configs} == {EnvironmentType.MODAL}


async def test_two_harnesses_give_each_task_a_trial_per_harness_in_plan_order(
    tmp_path: Path,
) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha", "beta")
    agent = _agent("pi@0.85.1", "opencode")
    found = await _engine(tmp_path, harbor).run("j", batch, 2, agent=agent)
    pi, opencode = agent.harness
    assert [(one.task.name, one.harness.name, one.attempt) for one in found] == [
        ("alpha", "pi", 0),
        ("alpha", "pi", 1),
        ("alpha", "opencode", 0),
        ("alpha", "opencode", 1),
        ("beta", "pi", 0),
        ("beta", "pi", 1),
        ("beta", "opencode", 0),
        ("beta", "opencode", 1),
    ]
    assert [one.harness for one in found] == [pi, pi, opencode, opencode] * 2
    by_name = {c.trial_name: c for c in harbor.configs}
    assert [by_name[one.name].agent.name for one in found] == [
        "pi",
        "pi",
        "opencode",
        "opencode",
    ] * 2
    assert by_name[found[0].name].agent.kwargs == {"version": "0.85.1"}
    assert by_name[found[2].name].agent.kwargs == {}


async def test_the_harness_pin_is_split_and_a_colon_name_is_left_whole(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "alpha")
    harbor = FakeTrials()
    await _engine(tmp_path, harbor).run("j", batch, 1, agent=_agent("pi"))
    assert harbor.configs[0].agent.name == "pi" and harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    await _engine(tmp_path, harbor).run("j", batch, 1, agent=_agent("acp:some.module:Agent@1"))
    assert harbor.configs[0].agent.name == "acp:some.module:Agent@1"
    assert harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    engine = _engine(tmp_path, harbor, kwargs={"version": "0.90"})
    await engine.run("j", batch, 1, agent=_agent("pi@0.85.1"))
    assert harbor.configs[0].agent.kwargs == {"version": "0.90"}


async def test_a_trial_that_could_not_run_keeps_its_slot(tmp_path: Path) -> None:
    harbor = FakeTrials(breaks=("beta",))
    batch = _batch(tmp_path, "alpha", "beta", "gamma")
    found = await _engine(tmp_path, harbor).run("run-0000", batch, 2, agent=_agent("pi"))
    assert len(found) == 6
    assert [one.task.name for one in found] == ["alpha"] * 2 + ["beta"] * 2 + ["gamma"] * 2
    assert [one.dir.is_dir() for one in found] == [True, True, False, False, True, True]
    assert [result_of(one.dir) is None for one in found] == [False, False, True, True, False, False]
    # The job directory holds only what Harbor's stand-in wrote.
    job_dir = tmp_path / "jobs" / "run-0000"
    assert sorted(p.name for p in job_dir.iterdir()) == sorted(
        one.name for one in found if one.dir.is_dir()
    )
    assert sorted(p.name for p in job_dir.rglob("*") if p.is_file()) == ["result.json"] * 4


async def test_one_concurrency_limit_is_shared_by_every_job_of_the_engine(tmp_path: Path) -> None:
    batch = _batch(tmp_path, *[f"t{n}" for n in range(6)])
    harbor = FakeTrials(dwell=0.01)
    engine = _engine(tmp_path, harbor, concurrency=2)
    await asyncio.gather(
        engine.run("j0", batch, 1, agent=_agent("pi")),
        engine.run("j1", batch, 1, agent=_agent("pi")),
    )
    assert harbor.peak == 2 and len(harbor.configs) == 12
    harbor = FakeTrials(dwell=0.005)
    await _engine(tmp_path, harbor, concurrency=1).run("j", batch, 1, agent=_agent("pi"))
    assert harbor.peak == 1


async def test_a_job_that_could_not_happen_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    harbor = FakeTrials()
    engine = _engine(tmp_path, harbor)
    batch = _batch(tmp_path, "alpha")
    for named in ("", "   "):
        with pytest.raises(ValueError, match="oracle"):
            await engine.run("j", batch, 1, agent=_agent(named))
    with pytest.raises(ValueError, match="attempts"):
        await engine.run("j", batch, 0, agent=_agent("pi"))
    with pytest.raises(ValueError, match="names no task"):
        await engine.run("j", [], 1, agent=_agent("pi"))
    assert harbor.configs == [] and not (tmp_path / "jobs").exists()


async def test_the_modal_builder_is_set_unless_the_shell_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(tmp_path, "alpha")
    await _engine(tmp_path, FakeTrials(), sandbox="docker").run("j", batch, 1, agent=_agent("pi"))
    assert MODAL_IMAGE_BUILDER not in os.environ
    await _engine(tmp_path, FakeTrials(), sandbox="modal").run("j", batch, 1, agent=_agent("pi"))
    assert os.environ[MODAL_IMAGE_BUILDER] == MODAL_IMAGE_BUILDER_VERSION
    monkeypatch.setenv(MODAL_IMAGE_BUILDER, "legacy")
    await _engine(tmp_path, FakeTrials(), sandbox="modal").run("j", batch, 1, agent=_agent("pi"))
    assert os.environ[MODAL_IMAGE_BUILDER] == "legacy"


async def test_every_finished_trial_is_reported_to_the_progress_with_its_mask(
    tmp_path: Path,
) -> None:
    class Counting(Reporter):
        def __init__(self) -> None:
            self.seen: dict[Path, bool] = {}

        def finished(self, trial: Path, *, masked: bool) -> None:
            self.seen[trial] = masked

    progress = Counting()
    batch = _batch(tmp_path, "alpha", "beta")
    engine = _engine(tmp_path, FakeTrials(breaks=("beta",)))
    engine.progress = progress
    found = await engine.run("j", batch, 1, agent=_agent("pi"))
    assert sorted(progress.seen) == sorted(one.dir for one in found)
    assert [progress.seen[one.dir] for one in found] == [False, True]


async def test_a_stray_cancellation_is_one_trials_mask_not_the_batchs(tmp_path: Path) -> None:
    async def stray(config: TrialConfig) -> None:
        if config.task.path.name == "beta":
            raise asyncio.CancelledError()
        (Path(config.trials_dir) / config.trial_name).mkdir(parents=True)

    batch = _batch(tmp_path, "alpha", "beta", "gamma")
    found = await _engine(tmp_path, stray).run("j", batch, 1, agent=_agent("pi"))
    assert [one.dir.is_dir() for one in found] == [True, False, True]


async def test_a_real_cancellation_still_stops_the_batch(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def slow(config: TrialConfig) -> None:
        started.set()
        await asyncio.sleep(10)

    engine = _engine(tmp_path, slow)
    running = asyncio.ensure_future(
        engine.run("j", _batch(tmp_path, "alpha"), 1, agent=_agent("pi"))
    )
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
    harbor = FakeTrials()
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="secret")
    engine = _engine(
        tmp_path,
        harbor,
        env={"OPENAI_BASE_URL": "https://api.openai.com/v1", "HTTPS_PROXY": "corp"},
    )
    endpoint = ProxyEndpoint(served(), tmp_path, proxy=proxy)
    found = await engine.run(
        "j", _batch(tmp_path, "alpha", "beta"), 2, agent=_agent("pi@0.85.1", endpoint=endpoint)
    )
    assert len(harbor.configs) == 4
    addresses = [config.agent.env["OPENAI_BASE_URL"] for config in harbor.configs]
    assert len(set(addresses)) == 4 and sorted(proxy.addressed) == sorted(
        c.trial_name for c in harbor.configs
    )
    for config, address in zip(harbor.configs, addresses, strict=True):
        assert address == f"http://host.docker.internal:8000/r/trial/{config.trial_name}/v1"
        assert config.agent.env["OPENAI_API_KEY"] == "secret"
        assert config.agent.env["HTTPS_PROXY"] == "corp", "the job's other env survives"
        assert config.agent.kwargs == {"version": "0.85.1", "model_api": "openai-completions"}
        assert config.agent.model_name == "openai/Qwen/Qwen3-8B"
        assert config.agent.extra_allowed_hosts == []
    assert {one.name for one in found} == {c.trial_name for c in harbor.configs}


def _dialled(
    tmp_path: Path, proxy: FakeProxy, config: TrialConfig, deployed: str, **rollout: object
) -> Connection:
    """What a served trial of `config` under `deployed` dials."""
    trial = Trial(
        "j",
        Path(config.task.path),
        0,
        harness(deployed),
        tmp_path / "jobs" / "j" / config.trial_name,
    )
    return ProxyEndpoint(served(**rollout), tmp_path, proxy=proxy).connection(trial)


def test_claude_code_is_handed_the_anthropic_wire_without_the_v1(tmp_path: Path) -> None:
    proxy = FakeProxy(origin="https://x.trycloudflare.com", token="k")
    config = _config(tmp_path, "alpha", harness="claude-code", env={"KEEP": "1"})
    pointed = connected(config, _dialled(tmp_path, proxy, config, "claude-code", fill_context=True))
    assert pointed.agent.env == {
        "KEEP": "1",
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": f"https://x.trycloudflare.com/r/trial/{config.trial_name}",
        "ANTHROPIC_API_KEY": "k",
    }
    assert pointed.agent.kwargs == {} and pointed.agent.name == "claude-code"
    assert pointed.agent.model_name == "anthropic/Qwen/Qwen3-8B"
    plain = connected(config, _dialled(tmp_path, proxy, config, "claude-code"))
    assert "DISABLE_COMPACT" not in plain.agent.env, "the profile's env only when filling"
    assert config.agent.env == {"KEEP": "1"}, "the job's config was not written into"


def test_opencode_gets_its_providers_env_and_keeps_a_blueprints_own_config(
    tmp_path: Path,
) -> None:
    """The trial gets the env that the profile's opencode config names. A blueprint's own
    `opencode_config` keeps its keys, and the profile's keys win where both set one."""
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="k")
    mine = {
        "provider": {"shipyard": {"models": {"Qwen/Qwen3-8B": {"limit": {"context": 32768}}}}},
        "agent": {"title": {"disable": False}, "build": {"steps": 40}},
    }
    config = _config(tmp_path, "alpha", harness="opencode", kwargs={"opencode_config": mine})
    pointed = connected(config, _dialled(tmp_path, proxy, config, "opencode"))
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

    dialled = Connection("m", host="host.docker.internal")
    pointed = connected(_config(tmp_path, "allow"), dialled)
    assert pointed.agent.extra_allowed_hosts == ["host.docker.internal"]
    had = _config(tmp_path, "allow").model_copy()
    had.agent.extra_allowed_hosts = ["host.docker.internal", "pypi.org"]
    assert connected(had, dialled).agent.extra_allowed_hosts == ["host.docker.internal", "pypi.org"]
    assert connected(_config(tmp_path, "public"), dialled).agent.extra_allowed_hosts == []
    assert connected(_config(tmp_path, "allow"), Connection("m")).agent.extra_allowed_hosts == []


def _config(tmp_path: Path, task: str, harness: str = "pi", **agent: object) -> TrialConfig:
    name, version = _harness_at_version(harness)
    made = AgentConfig(name=name, model_name="m", **agent)
    return _trial_config(
        tmp_path / "tasks" / task,
        trials_dir=tmp_path / "jobs" / "j",
        agent=made,
        sandbox="docker",
        verify=True,
    )
