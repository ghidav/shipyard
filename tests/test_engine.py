"""Tests for `shipyard.jobs.engine`: a job's trials run with Harbor under one concurrency
limit, each wired to its endpoint and holding its secrets, then recorded and judged."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.utils.env import redact_sensitive_value

from proxies import MODEL, made
from shipyard.config import Blueprint, Rollout
from shipyard.groups import Trial
from shipyard.jobs.admit import ENV_ERROR
from shipyard.jobs.engine import (
    DELIVERED,
    MODAL_IMAGE_BUILDER,
    MODAL_IMAGE_BUILDER_VERSION,
    HarborEngine,
    Job,
    _harness_at_version,
    _trial_config,
    allowlisted,
    connected,
    laid_over,
    quiet_litellm,
)
from shipyard.jobs.harnesses import PROFILES, harness
from shipyard.jobs.modules import PROMPT_TEMPLATE, Module, Modules
from shipyard.jobs.secrets import Pool, Secrets
from shipyard.models.endpoints import Connection, ProviderEndpoint, ProxyEndpoint
from tests.fakes import FakeProxy, FakeTrials, write_result

NAME = "BAUPLAN_API_KEY"
#: Pool keys longer than eight characters, so Harbor's redacted form of each is distinct.
KEYS = ("key-0000-aaaaaaaa", "key-0001-bbbbbbbb")
PROVIDER = ProviderEndpoint("some-model", "anthropic")


@pytest.fixture(autouse=True)
def _set_by_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine sets the Modal builder and each held secret's name in this process. Each
    test starts without them and leaves them as it found them."""
    for name in (MODAL_IMAGE_BUILDER, NAME):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    root = tmp_path / "tasks"
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return [root / name for name in names]


def _engine(
    tmp_path: Path,
    runner: Any,
    *,
    secrets: Secrets | None = None,
    env: Mapping[str, str] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    **rollout: Any,
) -> HarborEngine:
    """Build an engine from `[rollout]` fields; `env` and `kwargs` are the harness's settings."""
    table = {"sandbox": "docker", "concurrency": 4, "jobs_dir": str(tmp_path / "jobs")}
    return HarborEngine(
        Rollout(**{**table, **rollout}), env=env, kwargs=kwargs, secrets=secrets, run_trial=runner
    )


async def _run(
    engine: HarborEngine,
    name: str,
    tasks: list[Path],
    attempts: int,
    *,
    deployed: str = "pi",
    endpoint: Any = PROVIDER,
    modules: Modules | None = None,
) -> Job:
    job = Job(name, modules=modules, harness=harness(deployed))
    return await engine.run(job, tasks, attempts, endpoint=endpoint)


def served(**rollout: Any) -> Blueprint:
    """A blueprint with `[rollout]` from the keywords."""
    return Blueprint.model_validate(
        {
            "agent": {"model": MODEL, "harness": "pi@0.85.1"},
            "data": {"train": "d", "batch_size": 1},
            "rollout": rollout,
            "recipe": {"steps": 1, "learning_rate": 1e-5},
        }
    )


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
    pi = harness("pi@0.85.1")
    job = Job("run-0000", harness=pi)
    found = await engine.run(job, batch, 3, endpoint=PROVIDER)
    assert found is job and len(job.trials) == 6
    assert {one.job for one in job.trials} == {"run-0000"}
    assert [one.task for one in job.trials] == [batch[0]] * 3 + [batch[1]] * 3
    assert all(one.harness is pi for one in job.trials)
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
        assert config.environment.env == {} and config.verifier.env == {}, "no secrets held"
    # The trials come back in plan order under the job, named as Harbor names them.
    assert [one.dir.parent for one in job.trials] == [tmp_path / "jobs" / "run-0000"] * 6
    assert [one.name.split("__")[0] for one in job.trials] == ["alpha"] * 3 + ["beta"] * 3
    assert all(len(one.name.split("__")[1]) == 7 for one in job.trials)
    assert {one.name for one in job.trials} == {c.trial_name for c in harbor.configs}


async def test_every_job_of_an_engine_runs_under_its_tables_on_its_sandbox(
    tmp_path: Path,
) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha")
    engine = _engine(
        tmp_path, harbor, sandbox="modal", env={"A": "1"}, kwargs={"max_turns": 10}, timeout=900.0
    )
    await _run(engine, "j0", batch, 1, deployed="pi@0.85.1")
    await _run(engine, "j1", batch, 1, deployed="pi@0.85.1")
    for config in harbor.configs:
        assert config.agent.env == {"A": "1"}
        assert config.agent.kwargs == {"version": "0.85.1", "max_turns": 10}
        assert config.agent.override_setup_timeout_sec is None
        assert config.agent.override_timeout_sec == 900.0
        assert config.environment.type == EnvironmentType.MODAL
        assert config.verifier.disable is False
    assert [c.trials_dir.name for c in harbor.configs] == ["j0", "j1"]


async def test_each_job_runs_the_harness_it_is_given(tmp_path: Path) -> None:
    harbor = FakeTrials()
    batch = _batch(tmp_path, "alpha", "beta")
    engine = _engine(tmp_path, harbor)
    pinned = await _run(engine, "j0", batch, 2, deployed="pi@0.85.1")
    bare = await _run(engine, "j1", batch, 2, deployed="opencode")
    by_name = {c.trial_name: c for c in harbor.configs}
    for job, named, kwargs in ((pinned, "pi", {"version": "0.85.1"}), (bare, "opencode", {})):
        assert [one.task.name for one in job.trials] == ["alpha", "alpha", "beta", "beta"]
        assert {one.harness.name for one in job.trials} == {named}
        assert {by_name[one.name].agent.name for one in job.trials} == {named}
        assert all(by_name[one.name].agent.kwargs == kwargs for one in job.trials)


async def test_the_harness_pin_is_split_and_a_colon_name_is_left_whole(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "alpha")
    harbor = FakeTrials()
    await _run(_engine(tmp_path, harbor), "j", batch, 1, deployed="pi")
    assert harbor.configs[0].agent.name == "pi" and harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    await _run(_engine(tmp_path, harbor), "j", batch, 1, deployed="acp:some.module:Agent@1")
    assert harbor.configs[0].agent.name == "acp:some.module:Agent@1"
    assert harbor.configs[0].agent.kwargs == {}
    harbor = FakeTrials()
    engine = _engine(tmp_path, harbor, kwargs={"version": "0.90"})
    await _run(engine, "j", batch, 1, deployed="pi@0.85.1")
    assert harbor.configs[0].agent.kwargs == {"version": "0.90"}


async def test_a_trial_that_could_not_run_keeps_its_slot_and_is_masked(tmp_path: Path) -> None:
    harbor = FakeTrials(breaks=("beta",))
    batch = _batch(tmp_path, "alpha", "beta", "gamma")
    job = await _run(_engine(tmp_path, harbor), "run-0000", batch, 2)
    found = job.trials
    assert len(found) == 6
    assert [one.task.name for one in found] == ["alpha"] * 2 + ["beta"] * 2 + ["gamma"] * 2
    assert [one.dir.is_dir() for one in found] == [True, True, False, False, True, True]
    assert [one.result is None for one in found] == [False, False, True, True, False, False]
    assert [one.reward for one in found] == [1.0, 1.0, None, None, 1.0, 1.0]
    masks = [one.verdict.mask if one.verdict else "unjudged" for one in found]
    assert masks == [None, None, ENV_ERROR, ENV_ERROR, None, None]
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
    await asyncio.gather(_run(engine, "j0", batch, 1), _run(engine, "j1", batch, 1))
    assert harbor.peak == 2 and len(harbor.configs) == 12
    harbor = FakeTrials(dwell=0.005)
    await _run(_engine(tmp_path, harbor, concurrency=1), "j", batch, 1)
    assert harbor.peak == 1


async def test_a_job_that_could_not_happen_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    harbor = FakeTrials()
    engine = _engine(tmp_path, harbor)
    batch = _batch(tmp_path, "alpha")
    for named in ("", "   "):
        with pytest.raises(ValueError, match="oracle"):
            await _run(engine, "j", batch, 1, deployed=named)
    with pytest.raises(ValueError, match="attempts"):
        await _run(engine, "j", batch, 0)
    with pytest.raises(ValueError, match="names no task"):
        await _run(engine, "j", [], 1)
    assert harbor.configs == [] and not (tmp_path / "jobs").exists()


async def test_the_modal_builder_is_set_unless_the_shell_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(tmp_path, "alpha")
    await _run(_engine(tmp_path, FakeTrials(), sandbox="docker"), "j", batch, 1)
    assert MODAL_IMAGE_BUILDER not in os.environ
    await _run(_engine(tmp_path, FakeTrials(), sandbox="modal"), "j", batch, 1)
    assert os.environ[MODAL_IMAGE_BUILDER] == MODAL_IMAGE_BUILDER_VERSION
    monkeypatch.setenv(MODAL_IMAGE_BUILDER, "legacy")
    await _run(_engine(tmp_path, FakeTrials(), sandbox="modal"), "j", batch, 1)
    assert os.environ[MODAL_IMAGE_BUILDER] == "legacy"


async def test_the_job_and_every_finished_trial_are_reported_to_the_progress(
    tmp_path: Path,
) -> None:
    class Counting:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, int]] = []
            self.seen: list[str] = []
            self.closed = False

        def started(self, job: str, *, trials: int) -> None:
            self.events.append(("started", job, trials))

        def finished(self, job: str) -> None:
            self.seen.append(job)

        def done(self, job: str, *, masked: int) -> None:
            self.events.append(("done", job, masked))

        def close(self) -> None:
            self.closed = True

    progress = Counting()
    engine = _engine(tmp_path, FakeTrials(breaks=("beta",)))
    engine.progress = progress
    job = await _run(engine, "j", _batch(tmp_path, "alpha", "beta"), 2)
    assert progress.events == [("started", "j", 4), ("done", "j", 2)]
    assert progress.seen == ["j"] * len(job.trials)
    engine.close()
    assert progress.closed


async def test_a_stray_cancellation_is_one_trials_mask_not_the_jobs(tmp_path: Path) -> None:
    async def stray(config: TrialConfig) -> None:
        if config.task.path.name == "beta":
            raise asyncio.CancelledError()
        write_result(Path(config.trials_dir) / config.trial_name, reward=1.0)

    batch = _batch(tmp_path, "alpha", "beta", "gamma")
    job = await _run(_engine(tmp_path, stray), "j", batch, 1)
    assert [one.dir.is_dir() for one in job.trials] == [True, False, True]
    assert [one.reward for one in job.trials] == [1.0, None, 1.0]


async def test_a_real_cancellation_still_stops_the_job(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def slow(config: TrialConfig) -> None:
        started.set()
        await asyncio.sleep(10)

    running = asyncio.ensure_future(_run(_engine(tmp_path, slow), "j", _batch(tmp_path, "a"), 1))
    await asyncio.wait_for(started.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


def test_litellm_is_told_not_to_print_its_provider_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(suppress_debug_info=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    quiet_litellm()
    assert fake.suppress_debug_info is True


# ---------------------------------------------------------------- what a job carries


async def test_each_trial_holds_the_endpoints_records_of_it_and_is_judged_with_them(
    tmp_path: Path,
) -> None:
    proxy = FakeProxy(
        answers={"alpha": [made("one", "ok"), made("oneok", "two", seq=2)], "beta": []}
    )
    endpoint = ProxyEndpoint(served(), tmp_path, proxy=proxy)
    batch = _batch(tmp_path, "alpha", "beta")
    job = await _run(_engine(tmp_path, FakeTrials()), "j", batch, 1, endpoint=endpoint)
    alpha, beta = job.trials
    assert proxy.fetched == [alpha.name, beta.name]
    assert [one.seq for one in alpha.records or ()] == [1, 2] and beta.records == []
    assert alpha.reward == 1.0, "graded by its result.json"
    assert beta.verdict is not None and beta.verdict.mask == ENV_ERROR, "it never reached the proxy"
    unrecorded = await _run(_engine(tmp_path, FakeTrials()), "p", batch, 1)
    assert [(one.records, one.reward) for one in unrecorded.trials] == [(None, 1.0)] * 2


async def test_a_jobs_modules_are_delivered_beside_its_trials_and_laid_into_the_agent(
    tmp_path: Path,
) -> None:
    skill = "---\nname: tidy\ndescription: Keep the tree tidy.\n---\nTidy up.\n"
    modules = Modules(
        {
            "prompt": Module("prompt", "prompt", {"PROMPT.md": "Be brief."}),
            "tidy": Module("tidy", "skill", {"SKILL.md": skill}),
        }
    )
    harbor = FakeTrials()
    engine = _engine(tmp_path, harbor, kwargs={"thinking": "high"})
    batch = _batch(tmp_path, "alpha")
    await _run(engine, "j", batch, 2, deployed="pi@0.85.1", modules=modules)
    home = (tmp_path / "jobs" / "j" / DELIVERED).resolve()
    template = home / PROMPT_TEMPLATE
    for config in harbor.configs:
        assert config.agent.kwargs == {
            "version": "0.85.1",
            "thinking": "high",
            "prompt_template_path": str(template),
        }
        assert config.agent.skills == [str(home / "skills")]
    assert "Be brief." in template.read_text(encoding="utf-8")
    assert (home / "skills" / "tidy" / "SKILL.md").read_text(encoding="utf-8") == skill
    await _run(engine, "k", batch, 1, deployed="pi@0.85.1")
    assert harbor.configs[-1].agent.kwargs == {"version": "0.85.1", "thinking": "high"}
    assert harbor.configs[-1].agent.skills == []
    assert not (tmp_path / "jobs" / "k" / DELIVERED).exists()


# ------------------------------------------------------------------------- secrets


class Holding:
    """Fake Harbor trial runner that tracks held secrets and concurrency, and writes the
    secret where Harbor and an agent's log would. Each trial ends by raising `raises`, if set."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.configs: list[TrialConfig] = []
        self.holding: dict[str, str] = {}
        self.most = 0
        self.shared = False
        self.raises = raises

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        key = config.environment.env.get(NAME, "")
        self.shared |= key in self.holding.values()
        self.holding[config.trial_name] = key
        self.most = max(self.most, len(self.holding))
        trial = Path(config.trials_dir) / config.trial_name
        trial.mkdir(parents=True, exist_ok=True)
        shown = redact_sensitive_value(key)
        (trial / "config.json").write_text(f'{{"env": {{"{NAME}": "{shown}"}}}}\n')
        (trial / "agent.log").write_text(f"$ env\n{NAME}={key}\n")
        await asyncio.sleep(0.01)
        del self.holding[config.trial_name]
        if self.raises is not None:
            raise self.raises


class Minted(Secrets):
    """Fake `Secrets` that mints a key for each trial and records each release."""

    def __init__(self) -> None:
        self.acquired: list[Trial] = []
        self.released: list[tuple[Trial, Mapping[str, str]]] = []

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        self.acquired.append(trial)
        return {NAME: f"minted-for-{trial.name}"}

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        self.released.append((trial, held))


async def test_no_two_trials_in_flight_hold_one_identity_nor_more_run_than_there_are(
    tmp_path: Path,
) -> None:
    runner = Holding()
    pool = Pool([{NAME: key} for key in KEYS])
    engine = _engine(tmp_path, runner, secrets=pool, concurrency=4)
    await _run(engine, "j", _batch(tmp_path, *[f"t{n}" for n in range(6)]), 1)
    assert len(runner.configs) == 6 and runner.most == 2 and not runner.shared
    for config in runner.configs:
        key = config.environment.env[NAME]
        assert key in KEYS and config.verifier.env == {NAME: key}, "the verifier holds it too"
        assert NAME not in config.agent.env
    assert os.environ[NAME] == "", "the task's ${BAUPLAN_API_KEY} resolves, to no identity"


async def test_what_a_trial_held_is_scrubbed_from_its_directory(tmp_path: Path) -> None:
    engine = _engine(tmp_path, Holding(), secrets=Pool([{NAME: KEYS[0]}]))
    job = await _run(engine, "j", _batch(tmp_path, "alpha"), 1)
    [trial] = job.trials
    left = {path.name: path.read_text() for path in trial.dir.iterdir()}
    assert left == {
        "config.json": '{"env": {"BAUPLAN_API_KEY": "${BAUPLAN_API_KEY}"}}\n',
        "agent.log": "$ env\nBAUPLAN_API_KEY=${BAUPLAN_API_KEY}\n",
    }


@pytest.mark.parametrize("raises", [RuntimeError("the sandbox died"), asyncio.CancelledError()])
async def test_an_identity_comes_back_however_its_trial_ended(
    tmp_path: Path, raises: BaseException
) -> None:
    """A trial that raises, or is cancelled by something other than the run, still releases
    its identity and is scrubbed. With a single identity, the second job would otherwise hang."""
    pool = Pool([{NAME: KEYS[0]}])
    failed = await _run(
        _engine(tmp_path, Holding(raises), secrets=pool), "j0", _batch(tmp_path, "a", "b"), 1
    )
    assert all(KEYS[0] not in (one.dir / "agent.log").read_text() for one in failed.trials)
    runner = Holding()
    job = _run(_engine(tmp_path, runner, secrets=pool), "j1", _batch(tmp_path, "a", "b"), 1)
    await asyncio.wait_for(job, timeout=5)
    assert len(runner.configs) == 2


async def test_a_cancelled_job_still_scrubs_and_releases_what_its_trials_held(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def slow(config: TrialConfig) -> None:
        trial = Path(config.trials_dir) / config.trial_name
        trial.mkdir(parents=True)
        (trial / "agent.log").write_text(f"{NAME}={config.environment.env[NAME]}\n")
        started.set()
        await asyncio.sleep(10)

    pool = Pool([{NAME: KEYS[0]}])
    batch = _batch(tmp_path, "alpha")
    running = asyncio.ensure_future(_run(_engine(tmp_path, slow, secrets=pool), "j0", batch, 1))
    await asyncio.wait_for(started.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    [left] = (tmp_path / "jobs" / "j0").iterdir()
    assert (left / "agent.log").read_text() == f"{NAME}=${{{NAME}}}\n"
    runner = Holding()
    await asyncio.wait_for(_run(_engine(tmp_path, runner, secrets=pool), "j1", batch, 1), 5)
    assert len(runner.configs) == 1


async def test_secrets_of_ones_own_are_acquired_and_released_with_each_trial(
    tmp_path: Path,
) -> None:
    runner, minted = Holding(), Minted()
    engine = _engine(tmp_path, runner, secrets=minted)
    job = await _run(engine, "j", _batch(tmp_path, "alpha", "beta"), 2)
    assert sorted(t.dir for t in minted.acquired) == sorted(t.dir for t in job.trials)
    assert {trial.dir: held[NAME] for trial, held in minted.released} == {
        trial.dir: f"minted-for-{trial.name}" for trial in job.trials
    }
    assert {c.trial_name: c.environment.env[NAME] for c in runner.configs} == {
        trial.name: f"minted-for-{trial.name}" for trial in job.trials
    }
    assert not runner.shared


# ------------------------------------------------------------------- a served model


def _task_toml(tmp_path: Path, name: str, text: str) -> Path:
    task = tmp_path / "tasks" / name
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(text, encoding="utf-8")
    return task


def _config(tmp_path: Path, task: str, deployed: str = "pi", **agent: Any) -> TrialConfig:
    name, _ = _harness_at_version(deployed)
    return _trial_config(
        tmp_path / "tasks" / task,
        trials_dir=tmp_path / "jobs" / "j",
        agent=AgentConfig(name=name, model_name="m", **agent),
        sandbox="docker",
    )


def _dialled(
    tmp_path: Path, proxy: FakeProxy, config: TrialConfig, deployed: str, **rollout: Any
) -> Connection:
    """Return the connection a served trial of `config` under harness `deployed` dials."""
    trial = Trial(
        "j", Path(config.task.path), harness(deployed), tmp_path / "jobs" / "j" / config.trial_name
    )
    return ProxyEndpoint(served(**rollout), tmp_path, proxy=proxy).connection(trial)


async def test_every_trial_is_pointed_at_the_address_that_names_it(tmp_path: Path) -> None:
    harbor = FakeTrials()
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="secret")
    engine = _engine(
        tmp_path,
        harbor,
        env={"OPENAI_BASE_URL": "https://api.openai.com/v1", "HTTPS_PROXY": "corp"},
    )
    endpoint = ProxyEndpoint(served(), tmp_path, proxy=proxy)
    batch = _batch(tmp_path, "alpha", "beta")
    job = await _run(engine, "j", batch, 2, deployed="pi@0.85.1", endpoint=endpoint)
    assert len(harbor.configs) == 4
    addresses = [config.agent.env["OPENAI_BASE_URL"] for config in harbor.configs]
    # Every trial is addressed, and so is the job's preflight, once.
    named = [c.trial_name for c in harbor.configs]
    assert len(set(addresses)) == 4 and sorted(proxy.addressed) == sorted([*named, "preflight"])
    for config, address in zip(harbor.configs, addresses, strict=True):
        assert address == f"http://host.docker.internal:8000/r/trial/{config.trial_name}/v1"
        assert config.agent.env["OPENAI_API_KEY"] == "secret"
        assert config.agent.env["HTTPS_PROXY"] == "corp", "the job's other env survives"
        assert config.agent.kwargs == {"version": "0.85.1", "model_api": "openai-completions"}
        assert config.agent.model_name == f"openai/{MODEL}"
        assert config.agent.extra_allowed_hosts == []
    assert {one.name for one in job.trials} == {c.trial_name for c in harbor.configs}


async def test_a_job_lets_the_trials_of_an_allowlisted_task_reach_the_proxys_host(
    tmp_path: Path,
) -> None:
    allow = _task_toml(
        tmp_path, "allow", '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["x.io"]\n'
    )
    public = _task_toml(tmp_path, "public", '[environment]\nnetwork_mode = "public"\n')
    harbor = FakeTrials()
    endpoint = ProxyEndpoint(served(), tmp_path, proxy=FakeProxy(origin="https://p.example:8443"))
    await _run(_engine(tmp_path, harbor), "j", [allow, public], 1, endpoint=endpoint)
    hosts = {c.task.path.name: c.agent.extra_allowed_hosts for c in harbor.configs}
    assert hosts == {"allow": ["p.example"], "public": []}
    harbor = FakeTrials()
    await _run(_engine(tmp_path, harbor), "p", [allow], 1)
    assert harbor.configs[0].agent.extra_allowed_hosts == [], "a provider's model has no host"


def test_terminus_2_whose_calls_harbor_makes_here_dials_the_proxys_own_address(
    tmp_path: Path,
) -> None:
    """Its litellm runs in this process, which neither the trial env nor the container's name
    for the proxy reaches: the proxy's loopback address and its token go in as options."""
    proxy = FakeProxy(origin="https://x.trycloudflare.com", token="k")
    config = _config(tmp_path, "alpha", deployed="terminus-2")
    dialled = _dialled(tmp_path, proxy, config, "terminus-2")
    address = f"http://127.0.0.1:8000/r/trial/{config.trial_name}/v1"
    assert dialled.kwargs == {"api_base": address, "llm_call_kwargs": {"api_key": "k"}}


async def test_a_task_with_no_network_is_refused_unless_the_harness_calls_from_here(
    tmp_path: Path,
) -> None:
    sealed = _task_toml(tmp_path, "sealed", '[environment]\nnetwork_mode = "no-network"\n')
    runner = FakeTrials()
    engine = _engine(tmp_path, runner)
    with pytest.raises(ValueError, match="sealed run their agent with no network, so pi"):
        await _run(engine, "j", [sealed], 1)
    assert runner.configs == [], "no trial started"
    await _run(engine, "k", [sealed], 1, deployed="terminus-2")
    assert len(runner.configs) == 1, "terminus-2 calls the model from this machine"


async def test_a_config_the_harness_refuses_fails_the_job_once_before_any_trial(
    tmp_path: Path,
) -> None:
    runner = FakeTrials()
    engine = _engine(tmp_path, runner, kwargs={"prompt_template_path": "/m/PROMPT.md"})
    with pytest.raises(ValueError, match="the terminus-2 harness refuses its config") as refused:
        await _run(engine, "j", _batch(tmp_path, "alpha", "beta"), 2, deployed="terminus-2")
    assert "Unknown option 'prompt_template_path'" in str(refused.value)
    assert runner.configs == [], "no trial started"


def test_claude_code_is_handed_the_anthropic_wire_without_the_v1(tmp_path: Path) -> None:
    proxy = FakeProxy(origin="https://x.trycloudflare.com", token="k")
    config = _config(tmp_path, "alpha", deployed="claude-code", env={"KEEP": "1"})
    pointed = connected(config, _dialled(tmp_path, proxy, config, "claude-code", fill_context=True))
    assert pointed.agent.env == {
        "KEEP": "1",
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": f"https://x.trycloudflare.com/r/trial/{config.trial_name}",
        "ANTHROPIC_API_KEY": "k",
    }
    assert pointed.agent.kwargs == {} and pointed.agent.name == "claude-code"
    assert pointed.agent.model_name == f"anthropic/{MODEL}"
    plain = connected(config, _dialled(tmp_path, proxy, config, "claude-code"))
    assert "DISABLE_COMPACT" not in plain.agent.env, "the profile's env only when filling"
    assert config.agent.env == {"KEEP": "1"}, "the job's config was not written into"


def test_opencode_gets_its_providers_env_and_keeps_a_blueprints_own_config(
    tmp_path: Path,
) -> None:
    """The trial receives the env that the profile's opencode config names. A blueprint's own
    `opencode_config` keeps its keys, and the profile's keys win where both set one."""
    proxy = FakeProxy(origin="http://host.docker.internal:8000", token="k")
    mine = {
        "provider": {"shipyard": {"models": {MODEL: {"limit": {"context": 32768}}}}},
        "agent": {"title": {"disable": False}, "build": {"steps": 40}},
    }
    config = _config(tmp_path, "alpha", deployed="opencode", kwargs={"opencode_config": mine})
    pointed = connected(config, _dialled(tmp_path, proxy, config, "opencode"))
    address = f"http://host.docker.internal:8000/r/trial/{config.trial_name}/v1"
    assert pointed.agent.env == {"SHIPYARD_BASE_URL": address, "SHIPYARD_API_KEY": "k"}
    merged = pointed.agent.kwargs["opencode_config"]
    assert merged["provider"]["shipyard"] == {
        "npm": "@ai-sdk/openai-compatible",
        "env": ["SHIPYARD_API_KEY"],
        "options": {"baseURL": "${SHIPYARD_BASE_URL}"},
        "models": {MODEL: {"limit": {"context": 32768}}},
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
    had = _config(tmp_path, "allow")
    had.agent.extra_allowed_hosts = ["host.docker.internal", "pypi.org"]
    assert connected(had, dialled).agent.extra_allowed_hosts == ["host.docker.internal", "pypi.org"]
    assert connected(_config(tmp_path, "public"), dialled).agent.extra_allowed_hosts == []
    assert connected(_config(tmp_path, "allow"), Connection("m")).agent.extra_allowed_hosts == []
