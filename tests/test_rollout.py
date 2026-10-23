"""Tests for a rollout: one job, at one harness and one renderer, that runs one model on each task
and returns one group per task in the tasks' order. The job is named, judged and logged, and the
endpoint is checked for it alone; a recipe that wants several harnesses gathers several rollouts,
which share the run's one concurrency limit."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from proxies import MODEL, made
from shipyard import Group, Harness, Model, Trainer
from shipyard.config import Config, ModelTable, Rollout
from shipyard.groups import Connection, Trial
from shipyard.jobs.engine import HarborEngine
from shipyard.models.endpoints import NothingServed
from shipyard.records import Records, served_nothing
from shipyard.run import files
from shipyard.run.logs import Logs
from tests.fakes import FakeProxy, FakeTrials, Harbor
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService

PI, CLAUDE = Harness(name="pi@0.85.1"), Harness(name="claude-code@2.1.0")


class Answered:
    """An endpoint kept in memory: it answers every trial but those of the jobs in `silent`,
    and notes each check it is asked for."""

    provider, model, kind = "tinker", MODEL, "tinker"

    def __init__(self, *silent: str) -> None:
        self.silent = set(silent)
        self.checked: list[tuple[str, int]] = []

    def connection(self, trial: Trial) -> Connection:
        return Connection(model=f"anthropic/{MODEL}")

    async def open(self, trial: Trial) -> Connection:
        return self.connection(trial)

    async def close(self, trial: Trial) -> Records:
        return Records([] if trial.job in self.silent else [made()])

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        self.checked.append((job, len(trials)))
        why = served_nothing(one.records for one in trials)
        if why:
            raise NothingServed(f"job {job}: {why}")


class Flights(FakeTrials):
    """Fake Harbor trial runner that notes, as each trial starts, the jobs of the trials then in
    flight."""

    def __init__(self) -> None:
        super().__init__(dwell=0.01)
        self.flying: list[str] = []
        self.together: list[set[str]] = []

    async def __call__(self, config: TrialConfig) -> None:
        job = Path(config.trials_dir).name
        self.flying.append(job)
        self.together.append(set(self.flying))
        try:
            await super().__call__(config)
        finally:
            self.flying.remove(job)


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor's own check of each harness's config, which is not what these tests are about."""
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)


def _tasks(tmp_path: Path, *names: str) -> list[Path]:
    for name in names:
        (tmp_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
    return [tmp_path / "tasks" / name for name in names]


def _engine(home: Path) -> tuple[HarborEngine, FakeTrials, Path]:
    """Return an engine that logs to the run `<home>/r`, its fake trials, and that run."""
    config = Config.model_validate(
        {
            "models": {"agent": {"model": MODEL}},
            "data": {"train": "d", "batch_size": 1},
            "rollout": {"concurrency": 4, "jobs_dir": str(home / "jobs")},
            "recipe": {"steps": 1},
        }
    )
    run, trials = home / "r", FakeTrials()
    engine = HarborEngine(config.rollout, logs=Logs(config, run), run_trial=trials)
    return engine, trials, run


# ------------------------------------------------------------------ the engine's rollout


async def test_a_rollout_runs_one_job_and_returns_a_group_per_task_in_the_tasks_order(
    tmp_path: Path,
) -> None:
    beta, alpha = _tasks(tmp_path, "beta", "alpha")
    engine, _, run = _engine(tmp_path)
    answered = Answered()
    groups = await engine.rollout(
        [beta, alpha],
        attempts=2,
        harness=PI,
        endpoint=answered,
        renderer="low",
        step=3,
        role="agent",
    )
    assert [(g.task.name, len(g.trials)) for g in groups] == [("beta", 2), ("alpha", 2)]
    trials = [one for group in groups for one in group.trials]
    assert {(one.job, one.renderer, one.role) for one in trials} == {("r-0000", "low", "agent")}
    assert [(g.harness, g.renderer) for g in groups] == [("pi@0.85.1", "low")] * 2
    assert answered.checked == [("r-0000", 4)], "checked once, over the job's trials"
    (row,) = files.read(run / files.JOBS)
    assert {key: row[key] for key in ("job", "purpose", "harness", "renderer", "role", "step")} == {
        "job": "r-0000",
        "purpose": "training",
        "harness": "pi@0.85.1",
        "renderer": "low",
        "role": "agent",
        "step": 3,
    }
    assert (row["trials"], row["tasks"]) == (4, 2)


async def test_a_job_the_proxy_served_nothing_fails_its_rollout_though_one_beside_it_was_served(
    tmp_path: Path,
) -> None:
    """Each rollout's job is checked alone: one that reached nothing raises, and the one
    gathered beside it, which was served, still returns its groups."""
    tasks = _tasks(tmp_path, "alpha", "beta")
    engine, _, run = _engine(tmp_path)
    answered = Answered("r-0001")  # gather starts the rollouts in order: claude-code's is second
    served, silent = await asyncio.gather(
        engine.rollout(tasks, attempts=1, harness=PI, endpoint=answered),
        engine.rollout(tasks, attempts=1, harness=CLAUDE, endpoint=answered),
        return_exceptions=True,
    )
    assert isinstance(silent, NothingServed)
    assert str(silent).startswith("job r-0001: no trial reached the proxy")
    assert isinstance(served, list) and [g.task.name for g in served] == ["alpha", "beta"]
    assert sorted(answered.checked) == [("r-0000", 2), ("r-0001", 2)], "each over its own trials"
    jobs = {row["harness"]: row["job"] for row in files.read(run / files.JOBS)}
    assert jobs == {"pi@0.85.1": "r-0000", "claude-code@2.1.0": "r-0001"}, "both logged"


async def test_a_rollout_no_trial_could_run_is_refused_before_any_runs(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, "alpha")
    engine, trials, run = _engine(tmp_path)
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        await engine.rollout(tasks, attempts=0, harness=PI, endpoint=Answered())
    assert await engine.rollout([], attempts=1, harness=PI, endpoint=Answered()) == [], "no task"
    assert trials.configs == [] and not (tmp_path / "jobs").exists(), "no job was named"
    assert not (run / files.JOBS).exists()
    unlogged = HarborEngine(Rollout(jobs_dir=str(tmp_path / "jobs")), run_trial=trials)
    with pytest.raises(RuntimeError, match="build the engine with `logs`"):
        await unlogged.rollout(tasks, attempts=1, harness=PI, endpoint=Answered())


# ------------------------------------------------------------------ a recipe's rollouts


async def test_a_recipe_that_wants_two_harnesses_gathers_two_rollouts_under_the_one_limit(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    trials = harbor(Flights())
    rolled: list[list[Group]] = []

    class Both(GRPO):
        class Recipe(GRPO.Recipe):
            second: Harness

        async def step(self) -> None:
            tasks = self.draw()
            rolled[:] = await asyncio.gather(
                *(
                    self.rollout(tasks, model=self.agent, harness=one, attempts=2, modules=None)
                    for one in (self.recipe.harness, self.recipe.second)
                )
            )

    home = run_toml(
        tmp_path,
        'second = "claude-code@2.1.0"',
        **ONE_STEP,
        **{"concurrency = 2": "concurrency = 3"},
    )
    directory = await train(home, tmp_path / "runs", Both)
    assert trials.peak == 3, "eight trials, never more than [rollout] concurrency at once"
    assert any(len(jobs) == 2 for jobs in trials.together), "the two jobs ran at once"
    jobs = [(row["job"], row["harness"]) for row in files.read(directory / files.JOBS)]
    named = [job for job, _ in jobs]
    assert len(set(named)) == 2, "each rollout a job of its own name"
    assert sorted(harness for _, harness in jobs) == ["claude-code@2.1.0", "pi@0.85.1"]
    for groups, harness in zip(rolled, ("pi@0.85.1", "claude-code@2.1.0"), strict=True):
        assert [(g.harness, len(g.trials)) for g in groups] == [(harness, 2)] * 2, "one per task"
    assert [g.task for g in rolled[0]] == [g.task for g in rolled[1]]
    (row,) = rows(directory)
    assert (row["groups"], row["trials"]) == (4, 8), "the row counts every group of the step"
    assert sorted(row["jobs"]) == sorted(named)


async def test_a_trainer_rolls_out_another_model_through_its_own_harness_under_its_role(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    trials = harbor(FakeTrials(default={"reward": 1.0, "tokens": (100, 0, 10)}))

    class Critiqued(GRPO):
        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.critic = self.models["critic"]
            self.teacher = self.models["teacher"]

        async def step(self) -> None:
            tasks = self.draw()
            own = Harness(name="pi", env={"CRITIC": "1"}, kwargs={"style": "terse"})
            critic: dict[str, Any] = {"model": self.critic, "attempts": 1, "modules": None}
            await self.rollout(tasks[:1], harness=own, purpose="critique", **critic)
            with pytest.raises(ValueError, match=r"\[models.critic\] renderers holds none"):
                await self.rollout(tasks, harness="pi", renderer="low", **critic)
            await self.rollout(
                tasks[:1], model=self.teacher, harness="pi", attempts=1, modules=None
            )
            stray = Model("stray", ModelTable(model="claude-x", provider="anthropic"))
            with pytest.raises(ValueError, match="roll out a model of self.models"):
                await self.rollout(tasks, model=stray, harness="pi", attempts=1, modules=None)

    others = (
        '[models.critic]\nmodel = "claude-sonnet-5-5"\nprovider = "anthropic"\n\n'
        '[models.teacher]\nmodel = "Qwen/Qwen3-32B"\n\n[data]'
    )
    home = run_toml(tmp_path, **ONE_STEP, **{"[data]": others})
    directory = await train(home, tmp_path / "runs", Critiqued)
    config, taught = trials.configs
    assert config.agent.model_name == "anthropic/claude-sonnet-5-5"
    assert (config.agent.env, config.agent.kwargs) == ({"CRITIC": "1"}, {"style": "terse"}), (
        "the harness's own settings"
    )
    assert taught.agent.model_name == "openai/Qwen/Qwen3-32B"
    proxy, _ = fakes
    assert [lane for _, lane, _, _ in proxy.opened] == ["teacher"], "opened on its own lane"
    critique, teaching = files.read(directory / files.JOBS)
    assert (critique["purpose"], critique["role"], critique["harness"]) == (
        "critique",
        "critic",
        "pi",
    )
    assert critique["backend"] == "api", "the provider's own"
    assert (teaching["role"], teaching["backend"]) == ("teacher", "tinker")
    usage = list(files.read(directory / files.USAGE))
    assert {one["role"] for one in usage if one.get("provider") == "anthropic"} == {"critic"}
    (row,) = rows(directory)
    assert row["trials"] == 1, "only training rollouts count in the step's row, whoever ran them"


async def test_a_model_on_tinker_beside_a_providers_agent_is_served_by_the_runs_proxy(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(FakeTrials())

    class Taught(Trainer):
        Recipe = GRPO.Recipe

        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.agent = self.models["agent"]
            self.teacher = self.models["teacher"]

        async def step(self) -> None:
            tasks = self.draw()[:1]
            await self.rollout(
                tasks, model=self.teacher, harness=self.recipe.harness, attempts=1, modules=None
            )

    claude = {
        'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"',
        "[data]": '[models.teacher]\nmodel = "Qwen/Qwen3-32B"\n\n[data]',
    }
    directory = await train(run_toml(tmp_path, **ONE_STEP, **claude), tmp_path / "runs", Taught)
    [(_, settings, _)] = proxy.placed
    assert [(one["name"], one["model"]) for one in settings["lanes"]] == [
        ("teacher", "Qwen/Qwen3-32B")
    ], "the teacher's lane alone: a provider's agent has none"
    (config,) = trials.configs
    assert config.agent.model_name == "openai/Qwen/Qwen3-32B"
    (job,) = files.read(directory / files.JOBS)
    assert (job["role"], job["backend"]) == ("teacher", "tinker")
