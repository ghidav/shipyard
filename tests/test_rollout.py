"""Tests for `HarborEngine.rollout`: a rollout runs one job per harness and renderer, all at
once; each is named, judged and logged; the endpoint is checked once over all of them; and
the groups come back one per task, renderer and, unless mixed, harness, in the tasks' order."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL, made
from shipyard import Model, Trainer
from shipyard.config import Blueprint
from shipyard.groups import Connection, Trial
from shipyard.jobs.engine import HarborEngine
from shipyard.models.endpoints import NothingServed
from shipyard.records import served_nothing
from shipyard.run import files
from shipyard.run.logs import Logs
from tests.fakes import FakeProxy, FakeTrials, Harbor
from tests.runs import ONE_STEP, blueprint, rows, train
from tests.service import FakeService

TWO = ["pi", "claude-code"]


class Answered:
    """An endpoint kept in memory: it answers every trial but those of the jobs in `silent`,
    and notes each check it is asked for."""

    provider, model, kind = "tinker", MODEL, "tinker"

    def __init__(self, *silent: str) -> None:
        self.silent = set(silent)
        self.checked: list[tuple[str, int]] = []

    def connection(self, trial: Trial) -> Connection:
        return Connection(model=f"anthropic/{MODEL}")

    async def record(self, job: str, trials: Sequence[Trial]) -> dict[str, list[Any]]:
        return {one.name: [] if job in self.silent else [made()] for one in trials}

    def check(self, job: str, trials: Sequence[Trial]) -> None:
        self.checked.append((job, len(trials)))
        why = served_nothing(one.records for one in trials)
        if why:
            raise NothingServed(f"job {job}: {why}")


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harbor's own check of each harness's config, which is not what these tests are about."""
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)


def _tasks(tmp_path: Path, *names: str) -> list[Path]:
    for name in names:
        (tmp_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
    return [tmp_path / "tasks" / name for name in names]


def _engine(home: Path, *, mix: bool = False) -> tuple[HarborEngine, FakeTrials, Path]:
    """Return an engine that logs to the run `<home>/r`, its fake trials, and that run."""
    config = Blueprint.model_validate(
        {
            "agent": {"model": MODEL, "harness": TWO, "mix_harnesses": mix},
            "data": {"train": "d", "batch_size": 1, "group_size": 3},
            "rollout": {"concurrency": 4, "jobs_dir": str(home / "jobs")},
            "recipe": {"steps": 1},
        }
    )
    run, trials = home / "r", FakeTrials()
    engine = HarborEngine(config.rollout, logs=Logs(config, run), mix=mix, run_trial=trials)
    return engine, trials, run


async def test_a_rollout_groups_each_task_per_harness_unless_mixed_in_the_tasks_order(
    tmp_path: Path,
) -> None:
    beta, alpha = _tasks(tmp_path, "beta", "alpha")
    engine, _, run = _engine(tmp_path / "apart")
    groups = await engine.rollout(
        [beta, alpha], attempts=2, harnesses=TWO, renderers=[None], endpoint=Answered()
    )
    assert [(g.task.name, g.harness, len(g.trials)) for g in groups] == [
        ("beta", "pi", 2),
        ("beta", "claude-code", 2),
        ("alpha", "pi", 2),
        ("alpha", "claude-code", 2),
    ], "a group per task and harness, in the order of the tasks"
    assert [row["job"] for row in files.read(run / files.JOBS)] == ["r-0000", "r-0001"]
    mixed, _, _ = _engine(tmp_path / "mixed", mix=True)
    groups = await mixed.rollout(
        [beta, alpha], attempts=3, harnesses=TWO, renderers=[None], endpoint=Answered()
    )
    assert [(g.task.name, g.harness, len(g.trials)) for g in groups] == [
        ("beta", None, 3),
        ("alpha", None, 3),
    ], "each task's three attempts shared across the harnesses, in one group"


async def test_the_endpoint_is_checked_once_over_every_job_of_the_rollout(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, "alpha", "beta")
    engine, _, run = _engine(tmp_path)
    reached = Answered("r-0000")  # the first job reached nothing, the second did
    groups = await engine.rollout(
        tasks, attempts=1, harnesses=TWO, renderers=[None], endpoint=reached
    )
    assert reached.checked == [("r-0000, r-0001", 4)], "once, over every trial of the rollout"
    assert len(groups) == 4, "a job that reached nothing does not stop the rollout"
    dead = Answered("r-0002", "r-0003")
    with pytest.raises(NothingServed, match="^job r-0002, r-0003: no trial reached the proxy"):
        await engine.rollout(tasks, attempts=1, harnesses=TWO, renderers=[None], endpoint=dead)
    jobs = [row["job"] for row in files.read(run / files.JOBS)]
    assert jobs == ["r-0000", "r-0001", "r-0002", "r-0003"], "every job logged before the check"


async def test_a_rollout_no_trial_could_run_is_refused_before_any_runs(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, "alpha")
    engine, trials, _ = _engine(tmp_path)
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        await engine.rollout(
            tasks, attempts=0, harnesses=TWO, renderers=[None], endpoint=Answered()
        )
    with pytest.raises(ValueError, match="names no harness"):
        await engine.rollout(tasks, attempts=1, harnesses=[], renderers=[None], endpoint=Answered())
    assert trials.configs == []


async def test_a_trainer_rolls_out_another_model_through_its_own_harness_under_its_role(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    trials = harbor(FakeTrials(default={"reward": 1.0, "tokens": (100, 0, 10)}))

    class Critiqued(Trainer):
        def __init__(self, config: Blueprint, directory: Path) -> None:
            super().__init__(config, directory)
            self.agent.training = False
            self.critic = Model("claude-sonnet-5-5", provider="anthropic")
            self.teacher = Model("Qwen/Qwen3-32B")

        async def step(self) -> None:
            tasks, own = self.draw(), {"env": {"CRITIC": "1"}, "kwargs": {"style": "terse"}}
            critic = {"model": self.critic, "harness": "pi"}
            await self.rollout(tasks[:1], attempts=1, purpose="critique", **critic, **own)
            with pytest.raises(ValueError, match="name the harness it runs through"):
                await self.rollout(tasks, model=self.critic)
            with pytest.raises(ValueError, match="renders the agent's calls alone"):
                await self.rollout(tasks, renderer="low", **critic)
            await self.rollout(tasks[:1], attempts=1, model=self.teacher, harness="pi")
            stray = Model("claude-x", provider="anthropic")
            with pytest.raises(ValueError, match="keep the model on the trainer"):
                await self.rollout(tasks, model=stray, harness="pi")

    agents = {'harness = "pi@0.85.1"': 'harness = "pi@0.85.1"\nenv = { AGENT = "1" }'}
    directory = await train(blueprint(tmp_path, **ONE_STEP, **agents), tmp_path / "runs", Critiqued)
    config, taught = trials.configs
    assert config.agent.model_name == "anthropic/claude-sonnet-5-5"
    assert (config.agent.env, config.agent.kwargs) == ({"CRITIC": "1"}, {"style": "terse"}), (
        "its own harness settings, not [agent]'s"
    )
    assert taught.agent.model_name == "openai/Qwen/Qwen3-32B"
    assert "/r/lane/teacher/trial/" in taught.agent.env["OPENAI_BASE_URL"], "its own lane"
    critique, teaching = files.read(directory / files.JOBS)
    assert (critique["purpose"], critique["role"], critique["harness"]) == (
        "critique",
        "critic",
        "pi",
    )
    assert (teaching["role"], teaching["backend"]) == ("teacher", "tinker")
    usage = list(files.read(directory / files.USAGE))
    assert {one["role"] for one in usage if one.get("provider") == "anthropic"} == {"critic"}
    (row,) = rows(directory)
    assert row["trials"] == 0, "only the agent's training rollouts count in the step's row"


async def test_a_model_on_tinker_beside_a_providers_agent_is_served_by_the_runs_proxy(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, _ = fakes
    trials = harbor(FakeTrials())

    class Taught(Trainer):
        def __init__(self, config: Blueprint, directory: Path) -> None:
            super().__init__(config, directory)
            self.agent.training = False
            self.teacher = Model("Qwen/Qwen3-32B")

        async def step(self) -> None:
            await self.rollout(self.draw()[:1], attempts=1, model=self.teacher, harness="pi")

    claude = {'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"'}
    directory = await train(blueprint(tmp_path, **ONE_STEP, **claude), tmp_path / "runs", Taught)
    [(_, settings, _)] = proxy.placed
    assert [(one["name"], one["model"]) for one in settings["lanes"]] == [
        ("teacher", "Qwen/Qwen3-32B")
    ], "the teacher's lane alone: a provider's agent has none"
    (config,) = trials.configs
    assert config.agent.model_name == "openai/Qwen/Qwen3-32B"
    (job,) = files.read(directory / files.JOBS)
    assert (job["role"], job["backend"]) == ("teacher", "tinker")
