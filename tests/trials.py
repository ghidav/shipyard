"""Test helpers that replace Harbor: `result.json` writers, fake trial runners, fixture
datasets and blueprints, a trainer job runner, and readers for a run's records."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeVar

from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.cli import resolved
from shipyard.cli.preflight import Finding
from shipyard.cli.run import Run
from shipyard.modules import SKILL_FILE, Candidate
from shipyard.sampling.endpoint import endpoint_for
from shipyard.training.recipes.gepa.cycle import minibatches
from shipyard.training.recipes.gepa.reflect import COLLECTED
from shipyard.training.trainer import Trainer, carried
from shipyard.types import Agent, Group, Member, Policy, Trial, harness

FIXTURES = Path(__file__).parent / "tasks"
BLUEPRINTS = Path(__file__).parent / "blueprints"
VALID = Path(__file__).parent / "modules" / "valid"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
#: Token counts of the fake proxy's single record per trial: prompt "go" plus completion "ok"
#: is four tokens, three of which are model input.
SEQUENCE_TOKENS = 4
INPUT_TOKENS = 3
START = "tinker://fake/sampler_weights/start"
#: Blueprint replacement that adds a KL term to the recipe.
ANCHORED = {"learning_rate = 2e-5": "learning_rate = 2e-5\nkl_coef = 0.1"}
#: Keys of a step's row that a run with no KL term, no teacher and no token budget leaves out.
UNTAUGHT = ("anchor_kl", "teacher_kl", "taught", "untaught", "overlong")
#: Blueprint replacement that adds a teacher to the recipe.
TAUGHT = {
    "learning_rate = 2e-5": 'learning_rate = 2e-5\nteacher_coef = 0.5\nteacher = "Qwen/Qwen3-32B"'
}
#: Blueprint for a GEPA search over the fixture dataset on a hosted model, with Claude Code as
#: the reflection harness.
GEPA = """
[agent]
model = "some-hosted-model"
provider = "openrouter"
harness = "pi@0.85.1"

[data]
train = "fixture"
batch_size = 1
[rollout]
sandbox = "modal"
[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "anthropic/claude-sonnet-5"
minibatch = 1
patience = 2
"""


def fixture_tasks(tmp_path: Path) -> Path:
    """Copy the fixture datasets to `<tmp>/tasks`."""
    return shutil.copytree(FIXTURES, tmp_path / "tasks")


def check(blueprint: Path) -> list[Finding]:
    """Return the findings `shipyard check` reports for the blueprint."""
    return resolved.report(blueprint).findings


def write_blueprint(tmp_path: Path, text: str) -> Path:
    """Write `text` to `<tmp>/bp/run.toml` and return the blueprint directory, which is the
    form `load` and `check` take."""
    home = tmp_path / "bp"
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


def dataset(root: Path, name: str, *tasks: str) -> Path:
    """Create a dataset of copies of the fixture `alpha` task under the given names."""
    home = root / name
    for task in tasks:
        shutil.copytree(FIXTURES / "fixture" / "alpha", home / task)
    return home


#: Blueprint replacements that put all four fixture tasks in one batch and run one step.
ONE_STEP = {"batch_size = 2": "batch_size = 4", "steps = 2": "steps = 1"}


def blueprint(tmp_path: Path, kind: str = "dapo", *, tasks: int = 4, **replacements: str) -> Path:
    """Write the recipe's fixture blueprint over `tasks` tasks named a, b, c, and so on, with
    two tasks per batch and groups of two."""
    if not (tmp_path / "tasks" / "four").is_dir():
        dataset(tmp_path / "tasks", "four", *"abcdefgh"[:tasks])
    text = (BLUEPRINTS / kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"aime-train"', '"four"').replace("group_size = 4", "group_size = 2")
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp" / kind
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


def write_result(
    trial_dir: Path,
    *,
    reward: float | None = None,
    rewards: Mapping[str, Any] | None = None,
    exception: str | None = None,
    tokens: tuple[int | None, int | None, int | None] | None = None,
    steps: list[tuple[int | None, int | None, int | None]] | None = None,
    provider: str | None = None,
    seconds: float | None = None,
    raw: str | None = None,
) -> Path:
    """Create `trial_dir` with a `result.json` in the shape Harbor's `TrialResult` dumps,
    built from the given parts."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "result.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return trial_dir
    payload: dict[str, Any] = {
        "task_name": trial_dir.name.split("__")[0],
        "trial_name": trial_dir.name,
        "agent_info": {"name": "pi", "version": "0.85.1", "model_info": None},
        "agent_result": None,
        "verifier_result": None,
        "exception_info": None,
    }
    if rewards is not None:
        payload["verifier_result"] = {"rewards": dict(rewards)}
    elif reward is not None:
        payload["verifier_result"] = {"rewards": {"reward": reward}}
    if exception is not None:
        payload["exception_info"] = {
            "exception_type": exception,
            "exception_message": "",
            "exception_traceback": "",
            "occurred_at": T0.isoformat(),
        }
    if tokens is not None:
        payload["agent_result"] = _context(tokens)
    if steps is not None:
        payload["step_results"] = [
            {"step_name": f"step-{n}", "agent_result": _context(each)}
            for n, each in enumerate(steps)
        ]
    if provider is not None:
        payload["agent_info"]["model_info"] = {"name": "m", "provider": provider}
    if seconds is not None:
        payload["environment_setup"] = _phase(T0, 1.0)
        payload["agent_execution"] = _phase(T0 + timedelta(seconds=1), seconds - 2.0)
        payload["verifier"] = _phase(T0 + timedelta(seconds=seconds - 1), 1.0)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return trial_dir


def _context(tokens: tuple[int | None, int | None, int | None]) -> dict[str, Any]:
    n_input, n_cache, n_output = tokens
    return {"n_input_tokens": n_input, "n_cache_tokens": n_cache, "n_output_tokens": n_output}


def _phase(start: datetime, seconds: float) -> dict[str, str]:
    return {
        "started_at": start.isoformat(),
        "finished_at": (start + timedelta(seconds=seconds)).isoformat(),
    }


class FakeTrials:
    """Fake for Harbor's `Trial.create` and `run`. It records every config and the peak
    concurrency, and writes each trial's result from `outcomes` by task name. A task in
    `breaks` raises instead, and an outcome of None leaves the trial directory empty."""

    def __init__(
        self,
        *,
        dwell: float = 0.0,
        breaks: tuple[str, ...] = (),
        outcomes: Mapping[str, dict[str, Any] | None] | None = None,
        default: dict[str, Any] | None = None,
    ) -> None:
        self.configs: list[TrialConfig] = []
        self.dwell = dwell
        self.breaks = breaks
        self.outcomes = dict(outcomes or {})
        self.default = {"reward": 1.0} if default is None else default
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.dwell)
            task = Path(str(config.task.path)).name
            if task in self.breaks:
                raise RuntimeError("the container never came up")
            trial_dir = Path(config.trials_dir) / config.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            outcome = self.outcomes.get(task, self.default)
            if outcome is not None:
                write_result(trial_dir, **outcome)
        finally:
            self.in_flight -= 1


class Fakes:
    """Fake Harbor trial runner for both kinds of job a GEPA search runs. A policy trial
    scores 1 when its skill names its task, and a reflection trial publishes a skill naming
    the task of its first trace."""

    def __init__(self) -> None:
        self.configs: list[TrialConfig] = []
        self.reflected: list[list[str]] = []

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        task = Path(str(config.task.path))
        trial = Path(config.trials_dir) / config.trial_name
        if task.name == "reflection":
            traces = sorted((task / "environment" / "traces").iterdir())
            self.reflected.append([trace.name for trace in traces])
            named = traces[0].read_text().splitlines()[0].removeprefix("task: ")
            published = trial.joinpath(*COLLECTED)
            published.mkdir(parents=True)
            skill = (
                f"---\nname: solving\ndescription: what to do.\n---\n\nmention {Path(named).name}\n"
            )
            (published / SKILL_FILE).write_text(skill)
            write_result(trial, tokens=(1000, 100, 50), provider="anthropic")
            return
        text = (Path(config.agent.skills[0]) / "solving" / SKILL_FILE).read_text()
        trial.mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "agent" / "pi.txt").write_text(f"working on {task.name}")
        write_result(trial, reward=1.0 if f"mention {task.name}\n" in text else 0.0)


class Alternating:
    """Fake trial runner that rewards each task 1, 0, 1, 0 in call order, so every group of
    two has one success and one failure."""

    def __init__(self) -> None:
        self.configs: list[TrialConfig] = []
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        write_result(Path(config.trials_dir) / config.trial_name, reward=float(self.seen[task] % 2))


class Stopped(BaseException):
    pass


class Interrupting:
    """Fake trial runner with alternating rewards whose `after`-th trial raises `Stopped`,
    as a signal would."""

    def __init__(self, after: int) -> None:
        self.inner = Alternating()
        self.after = after
        self.calls = 0

    async def __call__(self, config: TrialConfig) -> None:
        self.calls += 1
        if self.calls == self.after:
            raise Stopped("SIGINT")
        await self.inner(config)


Fake = TypeVar("Fake")


class Harbor(Protocol):
    """Type of the `harbor` fixture, which installs a fake trial runner and returns it."""

    def __call__(self, fake: Fake, /) -> Fake: ...


async def job(
    run: Run,
    batch: list[Path],
    *,
    rollouts: int,
    step: int = 0,
    modules: Candidate | None = None,
    agent: Agent | None = None,
) -> list[Group]:
    """Run one trainer job and return its groups. If `agent` is given, the caller holds its
    policy open and its endpoint is reopened first to check that it is still up; otherwise an
    agent is built for the run and its policy is opened only for the job."""
    trainer = Trainer(run.config, run.directory)
    trainer.steps = step
    if agent is not None:
        await agent.policy.endpoint.open()
        return await trainer.job(agent, batch, rollouts)
    agent = agent_for(run, modules)
    async with agent.policy:
        return await trainer.job(agent, batch, rollouts)


def agent_for(run: Run, modules: Candidate | None = None) -> Agent:
    """Build the agent a run's jobs use: the run's harness over its own endpoint, carrying
    the run's modules unless `modules` is given."""
    held = carried(run.config, run.directory) if modules is None else modules
    policy = Policy(endpoint_for(run.config, run.directory))
    return Agent(policy, [harness(run.config.agent.harness)], held)


def members_of(found: list[Group]) -> list[Member]:
    return [one for group in found for one in group.members]


def job_of(found: list[Group]) -> str:
    return members_of(found)[0].trial.job


def trials_of(found: list[Group]) -> list[Path]:
    return [one.trial.dir for one in members_of(found)]


def trial(directory: Path, *, job: str = "j", task: Path | None = None) -> Trial:
    """Build a `Trial` over `directory` for readers that need only its name and path."""
    return Trial(job, Path(directory.name) if task is None else task, 0, harness("pi"), directory)


def rows(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.METRICS))


def checkpoints(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.CHECKPOINTS))


def drawn(feedback: Sequence[Path], size: int, rng: random.Random, rounds: int) -> list[list[str]]:
    """Return the task names in the minibatches of a search's first `rounds` rounds. These
    are what `minibatches(feedback, size, rng)` yields when each round's parent is drawn from
    the same `rng` just before its minibatch, as `evolve` does."""
    sampled = minibatches(feedback, size, rng)
    found = []
    for _ in range(rounds):
        rng.random()
        found.append([task.name for task in next(sampled)])
    return found
