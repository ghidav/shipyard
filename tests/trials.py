"""Trials and blueprints without Harbor: `result.json` files in the shape `TrialResult`
dumps, runners that stand in for `Trial.create` + `run` and write them, fixture datasets and
`run.toml` writers, one job through the trainer, and a run's rows read back."""

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

from shipyard import record, resolved
from shipyard.modules import SKILL_FILE, Candidate
from shipyard.preflight import Finding
from shipyard.recipes.gepa.cycle import minibatches
from shipyard.recipes.gepa.reflect import COLLECTED
from shipyard.run import Run
from shipyard.sampling.endpoint import endpoint_for
from shipyard.training.trainer import Trainer
from shipyard.types import Agent, Group, Member, Policy, Trial, harness

FIXTURES = Path(__file__).parent / "tasks"
BLUEPRINTS = Path(__file__).parent / "blueprints"
VALID = Path(__file__).parent / "modules" / "valid"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
#: The fake proxy's one record per trial: prompt "go" + completion "ok" is four tokens,
#: of which the model reads three.
SEQUENCE_TOKENS = 4
INPUT_TOKENS = 3
START = "tinker://fake/sampler_weights/start"
#: A blueprint replacement that gives the recipe a KL term.
ANCHORED = {"learning_rate = 2e-5": "learning_rate = 2e-5\nkl_coef = 0.1"}
#: A gepa search over the fixture dataset on a hosted model, Claude Code reflecting.
GEPA = """
[model]
name = "some-hosted-model"
provider = "openrouter"
[data]
train = "fixture"
batch_size = 1
[rollout]
harness = "pi@0.85.1"
sandbox = "modal"
[recipe]
kind = "gepa"
reflection_harness = "claude-code"
reflection_model = "anthropic/claude-sonnet-5"
minibatch = 1
patience = 2
"""


def fixture_tasks(tmp_path: Path) -> Path:
    """The fixture datasets copied to `<tmp>/tasks`."""
    return shutil.copytree(FIXTURES, tmp_path / "tasks")


def check(blueprint: Path) -> list[Finding]:
    """The findings `shipyard check` prints for the blueprint."""
    return resolved.report(blueprint).findings


def write_blueprint(tmp_path: Path, text: str) -> Path:
    """Writes `text` to `<tmp>/bp/run.toml` and returns the blueprint directory, the form
    `load` and `check` take."""
    home = tmp_path / "bp"
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


def dataset(root: Path, name: str, *tasks: str) -> Path:
    """A dataset of copies of the fixture `alpha` task, under the names given."""
    home = root / name
    for task in tasks:
        shutil.copytree(FIXTURES / "fixture" / "alpha", home / task)
    return home


#: The fixture's four tasks in one batch, and one step to take them all.
ONE_STEP = {"batch_size = 2": "batch_size = 4", "steps = 2": "steps = 1"}


def blueprint(tmp_path: Path, kind: str = "dapo", *, tasks: int = 4, **replacements: str) -> Path:
    """The recipe's fixture over `tasks` tasks named a, b, c..., two per batch, groups of two."""
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
    """A trial directory holding a `result.json` built from the parts given."""
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
    """Stands in for Harbor's `Trial.create` + `run`. Records every config and the peak
    concurrency, and writes each trial's directory and result as `outcomes` says by task
    name. A task in `breaks` raises instead. An outcome of None leaves the directory empty."""

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
    """Harbor's trial replaced for both kinds of job a search runs: a policy trial scores 1
    when the skill it was handed names its task, and a reflection trial publishes a skill
    naming the task its first trace is about."""

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
    """Rewards 1, 0, 1, 0 per task in call order, so every group of two has spread."""

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
    """Alternating rewards, and the `after`-th trial stops the run as a signal would."""

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
    """The `harbor` fixture: stands a fake in for Harbor's trial and returns it."""

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
    """One job through the trainer under `agent`, whose policy the caller holds open; its
    endpoint is checked still up first. Without one, an agent made for the run, its policy
    opened for the job and closed after."""
    trainer = Trainer(run)
    trainer.steps = step
    if agent is not None:
        await agent.policy.endpoint.open()
        return await trainer.job(agent, batch, rollouts)
    agent = agent_for(run, modules)
    async with agent.policy:
        return await trainer.job(agent, batch, rollouts)


def agent_for(run: Run, modules: Candidate | None = None) -> Agent:
    """The agent a run's jobs carry: its harness over an endpoint of its own, carrying the
    run's modules unless handed others."""
    carried = run.carried if modules is None else modules
    policy = Policy(endpoint_for(run.config, run.directory))
    return Agent(policy, [harness(run.config.rollout.harness)], carried)


def members_of(found: list[Group]) -> list[Member]:
    return [one for group in found for one in group.members]


def job_of(found: list[Group]) -> str:
    return members_of(found)[0].trial.job


def trials_of(found: list[Group]) -> list[Path]:
    return [one.trial.dir for one in members_of(found)]


def trial(directory: Path, *, job: str = "j", task: Path | None = None) -> Trial:
    """A trial over a directory, for the readers that need no more than its name and dir."""
    return Trial(job, Path(directory.name) if task is None else task, 0, harness("pi"), directory)


def rows(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.METRICS))


def checkpoints(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.CHECKPOINTS))


def drawn(feedback: Sequence[Path], size: int, rng: random.Random, rounds: int) -> list[list[str]]:
    """By task name, the minibatches of a search's first `rounds` rounds: what
    `minibatches(feedback, size, rng)` yields, each round's parent drawn from the same `rng`
    just before its minibatch, as `evolve` draws them."""
    sampled = minibatches(feedback, size, rng)
    found = []
    for _ in range(rounds):
        rng.random()
        found.append([task.name for task in next(sampled)])
    return found
