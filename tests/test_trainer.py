"""Tests for `shipyard`: the four primitives, the base step, the recipes, and the run around
them (checkpoints, resuming, evaluation, costs, the CLI)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proxies import MODEL, made
from shipyard import Blueprint, Model, Trainer, Trial, load
from shipyard.cli import app
from shipyard.config import ConfigError
from shipyard.jobs.engine import split
from shipyard.jobs.harnesses import harness
from shipyard.jobs.modules import PROMPT, PROMPT_FILE, Module, Modules
from shipyard.models.backends import connect
from shipyard.models.endpoints import NothingServed
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.costs import Costs
from shipyard.run.data import Generator, tasks
from shipyard.run.directory import (
    continued,
    open_run,
    reopened,
    resumed,
    running,
    trainer_class,
    trainer_in,
)
from tests.fakes import (
    INPUT_TOKENS,
    SEQUENCE_TOKENS,
    START,
    Alternating,
    FakeProxy,
    FakeTrials,
    Harbor,
    Spread,
    write_result,
)
from tests.runs import ONE_STEP, blueprint, rows, train
from tests.service import FakeService, FlatAnchor

RECIPES = Path(__file__).parents[1] / "recipes"
#: DAPO's own `[recipe]` keys, at the paper's values.
DAPO = ("clip_low = 0.2", "clip_high = 0.28", "overlong_penalty = 0.5", "overlong_buffer = 0.2")


def _advantages(service: FakeService) -> list[float]:
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    return sorted(
        {round(a, 4) for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a}
    )


def _old(service: FakeService) -> set[float]:
    """Return the old policy's logprobs every training pass took its ratio against."""
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    return {round(a, 4) for d in trained for a in d.loss_fn_inputs["logprobs"].tolist() if a}


def _forwards(service: FakeService) -> list[list[int]]:
    return [c[1] for c in service.client.calls if c[0] == "forward"]


async def test_the_base_step_draws_rolls_out_scores_and_updates(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    directory = await train(blueprint(tmp_path), tmp_path / "runs")
    client = service.client
    assert service.calls == [("lora", MODEL, 32)]
    # No reference pass: the sampler's logprobs are the old policy's. Then the pass and the
    # optimizer step.
    steps = [c[0] for c in client.calls if c[0] in ("forward", "forward_backward", "optim_step")]
    assert steps == ["forward_backward", "optim_step"] * 2
    assert _old(service) == {-0.5}, "the sampler's logprobs, as the proxy recorded them"
    assert _advantages(service) == [-0.5, 0.5], "each reward less its group's mean"
    assert proxy.swapped == [START] + [f"tinker://fake/sampler_weights/sample-{n}" for n in (1, 2)]
    for row in rows(directory):
        assert row["trained"] is True and (row["groups"], row["trials"]) == (2, 4)
        assert (row["uniform"], row["sequences"]) == (0, 4)
        assert row["train_tokens"] == 4 * INPUT_TOKENS and row["loss"] == "importance_sampling"
        assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5), (
            "the sampler's -0.5, the trainer's -0.3"
        )
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [(c["tag"], c["step"], c["weights"]) for c in saved] == [
        ("step-1", 1, {"agent": "tinker://fake/weights/step-1"}),
        ("step-2", 2, {"agent": "tinker://fake/weights/step-2"}),
        ("final", 2, {"agent": "tinker://fake/weights/final"}),
    ], "each trained model's weights, by its role"
    assert [c["state"] for c in saved] == [
        f"{files.STATES}/step-1.json",
        f"{files.STATES}/step-2.json",
        f"{files.STATES}/step-2.json",
    ], "final is the state after the last step, with weights of its own"
    states = [files.read_json(directory / c["state"]) for c in saved]
    assert [(s["steps"], s["updates"], s["drawn"]) for s in states] == [
        (1, {"agent": 1}, 1),
        (2, {"agent": 2}, 2),
        (2, {"agent": 2}, 2),
    ]
    assert [row["step"] for row in rows(directory)] == [1, 2], "a step's row, counted from 1"
    jobs = list(files.read(directory / files.JOBS))
    assert [(job["step"], Path(job["dir"]).name) for job in jobs] == [
        (1, f"{directory.name}-0000"),
        (2, f"{directory.name}-0001"),
    ], "each job's row says its step and where its trials are"
    assert all(Path(job["dir"]).is_absolute() and Path(job["dir"]).is_dir() for job in jobs)
    costs = files.read_json(directory / files.COSTS)
    # Two jobs of four "go" prompts and "ok" replies, and the two training passes.
    assert costs["tokens"] == {"tinker": {MODEL: {"prefill": 16, "sample": 16, "train": 24}}}
    usage = list(files.read(directory / files.USAGE))
    assert Costs.of(usage).to_dict() == costs, "the costs are the usage ledger's totals"
    tokens = [(row["step"], row.get("job"), "train" in row) for row in usage if "provider" in row]
    assert tokens == [
        (1, f"{directory.name}-0000", False),
        (1, None, True),
        (2, f"{directory.name}-0001", False),
        (2, None, True),
    ], "each step's rollouts, by job, then its training pass"
    assert {row["role"] for row in usage} == {"agent"}
    note = files.read_json(directory / files.RUN)["process"]
    assert note["failed"] is False


async def test_a_sequence_recorded_without_logprobs_is_scored_at_the_current_weights(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    proxy.default = [replace(made(), inference_logprobs=())]
    harbor(Alternating())
    await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs")
    steps = [c[0] for c in service.client.calls if c[0] in ("forward", "forward_backward")]
    assert steps == ["forward", "forward_backward"], "a reference pass for what the proxy missed"
    assert _forwards(service) == [[INPUT_TOKENS] * 4]


async def test_a_step_with_nothing_to_teach_takes_no_gradient(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(FakeTrials())
    directory = await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs")
    (row,) = rows(directory)
    assert row["trained"] is False and row["uniform"] == 2 and "sequences" not in row
    assert not any(c[0] in ("forward", "forward_backward") for c in service.client.calls)
    saved = [c["weights"] for c in files.read(directory / files.CHECKPOINTS)]
    assert saved == [{}, {}], "no update, so no weights to keep: the state alone"


async def test_a_run_that_trains_its_agent_with_no_learning_rate_is_refused_before_it_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    home = blueprint(tmp_path, **ONE_STEP, **{"learning_rate = 2e-5\n": ""})
    assert load(home).recipe.learning_rate is None, "a recipe that never updates leaves it out"
    with pytest.raises(ValueError, match=r"^\[recipe\] learning_rate: the agent trains"):
        await train(home, tmp_path / "runs", Trainer)
    assert service.calls == [] and proxy.placed == [], "nothing opened"


async def test_a_recipe_can_grade_the_rollouts_that_ran_out_of_time(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(FakeTrials(default={"reward": 1.0, "exception": "AgentTimeoutError"}))
    masked = await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs")
    graded = await train(
        blueprint(tmp_path, "grade_timeouts = true", **ONE_STEP), tmp_path / "runs"
    )
    counted = [(row["graded"], row["masked"]) for row in (*rows(masked), *rows(graded))]
    assert counted == [(0, 4), (4, 0)], "masked by default; graded when the recipe says so"


async def test_a_job_nothing_reached_stops_the_run_before_training_and_closes_it_as_errored(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    proxy.default = []  # no trial reaches the proxy, so each trial's records come back empty
    harbor(FakeTrials())
    directory, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    with pytest.raises(NothingServed, match="no trial reached the proxy"):
        with running(directory), Trainer(config, directory) as trainer:
            await trainer.run()
    assert not any(c[0] == "forward_backward" for c in service.client.calls)
    dead = "no trial reached the proxy; the endpoint is dead or unreachable"
    assert service.closed == [("errored", f"NothingServed: job {directory.name}-0000: {dead}")]
    assert len(list(files.read(directory / files.JOBS))) == 1, "the job's row comes first"
    assert not (directory / files.METRICS).exists()
    assert not (directory / files.CHECKPOINTS).exists(), "no final checkpoint"
    assert proxy.closed == 1


async def test_the_opd_example_trains_every_rollout_on_the_teachers_gap(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    teacher = service.teachers["Qwen/Qwen3-32B"] = FlatAnchor(-0.2)
    harbor(FakeTrials())
    home = blueprint(tmp_path, 'teacher = "Qwen/Qwen3-32B"', **ONE_STEP)
    directory = await train(home, tmp_path / "runs", RECIPES / "opd/trainer.py")
    assert ("sampling", "Qwen/Qwen3-32B", None) in service.calls and len(teacher.asked) == 4
    assert _advantages(service) == [0.3], "equal rewards, and the teacher's gap on every token"
    (row,) = rows(directory)
    assert row["trained"] is True and row["uniform"] == 0
    assert row["teacher_kl"] == pytest.approx(-0.3, abs=1e-5)
    spent = files.read_json(directory / files.COSTS)["tokens"]["tinker"]["Qwen/Qwen3-32B"]
    assert spent == {"prefill": 4 * SEQUENCE_TOKENS, "sample": 4}
    usage = list(files.read(directory / files.USAGE))
    assert {row["role"] for row in usage if row.get("model") == "Qwen/Qwen3-32B"} == {"teacher"}


def test_a_trial_reads_its_task_what_the_agent_did_and_what_the_tests_printed(
    tmp_path: Path,
) -> None:
    task, ran = tmp_path / "task", tmp_path / "trial"
    (ran / "agent").mkdir(parents=True)
    (ran / "verifier").mkdir()
    task.mkdir()
    (task / "instruction.md").write_text("Write hello.txt.")
    (ran / "verifier" / "test-stdout.txt").write_text("1 failed")
    picture = {"type": "image", "source": {"media_type": "image/png", "path": "shot.png"}}
    steps = [
        {"step_id": 1, "source": "user", "message": "Write hello.txt."},
        {
            "step_id": 2,
            "source": "agent",
            "message": [{"type": "text", "text": "Writing it."}, picture],
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "bash", "arguments": {"cmd": "ls"}}
            ],
            "observation": {"results": [{"source_call_id": "c1", "content": "app"}]},
        },
    ]
    trajectory = {"schema_version": "ATIF-v1.6", "session_id": "s", "steps": steps}
    (ran / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    trial = Trial("job", task, harness("pi"), ran)
    assert trial.instruction == "Write hello.txt." and trial.test_output == "1 failed"
    assert [one["step_id"] for one in trial.steps] == [1, 2], "as Harbor wrote them"
    assert trial.transcript == (
        "[user]\nWrite hello.txt.\n\n"
        '[agent]\nWriting it.\n[image]\n[call] bash {"cmd": "ls"}\n[result]\napp'
    )
    gone = Trial("job", tmp_path / "none", harness("pi"), tmp_path / "none")
    assert (gone.instruction, gone.transcript, gone.test_output) == ("", "", "")
    assert gone.steps == []


async def test_dapo_keeps_drawing_until_a_batchs_worth_of_groups_have_spread(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    home = blueprint(tmp_path, *DAPO, **ONE_STEP)
    plan = Generator(tasks("four", root=tmp_path / "tasks"), 2, 0)
    first, second = [[one.name for one in batch] for batch in islice(plan.batches(), 2)]
    harbor(Spread({first[0], *second}))
    directory = await train(home, tmp_path / "runs", RECIPES / "dapo/trainer.py")
    (row,) = rows(directory)
    assert len(row["jobs"]) == 2 and (row["groups"], row["uniform"]) == (4, 1)
    assert row["sequences"] == 6 and row["loss"] == "ppo"


def test_dapos_overlong_penalty_grows_over_the_last_share_of_the_budget_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollout's budget is its model calls together: `overlong_budget`, or else the proxy's
    per-rollout budget, which no rollout opened yet has."""
    monkeypatch.chdir(tmp_path)
    home = blueprint(tmp_path, *DAPO, "overlong_budget = 1000")
    directory, config = open_run(home, root=tmp_path / "runs", trainer=RECIPES / "dapo/trainer.py")
    dapo = trainer_class(directory)(config, directory)
    # 0.5 at the budget, over its last fifth: from 800 tokens on.
    assert [dapo.overlong(n) for n in (800, 900, 1000, 4000)] == [0.0, -0.25, -0.5, -0.5]
    unset, config = open_run(
        blueprint(tmp_path, *DAPO), root=tmp_path / "runs", trainer=RECIPES / "dapo/trainer.py"
    )
    assert trainer_class(unset)(config, unset).overlong(10**6) == 0.0


def _sent(service: FakeService) -> set[tuple[Any, ...]]:
    """Return the loss and loss config of every training pass."""
    return {(c[2], tuple(c[3].items())) for c in service.client.calls if c[0] == "forward_backward"}


def _adam(service: FakeService) -> Any:
    """Return the Adam settings of the run's only optimizer step."""
    (adam,) = [c[1] for c in service.client.calls if c[0] == "optim_step"]
    return adam


#: The prompt the agent starts from, and the one the fake reflector rewrites any prompt to.
SEED, CAREFUL = "Solve the task.", "Be careful: check the answer, then finish."
#: Has the agent carry the modules in the blueprint's `modules/`.
CARRIED = {'harness = "pi@0.85.1"': 'harness = "pi@0.85.1"\nmodules = "modules"'}
#: The file a reflection task holds the module in, and its trial's tests keep it in.
REFLECTED = "module.md"
#: A GEPA search of the four tasks, its reflector a provider's model through `pi`.
SEARCH = (
    "budget = 12",
    "pareto = 2",
    "minibatch = 2",
    'reflector = { model = "claude-sonnet-5-5", provider = "anthropic", harness = "pi" }',
)


class Careful:
    """Fake trial runner. An agent's trial is rewarded 1 when the prompt delivered to it says
    "careful", and 0 otherwise. A reflector's trial, at a task that holds a module, notes the
    text it was handed and rewrites it to `answer`, which its tests keep."""

    def __init__(self, answer: str = CAREFUL) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.handed: list[str] = []

    async def __call__(self, config: Any) -> None:
        trial = Path(config.trials_dir) / config.trial_name
        module = Path(str(config.task.path)) / "environment" / REFLECTED
        if module.is_file():
            self.handed.append(module.read_text(encoding="utf-8"))
            write_result(trial, reward=1.0, tokens=(100, 0, 10))
            (trial / "verifier").mkdir(exist_ok=True)
            (trial / "verifier" / REFLECTED).write_text(self.answer, encoding="utf-8")
            return
        template = (config.agent.kwargs or {}).get("prompt_template_path")
        said = Path(template).read_text(encoding="utf-8") if template else ""
        self.prompts.append(said)
        write_result(trial, reward=float("careful" in said.lower()))


def _seeded(home: Path) -> None:
    """Seed the modules in `home` with the prompt `SEED`."""
    (home / "modules").mkdir()
    (home / "modules" / "PROMPT.md").write_text(f"{SEED}\n", encoding="utf-8")


async def test_the_gepa_example_searches_the_prompt_and_keeps_the_best(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    pytest.importorskip("gepa")
    trials = harbor(Careful())
    unrated = {"learning_rate = 2e-5\n": ""}  # a search takes no gradient step
    home = blueprint(tmp_path, *SEARCH, **CARRIED, **ONE_STEP, **unrated)
    _seeded(home)
    directory = await train(home, tmp_path / "runs", RECIPES / "gepa/trainer.py")
    best = (directory / "modules" / "best" / "PROMPT.md").read_text(encoding="utf-8")
    assert best.strip() == CAREFUL
    assert trials.handed and SEED in trials.handed[0], "the reflector read the seed's text"
    task = directory / "gepa" / "reflections" / "0001" / "00"
    assert (task / "environment" / "attempts").is_dir() and (task / "tests" / "test.sh").is_file()
    assert any(SEED in one for one in trials.prompts), "the seed was rolled out"
    assert any("careful" in one for one in trials.prompts), "and so was its rewrite"
    jobs = list(files.read(directory / files.JOBS))
    assert {(job["purpose"], job["role"]) for job in jobs} == {
        ("search", "agent"),
        ("reflection", "reflector"),
    }
    (row,) = rows(directory)
    assert row["best_score"] == 1.0 and row["candidates"] == 2
    assert row["reflections"] == len(trials.handed)
    spent = files.read_json(directory / files.COSTS)["tokens"]
    assert spent["anthropic"]["claude-sonnet-5-5"]["sample"] == 10 * len(trials.handed)
    usage = list(files.read(directory / files.USAGE))
    asked = {one["role"] for one in usage if one.get("provider") == "anthropic"}
    assert asked == {"reflector"}, "the reflector's spend is filed under its role"
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [(c["tag"], c["weights"]) for c in saved] == [("step-1", {}), ("final", {})], (
        "a search moves no weights, and saves the modules it found"
    )
    state = Trainer.State.model_validate(files.read_json(directory / saved[-1]["state"]))
    assert state.modules is not None and state.modules.texts[PROMPT].strip() == CAREFUL


async def test_a_gepa_proposal_that_cannot_be_delivered_scores_nothing_and_the_search_goes_on(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    pytest.importorskip("gepa")
    trials = harbor(Careful(answer=f"{CAREFUL} {{% endraw %}}"))
    unrated = {"learning_rate = 2e-5\n": ""}
    home = blueprint(tmp_path, *SEARCH, **CARRIED, **ONE_STEP, **unrated)
    _seeded(home)
    directory = await train(home, tmp_path / "runs", RECIPES / "gepa/trainer.py")
    assert trials.handed, "the reflector proposed"
    assert trials.prompts and not any("careful" in one for one in trials.prompts), (
        "no proposal was rolled out"
    )
    best = (directory / "modules" / "best" / "PROMPT.md").read_text(encoding="utf-8")
    assert best.strip() == SEED


async def test_a_run_that_stopped_goes_on_as_a_new_run_from_its_last_checkpoint(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())

    class Crashing(Trainer):
        async def step(self) -> None:
            if self.state.steps == 1:
                raise RuntimeError("the machine went away")
            await super().step()

    origin, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="went away"):
        with running(origin), Crashing(config, origin) as crashing:
            await crashing.run()
    assert files.read_json(origin / files.RUN)["process"]["failed"] is True
    written = (files.METRICS, files.CHECKPOINTS, files.USAGE, files.COSTS)
    before = {name: (origin / name).read_bytes() for name in written}

    going = resumed(origin)
    assert going is not None
    directory, config, checkpoint = going
    assert checkpoint.tag == "step-1"
    assert directory.parent == origin.parent and directory.name.startswith("four__")
    trainer = Trainer(config, directory)
    trainer.restore(origin, checkpoint)
    with running(directory), trainer:
        await trainer.run()
    noted = files.read_json(directory / files.RUN)
    assert noted["origin"] == {"run": str(origin.resolve()), "tag": "step-1", "first": origin.name}
    assert noted["process"]["failed"] is False
    assert [one["step"] for one in rows(directory)] == [2], "its own rows, from where it went on"
    assert [c["tag"] for c in files.read(directory / files.CHECKPOINTS)] == ["step-2", "final"]
    assert {row["step"] for row in files.read(directory / files.USAGE)} == {2}, "its own costs"
    assert {name: (origin / name).read_bytes() for name in written} == before, (
        "the other is as it was"
    )
    loaded = service.client.calls.index(
        ("load_state_with_optimizer", "tinker://fake/weights/step-1")
    )
    after = [c[1] for c in service.client.calls[loaded:] if c[0] == "save_weights_for_sampler"]
    assert after[0] == "start", "the weights it loaded, saved for the proxy"
    assert proxy.swapped[2] == "tinker://fake/sampler_weights/start", "serves the weights it loaded"
    sampled = [c[1] for c in service.client.calls if c[0] == "save_weights_for_sampler"]
    assert sampled == ["start", "sample-1", "start", "sample-2", "final"], "a step's is its state"


class Noting(Trainer):
    """A recipe that moves no weights: it notes the tasks it draws in its state, and carries a
    prompt it rewrites each step. It stops at step `stop`, as a machine that went away would."""

    stop: int | None = None

    class State(Trainer.State):
        seen: list[str] = []

    async def step(self) -> None:
        if self.state.steps == self.stop:
            raise RuntimeError("the machine went away")
        self.state.seen += [task.name for task in self.draw()]
        self.state.modules = Modules(
            {PROMPT: Module(PROMPT, "prompt", {PROMPT_FILE: f"{self.state.steps}.\n"})}
        )


async def test_a_run_that_moves_no_weights_saves_its_state_and_a_resume_goes_on_exactly(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    longer = {"steps = 2": "steps = 3"}
    whole = await train(blueprint(tmp_path, **longer), tmp_path / "runs", Noting)

    class Stopping(Noting):
        stop = 2

    origin, config = open_run(blueprint(tmp_path, **longer), root=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="went away"):
        with running(origin), Stopping(config, origin) as stopping:
            await stopping.run()
    going = resumed(origin)
    assert going is not None
    directory, config, checkpoint = going
    assert (checkpoint.tag, checkpoint.weights) == ("step-2", {}), "a state, no weights to save"
    trainer = Noting(config, directory)
    trainer.restore(origin, checkpoint)
    assert (trainer.state.steps, trainer.state.drawn, len(trainer.state.seen)) == (2, 2, 4)
    assert trainer.state.modules is not None and trainer.state.modules.texts[PROMPT] == "1.\n"
    with running(directory), trainer:
        await trainer.run()
    tags = [c["tag"] for c in files.read(directory / files.CHECKPOINTS)]
    assert tags == ["step-3", "final"]
    done = files.read_json(whole / files.STATES / "step-3.json")
    assert trainer.state == Noting.State.model_validate(done), "as if it had never stopped"
    assert not [c for c in service.client.calls if "state" in c[0]], "nothing to save or load"
    _, step = reopened(directory, tag="step-3")
    again, _ = continued(directory, step)
    noted = files.read_json(again / files.RUN)["origin"]
    assert (noted["run"], noted["first"]) == (str(directory), origin.name), "a chain's first run"


async def test_a_run_checkpoints_every_few_steps_and_saves_weights_once_updates_moved_them(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(FakeTrials())  # every reward 1, so no group teaches anything, and nothing updates
    every = {"steps = 2": "steps = 3", "[recipe]": "[checkpoints]\nevery = 2\n\n[recipe]"}
    directory = await train(blueprint(tmp_path, **every), tmp_path / "runs")
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [(c["tag"], c["step"], c["weights"]) for c in saved] == [
        ("step-2", 2, {}),
        ("final", 3, {}),
    ]
    assert not [c for c in service.client.calls if c[0] == "save_state"]


def test_a_checkpoints_state_must_fit_the_recipes_and_drops_what_the_recipe_no_longer_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    saved = directory / files.STATES / "step-1.json"
    row = Checkpoint("step-1", 1, f"{files.STATES}/step-1.json")
    files.write_json(saved, {"steps": 1, "drawn": 1, "seen": ["a", "b"], "gone": 3})
    trainer = Noting(config, directory)
    with caplog.at_level("WARNING"):
        trainer.restore(directory, row)
    assert (trainer.state.steps, trainer.state.drawn, trainer.state.seen) == (1, 1, ["a", "b"])
    assert "gone" in caplog.text, "a field the recipe dropped is said to be dropped"
    files.write_json(saved, {"steps": 1, "seen": "a, b"})
    with pytest.raises(ValueError, match="does not fit Noting.State"):
        Noting(config, directory).restore(directory, row)
    with pytest.raises(ValueError, match="its state is missing or unreadable"):
        Noting(config, directory).restore(
            directory, replace(row, state=f"{files.STATES}/step-9.json")
        )


def test_resuming_a_finished_run_opens_none_and_one_with_more_steps_goes_on_from_it(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(Alternating())
    runner, runs = CliRunner(), tmp_path / "r"
    done = runner.invoke(app, ["run", str(blueprint(tmp_path, **ONE_STEP)), "--root", str(runs)])
    assert done.exit_code == 0, done.output
    (directory,) = runs.iterdir()
    calls = len(service.calls)
    again = runner.invoke(app, ["resume", str(directory)])
    assert again.exit_code == 0 and "nothing is left to run" in again.output
    assert [one.name for one in runs.iterdir()] == [directory.name] and len(service.calls) == calls
    # Given more steps, the finished run is a checkpoint like any other.
    toml = directory / files.CONFIG
    toml.write_text(toml.read_text().replace("steps = 1", "steps = 2"))
    more = runner.invoke(app, ["resume", str(directory)])
    assert more.exit_code == 0, more.output
    (other,) = [one for one in runs.iterdir() if one != directory]
    assert f"run {other.name} goes on from {directory.name} at final" in more.output
    assert [one["step"] for one in rows(other)] == [2], "the step after the one it went on from"


async def test_a_tasks_trials_count_as_one_group_whatever_jobs_they_ran_in(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """As FST's are: each task once under each of two texts, one job per text."""
    harbor(Alternating())

    class Twice(Trainer):
        async def step(self) -> None:
            batch = self.draw()
            for _ in range(2):
                await self.rollout(batch, attempts=1)

    directory = await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Twice)
    (row,) = rows(directory)
    assert len(row["jobs"]) == 2 and (row["groups"], row["trials"]) == (2, 4)
    assert row["reward_spread"] == 0.5, "each task's 1 and 0, across the jobs"


async def test_a_recipe_metric_named_like_a_row_key_is_logged_beside_it(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())

    class Measuring(Trainer):
        async def step(self) -> None:
            await super().step()
            self.metrics.update(entropy=1.5, trials=9, own=1)

    directory = await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Measuring)
    (row,) = rows(directory)
    assert row["entropy"] != 1.5 and row["trials"] == 4, "the run's own keys are kept"
    assert (row["recipe_entropy"], row["recipe_trials"], row["own"]) == (1.5, 9, 1)


async def test_the_test_set_is_evaluated_every_few_steps_and_not_again_at_the_end(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())
    with_test = {"group_size = 2": 'group_size = 2\ntest = "four"\ntest_group_size = 1'}
    home = blueprint(tmp_path, "evaluate_every = 1", **ONE_STEP, **with_test)
    directory = await train(home, tmp_path / "runs")
    (row,) = rows(directory)
    assert (row["step"], row["eval_tasks"], row["eval_trials"]) == (1, 4, 4), "in its step's row"
    jobs = [one["purpose"] for one in files.read(directory / files.JOBS)]
    assert jobs == ["training", "evaluation", "evaluation"]
    assert row["groups"] == 2, "an evaluation's groups are not the step's"


def test_the_recipe_table_holds_the_keys_its_trainer_declares_and_every_table_is_checked(
    tmp_path: Path,
) -> None:
    class Anchored(Trainer):
        class Recipe(Trainer.Recipe):
            anchor_coef: float
            teacher: str | None = None

    loaded = load(blueprint(tmp_path, "anchor_coef = 0.01"), Anchored.Recipe)
    assert loaded.recipe.anchor_coef == 0.01 and loaded.recipe.teacher is None
    with pytest.raises(ConfigError) as refused:
        load(blueprint(tmp_path, "anchor_cof = 0.01"), Anchored.Recipe)
    assert set(refused.value.problems) == {
        "[recipe] anchor_coef: field required",
        "[recipe] anchor_cof: unknown key",
    }
    with pytest.raises(ConfigError) as refused:
        load(blueprint(tmp_path, "anchor_coef = 0.01", 'kind = "dapo"'))
    assert set(refused.value.problems) == {
        "[recipe] anchor_coef: unknown key",
        "[recipe] kind: unknown key",
    }, "the base trainer reads no keys of its own"
    with pytest.raises(ConfigError) as refused:
        load(blueprint(tmp_path, **{"steps = 2\n": "", "[rollout]": "[rolout]"}))
    assert set(refused.value.problems) == {
        "[recipe] steps: field required",
        "[rolout]: unknown table",
    }
    # [agent] is the run's own policy: Tinker serves it, unless it names a provider.
    assert load(blueprint(tmp_path)).agent.provider == "tinker"
    named = load(
        blueprint(tmp_path, **{'harness = "pi@0.85.1"': 'harness = "pi"\nprovider = "anthropic"'})
    )
    assert named.agent.provider == "anthropic"
    with pytest.raises(ConfigError, match=r"^\[agent\] weights: unknown key$"):
        load(
            blueprint(
                tmp_path, **{'harness = "pi@0.85.1"': 'harness = "pi"\nweights = "tinker://w"'}
            )
        )
    # The run reads no harness log for its turns, and `train` has no older name.
    with pytest.raises(ConfigError) as refused:
        load(
            blueprint(tmp_path, **{"concurrency = 2": "check_turns = true", "train =": "dataset ="})
        )
    assert set(refused.value.problems) == {
        "[rollout] check_turns: unknown key",
        "[data] dataset: unknown key",
        "[data] train: field required",
    }
    with pytest.raises(ConfigError, match=r"^\[data\] train: a dataset name or a list of them$"):
        load(blueprint(tmp_path, **{'train = "four"': "train = 4"}))


class Holding(Trainer):
    """A trainer holding a model of every kind: as attributes, in a dict and in a list."""

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.critics = {"main": Model("claude-sonnet-5-5", provider="anthropic")}
        self.teachers = [Model("Qwen/Qwen3-32B"), Model("tinker://r/sampler_weights/final")]


def test_the_run_makes_its_agent_and_finds_the_models_a_recipe_holds(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    directory, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    trainer = Holding(config, directory)
    agent = trainer.policy
    assert (agent.provider, agent.name) == ("tinker", MODEL)
    assert agent.training, "the run trains its agent unless its recipe says otherwise"
    held = trainer._held()
    assert held == [
        ("agent", agent),
        ("critics[main]", trainer.critics["main"]),
        ("teachers[0]", trainer.teachers[0]),
        ("teachers[1]", trainer.teachers[1]),
    ], "each named for where the trainer holds it, as its spend is filed"
    assert all(one.endpoint is None for _, one in held), "nothing built yet"
    assert not any(one.training for _, one in held[1:]), "the run trains its agent alone"
    with pytest.raises(AttributeError):
        trainer.policy = Model("Qwen/Qwen3-32B")  # type: ignore[misc]


async def test_a_model_is_used_once_the_run_binds_it_to_what_it_runs_on(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    directory, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    trainer = Holding(config, directory)
    agent, teacher = trainer.policy, trainer.teachers[0]
    with pytest.raises(RuntimeError, match="not connected yet"):
        await agent.logprobs([])
    connected = connect(config, directory, trainer._held(), trainer.logs.spend)
    assert connected.session is not None and connected.session.trains
    assert agent.endpoint is not None and teacher.endpoint is not None
    assert list(agent.endpoint.proxy.lanes) == ["agent", "teachers-0", "teachers-1"], (
        "the agent's lane, then a lane for each other model on Tinker"
    )
    document = teacher.endpoint.document
    assert (document["name"], document["model"]) == ("teachers-0", "Qwen/Qwen3-32B")
    assert "weights" not in document, "its base model, until the run points it"
    assert [(one.role, one.load) for one in connected.others] == [
        ("teachers[0]", "Qwen/Qwen3-32B"),
        ("teachers[1]", "tinker://r/sampler_weights/final"),
    ], "each model on Tinker opens a session of its own"
    with pytest.raises(RuntimeError, match="already open"):
        teacher.training = True


async def test_a_providers_agent_is_rolled_out_by_its_harness_and_never_trained(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    trials = harbor(FakeTrials(default={"reward": 1.0, "tokens": (100, 40, 10)}))
    claude = {'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"'}
    home = blueprint(tmp_path, **claude, **ONE_STEP)
    with pytest.raises(ValueError, match="anthropic serves it, .* `self.policy.training = False`"):
        await train(home, tmp_path / "runs", Trainer)
    assert service.calls == [] and proxy.placed == [] and trials.configs == [], "nothing opened"

    class Measured(Trainer):
        def __init__(self, config: Blueprint, directory: Path) -> None:
            super().__init__(config, directory)
            self.policy.training = False

        async def step(self) -> None:
            groups = await self.rollout(self.draw())
            with pytest.raises(ValueError, match="no weights to train"):
                await self.update(groups)

    directory = await train(home, tmp_path / "runs", Measured)
    assert service.calls == [] and proxy.placed == [], "no Tinker session and no proxy"
    named = [one.agent.model_name for one in trials.configs]
    assert named == ["anthropic/claude-sonnet-5-5"] * 4, "the harness calls the provider"
    [job] = list(files.read(directory / files.JOBS))
    assert job["backend"] == "provider" and job["graded"] == 4
    spent = files.read_json(directory / files.COSTS)["tokens"]
    assert spent == {
        "anthropic": {"claude-sonnet-5-5": {"prefill": 240, "cache": 160, "sample": 40}}
    }
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [(c["tag"], c["weights"]) for c in saved] == [("step-1", {}), ("final", {})], (
        "no weights to save, but a state to go on from"
    )
    (row,) = rows(directory)
    assert row["trained"] is False

    class Taught(Measured):
        def __init__(self, config: Blueprint, directory: Path) -> None:
            super().__init__(config, directory)
            self.teacher = Model("Qwen/Qwen3-32B")

    await train(home, tmp_path / "runs", Taught)
    assert service.calls == [("sampling", "Qwen/Qwen3-32B", None)], "its own session, alone"


def test_the_agent_takes_a_harness_or_several_and_mixing_them_needs_a_share_for_each(
    tmp_path: Path,
) -> None:
    pi = 'harness = "pi@0.85.1"'
    assert load(blueprint(tmp_path)).agent.harness == ["pi@0.85.1"], "read as a list"
    two = load(blueprint(tmp_path, **{pi: 'harness = ["pi@0.85.1", "claude-code@2.1.0"]'}))
    assert two.agent.harness == ["pi@0.85.1", "claude-code@2.1.0"] and not two.agent.mix_harnesses
    with pytest.raises(ConfigError, match=r"^\[agent\] harness: each harness is named once$"):
        load(blueprint(tmp_path, **{pi: 'harness = ["pi", "pi"]'}))
    with pytest.raises(ConfigError, match=r"^\[agent\] harness: a harness or a list of them$"):
        load(blueprint(tmp_path, **{pi: "harness = []"}))
    three = 'harness = ["pi", "claude-code", "opencode"]\nmix_harnesses = true'
    with pytest.raises(ConfigError) as refused:
        load(blueprint(tmp_path, **{pi: three}))
    assert refused.value.problems == [
        "[data] group_size: 2 is fewer than the 3 harnesses [agent] mix_harnesses splits it across"
    ]


def test_split_runs_each_task_on_every_harness_and_mixing_shares_a_groups_attempts() -> None:
    a, b = Path("a"), Path("b")
    two, none = ["pi", "cc"], [None]
    assert split([a, b], 4, two, none, mix=False) == [
        ("pi", None, [a, b], 4),
        ("cc", None, [a, b], 4),
    ], "a group of four per harness"
    assert split([a, b], 5, two, none, mix=True) == [
        ("pi", None, [a, b], 3),
        ("cc", None, [a, b], 2),
    ], "one group of five, shared"
    assert split([a], 1, two, none, mix=True) == [("pi", None, [a], 1)], "one attempt: the first"
    assert split([], 4, two, none, mix=False) == [], "no tasks, no jobs"
    alone = [("pi", None, [a, b], 4)]
    assert split([a, b], 4, ["pi"], none, mix=False) == split([a, b], 4, ["pi"], none, mix=True)
    assert split([a, b], 4, ["pi"], none, mix=True) == alone


def test_a_recipe_folder_runs_with_the_trainer_beside_its_run_toml(tmp_path: Path) -> None:
    home, runs = blueprint(tmp_path), tmp_path / "runs"
    plain, _ = open_run(home, root=runs)
    assert trainer_class(plain) is Trainer, "no trainer.py beside it: the base trainer"
    (home / "trainer.py").write_bytes((RECIPES / "dapo" / "trainer.py").read_bytes())
    with pytest.raises(ConfigError, match=r"\[recipe\] clip_low: field required"):
        open_run(home, root=runs)
    assert [one.name for one in runs.iterdir()] == [plain.name], "a refused blueprint opens no run"
    beside, _ = open_run(blueprint(tmp_path, *DAPO), root=runs)
    assert trainer_class(beside).__name__ == "DAPOTrainer"
    taught = blueprint(tmp_path, 'teacher = "Qwen/Qwen3-32B"')
    named, _ = open_run(taught / "run.toml", root=runs, trainer=RECIPES / "opd" / "trainer.py")
    assert trainer_class(named).__name__ == "OPDTrainer", "a trainer named outright wins"


def _recipe(folder: Path, bonus: float) -> Path:
    """Write a recipe folder whose trainer imports a file beside it. Return the trainer."""
    folder.mkdir(parents=True)
    (folder / "helpers.py").write_text(f"BONUS = {bonus}\n")
    (folder / "trainer.py").write_text(
        "from shipyard import Trainer\n\nfrom .helpers import BONUS\n\n\n"
        "class Helped(Trainer):\n    bonus = BONUS\n"
    )
    return folder / "trainer.py"


def test_a_recipe_imports_the_files_beside_its_trainer_each_folder_its_own(
    tmp_path: Path,
) -> None:
    one, two = _recipe(tmp_path / "one", 0.25), _recipe(tmp_path / "two", 0.5)
    assert (trainer_in(one).bonus, trainer_in(two).bonus) == (0.25, 0.5)


def test_a_run_keeps_its_recipes_place_and_a_resume_runs_the_code_as_it_is(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _recipe(tmp_path / "recipe", 0.25)
    directory, _ = open_run(blueprint(tmp_path), root=tmp_path / "runs", trainer=trainer)
    assert not (directory / files.TRAINER).exists(), "the code stays in the recipe's folder"
    assert files.read_json(directory / files.RUN)["recipe"]["trainer"] == str(trainer.resolve())
    row = {"tag": "step-1", "step": 1, "updates": 1, "drawn": 1, "state_path": "tinker://s"}
    files.append(directory / files.CHECKPOINTS, row)
    reopened(directory)
    assert "changed since the run started" not in caplog.text
    (trainer.parent / "helpers.py").write_text("BONUS = 0.75\n")
    reopened(directory)
    assert "changed since the run started" in caplog.text, "said, and the run goes on"
    shutil.rmtree(trainer.parent)
    with pytest.raises(ValueError, match="which is gone"):
        reopened(directory)


@pytest.mark.parametrize(
    "recipe", sorted(one.name for one in RECIPES.iterdir() if (one / "run.toml").is_file())
)
def test_each_recipes_run_toml_sets_exactly_the_keys_its_trainer_declares(recipe: str) -> None:
    try:
        made = trainer_in(RECIPES / recipe / "trainer.py")
    except ModuleNotFoundError as missing:
        pytest.skip(f"{recipe} needs {missing.name}")
    load(RECIPES / recipe, made.Recipe)


class Signalling:
    """Fake trial runner that sends this process `signum`, as `kill`, a scheduler or a closed
    terminal would, and waits as a trial in flight does."""

    def __init__(self, signum: int) -> None:
        self.signum = signum

    async def __call__(self, config: Any) -> None:
        os.kill(os.getpid(), self.signum)
        await asyncio.sleep(5)


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_run_sent_sigterm_or_sighup_unwinds_closes_what_it_opened_and_records_the_stop(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor, name: str
) -> None:
    proxy, service = fakes
    signum = getattr(signal, name)
    harbor(Signalling(signum))
    before = signal.getsignal(signum)
    stopped = CliRunner().invoke(
        app, ["run", str(blueprint(tmp_path)), "--root", str(tmp_path / "r")]
    )
    assert stopped.exit_code == 130, stopped.output
    assert f"stopped: {name}" in stopped.output
    assert signal.getsignal(signum) is before, "the shell's handler is back"
    (directory,) = (tmp_path / "r").iterdir()
    note = files.read_json(directory / files.RUN)["process"]
    assert note["failed"] is True and note["error"] == f"stopped: {name}"
    assert proxy.closed == 1, "the proxy, and its tunnel, are stopped"
    assert service.closed and service.closed[-1][0] == "interrupted"


def test_the_cli_runs_a_blueprint_and_refuses_to_resume_with_no_checkpoint(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())
    runner = CliRunner()
    done = runner.invoke(
        app, ["run", str(blueprint(tmp_path, **ONE_STEP)), "--root", str(tmp_path / "r")]
    )
    assert done.exit_code == 0, done.output
    (directory,) = (tmp_path / "r").iterdir()
    assert files.read_json(directory / files.RUN)["process"]["failed"] is False
    empty = tmp_path / "r" / "empty"
    empty.mkdir()
    (empty / files.CONFIG).write_text((directory / files.CONFIG).read_text())
    refused = runner.invoke(app, ["resume", str(empty)])
    assert refused.exit_code == 1 and "has no checkpoint 'last'" in refused.output
