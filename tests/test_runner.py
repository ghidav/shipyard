"""Test the runner on fakes: a dapo run, two epochs, the anchor, each kind's class, the refusals."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record, session
from shipyard.data import Generator, tasks
from shipyard.recipes import run_recipe, trainer_for
from shipyard.run import Run
from shipyard.runner import Runner
from shipyard.training.trainer import ROW, Trainer
from shipyard.types import SERVE_TTL
from tests.records import FakeProxy
from tests.trainers import FakeService, capable
from tests.trials import FIXTURES, write_result

BLUEPRINTS = Path(__file__).parent / "blueprints"
MODEL = "Qwen/Qwen3-8B"
#: The fake proxy's one record per trial: prompt "go" + completion "ok" is four tokens,
#: of which the model reads three.
SEQUENCE_TOKENS = 4
INPUT_TOKENS = 3
START = "tinker://fake/sampler_weights/start"
#: A blueprint replacement that gives the recipe a KL term.
ANCHORED = {"learning_rate = 2e-5": "learning_rate = 2e-5\nkl_coef = 0.1"}


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


def _blueprint(tmp_path: Path, kind: str = "dapo", *, tasks: int = 4, **replacements: str) -> Path:
    """The recipe's fixture over `tasks` tasks named a, b, c..., two per batch, groups of two."""
    home = tmp_path / "tasks" / "four"
    if not home.is_dir():
        for name in "abcdefgh"[:tasks]:
            shutil.copytree(FIXTURES / "fixture" / "alpha", home / name)
    text = (BLUEPRINTS / kind / "run.toml").read_text(encoding="utf-8")
    text = text.replace('"aime-train"', '"four"').replace("group_size = 4", "group_size = 2")
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp" / kind
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProxy, FakeService]:
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True)))
    service.client.trained = -0.3
    monkeypatch.setattr(session, "service_client", lambda metadata=None: service)
    return proxy, service


async def _run(home: Path, tmp_path: Path, runner: Any) -> tuple[Run, Runner]:
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = runner
    started = Runner.of(opened, trainer_for(opened.config.recipe))
    with opened:
        await started.run()
    return opened, started


def _two(tmp_path: Path) -> None:
    for name in ("x", "y"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "two" / name)


def _rows(opened: Run) -> list[dict[str, Any]]:
    return list(record.read(opened.directory / record.METRICS))


def _checkpoints(opened: Run) -> list[dict[str, Any]]:
    return list(record.read(opened.directory / record.CHECKPOINTS))


async def test_a_two_step_dapo_run_opens_a_client_serves_samples_steps_and_checkpoints(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    client = service.client
    opened, started = await _run(_blueprint(tmp_path), tmp_path, Alternating())
    # The client: a LoRA at the blueprint's rank, named after the run, no anchor, the
    # session finished.
    assert service.calls == [("lora", MODEL, 32)]
    assert service.metadata == [{"shipyard_run": opened.id, "shipyard_recipe": "dapo"}]
    assert service.closed == [("success", None)]
    trainer = started.trainer
    assert (trainer.steps, trainer.updates, trainer.epoch) == (2, 2, 0)
    # The start served before the first job, then the weights after each update.
    published = [
        c for c in client.calls if c[0] == "save_weights_for_sampler" and c[2] == SERVE_TTL
    ]
    assert published == [
        ("save_weights_for_sampler", name, SERVE_TTL) for name in ("start", "sample-1", "sample-2")
    ]
    assert proxy.swapped == [START] + [f"tinker://fake/sampler_weights/sample-{n}" for n in (1, 2)]
    assert proxy.placed[0][0] == "local" and proxy.closed == 1
    # One reference pass per step, then dapo's 16 substeps capped at the step's two groups.
    assert [
        c[0] for c in client.calls if c[0] in ("forward", "forward_backward", "optim_step")
    ] == ["forward", "forward_backward", "optim_step", "forward_backward", "optim_step"] * 2
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert all(
        c[2:] == ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}) for c in sent
    )
    rows = _rows(opened)
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [row["step"] for row in rows] == [0, 1] and [row["seq"] for row in rows] == [1, 2]
    assert [row["jobs"] for row in rows] == [[jobs[0]["job"]], [jobs[1]["job"]]]
    for row in rows:
        assert row["trained"] is True
        assert (row["groups"], row["rollouts"], row["graded"], row["masked"]) == (2, 4, 4, 0)
        assert (row["degenerate"], row["sequences"], row["sequences_per_rollout"]) == (0, 4, 1.0)
        assert row["train_tokens"] == 4 * INPUT_TOKENS
        assert (row["reward_mean"], row["reward_spread"], row["length_mean"]) == (0.5, 0.5, 2.0)
        assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5)
        assert row["learning_rate"] == pytest.approx(2e-5), "no warm-up by default"
        assert (row["substeps"], row["loss_fn"]) == (2, "ppo")
        keys = [key for key in row if key not in ("at", "seq")]
        assert keys == [key for key in ROW if key not in ("anchor_kl", "overlong")]
    # `every = 1`: a checkpoint per update at the TTL, then `final`, kept with none.
    saved = _checkpoints(opened)
    assert [(c["tag"], c["step"], c["updates"], c["epoch"], c["ttl_hours"]) for c in saved] == [
        ("step-1", 1, 1, 0, 168.0),
        ("step-2", 2, 2, 0, 168.0),
        ("final", 2, 2, 0, None),
    ]
    assert [c for c in client.calls if c[0] == "save_state"] == [
        ("save_state", "step-1", 604800),
        ("save_state", "step-2", 604800),
        ("save_state", "final", None),
    ]
    costs = record.read_json(opened.directory / record.COSTS)
    tinker = costs["parties"]["tinker"]
    assert tinker["train_tokens"] == tinker["reference_tokens"] == 2 * 4 * INPUT_TOKENS
    assert "anchor_tokens" not in tinker and tinker["trials"] == 8
    assert list(costs["parties"]) == ["tinker"]
    assert [(job["batch"], job["purpose"], job["graded"]) for job in jobs] == [
        (0, "rollout", 4),
        (1, "rollout", 4),
    ]
    note = Run.read(opened.directory)
    assert note["failed"] is False and note["kind"] == "dapo"


async def test_each_epoch_is_one_pass_over_the_train_set_and_the_steps_count_through(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    home = _blueprint(tmp_path, **{"batch_size = 2": "batch_size = 2\nepochs = 2"})
    trials = Alternating()
    opened, started = await _run(home, tmp_path, trials)
    made = Generator(tasks("four", root=tmp_path / "tasks"), 2, 0)
    planned = [*made.batches(0), *made.batches(1)]
    ran = [Path(str(c.task.path)).name for c in trials.configs]
    assert [list(dict.fromkeys(ran[at : at + 4])) for at in range(0, len(ran), 4)] == [
        [task.name for task in batch] for batch in planned
    ]
    assert [row["step"] for row in _rows(opened)] == [0, 1, 2, 3]
    assert [(c["tag"], c["step"], c["epoch"]) for c in _checkpoints(opened)] == [
        ("step-1", 1, 0),
        ("step-2", 2, 0),
        ("step-3", 3, 1),
        ("step-4", 4, 1),
        ("final", 4, 1),
    ]
    assert [job["batch"] for job in record.read(opened.directory / record.JOBS)] == [0, 1, 2, 3]


async def test_a_run_that_names_a_test_set_evaluates_on_the_cadence(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _two(tmp_path)
    home = _blueprint(
        tmp_path,
        **{"batch_size = 2": 'batch_size = 2\ntest = "two"\nevaluate_every = 1\nepochs = 2'},
    )
    opened, started = await _run(home, tmp_path, Alternating())
    rows = _rows(opened)
    assert ["evaluation" if row.get("evaluation") else "step" for row in rows] == [
        "step",
        "step",
        "evaluation",
        "step",
        "step",
        "evaluation",
    ]
    evaluations = [row for row in rows if row.get("evaluation")]
    assert [
        (row["step"], row["epoch"], row["tasks"], row["rollouts"], row["graded"], row["mean"])
        for row in evaluations
    ] == [(2, 0, 2, 4, 4, 0.5), (4, 1, 2, 4, 4, 0.5)]
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["purpose"] for job in jobs] == ["rollout", "rollout", "evaluation"] * 2
    assert [job["batch"] for job in jobs] == [0, 1, 2, 2, 3, 4]
    assert [c["tag"] for c in _checkpoints(opened)][-1] == "final"


async def test_a_run_that_names_a_test_set_ends_with_an_evaluation(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _two(tmp_path)
    home = _blueprint(
        tmp_path, **{"batch_size = 2": 'batch_size = 2\ntest = "two"\ntest_group_size = 1'}
    )
    opened, started = await _run(home, tmp_path, Alternating())
    *steps, evaluation = _rows(opened)
    assert len(steps) == 2 and not any(row.get("evaluation") for row in steps)
    assert (evaluation["step"], evaluation["epoch"], evaluation["rollouts"]) == (2, 0, 2)
    assert [job["purpose"] for job in record.read(opened.directory / record.JOBS)][
        -1
    ] == "evaluation"


@pytest.mark.parametrize(
    ("kind", "loss", "bounds"),
    [
        ("cispo", "cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}),
        ("dr-grpo", "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}),
    ],
)
async def test_each_kind_reaches_its_trainer_class(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    kind: str,
    loss: str,
    bounds: dict[str, float],
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, kind, **{"batch_size = 2": "batch_size = 4"})
    opened, started = await _run(home, tmp_path, Alternating())
    sent = [c for c in service.client.calls if c[0] == "forward_backward"]
    assert sent and all(c[2:] == (loss, bounds) for c in sent)
    assert [row["loss_fn"] for row in _rows(opened)] == [loss]


async def test_a_trainer_class_of_ones_own_replaces_the_kinds(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    opened.run_trial = Alternating()
    started = Runner.of(opened, Trainer)
    with opened:
        agent = await started.run()
    assert agent.policy.client is not None and type(started.trainer) is Trainer
    assert [row["loss_fn"] for row in _rows(opened)] == ["importance_sampling"] * 2


async def test_a_kl_anchor_is_a_sampling_client_on_the_starting_weights(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4", **ANCHORED})
    opened, started = await _run(home, tmp_path, Alternating())
    assert service.calls == [("lora", MODEL, 32), ("sampling", MODEL, None)]
    assert len(service.anchor.asked) == 8
    (row,) = _rows(opened)
    # mu is the forward's -0.5 and the anchor says -0.1: the gap is -0.4 on every token.
    assert row["anchor_kl"] == pytest.approx(-0.4, abs=1e-5)
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["anchor_tokens"] == 8 * SEQUENCE_TOKENS


async def test_a_kl_anchor_from_a_checkpoint_samples_weights_published_from_its_state(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    state = "tinker://run:train:0/weights/final"
    home = _blueprint(
        tmp_path,
        **{
            "batch_size = 2": "batch_size = 4",
            "[model]\n": f'[model]\nfrom_checkpoint = "{state}"\n',
            **ANCHORED,
        },
    )
    await _run(home, tmp_path, Alternating())
    published = "tinker://fake/sampler_weights/anchor"
    assert service.calls == [("from_state", state), ("sampling", MODEL, published)]
    assert ("save_weights_for_sampler", "anchor", SERVE_TTL) in service.client.calls
    assert proxy.swapped[0] == START


async def test_a_run_whose_model_is_not_served_is_refused_before_a_session_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, **{"lora_rank = 32": 'lora_rank = 32\nprovider = "openrouter"'})
    opened = Run.open(home, root=tmp_path / "runs")
    with pytest.raises(ValueError, match="trains the weights this run serves"), opened:
        await run_recipe(opened)
    assert service.calls == [] and service.closed == []
    assert Run.read(opened.directory)["failed"] is True


async def test_a_step_that_fails_closes_the_session_as_errored_and_the_run_as_failed(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    service.client.failing = RuntimeError("the backend went away")
    home = _blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4"})
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = Alternating()
    with pytest.raises(RuntimeError, match="went away"), opened:
        await run_recipe(opened)
    assert service.closed == [("errored", "RuntimeError: the backend went away")]
    assert _rows(opened) == [] and _checkpoints(opened) == [] and proxy.closed == 1
    note = Run.read(opened.directory)
    assert note["failed"] is True and note["error"] == "RuntimeError: the backend went away"


async def test_a_run_killed_mid_epoch_resumes_from_its_last_checkpoint(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    opened.run_trial = Interrupting(after=5)  # the second batch's first trial
    started = Runner.of(opened, trainer_for(opened.config.recipe))
    with pytest.raises(Stopped), opened:
        await started.run()
    assert [c["tag"] for c in _checkpoints(opened)] == ["step-1"]
    assert Run.read(opened.directory)["error"] == "Stopped: SIGINT"
    assert service.closed[-1][0] == "interrupted"

    resumed = Runner.from_run(opened.directory)
    row, consumed = resumed.position
    assert (row["tag"], consumed) == ("step-1", 1), "one batch of the epoch was stepped"
    trainer = resumed.trainer
    assert (trainer.steps, trainer.updates, trainer.epoch) == (1, 1, 0)
    again = resumed.opened
    again.run_trial = trials = Alternating()
    with again:
        await resumed.run()
    planned = list(Generator(tasks("four", root=tmp_path / "tasks"), 2, 0).batches(0))
    second = [task.name for task in planned[1]]
    assert list(dict.fromkeys(Path(str(c.task.path)).name for c in trials.configs)) == second
    rows = _rows(again)
    assert [(r["step"], r["seq"], r["trained"]) for r in rows] == [(0, 1, True), (1, 2, True)]
    assert [(c["tag"], c["step"], c["updates"]) for c in _checkpoints(again)] == [
        ("step-1", 1, 1),
        ("step-2", 2, 2),
        ("final", 2, 2),
    ]
    # The client came from the checkpoint's state with its optimizer, the proxy served its
    # sampler weights before the first job, and the weights after the update.
    assert service.calls[-1] == ("from_state_with_optimizer", "tinker://fake/weights/step-1")
    assert proxy.swapped == [
        START,
        "tinker://fake/sampler_weights/sample-1",
        "tinker://fake/sampler_weights/step-1",
        "tinker://fake/sampler_weights/sample-2",
    ]
    assert service.closed[-1] == ("success", None)
    note = Run.read(again.directory)
    assert note["resumed"] == "step-1" and note["failed"] is False and note["finished_at"]
    assert [n["error"] for n in note["earlier"]] == ["Stopped: SIGINT"]
    jobs = list(record.read(again.directory / record.JOBS))
    assert [(job["batch"], job["epoch"]) for job in jobs] == [(0, 0), (1, 0)], (
        "the interrupted job was never recorded; the resumed step drew its batch again"
    )


async def test_a_run_is_evaluated_at_a_checkpoint_as_one_more_row(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    opened, _ = await _run(_blueprint(tmp_path), tmp_path, Alternating())
    _two(tmp_path)
    reopened = Runner.from_run(opened.directory, checkpoint="step-1")
    reopened.opened.run_trial = Alternating()
    calls = len(service.calls)
    with reopened.opened:
        result = await reopened.evaluate(tasks("two", root=tmp_path / "tasks"))
    assert result is not None
    assert (result.step, result.epoch, result.tasks, result.rollouts) == (1, 0, 2, 4)
    assert result.mean == 0.5
    rows = _rows(reopened.opened)
    assert rows[-1]["evaluation"] is True and rows[-1]["checkpoint"] == "step-1"
    assert rows[-1]["seq"] == 3, "numbered after the run's own rows"
    assert len(service.calls) == calls, "no training client was opened"
    assert proxy.swapped[-1] == "tinker://fake/sampler_weights/step-1"
    jobs = list(record.read(reopened.opened.directory / record.JOBS))
    assert jobs[-1]["purpose"] == "evaluation"
    assert Run.read(reopened.opened.directory)["resumed"] == "step-1"
    with pytest.raises(ValueError, match="no checkpoint tagged 'step-9'"):
        Runner.from_run(opened.directory, checkpoint="step-9")
