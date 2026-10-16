"""Test the runner on fakes: a dapo run, two epochs, the anchor, each kind's class, the refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipyard import record
from shipyard.data import Generator, tasks
from shipyard.run import Run
from shipyard.runner import Runner, runner_in
from shipyard.training.trainer import ROW, Trainer
from shipyard.types import SERVE_TTL
from tests.proxies import MODEL
from tests.records import FakeProxy
from tests.trainers import FakeService
from tests.trials import (
    ANCHORED,
    INPUT_TOKENS,
    SEQUENCE_TOKENS,
    START,
    Alternating,
    Harbor,
    Interrupting,
    Stopped,
    blueprint,
    checkpoints,
    dataset,
    rows,
)


async def _run(home: Path, tmp_path: Path) -> tuple[Run, Runner]:
    opened = Run.open(home, root=tmp_path / "runs")
    started = Runner.from_run(opened)
    with opened:
        await started.run()
    return opened, started


async def test_a_two_step_dapo_run_opens_a_client_serves_samples_steps_andcheckpoints(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    client = service.client
    harbor(Alternating())
    opened, started = await _run(blueprint(tmp_path), tmp_path)
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
    logged = rows(opened)
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [row["step"] for row in logged] == [0, 1] and [row["seq"] for row in logged] == [1, 2]
    assert [row["jobs"] for row in logged] == [[jobs[0]["job"]], [jobs[1]["job"]]]
    for row in logged:
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
    saved = checkpoints(opened)
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
    assert [(job["step"], job["purpose"], job["graded"]) for job in jobs] == [
        (0, "training", 4),
        (1, "training", 4),
    ]
    note = Run.read(opened.directory)
    assert note["failed"] is False and note["kind"] == "dapo"


async def test_each_epoch_is_one_pass_over_the_train_set_and_the_steps_count_through(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    home = blueprint(tmp_path, **{"batch_size = 2": "batch_size = 2\nepochs = 2"})
    trials = Alternating()
    harbor(trials)
    opened, started = await _run(home, tmp_path)
    made = Generator(tasks("four", root=tmp_path / "tasks"), 2, 0)
    planned = [*made.batches(0), *made.batches(1)]
    ran = [Path(str(c.task.path)).name for c in trials.configs]
    assert [list(dict.fromkeys(ran[at : at + 4])) for at in range(0, len(ran), 4)] == [
        [task.name for task in batch] for batch in planned
    ]
    assert [row["step"] for row in rows(opened)] == [0, 1, 2, 3]
    assert [(c["tag"], c["step"], c["epoch"]) for c in checkpoints(opened)] == [
        ("step-1", 1, 0),
        ("step-2", 2, 0),
        ("step-3", 3, 1),
        ("step-4", 4, 1),
        ("final", 4, 1),
    ]
    assert [job["step"] for job in record.read(opened.directory / record.JOBS)] == [0, 1, 2, 3]


async def test_a_run_that_names_a_test_set_evaluates_on_the_cadence(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    dataset(tmp_path / "tasks", "two", "x", "y")
    home = blueprint(
        tmp_path,
        **{"batch_size = 2": 'batch_size = 2\ntest = "two"\nevaluate_every = 1\nepochs = 2'},
    )
    harbor(Alternating())
    opened, started = await _run(home, tmp_path)
    logged = rows(opened)
    assert ["evaluation" if row.get("evaluation") else "step" for row in logged] == [
        "step",
        "step",
        "evaluation",
        "step",
        "step",
        "evaluation",
    ]
    evaluations = [row for row in logged if row.get("evaluation")]
    assert [
        (
            row["step"],
            row["epoch"],
            row["tasks"],
            row["rollouts"],
            row["graded"],
            row["reward_mean"],
        )
        for row in evaluations
    ] == [(2, 0, 2, 4, 4, 0.5), (4, 1, 2, 4, 4, 0.5)]
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["purpose"] for job in jobs] == ["training", "training", "evaluation"] * 2
    assert [job["step"] for job in jobs] == [0, 1, 2, 2, 3, 4]
    assert [c["tag"] for c in checkpoints(opened)][-1] == "final"


async def test_a_run_that_names_a_test_set_ends_with_an_evaluation(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    dataset(tmp_path / "tasks", "two", "x", "y")
    home = blueprint(
        tmp_path, **{"batch_size = 2": 'batch_size = 2\ntest = "two"\ntest_group_size = 1'}
    )
    harbor(Alternating())
    opened, started = await _run(home, tmp_path)
    *steps, evaluation = rows(opened)
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
    harbor: Harbor,
) -> None:
    _, service = fakes
    home = blueprint(tmp_path, kind, **{"batch_size = 2": "batch_size = 4"})
    harbor(Alternating())
    opened, started = await _run(home, tmp_path)
    sent = [c for c in service.client.calls if c[0] == "forward_backward"]
    assert sent and all(c[2:] == (loss, bounds) for c in sent)
    assert [row["loss_fn"] for row in rows(opened)] == [loss]


async def test_a_trainer_class_of_ones_own_replaces_the_kinds(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    opened = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    harbor(Alternating())
    started = Runner.from_run(opened, Trainer)
    with opened:
        agent = await started.run()
    assert agent.policy.client is not None and type(started.trainer) is Trainer
    assert [row["loss_fn"] for row in rows(opened)] == ["importance_sampling"] * 2


async def test_a_kl_anchor_is_a_sampling_client_on_the_starting_weights(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    home = blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4", **ANCHORED})
    harbor(Alternating())
    opened, started = await _run(home, tmp_path)
    assert service.calls == [("lora", MODEL, 32), ("sampling", MODEL, None)]
    assert len(service.anchor.asked) == 8
    (row,) = rows(opened)
    # mu is the forward's -0.5 and the anchor says -0.1: the gap is -0.4 on every token.
    assert row["anchor_kl"] == pytest.approx(-0.4, abs=1e-5)
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["anchor_tokens"] == 8 * SEQUENCE_TOKENS


async def test_a_kl_anchor_from_a_checkpoint_samples_weights_published_from_its_state(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    state = "tinker://run:train:0/weights/final"
    home = blueprint(
        tmp_path,
        **{
            "batch_size = 2": "batch_size = 4",
            "[model]\n": f'[model]\nfrom_checkpoint = "{state}"\n',
            **ANCHORED,
        },
    )
    harbor(Alternating())
    await _run(home, tmp_path)
    published = "tinker://fake/sampler_weights/anchor"
    assert service.calls == [("from_state", state), ("sampling", MODEL, published)]
    assert ("save_weights_for_sampler", "anchor", SERVE_TTL) in service.client.calls
    assert proxy.swapped[0] == START


async def test_a_resumed_run_anchors_kl_to_its_start_before_loading_the_checkpoint(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    state = "tinker://run:train:0/weights/final"
    home = blueprint(
        tmp_path, **{"[model]\n": f'[model]\nfrom_checkpoint = "{state}"\n', **ANCHORED}
    )
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Interrupting(after=5))
    with pytest.raises(Stopped), opened:
        await Runner.from_run(opened).run()
    resumed = Runner.from_checkpoint(opened.directory)
    harbor(Alternating())
    with resumed.trainer.run:
        await resumed.run()
    published = "tinker://fake/sampler_weights/anchor"
    assert service.calls[-2:] == [("from_state", state), ("sampling", MODEL, published)]
    calls = service.client.calls
    published_anchor = ("save_weights_for_sampler", "anchor", SERVE_TTL)
    anchored = len(calls) - 1 - calls[::-1].index(published_anchor)
    loaded = calls.index(("load_state_with_optimizer", "tinker://fake/weights/step-1"))
    assert anchored < loaded, "the anchor is the blueprint's start, not the resume point"


async def test_a_resumed_run_continues_its_warm_up(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    home = blueprint(tmp_path, **{"learning_rate = 2e-5": "learning_rate = 2e-5\nwarmup = 20"})
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Interrupting(after=5))
    with pytest.raises(Stopped), opened:
        await Runner.from_run(opened).run()
    resumed = Runner.from_checkpoint(opened.directory)
    harbor(Alternating())
    with resumed.trainer.run:
        await resumed.run()
    rates = [row["learning_rate"] for row in rows(resumed.trainer.run)]
    assert rates == pytest.approx([2e-5 / 20, 2e-5 * 2 / 20]), "the second update of twenty"


async def test_a_run_continuing_an_optimizer_state_skips_the_warm_up(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    state = "tinker://run:train:0/weights/final"
    home = blueprint(
        tmp_path,
        **{
            "[model]\n": f'[model]\nfrom_checkpoint = "{state}"\nrestore_optimizer = true\n',
            "learning_rate = 2e-5": "learning_rate = 2e-5\nwarmup = 20",
        },
    )
    harbor(Alternating())
    opened, _ = await _run(home, tmp_path)
    assert [row["learning_rate"] for row in rows(opened)] == pytest.approx([2e-5, 2e-5])


async def test_a_step_that_fails_closes_the_session_as_errored_and_the_run_as_failed(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    service.client.failing = RuntimeError("the backend went away")
    home = blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4"})
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Alternating())
    with pytest.raises(RuntimeError, match="went away"), opened:
        await Runner.from_run(opened).run()
    assert service.closed == [("errored", "RuntimeError: the backend went away")]
    assert rows(opened) == [] and checkpoints(opened) == [] and proxy.closed == 1
    note = Run.read(opened.directory)
    assert note["failed"] is True and note["error"] == "RuntimeError: the backend went away"


async def test_a_run_killed_mid_epoch_resumes_from_its_last_checkpoint(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    opened = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    harbor(Interrupting(after=5))  # the second batch's first trial
    started = Runner.from_run(opened)
    with pytest.raises(Stopped), opened:
        await started.run()
    assert [c["tag"] for c in checkpoints(opened)] == ["step-1"]
    assert Run.read(opened.directory)["error"] == "Stopped: SIGINT"
    assert service.closed[-1][0] == "interrupted"

    resumed = Runner.from_checkpoint(opened.directory)
    row, consumed = resumed.checkpoint, resumed.consumed
    assert (row["tag"], consumed) == ("step-1", 1), "one batch of the epoch was stepped"
    trainer = resumed.trainer
    assert (trainer.steps, trainer.updates, trainer.epoch) == (1, 1, 0)
    again = resumed.trainer.run
    trials = harbor(Alternating())
    with again:
        await resumed.run()
    planned = list(Generator(tasks("four", root=tmp_path / "tasks"), 2, 0).batches(0))
    second = [task.name for task in planned[1]]
    assert list(dict.fromkeys(Path(str(c.task.path)).name for c in trials.configs)) == second
    logged = rows(again)
    assert [(r["step"], r["seq"], r["trained"]) for r in logged] == [(0, 1, True), (1, 2, True)]
    assert [(c["tag"], c["step"], c["updates"]) for c in checkpoints(again)] == [
        ("step-1", 1, 1),
        ("step-2", 2, 2),
        ("final", 2, 2),
    ]
    # The client opened as the blueprint says and loaded the checkpoint's state with its
    # optimizer; the proxy served its sampler weights before the first job, and the weights
    # after the update.
    assert service.calls[-1] == ("lora", MODEL, 32)
    assert ("load_state_with_optimizer", "tinker://fake/weights/step-1") in service.client.calls
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
    assert [(job["step"], job["epoch"]) for job in jobs] == [(0, 0), (1, 0)], (
        "the interrupted job was never recorded; the resumed step drew its batch again"
    )


async def test_a_run_is_evaluated_at_a_checkpoint_as_one_more_row(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    opened, _ = await _run(blueprint(tmp_path), tmp_path)
    dataset(tmp_path / "tasks", "two", "x", "y")
    reopened = Runner.from_checkpoint(opened.directory, tag="step-1")
    harbor(Alternating())
    calls = len(service.calls)
    with reopened.trainer.run:
        result = await reopened.evaluate(tasks("two", root=tmp_path / "tasks"))
    assert result is not None
    assert (result.step, result.epoch, result.tasks, result.rollouts) == (1, 0, 2, 4)
    assert result.reward_mean == 0.5
    logged = rows(reopened.trainer.run)
    assert logged[-1]["evaluation"] is True and logged[-1]["checkpoint"] == "step-1"
    assert logged[-1]["seq"] == 3, "numbered after the run's own rows"
    assert len(service.calls) == calls, "no training client was opened"
    assert proxy.swapped[-1] == "tinker://fake/sampler_weights/step-1"
    jobs = list(record.read(reopened.trainer.run.directory / record.JOBS))
    assert jobs[-1]["purpose"] == "evaluation"
    assert Run.read(reopened.trainer.run.directory)["resumed"] == "step-1"
    with pytest.raises(ValueError, match="no checkpoint tagged 'step-9'"):
        Runner.from_checkpoint(opened.directory, tag="step-9")


RUNNER_FILE = """\
from shipyard.runner import Runner
from shipyard.recipes.dapo import DAPOTrainer


class Marked(DAPOTrainer):
    def log(self, found, update, **more):
        super().log(found, update, marked=True, **more)


class MyRunner(Runner):
    trainer = Marked
"""


def _file(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_a_runner_file_defines_one_runner_subclass(tmp_path: Path) -> None:
    found = runner_in(_file(tmp_path / "one.py", RUNNER_FILE))
    assert found.__name__ == "MyRunner" and found.trainer.__name__ == "Marked"
    assert runner_in(tmp_path / "one.py") is found, "a path is loaded once"
    with pytest.raises(ValueError, match="defines 0 Runner subclasses"):
        runner_in(_file(tmp_path / "none.py", "x = 1\n"))
    two = RUNNER_FILE + "\n\nclass Other(Runner):\n    pass\n"
    with pytest.raises(ValueError, match="defines 2 Runner subclasses"):
        runner_in(_file(tmp_path / "two.py", two))
    with pytest.raises(ZeroDivisionError):
        runner_in(_file(tmp_path / "broken.py", "1 / 0\n"))
    with pytest.raises(ValueError, match="not a Python file"):
        runner_in(_file(tmp_path / "run.toml", ""))


async def test_a_run_under_a_runner_file_trains_under_its_trainer_and_resumes_under_the_copy(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    file = _file(tmp_path / "run.py", RUNNER_FILE)
    opened = Run.open(blueprint(tmp_path), root=tmp_path / "runs", runner=file)
    assert (opened.directory / record.RUNNER).read_bytes() == file.read_bytes()
    harbor(Alternating())
    started = runner_in(file).from_run(opened)
    with opened:
        await started.run()
    assert type(started).__name__ == "MyRunner" and type(started.trainer).__name__ == "Marked"
    assert rows(opened) and all(row["marked"] for row in rows(opened))
    file.unlink()
    resumed = Runner.from_checkpoint(opened.directory)
    assert type(resumed).__name__ == "MyRunner" and type(resumed.trainer).__name__ == "Marked"
    assert resumed.checkpoint is not None and resumed.checkpoint["tag"] == "final"
