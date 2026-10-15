"""Test the procedure on fakes: an epoch, DAPO's accumulation, the anchor, the provider job."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.data import Generator, tasks
from shipyard.run import Run
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.training.client import TinkerClient
from shipyard.training.trainer import ROW, CISPOTrainer, DAPOTrainer, DrGRPOTrainer, Trainer
from shipyard.types import SERVE_TTL, Agent, Policy, harness
from tests.records import FakeProxy
from tests.trainers import FakeService
from tests.trials import FIXTURES, FakeTrials, write_result

BLUEPRINTS = Path(__file__).parent / "blueprints"
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
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        write_result(Path(config.trials_dir) / config.trial_name, reward=float(self.seen[task] % 2))


class Spread:
    """A task in `spread` scores 1, 0, 1, 0 in call order; any other scores 1."""

    def __init__(self, spread: set[str]) -> None:
        self.spread = spread
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        reward = float(self.seen[task] % 2) if task in self.spread else 1.0
        write_result(Path(config.trials_dir) / config.trial_name, reward=reward)


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
    service = FakeService()
    service.client.trained = -0.3
    return proxy, service


def _agent(run: Run, service: FakeService | None) -> Agent:
    client = None if service is None else TinkerClient(service.client, service=service)
    return Agent(Policy(run.endpoint, client), [harness(run.config.rollout.harness)])


def _plan(tmp_path: Path) -> list[list[str]]:
    planned = Generator(tasks("four", root=tmp_path / "tasks"), 2, 0).batches(0)
    return [[task.name for task in batch] for batch in planned]


async def _train(run: Run, trainer: Trainer, agent: Agent) -> None:
    with run:
        async with agent.policy:
            await trainer.train(agent, Generator.of(run.config).batches(0))


def _rows(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.METRICS))


def _checkpoints(run: Run) -> list[dict[str, Any]]:
    return list(record.read(run.directory / record.CHECKPOINTS))


async def test_an_epoch_serves_the_start_samples_steps_syncs_and_checkpoints(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    run = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    run.run_trial = Alternating()
    agent = _agent(run, service)
    trainer = DAPOTrainer(run)
    await _train(run, trainer, agent)

    client = service.client
    assert (trainer.steps, trainer.updates) == (2, 2)
    # The start served before the first job, then the weights after each update.
    published = [
        c for c in client.calls if c[0] == "save_weights_for_sampler" and c[2] == SERVE_TTL
    ]
    assert published == [
        ("save_weights_for_sampler", "start", SERVE_TTL),
        ("save_weights_for_sampler", "sample-1", SERVE_TTL),
        ("save_weights_for_sampler", "sample-2", SERVE_TTL),
    ]
    assert proxy.swapped == [START] + [f"tinker://fake/sampler_weights/sample-{n}" for n in (1, 2)]
    assert proxy.closed == 1 and service.closed == [("success", None)]
    # One reference pass per step, then dapo's substeps capped at the step's two groups,
    # each part submitted before the one before is read.
    stepped = [c[0] for c in client.calls if c[0] not in ("save_weights_for_sampler", "save_state")]
    assert (
        stepped
        == [
            "forward",
            "forward_backward",
            "optim_step",
            "forward_backward",
            "optim_step",
            "consumed",
            "consumed",
            "consumed",
            "consumed",
        ]
        * 2
    )
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert all(
        c[2:] == ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}) for c in sent
    )
    assert all(c[1] == [INPUT_TOKENS] * 2 for c in sent)
    rows = _rows(run)
    jobs = list(record.read(run.directory / record.JOBS))
    assert [row["step"] for row in rows] == [0, 1]
    assert [row["jobs"] for row in rows] == [[jobs[0]["job"]], [jobs[1]["job"]]]
    for row in rows:
        assert row["trained"] is True
        assert (row["groups"], row["rollouts"], row["graded"], row["masked"]) == (2, 4, 4, 0)
        assert (row["degenerate"], row["sequences"], row["sequences_per_rollout"]) == (0, 4, 1.0)
        assert row["train_tokens"] == 4 * INPUT_TOKENS
        assert (row["reward_mean"], row["reward_spread"], row["length_mean"]) == (0.5, 0.5, 2.0)
        assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5)
        assert row["kl_v2"] == pytest.approx(0.02, abs=1e-5)
        assert row["entropy"] == pytest.approx(0.5, abs=1e-5)
        assert row["learning_rate"] == pytest.approx(2e-5), "no warm-up by default"
        assert (row["substeps"], row["loss_fn"]) == (2, "ppo")
        assert row["seconds"] >= 0.0 and "anchor_kl" not in row and "overlong" not in row
        keys = [key for key in row if key not in ("at", "seq")]
        assert keys == [key for key in ROW if key not in ("anchor_kl", "overlong")]
    # `every = 1`: a checkpoint per update at the TTL, written down with the position.
    saved = _checkpoints(run)
    assert [(c["tag"], c["step"], c["updates"], c["epoch"]) for c in saved] == [
        ("step-1", 1, 1, 0),
        ("step-2", 2, 2, 0),
    ]
    for row in saved:
        assert row["state_path"] == f"tinker://fake/weights/{row['tag']}"
        assert row["sampler_path"] == f"tinker://fake/sampler_weights/{row['tag']}"
        assert row["ttl_hours"] == 168.0 and row["modules"] is None and row["harness"] == ["pi"]
    assert [c for c in client.calls if c[0] == "save_state"] == [
        ("save_state", "step-1", 604800),
        ("save_state", "step-2", 604800),
    ]
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["train_tokens"] == costs["reference_tokens"] == 2 * 4 * INPUT_TOKENS
    assert "anchor_tokens" not in costs and costs["trials"] == 8
    assert [(job["batch"], job["purpose"], job["served"]) for job in jobs] == [
        (0, "rollout", "tinker"),
        (1, "rollout", "tinker"),
    ]
    assert Run.read(run.directory)["proxy"]["placement"] == "local"


async def test_a_step_with_every_group_flat_logs_untrained_and_takes_no_gradient(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    run = Run.open(_blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4"}), root=tmp_path)
    run.run_trial = FakeTrials()
    trainer = DAPOTrainer(run)
    await _train(run, trainer, _agent(run, service))
    calls = [c[0] for c in service.client.calls]
    assert calls == ["save_weights_for_sampler"], "the start, and no step"
    (row,) = _rows(run)
    assert row["trained"] is False and row["step"] == 0
    assert (row["groups"], row["rollouts"], row["graded"], row["degenerate"]) == (4, 8, 8, 4)
    assert row["sequences"] == 0 and row["reward_mean"] == 1.0 and row["reward_spread"] == 0.0
    assert "train_tokens" not in row and "learning_rate" not in row and "kl_v1" not in row
    assert "sequences_per_rollout" not in row, "none credited: nothing to average"
    assert _checkpoints(run) == [] and proxy.swapped == [START]
    assert (trainer.steps, trainer.updates) == (1, 0)
    assert "train_tokens" not in record.read_json(run.directory / record.COSTS)["parties"]["tinker"]


async def test_dapo_holds_jobs_until_a_batchs_worth_of_groups_have_spread(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    first, second = _plan(tmp_path)
    run.run_trial = Spread({first[0], *second})
    trainer = DAPOTrainer(run)
    await _train(run, trainer, _agent(run, service))
    (row,) = _rows(run)
    jobs = list(record.read(run.directory / record.JOBS))
    assert row["step"] == 0 and row["trained"] is True
    assert row["jobs"] == [job["job"] for job in jobs] and len(jobs) == 2
    assert (row["groups"], row["rollouts"], row["degenerate"]) == (4, 8, 1)
    assert (row["sequences"], row["substeps"], row["train_tokens"]) == (6, 3, 6 * INPUT_TOKENS)
    assert [(job["batch"], job["trials"]) for job in jobs] == [(0, 4), (0, 4)]
    (forward,) = [c for c in service.client.calls if c[0] == "forward"]
    assert forward[1] == [INPUT_TOKENS] * 6, "the flat group is never sent"
    assert [c for c in service.client.calls if c[0] == "save_weights_for_sampler"] == [
        ("save_weights_for_sampler", "start", SERVE_TTL),
        ("save_weights_for_sampler", "sample-1", SERVE_TTL),
        ("save_weights_for_sampler", "step-1", 604800),
    ], "one update, sampled at the start throughout"
    assert [c["tag"] for c in _checkpoints(run)] == ["step-1"]


async def test_the_epochs_last_step_takes_what_it_holds(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(_blueprint(tmp_path, tasks=6), root=tmp_path / "runs")
    run.run_trial = FakeTrials()
    trainer = DAPOTrainer(run)
    await _train(run, trainer, _agent(run, service))
    (row,) = _rows(run)
    assert (row["step"], row["trained"], row["groups"], row["degenerate"]) == (0, False, 6, 6)
    assert len(row["jobs"]) == 3 and trainer.steps == 1


async def test_dr_grpo_keeps_the_base_sample_and_steps_once_per_batch(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(_blueprint(tmp_path, "dr-grpo"), root=tmp_path / "runs")
    first, second = _plan(tmp_path)
    run.run_trial = Spread({first[0], *second})
    trainer = DrGRPOTrainer(run)
    await _train(run, trainer, _agent(run, service))
    rows = _rows(run)
    assert [(row["step"], row["groups"], row["degenerate"], row["trained"]) for row in rows] == [
        (0, 2, 1, True),
        (1, 2, 0, True),
    ]
    sent = [c for c in service.client.calls if c[0] == "forward_backward"]
    assert [c[2:] for c in sent] == [
        ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
    ] * 2
    assert [row["substeps"] for row in rows] == [1, 1]


async def test_the_base_trainer_is_the_plain_policy_gradient(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    run.run_trial = Alternating()
    await _train(run, Trainer(run), _agent(run, service))
    client = service.client
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert [c[2:] for c in sent] == [("importance_sampling", {})] * 4
    # Rewards 1 and 0 in each group: the reward less the mean, not divided by the spread,
    # at the acted positions; the prompt's are 0.
    trained = [d for d in client.sent if "advantages" in d.loss_fn_inputs]
    assert len(trained) == 8
    assert {a for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a} == {0.5, -0.5}
    (optimizer,) = {c[1].model_dump_json() for c in client.calls if c[0] == "optim_step"}
    assert json.loads(optimizer) == {
        "learning_rate": 2e-5,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
    }
    assert [row["loss_fn"] for row in _rows(run)] == ["importance_sampling"] * 2


async def test_cispo_reads_the_samplers_logprobs_and_makes_no_forward(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(
        _blueprint(tmp_path, "cispo", **{"batch_size = 2": "batch_size = 4"}), root=tmp_path
    )
    run.run_trial = Alternating()
    await _train(run, CISPOTrainer(run), _agent(run, service))
    client = service.client
    assert not any(c[0] == "forward" for c in client.calls)
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert len(sent) == 4, "16 substeps, capped at the batch's four groups"
    assert all(
        c[2:] == ("cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}) for c in sent
    )
    (row,) = _rows(run)
    assert row["loss_fn"] == "cispo" and row["train_tokens"] == 24 and row["substeps"] == 4
    assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5), "mu is the record's -0.5"
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["train_tokens"] == 24 and "reference_tokens" not in costs


async def test_the_kl_anchor_is_subtracted_per_token_billed_and_reported(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4", **ANCHORED})
    run = Run.open(home, root=tmp_path)
    run.run_trial = Alternating()
    trainer = DAPOTrainer(run)
    trainer.anchor = service.anchor
    await _train(run, trainer, _agent(run, service))
    assert len(service.anchor.asked) == 8
    (row,) = _rows(run)
    # mu is the forward's -0.5 and the anchor says -0.1: the gap is -0.4 on every token.
    assert row["anchor_kl"] == pytest.approx(-0.4, abs=1e-5)
    keys = list(row)
    assert keys.index("entropy") < keys.index("anchor_kl") < keys.index("learning_rate")
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["anchor_tokens"] == 8 * SEQUENCE_TOKENS


async def test_a_kl_term_without_an_anchor_is_refused_at_the_step(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    run = Run.open(_blueprint(tmp_path, **ANCHORED), root=tmp_path / "runs")
    run.run_trial = Alternating()
    with pytest.raises(ValueError, match="needs the starting weights to anchor to"):
        await _train(run, DAPOTrainer(run), _agent(run, service))
    assert service.closed == [
        ("errored", "ValueError: kl_coef = 0.1 needs the starting weights to anchor to")
    ]


async def test_the_overlong_term_docks_against_the_budget_or_warns_once_that_it_is_off(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], caplog: pytest.LogCaptureFixture
) -> None:
    proxy, service = fakes
    home = _blueprint(tmp_path)
    run = Run.open(home, root=tmp_path / "runs")
    run.run_trial = Alternating()
    with caplog.at_level(logging.WARNING):
        await _train(run, DAPOTrainer(run), _agent(run, service))
    rows = _rows(run)
    assert len(rows) == 2 and all("overlong" not in row for row in rows)
    warned = [one.getMessage() for one in caplog.records if "overlong term" in one.getMessage()]
    assert warned == [
        "dapo's overlong term is off for this run: the proxy reports no token budget, so "
        "overlong_penalty = 0.5 docks no rollout"
    ]
    # Every fake trial samples two tokens: with a budget of two, each is docked in full.
    proxy.budget = 2
    caplog.clear()
    run = Run.open(home, root=tmp_path / "runs")
    run.run_trial = Alternating()
    with caplog.at_level(logging.WARNING):
        await _train(run, DAPOTrainer(run), _agent(run, service))
    assert [row["overlong"] for row in _rows(run)] == [4, 4]
    assert all(row["trained"] for row in _rows(run)), "1 - 0.5 against 0 - 0.5 still has spread"
    assert not [one for one in caplog.records if "overlong term" in one.getMessage()]


async def test_a_job_on_a_provider_agent_groups_the_verdicts_and_the_step_takes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _blueprint(tmp_path, **{"lora_rank = 32": 'lora_rank = 32\nprovider = "openrouter"'})
    run = Run.open(home, root=tmp_path / "runs")
    run.run_trial = Alternating()
    trainer = DAPOTrainer(run)
    agent = Agent(Policy(run.endpoint), [harness("pi@0.85.1"), harness("claude-code")])
    assert isinstance(run.endpoint, ProviderEndpoint)
    tasks = [tmp_path / "tasks" / "four" / name for name in ("a", "b")]
    with run:
        async with agent.policy:
            found = await trainer.job(agent, tasks, purpose="evaluation")
            assert await trainer.step(agent, found) is None
    assert [(group.task.name, group.harness.name) for group in found] == [
        ("a", "pi"),
        ("a", "claude-code"),
        ("b", "pi"),
        ("b", "claude-code"),
    ]
    for group in found:
        assert [one.verdict.reward for one in group.members] == [1.0, 0.0]
        assert all(one.sequences == () for one in group.members)
        assert group.shaped == [1.0, 0.0]
        assert group.advantages == pytest.approx([1.0, -1.0], abs=1e-5)
        assert group.reward_variance == 0.25
    (job,) = record.read(run.directory / record.JOBS)
    assert (job["purpose"], job["served"], job["trials"], job["tasks"]) == (
        "evaluation",
        "provider",
        8,
        2,
    )
    assert _rows(run) == []
