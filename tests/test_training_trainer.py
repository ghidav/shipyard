"""Tests for trainer steps across recipes, the KL anchor, and jobs on a provider agent."""

from __future__ import annotations

import json
import logging
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.cli.run import Run
from shipyard.data import Generator, tasks
from shipyard.sampling.endpoint import ProviderEndpoint, endpoint_for
from shipyard.training.client import TinkerClient
from shipyard.training.credit import Signals
from shipyard.training.recipes.cispo import CISPOTrainer
from shipyard.training.recipes.dapo import DAPOTrainer
from shipyard.training.recipes.dr_grpo import DrGRPOTrainer
from shipyard.training.recipes.opd import OPDTrainer
from shipyard.training.step import together
from shipyard.training.trainer import ROW, Trainer
from shipyard.types import SERVE_TTL, Agent, Datum, Group, Member, Policy, Sequence, harness
from tests.proxies import MODEL
from tests.records import FakeProxy, made
from tests.trainers import CharTokenizer, FakeService, FlatAnchor, Positional
from tests.trials import (
    ANCHORED,
    INPUT_TOKENS,
    ONE_STEP,
    SEQUENCE_TOKENS,
    START,
    TAUGHT,
    UNTAUGHT,
    Alternating,
    FakeTrials,
    Harbor,
    blueprint,
    checkpoints,
    rows,
    write_result,
)


class Spread:
    """Fake Harbor trial runner: tasks in `spread` score 1, 0, 1, 0 in call order, others 1."""

    def __init__(self, spread: set[str]) -> None:
        self.spread = spread
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        reward = float(self.seen[task] % 2) if task in self.spread else 1.0
        write_result(Path(config.trials_dir) / config.trial_name, reward=reward)


def _agent(run: Run, service: FakeService | None, *, anchored: bool = False) -> Agent:
    client = None if service is None else TinkerClient(service, run.config.agent.model)
    policy = Policy(endpoint_for(run.config, run.directory), client, anchored=anchored)
    return Agent(policy, [harness(run.config.agent.harness)])


def _plan(tmp_path: Path) -> list[list[str]]:
    """Return the task names of the first pass over the four-task dataset, two per batch."""
    made = Generator(tasks("four", root=tmp_path / "tasks"), 2, 0)
    return [[task.name for task in batch] for batch in islice(made.batches(), made.per_pass)]


async def _train(run: Run, trainer: Trainer, agent: Agent) -> None:
    """Advance `trainer` through its `[recipe] steps` steps under `agent`, as `train` does."""
    with run:
        async with agent.policy:
            while trainer.steps < trainer.recipe.steps:
                await trainer.advance(agent)


async def test_the_steps_serve_the_start_sample_step_sync_andcheckpoint(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    run = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    harbor(Alternating())
    agent = _agent(run, service)
    trainer = DAPOTrainer(run.config, run.directory)
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
    logged = rows(run)
    jobs = list(record.read(run.directory / record.JOBS))
    assert [row["step"] for row in logged] == [0, 1]
    assert [row["jobs"] for row in logged] == [[jobs[0]["job"]], [jobs[1]["job"]]]
    for row in logged:
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
        assert keys == [key for key in ROW if key not in UNTAUGHT]
    # `every = 1`: a checkpoint per update at the TTL, written down with the position.
    saved = checkpoints(run)
    assert [(c["tag"], c["step"], c["updates"], c["drawn"]) for c in saved] == [
        ("step-1", 1, 1, 1),
        ("step-2", 2, 2, 2),
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
    assert [(job["step"], job["purpose"], job["served"]) for job in jobs] == [
        (0, "training", "tinker"),
        (1, "training", "tinker"),
    ]
    assert Run.read(run.directory)["proxy"]["placement"] == "local"


async def test_a_step_with_every_group_flat_logs_untrained_and_takes_no_gradient(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    run = Run.open(blueprint(tmp_path, **ONE_STEP), root=tmp_path)
    harbor(FakeTrials())
    trainer = DAPOTrainer(run.config, run.directory)
    await _train(run, trainer, _agent(run, service))
    calls = [c[0] for c in service.client.calls]
    assert calls == ["save_weights_for_sampler"], "the start, and no step"
    (row,) = rows(run)
    assert row["trained"] is False and row["step"] == 0
    assert (row["groups"], row["rollouts"], row["graded"], row["degenerate"]) == (4, 8, 8, 4)
    assert row["sequences"] == 0 and row["reward_mean"] == 1.0 and row["reward_spread"] == 0.0
    assert "train_tokens" not in row and "learning_rate" not in row and "kl_v1" not in row
    assert "sequences_per_rollout" not in row, "none credited: nothing to average"
    assert checkpoints(run) == [] and proxy.swapped == [START]
    assert (trainer.steps, trainer.updates) == (1, 0)
    assert "train_tokens" not in record.read_json(run.directory / record.COSTS)["parties"]["tinker"]


async def test_groups_a_sample_left_uncredited_are_refused_before_the_step(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    harbor(Alternating())

    class Uncredited(Trainer):
        async def sample(self, agent: Agent) -> list[Group]:
            return await self.job(agent, self.draw())

    with pytest.raises(ValueError, match="Uncredited.sample returned groups it never credited"):
        await _train(run, Uncredited(run.config, run.directory), _agent(run, service))
    assert [c[0] for c in service.client.calls] == ["save_weights_for_sampler"], "no step"


async def test_dapo_holds_jobs_until_a_batchs_worth_of_groups_have_spread(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, **{"steps = 2": "steps = 1"}), root=tmp_path / "runs")
    first, second = _plan(tmp_path)
    harbor(Spread({first[0], *second}))
    trainer = DAPOTrainer(run.config, run.directory)
    await _train(run, trainer, _agent(run, service))
    (row,) = rows(run)
    jobs = list(record.read(run.directory / record.JOBS))
    assert row["step"] == 0 and row["trained"] is True
    assert row["jobs"] == [job["job"] for job in jobs] and len(jobs) == 2
    assert (row["groups"], row["rollouts"], row["degenerate"]) == (4, 8, 1)
    assert (row["sequences"], row["substeps"], row["train_tokens"]) == (6, 3, 6 * INPUT_TOKENS)
    assert [(job["step"], job["trials"]) for job in jobs] == [(0, 4), (0, 4)]
    (forward,) = [c for c in service.client.calls if c[0] == "forward"]
    assert forward[1] == [INPUT_TOKENS] * 6, "the flat group is never sent"
    assert [c for c in service.client.calls if c[0] == "save_weights_for_sampler"] == [
        ("save_weights_for_sampler", "start", SERVE_TTL),
        ("save_weights_for_sampler", "sample-1", SERVE_TTL),
        ("save_weights_for_sampler", "step-1", 604800),
    ], "one update, sampled at the start throughout"
    assert [c["tag"] for c in checkpoints(run)] == ["step-1"]


async def test_a_dapo_step_draws_at_most_a_passs_worth_and_takes_what_it_holds(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    home = blueprint(tmp_path, tasks=6, **{"steps = 2": "steps = 1"})
    run = Run.open(home, root=tmp_path / "runs")
    harbor(FakeTrials())
    trainer = DAPOTrainer(run.config, run.directory)
    await _train(run, trainer, _agent(run, service))
    (row,) = rows(run)
    assert (row["step"], row["trained"], row["groups"], row["degenerate"]) == (0, False, 6, 6)
    assert len(row["jobs"]) == 3 and (trainer.steps, trainer.drawn) == (1, 3)


async def test_dr_grpo_keeps_the_base_sample_and_steps_once_per_batch(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, "dr-grpo"), root=tmp_path / "runs")
    first, second = _plan(tmp_path)
    harbor(Spread({first[0], *second}))
    trainer = DrGRPOTrainer(run.config, run.directory)
    await _train(run, trainer, _agent(run, service))
    logged = rows(run)
    assert [(row["step"], row["groups"], row["degenerate"], row["trained"]) for row in logged] == [
        (0, 2, 1, True),
        (1, 2, 0, True),
    ]
    sent = [c for c in service.client.calls if c[0] == "forward_backward"]
    assert [c[2:] for c in sent] == [
        ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
    ] * 2
    assert [row["substeps"] for row in logged] == [1, 1]


async def test_the_base_trainer_is_the_plain_policy_gradient(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    harbor(Alternating())
    await _train(run, Trainer(run.config, run.directory), _agent(run, service))
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
    assert [row["loss_fn"] for row in rows(run)] == ["importance_sampling"] * 2


async def test_cispo_reads_the_samplers_logprobs_and_makes_no_forward(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, "cispo", **ONE_STEP), root=tmp_path)
    harbor(Alternating())
    await _train(run, CISPOTrainer(run.config, run.directory), _agent(run, service))
    client = service.client
    assert not any(c[0] == "forward" for c in client.calls)
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert len(sent) == 4, "16 substeps, capped at the batch's four groups"
    assert all(
        c[2:] == ("cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}) for c in sent
    )
    (row,) = rows(run)
    assert row["loss_fn"] == "cispo" and row["train_tokens"] == 24 and row["substeps"] == 4
    assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5), "the reference is the record's -0.5"
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["train_tokens"] == 24 and "reference_tokens" not in costs


async def test_the_kl_anchor_is_subtracted_per_token_billed_and_reported(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    home = blueprint(tmp_path, **ONE_STEP, **ANCHORED)
    run = Run.open(home, root=tmp_path)
    harbor(Alternating())
    trainer = DAPOTrainer(run.config, run.directory)
    await _train(run, trainer, _agent(run, service, anchored=True))
    assert len(service.anchor.asked) == 8
    (row,) = rows(run)
    # The reference is the forward's -0.5 and the anchor's -0.1: the gap is -0.4 per token.
    assert row["anchor_kl"] == pytest.approx(-0.4, abs=1e-5)
    keys = list(row)
    assert keys.index("entropy") < keys.index("anchor_kl") < keys.index("learning_rate")
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["anchor_tokens"] == 8 * SEQUENCE_TOKENS


async def test_opd_trains_every_group_on_the_teachers_gap_alone(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    """Every reward is 1, so no group has a reward signal. The teacher's -0.2 against the
    reference's -0.5 puts 0.3 on every target."""
    _, service = fakes
    teacher = service.teachers["Qwen/Qwen3-32B"] = FlatAnchor(-0.2)
    run = Run.open(blueprint(tmp_path, "opd"), root=tmp_path)
    harbor(FakeTrials())
    await _train(run, OPDTrainer(run.config, run.directory), _agent(run, service))
    opened = [c for c in service.calls if c[0] == "sampling"]
    assert opened == [("sampling", "Qwen/Qwen3-32B", None)], "opened once, on the first step"
    assert len(teacher.asked) == 8, "two steps of four sequences"
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    found = [a for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a]
    assert len(trained) == 8 and found == pytest.approx([0.3] * len(found))
    for row in rows(run):
        assert row["trained"] is True and row["degenerate"] == 0 and row["reward_mean"] == 1.0
        assert row["teacher_kl"] == pytest.approx(-0.3, abs=1e-5) and "anchor_kl" not in row
        keys = list(row)
        assert keys.index("entropy") < keys.index("teacher_kl") < keys.index("learning_rate")
    costs = record.read_json(run.directory / record.COSTS)["parties"]["tinker"]
    assert costs["teacher_tokens"] == 8 * SEQUENCE_TOKENS and "anchor_tokens" not in costs


async def test_a_dapo_step_with_a_teacher_adds_its_pull_and_keeps_flat_groups(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    """DAPO's advantage is 0 on groups whose rewards are all 1; the teacher's term, at
    teacher_coef = 0.5, gives them a gradient, so dapo draws no second job for spread."""
    _, service = fakes
    service.teachers["Qwen/Qwen3-32B"] = FlatAnchor(-0.2)
    run = Run.open(blueprint(tmp_path, **ONE_STEP, **TAUGHT), root=tmp_path)
    harbor(FakeTrials())
    await _train(run, DAPOTrainer(run.config, run.directory), _agent(run, service))
    (row,) = rows(run)
    assert len(row["jobs"]) == 1 and row["degenerate"] == 0 and row["trained"] is True
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    found = [a for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a]
    assert found and found == pytest.approx([0.15] * len(found))


# ---------------------------------------------------------------------- privileged text

#: The fixture tasks' instruction, as a harness sends it.
ASKED = "Write the word alpha to /app/answer.txt."
#: Blueprint replacement that makes the policy its own teacher, at teacher_coef 0.5.
CURRENT = {"learning_rate = 2e-5": 'learning_rate = 2e-5\nteacher_coef = 0.5\nteacher = "current"'}
FEEDBACK = "Feedback on an earlier attempt at this task:\nCheck the path.\n\n"


class Critic:
    """Fake Anthropic messages API: the same advice for every attempt, each prompt kept."""

    def __init__(self) -> None:
        self.read: list[str] = []

    async def create(self, **asked: Any) -> Any:
        self.read.append(asked["messages"][0]["content"])
        return SimpleNamespace(
            content=[SimpleNamespace(text="Check the path.")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )


class FeedbackDAPO(DAPOTrainer):
    """DAPO with Claude's feedback on failed rollouts, as a trainer file writes it."""

    claude: Any = None

    async def privileged(self, member: Member, group: Group) -> str | None:
        if member.verdict.reward == 1.0:
            return None
        reply = await self.claude.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{self.attempt(member)}\nWhat should the agent have done?",
                }
            ],
        )
        usage = reply.usage
        self.spent("anthropic", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        return f"Feedback on an earlier attempt at this task:\n{reply.content[0].text}\n\n"


def _critiqued(run: Run, critic: Critic) -> FeedbackDAPO:
    trainer = FeedbackDAPO(run.config, run.directory)
    trainer.claude = SimpleNamespace(messages=critic)
    return trainer


def _taught(
    fakes: tuple[FakeProxy, FakeService], monkeypatch: pytest.MonkeyPatch, prompt: str
) -> Positional:
    """Serve one call per rollout, `prompt` then "ok", and score with a positional teacher
    on every weights the run serves."""
    proxy, service = fakes
    proxy.default = [made(prompt=prompt, completion="ok")]
    teacher = Positional()
    for tag in ("start", "sample-1", "sample-2"):
        service.teachers[f"tinker://fake/sampler_weights/{tag}"] = teacher
    monkeypatch.setattr("shipyard.proxy.cookbook.tokenizer_for", lambda model: CharTokenizer())
    return teacher


def _advantages(service: FakeService) -> list[float]:
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    found = {round(a, 4) for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a}
    return sorted(found)


async def test_dapo_with_claudes_feedback_teaches_each_failure_through_the_current_weights(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each group has a success and a failure. Claude's feedback on the failure goes before
    the instruction in the teacher's copy, and the current weights score the policy's own
    tokens after it: "o" and "k" at -0.111 and -0.107 against the reference's -0.5."""
    _, service = fakes
    teacher = _taught(fakes, monkeypatch, f"<user>{ASKED}<assistant>")
    run = Run.open(blueprint(tmp_path, **CURRENT), root=tmp_path)
    harbor(Alternating())
    critic = Critic()
    agent = _agent(run, service)
    await _train(run, _critiqued(run, critic), agent)

    assert len(critic.read) == 4, "one failure per group, two groups a step, two steps"
    assert all("score: 0.000" in one and f"--- asked ---\n{ASKED}" in one for one in critic.read)
    read = [CharTokenizer().decode(one) for one in teacher.asked]
    assert read == [f"<user>{FEEDBACK}{ASKED}<assistant>ok"] * 4
    kept = sorted(Path(run.config.rollout.jobs_dir).rglob("privileged.txt"))
    assert [one.read_text() for one in kept] == [FEEDBACK] * 4
    opened = [c for c in service.calls if c[0] == "sampling"]
    assert opened == [
        ("sampling", None, START),
        ("sampling", None, "tinker://fake/sampler_weights/sample-1"),
    ], "the weights each step sampled with"
    assert agent.policy.samplers == {}, "a sampler on weights since replaced is dropped"
    # DAPO's -1 on the failure plus 0.5 x (teacher - reference); the success keeps its 1.
    assert _advantages(service) == [-0.8055, -0.8035, 1.0]
    for row in rows(run):
        assert (row["taught"], row["untaught"], row["degenerate"]) == (2, 0, 0)
        assert row["teacher_kl"] == pytest.approx(-0.391, abs=1e-5)
    costs = record.read_json(run.directory / record.COSTS)["parties"]
    assert (costs["anthropic"]["input_tokens"], costs["anthropic"]["output_tokens"]) == (400, 40)
    assert costs["tinker"]["teacher_tokens"] == 4 * len(read[0])


async def test_feedback_trains_a_group_that_failed_entirely_with_no_second_job(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, service = fakes
    _taught(fakes, monkeypatch, f"<user>{ASKED}<assistant>")
    run = Run.open(blueprint(tmp_path, **ONE_STEP, **CURRENT), root=tmp_path)
    harbor(FakeTrials(default={"reward": 0.0}))
    await _train(run, _critiqued(run, Critic()), _agent(run, service))
    (row,) = rows(run)
    assert len(row["jobs"]) == 1 and row["degenerate"] == 0 and row["taught"] == 8
    assert _advantages(service) == [0.1945, 0.1965], "the teacher's term alone"


@pytest.mark.parametrize("why", ["unmatched", "overflowed"])
async def test_a_sequence_the_teacher_cannot_read_after_its_text_is_counted_untaught(
    why: str,
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    proxy, service = fakes
    prompt = "<user>go<assistant>" if why == "unmatched" else f"<user>{ASKED}<assistant>"
    teacher = _taught(fakes, monkeypatch, prompt)
    proxy.budget = len(prompt) + 10
    run = Run.open(blueprint(tmp_path, **ONE_STEP, **CURRENT), root=tmp_path)
    harbor(Alternating())
    await _train(run, _critiqued(run, Critic()), _agent(run, service))
    (row,) = rows(run)
    assert (row["taught"], row["untaught"]) == (0, 4) and "teacher_kl" not in row
    assert teacher.asked == [] and row["trained"] is True, "DAPO still trains on the spread"
    counted = "4 with no instruction" if why == "unmatched" else "4 past the context of"
    said = [r.getMessage() for r in caplog.records if "got no teacher term" in r.getMessage()]
    assert len(said) == 1 and counted in said[0]


async def test_the_starting_weights_teach_after_a_fixed_text(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor opens for a "start" teacher without a KL term, and reads the text."""
    _, service = fakes
    _taught(fakes, monkeypatch, f"<user>{ASKED}<assistant>")
    start = {"learning_rate = 2e-5": 'learning_rate = 2e-5\nteacher_coef = 0.5\nteacher = "start"'}
    run = Run.open(blueprint(tmp_path, **ONE_STEP, **start), root=tmp_path)
    harbor(Alternating())

    class Prompted(DAPOTrainer):
        async def privileged(self, member: Member, group: Group) -> str | None:
            return "Think first.\n"

    trainer = Prompted(run.config, run.directory)
    with pytest.raises(ValueError, match="never opened"):
        await trainer.teacher(_agent(run, service).policy)
    await _train(run, trainer, trainer.agent())
    assert [c for c in service.calls if c[0] == "sampling"] == [("sampling", MODEL, None)]
    read = [CharTokenizer().decode(one) for one in service.anchor.asked]
    assert read == [f"<user>Think first.\n{ASKED}<assistant>ok"] * 8
    (row,) = rows(run)
    assert (row["taught"], row["untaught"]) == (8, 0) and "anchor_kl" not in row


async def test_a_recipe_that_trains_on_flat_groups_gets_their_datums(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    """Every reward is 1, so the base would drop every group; this recipe has a per-token
    signal of its own and keeps them."""
    _, service = fakes
    run = Run.open(blueprint(tmp_path, **ONE_STEP), root=tmp_path)
    harbor(FakeTrials())

    class EveryGroup(Trainer):
        def trains_on(self, group: Group) -> bool:
            return bool(group.measured)

        def token_advantages(
            self, member: Member, sequence: Sequence, signals: Signals, advantage: float
        ) -> list[float]:
            return [1.0] * len(sequence.targets)

    await _train(run, EveryGroup(run.config, run.directory), _agent(run, service))
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    assert len(trained) == 8
    assert {a for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a} == {1.0}
    (row,) = rows(run)
    assert row["trained"] is True and row["degenerate"] == 0


async def test_a_recipes_own_signal_reaches_its_token_advantages(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, **ONE_STEP), root=tmp_path)
    harbor(Alternating())
    seen: list[set[str]] = []

    class Bonus(Trainer):
        async def signals(self, policy: Policy, pairs: Any) -> list[Signals]:
            found = await super().signals(policy, pairs)
            for row, (_, sequence) in zip(found, pairs, strict=True):
                row["bonus"] = [0.25] * len(sequence.targets)
            return found

        def token_advantages(
            self, member: Member, sequence: Sequence, signals: Signals, advantage: float
        ) -> list[float]:
            seen.append(set(signals))
            base = super().token_advantages(member, sequence, signals, advantage)
            return [a + b for a, b in zip(base, signals["bonus"], strict=True)]

    await _train(run, Bonus(run.config, run.directory), _agent(run, service))
    assert seen and all(names == {"reference", "bonus"} for names in seen)
    trained = [d for d in service.client.sent if "advantages" in d.loss_fn_inputs]
    found = {a for d in trained for a in d.loss_fn_inputs["advantages"].tolist() if a}
    assert found == {0.75, -0.25}, "the rewards' +/-0.5, plus the bonus on every target"


async def test_a_recipe_sends_two_losses_before_one_optimizer_step(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, **ONE_STEP), root=tmp_path)
    harbor(Alternating())

    class Both(Trainer):
        async def forward_backward(self, client: Any, part: list[Datum]) -> Any:
            first = await client.forward_backward(part, "importance_sampling", None)
            second = await client.forward_backward(part, "cispo", {"clip_high_threshold": 4.0})
            return together(first, second)

    await _train(run, Both(run.config, run.directory), _agent(run, service))
    calls = [c[0] if c[0] != "forward_backward" else c[2] for c in service.client.calls]
    sent = [c for c in calls if c in ("importance_sampling", "cispo", "optim_step")]
    (row,) = rows(run)
    assert sent == ["importance_sampling", "cispo", "optim_step"] * row["substeps"]
    assert row["trained"] is True and row["kl_v1"] is not None, "the last result's logprobs"


async def test_a_kl_term_without_an_anchor_is_refused_at_the_step(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
) -> None:
    _, service = fakes
    run = Run.open(blueprint(tmp_path, **ANCHORED), root=tmp_path / "runs")
    harbor(Alternating())
    with pytest.raises(ValueError, match="needs the starting weights to anchor to"):
        await _train(run, DAPOTrainer(run.config, run.directory), _agent(run, service))
    assert service.closed == [
        ("errored", "ValueError: kl_coef = 0.1 needs the starting weights to anchor to")
    ]


async def test_the_overlong_term_docks_against_the_budget_or_warns_once_that_it_is_off(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    caplog: pytest.LogCaptureFixture,
    harbor: Harbor,
) -> None:
    proxy, service = fakes
    home = blueprint(tmp_path)
    run = Run.open(home, root=tmp_path / "runs")
    harbor(Alternating())
    with caplog.at_level(logging.WARNING):
        await _train(run, DAPOTrainer(run.config, run.directory), _agent(run, service))
    logged = rows(run)
    assert len(logged) == 2 and all("overlong" not in row for row in logged)
    warned = [one.getMessage() for one in caplog.records if "overlong term" in one.getMessage()]
    assert warned == [
        "dapo's overlong term is off for this run: the proxy reports no token budget, so "
        "overlong_penalty = 0.5 docks no rollout"
    ]
    # Every fake trial samples two tokens: with a budget of two, each is docked in full.
    proxy.budget = 2
    caplog.clear()
    run = Run.open(home, root=tmp_path / "runs")
    harbor(Alternating())
    with caplog.at_level(logging.WARNING):
        await _train(run, DAPOTrainer(run.config, run.directory), _agent(run, service))
    assert [row["overlong"] for row in rows(run)] == [4, 4]
    assert all(row["trained"] for row in rows(run)), "1 - 0.5 against 0 - 0.5 still has spread"
    assert not [one for one in caplog.records if "overlong term" in one.getMessage()]


async def test_a_job_on_a_provider_agent_groups_the_verdicts_and_the_step_takes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    monkeypatch.chdir(tmp_path)
    home = blueprint(tmp_path, **{"[agent]\n": '[agent]\nprovider = "openrouter"\n'})
    run = Run.open(home, root=tmp_path / "runs")
    harbor(Alternating())
    trainer = DAPOTrainer(run.config, run.directory)
    endpoint = endpoint_for(run.config, run.directory)
    agent = Agent(Policy(endpoint), [harness("pi@0.85.1"), harness("claude-code")])
    assert isinstance(endpoint, ProviderEndpoint)
    tasks = [tmp_path / "tasks" / "four" / name for name in ("a", "b")]
    with run:
        async with agent.policy:
            found = trainer.credited(await trainer.job(agent, tasks, purpose="evaluation"))
            assert (await trainer.step(agent, found)).update is None
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
    assert rows(run) == []
