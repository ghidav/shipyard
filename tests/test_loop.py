"""The gradient loop end to end on fakes: a two-step dapo run through `recipes.dapo.run`
with a FakeProxy, a stand-in for Harbor's trial and a fake Tinker service; the step
nothing could be credited in; a step refilled from the plan, and the cap on it; what cispo
and dr-grpo resolve to; the refusals."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record, serving, session
from shipyard.data import batches
from shipyard.recipes import run_recipe
from shipyard.run import Run
from shipyard.trainer import SERVE_TTL
from tests.records import FakeProxy
from tests.trainers import FakeService, capable
from tests.trials import FIXTURES, FakeTrials, write_result

BLUEPRINTS = Path(__file__).parent / "blueprints"
MODEL = "Qwen/Qwen3-8B"
#: The fake proxy's one record per trial: prompt "go" + completion "ok" is four tokens,
#: of which the model reads three.
SEQUENCE_TOKENS = 4
INPUT_TOKENS = 3


class Alternating:
    """Harbor's trial replaced: rewards 1, 0, 1, 0 per task in call order, so that every
    group of two has spread and no step is degenerate."""

    def __init__(self) -> None:
        self.configs: list[TrialConfig] = []
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        write_result(Path(config.trials_dir) / config.trial_name, reward=float(self.seen[task] % 2))


class Spread:
    """Harbor's trial replaced: a task in `spread` scores 1, 0, 1, 0 in call order, any other
    task 1 every time, so only the named tasks' groups carry a gradient."""

    def __init__(self, spread: set[str]) -> None:
        self.spread = spread
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        reward = float(self.seen[task] % 2) if task in self.spread else 1.0
        write_result(Path(config.trials_dir) / config.trial_name, reward=reward)


def _blueprint(tmp_path: Path, kind: str = "dapo", *, tasks: int = 4, **replacements: str) -> Path:
    """The recipe's fixture over a dataset of `tasks` tasks named a, b, c..., two tasks per
    batch, groups of two."""
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
    """The proxy and the Tinker service replaced, from a working directory of fixtures."""
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr(serving, "Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True)))
    service.client.trained = -0.3
    monkeypatch.setattr(session, "service_client", lambda metadata=None: service)
    return proxy, service


async def _run(home: Path, tmp_path: Path, runner: Any) -> Run:
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = runner
    with opened:
        await run_recipe(opened)
    return opened


def _plan(tmp_path: Path) -> list[list[str]]:
    """The batches the fixture's plan cuts, by task name, as the run will draw them."""
    planned = batches(["four"], size=2, seed=0, epochs=1, root=tmp_path / "tasks")
    return [[task.name for task in batch] for batch in planned]


def _rows(opened: Run) -> list[dict[str, Any]]:
    return list(record.read(opened.directory / record.METRICS))


def _checkpoints(opened: Run) -> list[dict[str, Any]]:
    return list(record.read(opened.directory / record.CHECKPOINTS))


async def test_a_two_step_dapo_run_publishes_points_samples_credits_steps_and_checkpoints(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    client = service.client
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    opened.run_trial = Alternating()
    pointed: list[list[str | None]] = []
    stamped: list[int | None] = []
    sampling = opened.sample

    async def sample(batch: Any, **named: Any) -> Any:
        pointed.append(list(proxy.swapped))
        rolled = await sampling(batch, **named)
        stamped.append(rolled.updates)
        return rolled

    opened.sample = sample  # type: ignore[method-assign]
    with opened:
        await run_recipe(opened)

    # The trainer: a LoRA at the blueprint's rank, no anchor, the session finished.
    assert service.calls == [("lora", MODEL, 32)]
    assert service.closed == [("success", None)]
    assert opened.trainer is not None and opened.trainer.updates == 2
    # Published before each sample, at the serve TTL, and the proxy pointed at it.
    published = [
        c for c in client.calls if c[0] == "save_weights_for_sampler" and c[2] == SERVE_TTL
    ]
    assert published == [
        ("save_weights_for_sampler", "sample-0", SERVE_TTL),
        ("save_weights_for_sampler", "sample-1", SERVE_TTL),
    ]
    paths = [f"tinker://fake/sampler_weights/sample-{n}" for n in (0, 1)]
    assert proxy.swapped == paths and pointed == [paths[:1], paths]
    assert proxy.placed[0][0] == "local" and proxy.closed == 1
    assert stamped == [0, 1], "each batch stamped with the updates applied before it"
    # One reference pass per step, then dapo's 16 substeps capped at the step's two groups,
    # one group each, with dapo's loss and clipping.
    assert [
        c[0] for c in client.calls if c[0] in ("forward", "forward_backward", "optim_step")
    ] == ["forward", "forward_backward", "optim_step", "forward_backward", "optim_step"] * 2
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert all(
        c[2:] == ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}) for c in sent
    )
    assert all(c[1] == [INPUT_TOKENS] * 2 for c in sent)
    # Two rows, in the row's order, with the step's measures and the update's.
    rows = _rows(opened)
    assert [row["step"] for row in rows] == [0, 1] and [row["seq"] for row in rows] == [1, 2]
    for row in rows:
        assert row["trained"] is True
        assert (row["groups"], row["rollouts"], row["graded"], row["masked"]) == (2, 4, 4, 0)
        assert (row["refills"], row["refill_rollouts"], row["surplus"]) == (0, 0, 0)
        assert (row["degenerate"], row["sequences"], row["sequences_per_rollout"]) == (0, 4, 1.0)
        assert row["train_tokens"] == 4 * INPUT_TOKENS
        assert (row["reward_mean"], row["reward_spread"], row["length_mean"]) == (0.5, 0.5, 2.0)
        assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5)
        assert row["kl_v2"] == pytest.approx(0.02, abs=1e-5)
        assert row["entropy"] == pytest.approx(0.5, abs=1e-5)
        assert row["learning_rate"] == pytest.approx(2e-5), "no warm-up by default"
        assert (row["substeps"], row["loss_fn"]) == (2, "ppo")
        assert row["seconds"] >= 0.0 and "anchor_kl" not in row
        keys = [key for key in row if key not in ("at", "seq")]
        assert keys == [
            "step",
            "trained",
            "groups",
            "rollouts",
            "refills",
            "refill_rollouts",
            "graded",
            "masked",
            "degenerate",
            "surplus",
            "sequences",
            "sequences_per_rollout",
            "train_tokens",
            "reward_mean",
            "reward_spread",
            "length_mean",
            "kl_v1",
            "kl_v2",
            "entropy",
            "learning_rate",
            "substeps",
            "loss_fn",
            "seconds",
        ]
    # `every = 1`: a checkpoint per step at the TTL, then `final`, kept with none.
    saved = _checkpoints(opened)
    assert [row["tag"] for row in saved] == ["step-1", "step-2", "final"]
    for row in saved:
        assert row["state_path"] == f"tinker://fake/weights/{row['tag']}"
        assert row["sampler_path"] == f"tinker://fake/sampler_weights/{row['tag']}"
        assert row["ttl_hours"] == (None if row["tag"] == "final" else 168.0)
    assert [c for c in client.calls if c[0] == "save_state"] == [
        ("save_state", "step-1", 604800),
        ("save_state", "step-2", 604800),
        ("save_state", "final", None),
    ]
    # The train and reference tokens on the tinker party, beside what sampling cost.
    costs = record.read_json(opened.directory / record.COSTS)
    tinker = costs["parties"]["tinker"]
    assert tinker["train_tokens"] == 2 * 4 * INPUT_TOKENS
    assert tinker["reference_tokens"] == 2 * 4 * INPUT_TOKENS
    assert "anchor_tokens" not in tinker and tinker["trials"] == 8
    assert list(costs["parties"]) == ["tinker"]
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["batch"] for job in jobs] == [0, 1] and all(job["graded"] == 4 for job in jobs)
    note = Run.read(opened.directory)
    assert note["failed"] is False and note["kind"] == "dapo"


#: What a run logs when its recipe has an overlong term and the proxy reports no budget.
OVERLONG_OFF = (
    "dapo's overlong term is off for this run: the proxy reports no token budget, so "
    "overlong_penalty = 0.5 docks no rollout"
)


def _overlong_off(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [one.getMessage() for one in caplog.records if "overlong term" in one.getMessage()]


async def test_the_proxys_token_budget_reaches_credit_and_the_row_counts_the_docked(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], caplog: pytest.LogCaptureFixture
) -> None:
    """Every fake trial samples two tokens; with a budget of two, each sampled to the limit
    and is docked DAPO's full penalty, which the row counts."""
    proxy, _ = fakes
    proxy.budget = 2
    with caplog.at_level(logging.WARNING):
        opened = await _run(_blueprint(tmp_path), tmp_path, Alternating())
    rows = _rows(opened)
    assert [row["overlong"] for row in rows] == [4, 4]
    assert all(row["trained"] for row in rows), "1 - 0.5 against 0 - 0.5 still has spread"
    assert _overlong_off(caplog) == [], "a budget to dock against"


async def test_an_overlong_term_with_no_budget_reported_warns_once_that_it_is_off(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], caplog: pytest.LogCaptureFixture
) -> None:
    """A proxy that reports no token budget, as a remote one may not: nothing is docked, no
    row counts `overlong`, and the run says once, not once a step, that the term is off."""
    proxy, _ = fakes
    assert proxy.budget is None
    with caplog.at_level(logging.WARNING):
        opened = await _run(_blueprint(tmp_path), tmp_path, Alternating())
    rows = _rows(opened)
    assert len(rows) == 2 and all("overlong" not in row for row in rows)
    assert _overlong_off(caplog) == [OVERLONG_OFF]
    (logged,) = [one for one in caplog.records if one.getMessage() == OVERLONG_OFF]
    assert logged.levelno == logging.WARNING


async def test_a_step_with_every_group_degenerate_logs_untrained_and_takes_no_gradient(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    home = _blueprint(tmp_path, **{"batch_size = 2": "batch_size = 4"})
    opened = await _run(home, tmp_path, FakeTrials())
    client = service.client
    assert not any(c[0] in ("forward", "forward_backward", "optim_step") for c in client.calls)
    (row,) = _rows(opened)
    assert row["trained"] is False and row["step"] == 0
    assert (row["groups"], row["rollouts"], row["graded"], row["degenerate"]) == (4, 8, 8, 4)
    assert row["sequences"] == 0 and row["reward_mean"] == 1.0 and row["reward_spread"] == 0.0
    assert "train_tokens" not in row and "learning_rate" not in row and "kl_v1" not in row
    assert "sequences_per_rollout" not in row, "none credited: nothing to average"
    assert [c["tag"] for c in _checkpoints(opened)] == ["final"]
    assert proxy.swapped == ["tinker://fake/sampler_weights/sample-0"]
    assert opened.trainer is not None and opened.trainer.updates == 0
    assert (
        "train_tokens" not in record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    )
    assert service.closed == [("success", None)]


async def test_a_step_short_of_groups_samples_the_plans_next_batch_until_it_is_full(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    """DAPO's dynamic sampling: the first batch has one group with a gradient, so the step
    samples the plan's next batch at the same weights; that one brings two, the batch of two
    is full, and the third is surplus. The plan is spent, so the run is one step."""
    _, service = fakes
    home = _blueprint(tmp_path)
    first, second = _plan(tmp_path)
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = Spread({first[0], *second})
    stamped: list[int | None] = []
    sampling = opened.sample

    async def sample(batch: Any, **named: Any) -> Any:
        rolled = await sampling(batch, **named)
        stamped.append(rolled.updates)
        return rolled

    opened.sample = sample  # type: ignore[method-assign]
    with opened:
        await run_recipe(opened)
    client = service.client
    assert stamped == [0, 0], "both rounds sampled the weights the step published"
    (row,) = _rows(opened)
    assert row["step"] == 0 and row["trained"] is True
    assert (row["groups"], row["rollouts"], row["refills"], row["refill_rollouts"]) == (4, 8, 1, 4)
    assert (row["degenerate"], row["surplus"], row["sequences"], row["substeps"]) == (1, 1, 4, 2)
    assert row["train_tokens"] == 4 * INPUT_TOKENS
    (forward,) = [c for c in client.calls if c[0] == "forward"]
    assert forward[1] == [INPUT_TOKENS] * 4, "the surplus group is never sent"
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [(job["batch"], job["purpose"], job["tasks"], job["trials"]) for job in jobs] == [
        (0, "rollout", 2, 4),
        (0, "refill", 2, 4),
    ]
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["trials"] == 8, "both rounds' rollouts are paid for"
    assert tinker["train_tokens"] == tinker["reference_tokens"] == 4 * INPUT_TOKENS
    assert opened.trainer is not None and opened.trainer.updates == 1
    assert [c["tag"] for c in _checkpoints(opened)] == ["step-1", "final"]


async def test_refill_stops_at_its_cap_and_when_the_plan_runs_out(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    """Six tasks, every group flat, `refill = 1`: step 0 samples its batch and one more and
    trains on nothing; step 1 has the last batch and no plan left to draw on."""
    _, service = fakes
    home = _blueprint(
        tmp_path, tasks=6, **{"learning_rate = 2e-5": "learning_rate = 2e-5\nrefill = 1"}
    )
    opened = await _run(home, tmp_path, FakeTrials())
    rows = _rows(opened)
    assert [(row["step"], row["trained"]) for row in rows] == [(0, False), (1, False)]
    assert [(row["groups"], row["refills"], row["refill_rollouts"]) for row in rows] == [
        (4, 1, 4),
        (2, 0, 0),
    ]
    assert [(row["degenerate"], row["surplus"]) for row in rows] == [(4, 0), (2, 0)]
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["batch"] for job in jobs] == [0, 0, 1]
    assert not any(c[0] in ("forward", "forward_backward") for c in service.client.calls)
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["trials"] == 12 and "train_tokens" not in tinker


async def test_without_refill_a_short_step_trains_on_what_it_has(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    home = _blueprint(tmp_path, **{"learning_rate = 2e-5": "learning_rate = 2e-5\nrefill = 0"})
    first, second = _plan(tmp_path)
    opened = await _run(home, tmp_path, Spread({first[0], *second}))
    rows = _rows(opened)
    assert [(row["step"], row["groups"], row["degenerate"]) for row in rows] == [
        (0, 2, 1),
        (1, 2, 0),
    ]
    assert not any("refills" in row or "surplus" in row for row in rows)


async def test_groups_of_one_never_refill_since_none_could_carry_a_gradient(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    home = _blueprint(tmp_path, **{"group_size = 2": "group_size = 1"})
    opened = await _run(home, tmp_path, FakeTrials())
    rows = _rows(opened)
    assert [(row["step"], row["trained"], row["groups"]) for row in rows] == [
        (0, False, 2),
        (1, False, 2),
    ]
    assert not any("refills" in row for row in rows)
    assert len(list(record.read(opened.directory / record.JOBS))) == 2


async def test_cispo_resolves_to_its_loss_and_reads_the_samplers_logprobs(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, "cispo", **{"batch_size = 2": "batch_size = 4"})
    opened = await _run(home, tmp_path, Alternating())
    client = service.client
    assert not any(c[0] == "forward" for c in client.calls), "reference = sampler: no forward"
    sent = [c for c in client.calls if c[0] == "forward_backward"]
    assert len(sent) == 4, "16 substeps, capped at the batch's four groups"
    assert all(
        c[2:] == ("cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}) for c in sent
    )
    (row,) = _rows(opened)
    assert row["trained"] is True and row["loss_fn"] == "cispo" and row["train_tokens"] == 24
    assert row["substeps"] == 4 and row["refills"] == 0
    assert row["kl_v1"] == pytest.approx(-0.2, abs=1e-5), "mu is the record's -0.5"
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["train_tokens"] == 24 and "reference_tokens" not in tinker


async def test_dr_grpo_resolves_to_symmetric_ppo_clipping_with_its_length_rule(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(tmp_path, "dr-grpo", **{"batch_size = 2": "batch_size = 4"})
    opened = await _run(home, tmp_path, Alternating())
    (sent,) = [c for c in service.client.calls if c[0] == "forward_backward"]
    assert sent[2:] == ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
    (row,) = _rows(opened)
    assert row["trained"] is True and row["loss_fn"] == "ppo" and row["substeps"] == 1
    assert "refills" not in row and "surplus" not in row, "dr-grpo samples no more"
    assert [c["tag"] for c in _checkpoints(opened)] == ["step-1", "final"]


async def test_a_kl_anchor_is_a_sampling_client_on_the_starting_weights(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    home = _blueprint(
        tmp_path,
        **{
            "batch_size = 2": "batch_size = 4",
            "learning_rate = 2e-5": "learning_rate = 2e-5\nkl_coef = 0.1",
        },
    )
    opened = await _run(home, tmp_path, Alternating())
    assert service.calls == [("lora", MODEL, 32), ("sampling", MODEL, None)]
    assert len(service.anchor.asked) == 8
    (row,) = _rows(opened)
    # mu is the forward's -0.5 and the anchor says -0.1: the gap is -0.4 on every token.
    assert row["anchor_kl"] == pytest.approx(-0.4, abs=1e-5)
    keys = list(row)
    assert keys.index("entropy") < keys.index("anchor_kl") < keys.index("learning_rate")
    tinker = record.read_json(opened.directory / record.COSTS)["parties"]["tinker"]
    assert tinker["anchor_tokens"] == 8 * SEQUENCE_TOKENS


async def test_a_kl_anchor_from_a_checkpoint_samples_weights_published_from_its_state(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    """Tinker samples only a sampler path, and `from_checkpoint` of a training run is a state
    path: the anchor gets the loaded state published, and the proxy starts on the base."""
    proxy, service = fakes
    state = "tinker://run:train:0/weights/final"
    home = _blueprint(
        tmp_path,
        **{
            "batch_size = 2": "batch_size = 4",
            "learning_rate = 2e-5": "learning_rate = 2e-5\nkl_coef = 0.1",
            "[model]\n": f'[model]\nfrom_checkpoint = "{state}"\n',
        },
    )
    await _run(home, tmp_path, Alternating())
    published = "tinker://fake/sampler_weights/anchor"
    assert service.calls == [("from_state", state), ("sampling", MODEL, published)]
    assert any(call[:2] == ("save_weights_for_sampler", "anchor") for call in service.client.calls)


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
