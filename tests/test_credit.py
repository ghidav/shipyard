"""What a rollout is worth under each preset, what the batch leaves out and says so,
where mu comes from, the KL anchor, and the refusal of weights that moved."""

from __future__ import annotations

from pathlib import Path

import pytest
import tinker

from shipyard.admit import Verdict
from shipyard.credit import LENGTH_CAP, advantages, credit, reference_logprobs, shaped
from shipyard.pack import Group, Member, Sequence
from shipyard.recipes import cispo, dapo, dr_grpo
from shipyard.recipes.train import row
from shipyard.trainer import Trainer
from tests.trainers import FakeTrainingClient, FlatAnchor, batch, preset

PROMPT = (1, 2, 3)
ACTION = (10, 11)


def _sequence(
    prompt: tuple[int, ...] = PROMPT, action: tuple[int, ...] = ACTION, *, scored: bool = True
) -> Sequence:
    tokens = (*prompt, *action)
    spans = ((len(prompt), len(tokens)),)
    return Sequence(tokens, spans, tuple(-0.5 for _ in action) if scored else None)


def _member(
    reward: float | None,
    *,
    mask: str | None = None,
    wrote: int | None = None,
    sequences: list[Sequence] | None = None,
) -> Member:
    found = () if mask else (tuple(sequences) if sequences is not None else (_sequence(),))
    if wrote is None:
        wrote = sum(len(one.targets) for one in found)
    return Member(Verdict(reward, mask, None), found, wrote)


def _group(*members: Member, updates: int | None = None, task: str = "t") -> Group:
    return Group(Path(task), tuple(members), updates)


def _trainer(updates: int = 0) -> Trainer:
    """A trainer whose forward scores each position minus the token it predicts."""
    return Trainer(FakeTrainingClient(positional=True), updates=updates)


def _read(client: FakeTrainingClient) -> list[list[int]]:
    """The whole sequence each forward pass scored: what it read, then its last target."""
    return [
        [*one.model_input.to_ints(), int(one.loss_fn_inputs["target_tokens"].tolist()[-1])]
        for one in client.sent
    ]


def _credited(one: tinker.Datum) -> list[float]:
    """The advantages on the tokens the policy sampled."""
    mask = one.loss_fn_inputs["mask"].tolist()
    found = one.loss_fn_inputs["advantages"].tolist()
    return [a for a, m in zip(found, mask, strict=True) if m]


def _advantage_of(one: tinker.Datum) -> float:
    (value,) = {round(a, 6) for a in _credited(one)}
    return value


# ------------------------------------------------------------------ the presets


def test_the_three_recipes_resolve_to_their_presets() -> None:
    from shipyard.config import load

    root = Path(__file__).parent / "blueprints"
    found = dapo.preset(load(root / "dapo").recipe)
    assert (found.name, found.normalize, found.loss_fn) == ("dapo", True, "ppo")
    assert found.loss_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
    assert (found.length_penalty, found.kl_coef, found.reference) == (0.0, 0.0, "trainer")
    assert (found.learning_rate, found.substeps) == (2e-5, 1)
    assert found.clipping == "clip 0.2 / 0.28"
    found = dr_grpo.preset(load(root / "dr-grpo").recipe)
    assert (found.name, found.normalize, found.loss_fn) == ("dr-grpo", False, "ppo")
    assert found.loss_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    assert (found.length_penalty, found.length_floor) == (0.2, 0)
    assert found.clipping == "clip 0.2 / 0.2"
    found = cispo.preset(load(root / "cispo").recipe)
    assert (found.name, found.normalize, found.loss_fn) == ("cispo", True, "cispo")
    assert found.loss_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 1.2}
    assert found.reference == "sampler"
    assert found.clipping == "weight truncated above 1.2, no lower bound"


async def test_dapo_divides_by_the_spread_and_dr_grpo_does_not() -> None:
    groups = [_group(_member(1.0), _member(0.0))]
    normalized = await credit(groups, preset(), _trainer())
    assert [_advantage_of(one) for one in normalized.datums] == pytest.approx([1.0, -1.0], abs=1e-4)
    plain = await credit(groups, preset(name="dr-grpo", normalize=False), _trainer())
    assert [_advantage_of(one) for one in plain.datums] == [0.5, -0.5]
    narrow = [_group(_member(0.51), _member(0.50))]
    assert [_advantage_of(one) for one in (await credit(narrow, preset(), _trainer())).datums] == (
        pytest.approx([1.0, -1.0], abs=1e-3)
    )
    assert advantages([0.51, 0.50], normalize=False) == pytest.approx([0.005, -0.005])


# ------------------------------------------------------------ degenerate groups


async def test_degenerate_groups_are_dropped_before_any_reference_pass() -> None:
    trainer = _trainer()
    groups = [
        _group(_member(1.0), _member(1.0), _member(1.0)),
        _group(_member(0.0), _member(0.0)),
        _group(_member(1.0)),
    ]
    found = await credit(groups, preset(), trainer)
    assert trainer.client.calls == [], "no forward pass was bought for a flat group"
    assert found.empty and found.degenerate == 3 and found.groups == 3
    assert found.rollouts == 6 and found.graded == 6 and found.masked == 0
    assert found.credited == 0 and found.sequences == 0 and found.reference_tokens == 0
    assert found.sequences_per_rollout is None, "nothing credited: nothing to average"
    assert found.reward_mean == pytest.approx(4 / 6) and found.reward_spread == 0.0


async def test_a_masked_member_enters_no_baseline_and_a_kept_group_counts_beside_a_dropped() -> (
    None
):
    groups = [
        _group(_member(1.0), _member(0.0), _member(0.0, mask="grading_error")),
        _group(_member(1.0), _member(1.0)),
    ]
    found = await credit(groups, preset(), _trainer())
    assert [_advantage_of(one) for one in found.datums] == pytest.approx([1.0, -1.0], abs=1e-4)
    assert found.masked == 1 and found.graded == 4 and found.rollouts == 5
    assert found.degenerate == 1 and found.groups == 2 and found.credited == 2
    assert found.sequences == 2 and found.sequences_per_rollout == 1.0
    assert found.reward_mean == pytest.approx(0.75)
    assert found.reward_spread == pytest.approx(0.25)


async def test_nothing_measured_reports_no_reward_rather_than_a_zero() -> None:
    groups = [_group(_member(None, mask="env_error"), _member(None, mask="timeout"))]
    found = await credit(groups, preset(), _trainer())
    assert found.empty and found.masked == 2 and found.graded == 0 and found.degenerate == 0
    assert found.reward_mean is None and found.reward_spread is None
    assert found.length_mean is None and found.sequences_per_rollout is None
    logged = row(4, found, None)
    assert "reward_mean" not in logged and "length_mean" not in logged
    assert logged == {
        "step": 4,
        "trained": False,
        "groups": 1,
        "rollouts": 2,
        "graded": 0,
        "masked": 2,
        "degenerate": 0,
        "sequences": 0,
    }


def test_sequences_per_rollout_averages_over_the_credited_members() -> None:
    one = tinker.Datum(model_input=tinker.ModelInput.from_ints([1, 2, 3]), loss_fn_inputs={})
    assert batch(one, one, one, credited=2).sequences_per_rollout == 1.5
    assert batch(graded=4, degenerate=2).sequences_per_rollout is None
    assert not batch(one).empty and batch().empty


# ---------------------------------------------------------- the dr-grpo length rule

DR = dict(name="dr-grpo", normalize=False)


async def test_a_tie_on_reward_is_broken_by_length_only_when_the_penalty_is_on() -> None:
    tie = [_group(_member(1.0, wrote=30), _member(1.0, wrote=10))]
    plain = await credit(tie, preset(**DR), _trainer())
    assert plain.empty and plain.degenerate == 1
    docked = await credit(tie, preset(**DR, length_penalty=0.5), _trainer())
    assert not docked.empty and docked.degenerate == 0
    # r - min(0.5 L / 20, 0.5) = [0.5, 0.75]; centred, the long one is -0.125.
    assert [_advantage_of(one) for one in docked.datums] == [-0.125, 0.125]
    assert shaped([1.0, 1.0], [30, 10], penalty=0.5, floor=0) == [0.5, 0.75]


async def test_length_is_compared_among_the_solved_and_failures_are_left_alone() -> None:
    docked = preset(**DR, length_penalty=0.2)
    group = _group(_member(1.0, wrote=30), _member(1.0, wrote=10), _member(0.0, wrote=50))
    found = await credit([group], docked, _trainer())
    # Mean solved length 20: docked 0.3 and 0.1; shaped [0.7, 0.9, 0.0], mean 0.5333.
    assert [_advantage_of(one) for one in found.datums] == pytest.approx(
        [0.1667, 0.3667, -0.5333], abs=1e-4
    )
    short = _group(_member(1.0, wrote=30), _member(1.0, wrote=10), _member(0.0, wrote=2))
    again = await credit([short], docked, _trainer())
    assert [_advantage_of(one) for one in again.datums] == pytest.approx(
        [0.1667, 0.3667, -0.5333], abs=1e-4
    )
    assert found.reward_mean == pytest.approx(2 / 3) and found.length_mean == 30.0


async def test_a_lone_success_and_a_group_of_failures_are_not_shaped() -> None:
    docked = preset(**DR, length_penalty=0.2)
    lone = _group(_member(1.0, wrote=30), _member(0.0, wrote=10))
    assert [_advantage_of(one) for one in (await credit([lone], docked, _trainer())).datums] == [
        0.5,
        -0.5,
    ]
    failures = _group(_member(0.0, wrote=30), _member(0.0, wrote=10))
    found = await credit([failures], docked, _trainer())
    assert found.empty and found.degenerate == 1
    assert shaped([0.0, 0.0], [30, 10], penalty=0.2, floor=0) == [0.0, 0.0]


async def test_the_longest_solved_answer_still_beats_every_failure() -> None:
    docked = preset(**DR, length_penalty=2.0)
    group = _group(_member(1.0, wrote=1000), _member(1.0, wrote=10), _member(0.0, wrote=5))
    longest, shortest, failure = (
        _advantage_of(one) for one in (await credit([group], docked, _trainer())).datums
    )
    # Mean solved length 505: 2.0 * 1000 / 505 = 3.96, capped at 0.5; the short one pays 0.04.
    assert longest - failure == pytest.approx(LENGTH_CAP, abs=1e-4)
    assert shortest > longest > failure


async def test_under_the_floor_nothing_is_docked_and_over_it_only_the_excess_is() -> None:
    tie = [_group(_member(1.0, wrote=30), _member(1.0, wrote=10))]
    docked = await credit(tie, preset(**DR, length_penalty=0.5, length_floor=10), _trainer())
    # Excess [20, 0] over the solved mean 20 docks [0.5, 0] at 0.5: centred [-0.25, 0.25].
    assert [_advantage_of(one) for one in docked.datums] == [-0.25, 0.25]
    spared = await credit(tie, preset(**DR, length_penalty=0.5, length_floor=30), _trainer())
    assert spared.empty and spared.degenerate == 1


async def test_length_is_on_the_row_whether_or_not_it_is_penalised() -> None:
    pair = _group(_member(1.0, wrote=4), _member(0.0, wrote=2))
    found = await credit([pair], preset(), _trainer())
    assert found.length_mean == 3.0 and row(0, found, None)["length_mean"] == 3.0


# ---------------------------------------------------------------- the reference


async def test_the_trainer_reference_is_one_batched_forward_aligned_by_the_shift() -> None:
    chained = Sequence((1, 2, 3, 4, 5, 6, 7, 8), ((2, 3), (4, 6), (7, 8)), None)
    single = _sequence((1, 2), (9,))
    trainer = _trainer()
    scored = await reference_logprobs(trainer, [chained, single])
    assert scored == [[-3.0, -5.0, -6.0, -8.0], [-9.0]]
    assert _read(trainer.client) == [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 9]]
    assert [call[0] for call in trainer.client.calls] == ["forward"], "one batched pass"
    first, _ = trainer.client.sent
    assert first.model_input.to_ints() == [1, 2, 3, 4, 5, 6, 7]
    assert first.loss_fn_inputs["target_tokens"].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert first.loss_fn_inputs["weights"].tolist() == [1.0] * 7
    assert await reference_logprobs(trainer, []) == [] and len(trainer.client.sent) == 2
    with pytest.raises(ValueError, match="none was given"):
        await reference_logprobs(None, [single])
    found = await credit(
        [_group(_member(1.0, sequences=[chained]), _member(0.0, sequences=[single]))],
        preset(),
        trainer,
    )
    assert found.reference_tokens == 7 + 2 and found.anchor_tokens == 0
    assert found.datums[0].loss_fn_inputs["logprobs"].tolist() == [0, -3.0, 0, -5.0, -6.0, 0, -8.0]
    assert found.datums[1].loss_fn_inputs["logprobs"].tolist() == [0, -9.0]
    assert found.sequences == 2 and found.sequences_per_rollout == 1.0


async def test_the_sampler_reference_reads_the_records_logprobs_and_makes_no_forward() -> None:
    trainer = _trainer(updates=7)
    groups = [_group(_member(1.0), _member(0.0), updates=2)]
    found = await credit(
        groups, preset(name="cispo", loss_fn="cispo", reference="sampler"), trainer
    )
    assert trainer.client.calls == [] and found.reference_tokens == 0
    assert found.datums[0].loss_fn_inputs["logprobs"].tolist() == [0, 0, -0.5, -0.5]
    unscored = [_group(_member(1.0, sequences=[_sequence(scored=False)]), _member(0.0))]
    with pytest.raises(ValueError, match="carries none"):
        await credit(unscored, preset(reference="sampler"), trainer)


async def test_the_kl_anchor_is_subtracted_per_token_and_reported() -> None:
    anchor = FlatAnchor(-0.1)
    groups = [_group(_member(1.0), _member(0.0))]
    found = await credit(groups, preset(kl_coef=0.5), _trainer(), anchor)
    # mu is -10 and -11 (the trainer's positional score); the anchor says -0.1 everywhere.
    assert anchor.asked == [[1, 2, 3, 10, 11], [1, 2, 3, 10, 11]]
    assert found.anchor_tokens == 10 and found.reference_tokens == 8
    gaps = [-10.0 + 0.1, -11.0 + 0.1]
    assert _credited(found.datums[0]) == pytest.approx([1.0 - 0.5 * g for g in gaps], abs=1e-4)
    assert _credited(found.datums[1]) == pytest.approx([-1.0 - 0.5 * g for g in gaps], abs=1e-4)
    assert found.anchor_kl == pytest.approx(sum(gaps) / 2)
    assert row(0, found, None)["anchor_kl"] == pytest.approx(sum(gaps) / 2)
    off = await credit(groups, preset(), _trainer(), anchor)
    assert len(anchor.asked) == 2 and off.anchor_kl is None and off.anchor_tokens == 0
    with pytest.raises(ValueError, match="anchor"):
        await credit(groups, preset(kl_coef=0.5), _trainer(), None)


async def test_credit_refuses_rollouts_sampled_at_weights_the_trainer_has_moved_from() -> None:
    groups = [_group(_member(1.0), _member(0.0), updates=2, task="alpha")]
    with pytest.raises(ValueError, match=r"applied 3 update\(s\).*alpha"):
        await credit(groups, preset(), _trainer(updates=5))
    still = await credit(groups, preset(), _trainer(updates=2))
    assert len(still.datums) == 2
    unstamped = [_group(_member(1.0), _member(0.0))]
    assert len((await credit(unstamped, preset(), _trainer(updates=5))).datums) == 2
    sampled = preset(reference="sampler")
    assert len((await credit(groups, sampled, _trainer(updates=5))).datums) == 2
