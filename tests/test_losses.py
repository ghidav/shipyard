"""Tests for the losses `update` trains with: Tinker's table of them, the check that refuses a
loss or a setting none of them has before anything is scored or sent, and the inputs each loss
is sent, on Tinker and on Together alike."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import tinker

from shipyard import Trainer
from shipyard.groups import Sequence, Trial
from shipyard.jobs.harnesses import harness
from shipyard.models.clients.tinker import LOSSES, check_loss, datum, sent
from shipyard.models.clients.together import LOSS_TYPES, sample
from tests.fakes import Alternating, FakeProxy, Harbor, made
from tests.runs import ONE_STEP, blueprint, train
from tests.service import FakeService

TRIAL = Trial("j", Path("t"), harness("pi"), Path("t"))


def _datum(advantage: float) -> tinker.Datum:
    """A datum of four tokens whose last two the model wrote, each at `advantage`."""
    sequence = Sequence((1, 2, 3, 4), ((2, 4),), None, (), TRIAL)
    return datum(sequence, logprobs=[-0.5, -0.5], advantages=[advantage, advantage])


# ------------------------------------------------------------------------ the table


def test_a_loss_is_one_of_tinkers_and_takes_only_its_own_settings() -> None:
    for name, known in LOSSES.items():
        check_loss(name, None)
        check_loss(name, dict.fromkeys(known.settings, 1.0))
    with pytest.raises(ValueError, match="^no loss 'grpo': the losses are importance_sampling, "):
        check_loss("grpo", None)
    with pytest.raises(
        ValueError,
        match="^the ppo loss takes clip_high_threshold, clip_low_threshold; got clip_low$",
    ):
        check_loss("ppo", {"clip_low": 0.8, "clip_high_threshold": 1.28})
    for bare in ("importance_sampling", "cross_entropy"):
        with pytest.raises(ValueError, match=f"^the {bare} loss takes no settings; got beta$"):
            check_loss(bare, {"beta": 0.05})
    assert set(LOSS_TYPES) == set(LOSSES), "Together trains with each of Tinker's losses"


def test_each_loss_is_sent_only_what_it_reads_and_never_the_mask() -> None:
    made_one = _datum(0.5)
    ratio = {"target_tokens", "logprobs", "advantages"}
    for name in ("importance_sampling", "ppo", "cispo", "dro"):
        assert set(sent(made_one, name).loss_fn_inputs) == ratio
        assert set(sample(made_one, name)["loss_fn_inputs"]) == ratio, "Together's alike"
    weighed = sent(made_one, "cross_entropy").loss_fn_inputs
    assert set(weighed) == {"target_tokens", "weights"}
    advantages = made_one.loss_fn_inputs["advantages"].tolist()
    assert weighed["weights"].tolist() == advantages == [0.0, 0.5, 0.5], "a target's advantage"
    on_together = sample(made_one, "cross_entropy")["loss_fn_inputs"]
    assert set(on_together) == {"target_tokens", "weights"}
    assert on_together["weights"]["data"] == [0.0, 0.5, 0.5]
    assert "mask" in made_one.loss_fn_inputs, "the mask stays here, for the drift metrics"


# ------------------------------------------------------------------------- in a run


class Misspelled(Trainer):
    """The base step, with PPO's lower clip threshold misspelled."""

    async def step(self) -> None:
        groups = await self.rollout(self.draw())
        await self.score(groups)
        await self.update(groups, loss="ppo", loss_config={"clip_low": 0.8})


async def test_update_refuses_a_misspelled_setting_before_anything_is_scored_or_sent(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    """The proxy recorded no logprobs, so an update would score the sequences first."""
    proxy, service = fakes
    proxy.default = [replace(made(), inference_logprobs=())]
    harbor(Alternating())
    with pytest.raises(ValueError, match="^the ppo loss takes .*; got clip_low$"):
        await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Misspelled)
    passes = [
        call[0] for call in service.client.calls if call[0] in ("forward", "forward_backward")
    ]
    assert passes == [], "nothing was scored or trained"


class Imitating(Trainer):
    """The base step trained with `cross_entropy`: each graded trial's tokens at weight 1."""

    async def step(self) -> None:
        groups = await self.rollout(self.draw())
        for group in groups:
            for trial in group.graded:
                trial.set_advantage(1.0)
        await self.update(groups, loss="cross_entropy")


async def test_update_with_cross_entropy_weighs_each_target_by_its_advantage(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(Alternating())
    await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Imitating)
    passes = [call for call in service.client.calls if call[0] == "forward_backward"]
    assert passes and all(call[2] == "cross_entropy" for call in passes)
    trained = service.client.sent
    assert trained and all(
        set(one.loss_fn_inputs) == {"target_tokens", "weights"} for one in trained
    )
    assert {w for one in trained for w in one.loss_fn_inputs["weights"].tolist()} == {0.0, 1.0}
