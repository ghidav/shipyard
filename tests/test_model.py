"""Tests for `shipyard.Model`: what a model can do follows from what it is, the run trains a
model made with `train=True` (its agent, unless the recipe says otherwise), and serves any
other as it is. What cannot be trained or opened is refused when the run starts, before
anything opens."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from proxies import MODEL
from shipyard import Blueprint, Model, Trainer
from shipyard.models import context
from shipyard.models.clients.tinker import TinkerClient
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.directory import open_run
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import ONE_STEP, blueprint, rows, train
from tests.service import FakeService, capable

#: A blueprint's `[recipe]` without its learning rate, as a recipe that never updates has it.
UNRATED = {"learning_rate = 2e-5\n": ""}
SAMPLER = "tinker://r/sampler_weights/final"
HARNESS = 'harness = "pi@0.85.1"'


def starting(checkpoint: str) -> dict[str, str]:
    """Return the blueprint replacement that starts the agent from `checkpoint`."""
    return {HARNESS: f'{HARNESS}\nfrom_checkpoint = "{checkpoint}"'}


class Scored(Trainer):
    """A trainer that never updates: it scores its agent's rollouts, served as they are."""

    def __init__(self, config: Blueprint, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent.training = False

    async def step(self) -> None:
        groups = await self.rollout(self.draw())
        seqs = [s for g in groups for t in g.graded for s in t.sequences]
        scored = await self.agent.logprobs(seqs)
        self.metrics["agent_logprob"] = float(scored[0].mean())  # type: ignore[union-attr]


def test_what_a_model_can_do_follows_from_what_it_is() -> None:
    assert Model(MODEL).trainable and Model(MODEL, provider="together").trainable
    assert Model("tinker://r/weights/step-3").trainable, "a training state goes on training"
    assert not Model(SAMPLER).trainable, "weights saved for sampling are sampled, not trained"
    assert not Model("claude-sonnet-5-5", provider="anthropic").trainable
    assert Model(MODEL).training is False, "a model is not trained unless asked"
    model = Model(MODEL, train=True)
    assert model.training is True
    model.training = False
    assert model.training is False, "settable, as in PyTorch, until the run starts"
    assert Model("claude-sonnet-5-5", provider="anthropic", train=True).training, (
        "whether it can be trained is checked when the run starts"
    )
    with pytest.raises(ValueError, match="needs its name"):
        Model("")


async def test_an_agent_the_run_does_not_train_is_served_and_scored_as_it_is(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    home = blueprint(tmp_path, **ONE_STEP, **UNRATED)
    directory = await train(home, tmp_path / "runs", Scored)
    assert not [c for c in service.calls if c[0] in ("lora", "from_state")], "no training client"
    assert ("sampling", MODEL, None) in service.calls and len(service.anchor.asked) == 4
    assert proxy.swapped == [None], "the proxy serves the base model"
    (row,) = rows(directory)
    assert row["trained"] is False and row["agent_logprob"] == pytest.approx(-0.1)
    usage = [one for one in files.read(directory / files.USAGE) if "model" in one]
    assert {(one["role"], one["model"]) for one in usage} == {("agent", MODEL)}
    assert "train" not in files.read_json(directory / files.COSTS)["tokens"]["tinker"][MODEL]
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [c["training_state"] for c in saved] == [{}, {}], "nothing moved, so nothing to save"


async def test_an_agent_served_as_it_is_starts_from_weights_saved_for_sampling(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    home = blueprint(tmp_path, **ONE_STEP, **UNRATED, **starting(SAMPLER))
    await train(home, tmp_path / "runs", Scored)
    assert proxy.swapped == [SAMPLER] and ("sampling", None, SAMPLER) in service.calls
    refused = blueprint(tmp_path, **ONE_STEP, **UNRATED, **starting("tinker://r/weights/s"))
    with pytest.raises(ValueError, match="does not train its agent serves it from weights saved"):
        await train(refused, tmp_path / "runs", Scored)
    trained = blueprint(tmp_path, **ONE_STEP, **starting(SAMPLER))
    with pytest.raises(ValueError, match="training goes on from a training state"):
        await train(trained, tmp_path / "runs")


async def test_what_the_run_cannot_train_is_refused_before_anything_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    """Each refusal is in tests/test_connect.py; this is the run around one of them."""
    proxy, service = fakes

    class Late(Trainer):
        async def step(self) -> None:
            self.agent.training = False

    claude = {'model = "Qwen/Qwen3-8B"': 'model = "claude-sonnet-5-5"\nprovider = "anthropic"'}
    with pytest.raises(ValueError, match="anthropic serves it, .* `self.agent.training = False`"):
        await train(blueprint(tmp_path, **ONE_STEP, **claude), tmp_path / "runs")
    assert service.calls == [] and proxy.placed == [], "nothing opened"
    with pytest.raises(RuntimeError, match="already open: set `training` in the trainer's"):
        await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Late)


async def test_a_client_that_only_serves_scores_through_a_sampling_client_on_its_weights() -> None:
    sampled = f"{MODEL}:sampling-nvfp4"
    service = FakeService(capabilities=capable((MODEL, True), (sampled, False)))
    serving = TinkerClient(sampled, trains=False, service=service)
    await serving.open()
    assert serving.client is None, "a model sampled alone opens no training client"
    assert service.calls == [("sampling", sampled, None)] and serving.scorer is service.anchor
    assert await serving.publish("start") is None, "the base model, as it is"
    started = TinkerClient(MODEL, from_checkpoint=SAMPLER, trains=False, service=service)
    await started.open()
    assert service.calls[-1] == ("sampling", None, SAMPLER), "scored at the weights it serves"
    assert await started.publish("start") == SAMPLER


def test_a_checkpoint_from_before_roles_restores_as_the_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, config = open_run(blueprint(tmp_path), root=tmp_path / "runs")
    state = f"{files.STATES}/step-1.json"
    files.write_json(directory / state, {"steps": 1, "updates": 1, "drawn": 1})
    weights = "tinker://fake/weights/step-1"
    trainer = Trainer(config, directory)
    old = {"tag": "step-1", "step": 1, "state": state, "weights": weights, "sampler": None}
    trainer.restore(directory, Checkpoint.of(old))
    assert trainer.state.updates == {"agent": 1} and trainer._training_state == weights
    assert Trainer.State(updates=0).updates == {}  # type: ignore[arg-type]


async def test_a_model_that_reads_other_tokens_is_refused_before_the_agent_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy, service = fakes
    monkeypatch.setattr("shipyard.models.context.shares_tokenizer", lambda model, other: False)

    class Taught(Trainer):
        def __init__(self, config: Blueprint, directory: Path) -> None:
            super().__init__(config, directory)
            self.teacher = Model("Qwen/Qwen3-32B")

    with pytest.raises(ValueError, match="^teacher = Model.* reads other token ids than the agent"):
        await train(blueprint(tmp_path, **ONE_STEP), tmp_path / "runs", Taught)
    assert service.calls == [("sampling", "Qwen/Qwen3-32B", None)], "the agent never opened"
    assert proxy.placed == [] and [status for status, _ in service.closed] == ["errored"]


def test_two_models_share_a_tokenizer_when_they_read_the_same_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabs = {"a": {"x": 0, "y": 1}, "b": {"x": 0, "y": 1}, "c": {"x": 1, "y": 0}}
    loaded: list[str] = []

    def tokenizer(model: str) -> Any:
        loaded.append(model)
        return SimpleNamespace(get_vocab=lambda: vocabs[model])

    monkeypatch.setattr(context, "tokenizer", tokenizer)
    assert context.shares_tokenizer("a", "a") and loaded == [], "one model needs no download"
    assert context.shares_tokenizer("a", "b") and not context.shares_tokenizer("a", "c")
