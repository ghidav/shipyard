"""Tests for `shipyard.Model`: what a model can do follows from its `[models.<role>]` table; the
run trains a model the recipe marks `training`, and serves and scores any other as it is. What
cannot be trained or opened is refused when the run starts, before anything opens."""

from __future__ import annotations

from collections.abc import Sequence as Values
from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard import Config, Model, Trainer
from shipyard.config import ModelTable
from shipyard.groups import Sequence, Trial
from shipyard.jobs.harnesses import Harness
from shipyard.model import Missing, Models
from shipyard.models.backends import API
from shipyard.models.clients.tinker import TinkerClient
from shipyard.run import files
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import GRPO, ONE_STEP, rows, run_toml, train
from tests.service import FakeService, capable
from tests.test_connect import CLAUDE, ON_ANTHROPIC, agent_keys

#: A config's `[models.agent]` without its learning rate, as a recipe that never trains it has it.
UNRATED = {"learning_rate = 2e-5\n": ""}
SAMPLER = "tinker://r/sampler_weights/final"
#: The trial a test's sequences belong to; scoring reads only their tokens.
TRIAL = Trial("j", Path("t"), Harness(name="pi"), Path("t"))


def _model(role: str = "agent", name: str = MODEL, **keys: Any) -> Model:
    """A model as the core builds it from a `[models.<role>]` table of `name` and `keys`."""
    return Model(role, ModelTable(model=name, **keys))


class Scored(Trainer):
    """A recipe that never trains: it scores its agent's rollouts, served as they are."""

    Recipe = GRPO.Recipe

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]

    async def step(self) -> None:
        groups = await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,  # type: ignore[attr-defined]
            attempts=self.recipe.group_size,  # type: ignore[attr-defined]
            modules=None,
        )
        seqs = [s for g in groups for t in g.graded for s in t.sequences]
        scored = await self.agent.logprobs(seqs)
        self.metrics["agent_logprob"] = float(scored[0].mean())


def test_what_a_model_can_do_follows_from_its_table() -> None:
    agent = _model()
    assert (agent.role, agent.name, agent.provider, agent.renderers) == (
        "agent",
        MODEL,
        "tinker",
        [],
    )
    assert agent.trainable and _model(provider="together").trainable
    claude = _model("critic", CLAUDE, provider="anthropic")
    assert not claude.trainable, "its provider only serves it"
    assert agent.training is False, "a model is not trained unless the recipe says so"
    agent.training = True
    assert agent.training, "settable, as in PyTorch, until the run starts"
    claude.training = True
    assert claude.training, "whether it can be trained is checked when the run connects it"
    assert (claude.backend, claude.endpoint, claude.budget) == (None, None, None)
    claude.bind(API(CLAUDE, "anthropic", lambda *billed, **tokens: None))
    assert (claude.endpoint.kind, claude.endpoint.routed) == ("api", f"anthropic/{CLAUDE}")
    with pytest.raises(RuntimeError, match="already open: set `training` in the trainer's"):
        claude.training = False


def test_a_model_with_renderers_whose_provider_is_an_api_is_refused_when_built() -> None:
    with pytest.raises(
        ValueError,
        match=r"^\[models.critic\] renderers: the proxy renders the calls of a model on Tinker "
        f"or Together, and anthropic serves {CLAUDE} to its harnesses itself",
    ):
        _model("critic", CLAUDE, provider="anthropic", renderers=["low"])
    for provider in ("tinker", "together"):
        assert _model(provider=provider, renderers="low").renderers == ["low"], "a name is one"


def test_a_recipe_reads_its_models_by_role_and_a_role_run_toml_does_not_name_is_missing() -> None:
    models = Models({"agent": ModelTable(model=MODEL), "teacher": ModelTable(model=CLAUDE)})
    assert [one.role for one in models.all] == ["agent", "teacher"]
    assert models.unread == ["agent", "teacher"], "listing them reads none"
    assert models["agent"] is models["agent"] and models.unread == ["teacher"]
    with pytest.raises(Missing) as missing:
        models["judge"]
    assert str(missing.value) == (
        "run.toml has no [models.judge], which the recipe reads: it names agent, teacher"
    )
    assert isinstance(missing.value, KeyError) and "judge" not in models


async def test_logprobs_scores_exactly_the_sequences_it_is_given() -> None:
    class Scorer:
        """A backend that scores each target by minus its position, and keeps what it was
        asked."""

        def __init__(self) -> None:
            self.asked: list[list[Sequence]] = []

        async def logprobs(self, sequences: Values[Sequence]) -> list[list[float]]:
            self.asked.append(list(sequences))
            return [[-float(at) for at in one.targets] for one in sequences]

    model = _model()
    with pytest.raises(RuntimeError, match="not connected yet"):
        await model.logprobs([])
    scorer = Scorer()
    model.bind(scorer)  # type: ignore[arg-type]
    one = Sequence((1, 2, 3, 4), ((2, 4),), None, (), TRIAL)
    two = Sequence((1, 2, 3, 4, 5, 6), ((1, 2), (4, 6)), None, (), TRIAL)
    found = await model.logprobs([one, two])
    ((first, second),) = scorer.asked
    assert first is one and second is two, "the sequences themselves, nothing placed in them"
    assert [row.tolist() for row in found] == [[-2.0, -3.0], [-1.0, -4.0, -5.0]]
    assert await model.logprobs([]) == [] and len(scorer.asked) == 1, "none asks nothing"


async def test_a_model_the_run_does_not_train_is_served_and_scored_as_it_is(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **UNRATED)
    directory = await train(home, tmp_path / "runs", Scored)
    assert not [c for c in service.calls if c[0] in ("lora", "from_state")], "no training client"
    assert ("sampling", MODEL, None) in service.calls and len(service.anchor.asked) == 4
    assert set(proxy.weights) == {None}, "its trials are opened on the base model"
    (row,) = rows(directory)
    assert row["trained"] is False and row["agent_logprob"] == pytest.approx(-0.1)
    usage = [one for one in files.read(directory / files.USAGE) if "model" in one]
    assert {(one["role"], one["model"]) for one in usage} == {("agent", MODEL)}
    assert "train" not in files.read_json(directory / files.COSTS)["tokens"]["tinker"][MODEL]
    saved = list(files.read(directory / files.CHECKPOINTS))
    assert [c["training_state"] for c in saved] == [{}, {}], "nothing moved, so nothing to save"


async def test_a_model_served_as_it_is_starts_from_weights_saved_for_sampling(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    proxy, service = fakes
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **UNRATED, **agent_keys(from_checkpoint=SAMPLER))
    await train(home, tmp_path / "runs", Scored)
    assert set(proxy.weights) == {SAMPLER} and ("sampling", None, SAMPLER) in service.calls
    state = agent_keys(from_checkpoint="tinker://r/weights/s")
    refused = run_toml(tmp_path, **ONE_STEP, **UNRATED, **state)
    with pytest.raises(ValueError, match="does not train is served from weights saved"):
        await train(refused, tmp_path / "runs", Scored)
    trained = run_toml(tmp_path, **ONE_STEP, **agent_keys(from_checkpoint=SAMPLER))
    with pytest.raises(ValueError, match="training goes on from a training state"):
        await train(trained, tmp_path / "runs")


async def test_what_the_run_cannot_train_is_refused_before_anything_opens(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    """Each refusal is in tests/test_connect.py; this is the run around one of them."""
    proxy, service = fakes

    class Late(Scored):
        async def step(self) -> None:
            self.agent.training = True

    with pytest.raises(ValueError, match=r"^\[models.agent\]: the recipe trains claude-sonnet"):
        await train(run_toml(tmp_path, **ONE_STEP, **ON_ANTHROPIC), tmp_path / "runs")
    assert service.calls == [] and proxy.placed == [], "nothing opened"
    with pytest.raises(RuntimeError, match="already open: set `training` in the trainer's"):
        await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs", Late)


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
