"""Tests for `shipyard.models.backends.connect`: each model run.toml names gets its backend by its
provider, each model on Tinker or Together a session of its own and a lane in the run's proxy, or
the run is refused, before anything opens."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard.model import Models
from shipyard.models.backends import API, Connected, connect
from shipyard.models.clients.tinker import SERVE_TTL
from shipyard.run.directory import open_run
from tests.fakes import START, FakeProxy
from tests.runs import GRPO, run_toml
from tests.service import FakeService

CLAUDE = "claude-sonnet-5-5"
#: A model the fake Tinker service serves beside the agent's, for sampling alone.
TEACHER = "Qwen/Qwen3-32B"
#: The config replacement that makes the agent a provider's model.
ON_ANTHROPIC = {f'model = "{MODEL}"': f'model = "{CLAUDE}"\nprovider = "anthropic"'}


def table(role: str, model: str, **keys: Any) -> str:
    """Return a `[models.<role>]` table of `model`, with `keys`, as run.toml writes it."""
    written = "".join(f"{key} = {json.dumps(value)}\n" for key, value in keys.items())
    return f'\n[models.{role}]\nmodel = "{model}"\n{written}'


def agent_keys(**keys: Any) -> dict[str, str]:
    """Return the config replacement that adds `keys` to `[models.agent]`."""
    written = "".join(f"{key} = {json.dumps(value)}\n" for key, value in keys.items())
    return {"lora_rank = 32\n": f"lora_rank = 32\n{written}"}


def connected(
    tmp_path: Path,
    *tables: str,
    trained: tuple[str, ...] = ("agent",),
    states: Mapping[str, str] | None = None,
    **replacements: str,
) -> tuple[Connected, Models]:
    """Connect the models of a run whose run.toml adds `tables`, the recipe training the
    `trained` ones, and return what connected them and the models."""
    home = run_toml(tmp_path, *tables, **replacements)
    directory, config = open_run(home, root=tmp_path / "runs", trainer=GRPO)
    models = Models(config.models)
    for role in trained:
        models[role].training = True
    made = connect(
        config, directory, models.all, lambda *billed, **tokens: None, training_states=states
    )
    return made, models


async def test_each_model_is_bound_by_its_provider_and_each_lane_bills_as_what_it_serves(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    made, models = connected(
        tmp_path, table("teacher", TEACHER), table("critic", CLAUDE, provider="anthropic")
    )
    agent, teacher, critic = models.all
    ours, theirs = made.sessions
    assert (agent.backend, teacher.backend) == (ours, theirs)
    assert (ours.trains, theirs.trains) == (True, False), "a session trains only a trained model"
    assert list(made.proxy.lanes) == ["agent", "teacher"], "a lane each, named by its role"
    for model, session in ((agent, ours), (teacher, theirs)):
        assert model.endpoint is session.endpoint and session.endpoint.opened is session
        assert (model.endpoint.provider, model.endpoint.kind) == ("tinker", "tinker")
    assert isinstance(critic.backend, API)
    assert (critic.endpoint.provider, critic.endpoint.kind) == ("anthropic", "api")
    assert service.calls == [] and proxy.placed == [], "nothing opens until the run starts"


async def test_a_model_on_tinker_opens_beside_a_providers_agent(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    made, models = connected(tmp_path, table("teacher", TEACHER), trained=(), **ON_ANTHROPIC)
    agent, teacher = models.all
    (session,) = made.sessions
    assert isinstance(agent.backend, API) and teacher.backend is session
    assert list(made.proxy.lanes) == ["teacher"], "the proxy serves the model on Tinker alone"
    alone, _ = connected(tmp_path, trained=(), **ON_ANTHROPIC)
    assert alone.sessions == [] and alone.proxy is None, "no model to serve, no proxy"


async def test_every_model_opens_a_session_the_untrained_first_then_the_trained_then_the_proxy(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model the run cannot open refuses it before a trained model's GPUs start; the proxy
    starts once every session has opened, and then each model serves its starting weights."""
    proxy, service = fakes
    started: list[tuple[list[Any], list[Any]]] = []
    local = proxy.local

    async def starting(settings: Any, **named: Any) -> FakeProxy:
        started.append((list(service.calls), list(service.client.calls)))
        return await local(settings, **named)

    monkeypatch.setattr(proxy, "local", starting)
    made, _ = connected(tmp_path, table("teacher", TEACHER))
    async with made:
        opened = [("sampling", TEACHER, None), ("lora", MODEL, 32)]
        assert service.calls == opened, "the teacher, named after the agent, opens first"
        ((calls, saves),) = started
        assert calls == opened and not any(c[0] == "save_weights_for_sampler" for c in saves), (
            "the proxy starts once both have, before any serves"
        )
        ((_, settings, _),) = proxy.placed
        assert [one["name"] for one in settings["lanes"]] == ["agent", "teacher"]
        told = {name: lane.told for name, lane in made.proxy.lanes.items()}
        assert told == {"teacher": None, "agent": START}, (
            "each serves its starting weights: the teacher its base model, the agent a save"
        )
        assert ("save_weights_for_sampler", "start", SERVE_TTL) in service.client.calls
    assert [status for status, _ in service.closed] == ["success", "success"], (
        "each session ends with the run"
    )


async def test_any_number_of_models_train_each_on_its_own_training_state(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    _, service = fakes
    service.capabilities = None  # the critic's model is not one the fake service lists
    states = {"agent": "tinker://r/weights/step-2", "critic": "tinker://r/weights/critic-2"}
    critic = table("critic", "Qwen/Qwen3-4B", learning_rate=1e-5)
    made, _ = connected(
        tmp_path, critic, table("teacher", TEACHER), trained=("agent", "critic"), states=states
    )
    assert [(one.trains, one.training_state) for one in made.sessions] == [
        (True, states["agent"]),
        (True, states["critic"]),
        (False, None),
    ]
    async with made:
        loaded = [
            call[1] for call in service.client.calls if call[0] == "load_state_with_optimizer"
        ]
        assert loaded == [states["agent"], states["critic"]], "each opens on its own"
        assert [call[0] for call in service.calls] == ["sampling", "lora", "lora"]


@pytest.mark.parametrize(
    ("tables", "trained", "replacements", "refused"),
    [
        (
            [table("critic", CLAUDE, provider="anthropic")],
            ("agent", "critic"),
            {},
            r"^\[models.critic\]: the recipe trains claude-sonnet-5-5, and anthropic serves it, "
            "so no run can train it",
        ),
        (
            [],
            ("agent",),
            agent_keys(from_checkpoint="tinker://r/sampler_weights/final"),
            "holds weights saved for sampling, and training goes on from a training state",
        ),
        (
            [table("teacher", TEACHER, from_checkpoint="tinker://r/weights/step-3")],
            ("agent",),
            {},
            "a model the recipe does not train is served from weights saved for sampling",
        ),
    ],
    ids=["a-model-its-provider-serves", "sampler-weights-trained", "a-training-state-served"],
)
async def test_what_the_run_cannot_train_or_open_is_refused_before_anything_opens(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    tables: list[str],
    trained: tuple[str, ...],
    replacements: dict[str, str],
    refused: str,
) -> None:
    proxy, service = fakes
    with pytest.raises(ValueError, match=refused):
        connected(tmp_path, *tables, trained=trained, **replacements)
    assert service.calls == [] and proxy.placed == []
