"""Tests for a model on a server that speaks Tinker's API and is not Thinking Machines' own, such
as a SkyRL server or a Modal Spindle deployment: `[models.<role>] base_url` and `api_key_env`,
which the model's client and its lane in the proxy both reach it by, and the checkpoint paths
and prompt logprobs such a server answers with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard.config import Config, ModelTable, shown
from shipyard.groups import Sequence, Trial
from shipyard.jobs.harnesses import Harness
from shipyard.model import Model
from shipyard.models.clients import client_for
from shipyard.models.clients.tinker import TinkerClient
from shipyard.models.endpoints import ProxyEndpoint
from tests.service import FakeService, FlatAnchor

SERVER = "https://spindle.example.modal.run"
#: The variable a run.toml names as holding the server's key.
KEY = "SPINDLE_API_KEY"
#: What a SkyRL server names weights saved for sampling, and a training state (20 Nov 2026).
SKYRL_SAMPLER = "tinker://model_93c58c3d/final"
SKYRL_STATE = "tinker://model_93c58c3d/weights/step-1"
TRIAL = Trial("j", Path("t"), Harness(name="pi"), Path("t"))


def config(**agent: Any) -> Config:
    return Config.model_validate(
        {
            "models": {"agent": {"model": MODEL, **agent}},
            "data": {"train": "d", "batch_size": 1},
            "recipe": {"steps": 1},
        }
    )


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """What each `tinker.ServiceClient` a client made was given to reach its server."""
    found: list[dict[str, str]] = []

    def service_client(metadata: dict[str, str] | None = None, **server: str) -> FakeService:
        found.append(server)
        return FakeService()

    monkeypatch.setattr("shipyard.models.clients.tinker.service_client", service_client)
    return found


# ------------------------------------------------------------------- the server


def test_a_model_is_reached_on_the_server_its_table_names_with_that_servers_key(
    monkeypatch: pytest.MonkeyPatch, reached: list[dict[str, str]]
) -> None:
    monkeypatch.setenv(KEY, "tml-own")
    monkeypatch.setenv("TINKER_BASE_URL", "https://elsewhere.example")
    made = client_for(config(base_url=SERVER, api_key_env=KEY).models["agent"], "r")
    assert reached == [{"base_url": SERVER, "api_key": "tml-own"}]
    assert made.provider == "tinker@spindle.example.modal.run", "billed to its own host"
    plain = client_for(config().models["agent"], "r")
    assert reached[-1] == {}, "the environment's server, as the SDK reads it"
    assert plain.provider == "tinker@elsewhere.example"


def test_two_models_of_one_run_sit_on_two_servers(
    monkeypatch: pytest.MonkeyPatch, reached: list[dict[str, str]]
) -> None:
    monkeypatch.setenv(KEY, "tml-own")
    agent = client_for(ModelTable(model=MODEL, base_url=SERVER, api_key_env=KEY), "r")
    teacher = client_for(ModelTable(model="Qwen/Qwen3-32B"), "r", trains=False)
    assert reached == [{"base_url": SERVER, "api_key": "tml-own"}, {}]
    assert (agent.provider, teacher.provider) == ("tinker@spindle.example.modal.run", "tinker")


def test_a_key_variable_that_is_not_set_is_refused_before_anything_is_reached(
    monkeypatch: pytest.MonkeyPatch, reached: list[dict[str, str]]
) -> None:
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-tinkers-own")
    with pytest.raises(ValueError, match=f"{KEY} is not set: it holds the key of .*{SERVER}"):
        TinkerClient(MODEL, base_url=SERVER, api_key_env=KEY)
    assert reached == [], "Tinker's own key went nowhere"


async def test_the_proxy_is_told_the_server_and_the_variable_never_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(KEY, "tml-own")
    held = config(base_url=SERVER, api_key_env=KEY)
    lane = ProxyEndpoint(held, tmp_path / "dapo__abc").lane("agent", MODEL, max_context=4096)
    client = TinkerClient(MODEL, base_url=SERVER, api_key_env=KEY, service=FakeService())
    lane.backend = await client.serving()
    document = lane.document
    assert (document["base_url"], document["api_key_env"]) == (SERVER, KEY)
    assert "tml-own" not in json.dumps(document), "the proxy reads the key from its environment"
    assert await TinkerClient(MODEL, service=FakeService()).serving() == {}


def test_a_server_is_named_with_its_key_and_only_for_a_model_on_tinker() -> None:
    with pytest.raises(ValueError, match=r"^\[models.agent\] base_url: name the variable"):
        Model("agent", ModelTable(model=MODEL, base_url=SERVER))
    for provider in ("together", "anthropic"):
        elsewhere = ModelTable(model=MODEL, provider=provider, base_url=SERVER, api_key_env=KEY)
        with pytest.raises(ValueError, match=f"speaks Tinker's API.* is on {provider}"):
            Model("agent", elsewhere)
    Model("agent", ModelTable(model=MODEL, api_key_env=KEY))  # another key on Tinker itself


def test_run_toml_shows_the_server_and_the_variable() -> None:
    text = shown(config(base_url=SERVER, api_key_env=KEY))
    assert f'base_url = "{SERVER}"\n' in text and f'api_key_env = "{KEY}"\n' in text
    assert "# base_url = none\n" in shown(config())


# ------------------------------------------------------------------- what such a server answers


async def test_a_skyrl_servers_sampler_weights_start_a_model_the_recipe_does_not_train() -> None:
    service = FakeService()
    served = TinkerClient(MODEL, from_checkpoint=SKYRL_SAMPLER, trains=False, service=service)
    await served.open()
    assert service.calls[-1] == ("sampling", None, SKYRL_SAMPLER)
    with pytest.raises(ValueError, match="training goes on from a training state"):
        TinkerClient(MODEL, from_checkpoint=SKYRL_SAMPLER, service=service)
    with pytest.raises(ValueError, match=f"{SKYRL_STATE} is a training state"):
        TinkerClient(MODEL, from_checkpoint=SKYRL_STATE, trains=False, service=service)
    await TinkerClient(MODEL, from_checkpoint=SKYRL_STATE, service=service).open()
    assert service.calls[-1] == ("from_state", SKYRL_STATE)


class OneShort(FlatAnchor):
    """A sampling client that scores a prompt without Tinker's leading None, as SkyRL's CPU
    backend did (20 Nov 2026): one logprob fewer than the prompt has tokens."""

    def scores(self, found: list[int]) -> list[float | None]:
        return [self.logprob] * (len(found) - 1)


async def test_prompt_logprobs_a_token_short_are_refused_and_not_read_a_token_off() -> None:
    chain = Sequence((1, 2, 3, 4), ((2, 4),), None, (), TRIAL)
    scoring = TinkerClient(MODEL, trains=False, service=FakeService())
    await scoring.open()
    assert (await scoring.compute_logprobs([chain]))[0] == [[-0.1, -0.1]], "Tinker's own shape"
    short = TinkerClient(MODEL, trains=False, service=FakeService(anchor=OneShort()))
    await short.open()
    with pytest.raises(RuntimeError, match="scored 3 of a prompt's 4 tokens"):
        await short.compute_logprobs([chain])
