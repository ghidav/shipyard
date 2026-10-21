"""Tests for `shipyard.models.backends.connect`: each model a trainer holds gets its backend by its
provider, or the run is refused, before anything opens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard import Model
from shipyard.models.backends import connect
from shipyard.run.directory import open_run
from tests.fakes import FakeProxy
from tests.runs import blueprint
from tests.service import FakeService

CLAUDE = "claude-sonnet-5-5"
#: The blueprint replacement that makes the agent a provider's model.
ON_ANTHROPIC = {f'model = "{MODEL}"': f'model = "{CLAUDE}"\nprovider = "anthropic"'}


def _connected(tmp_path: Path, held: list[tuple[str, Model]], **replacements: str) -> Any:
    directory, config = open_run(blueprint(tmp_path, **replacements), root=tmp_path / "runs")
    return connect(config, directory, held, lambda *billed, **tokens: None)


async def test_each_model_is_bound_by_its_provider_and_each_lane_bills_as_what_it_serves(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    proxy, service = fakes
    agent, teacher = Model(MODEL, train=True), Model("Qwen/Qwen3-32B")
    critic = Model(CLAUDE, provider="anthropic")
    connected = _connected(tmp_path, [("agent", agent), ("teacher", teacher), ("critic", critic)])
    session, (sampling,) = connected.session, connected.others
    assert agent.endpoint is session.endpoint and session.endpoint.opened is session
    assert (agent.endpoint.provider, agent.endpoint.kind) == ("tinker", "tinker")
    assert teacher.endpoint is sampling.endpoint and sampling.endpoint.opened is sampling
    assert (critic.endpoint.provider, critic.endpoint.kind) == ("anthropic", "provider")
    assert service.calls == [] and proxy.placed == [], "nothing opens until the run starts"


async def test_a_model_on_tinker_opens_beside_a_providers_agent(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService]
) -> None:
    agent, teacher = Model(CLAUDE, provider="anthropic"), Model("Qwen/Qwen3-32B")
    connected = _connected(tmp_path, [("agent", agent), ("teacher", teacher)], **ON_ANTHROPIC)
    assert connected.session is None and agent.endpoint.kind == "provider"
    assert list(connected.proxy.lanes) == ["teacher"], "the proxy serves the model on Tinker"
    (sampling,) = connected.others
    assert sampling.reader is None, "a provider's agent writes no token ids to score"


@pytest.mark.parametrize(
    ("held", "replacements", "refused"),
    [
        (
            [("agent", Model(CLAUDE, provider="anthropic", train=True))],
            ON_ANTHROPIC,
            "anthropic serves it, so no run can train it",
        ),
        (
            [("agent", Model("tinker://r/sampler_weights/final", train=True))],
            {},
            "it is weights saved for sampling, so no run can train it",
        ),
        (
            [("agent", Model(MODEL, train=True)), ("critic", Model("Qwen/Qwen3-4B", train=True))],
            {},
            r"critic = Model\('Qwen/Qwen3-4B'\) is training, and a run trains one model",
        ),
        (
            [("agent", Model(MODEL, train=True)), ("teacher", Model(MODEL, provider="together"))],
            {},
            "a model on Together opens as the agent alone",
        ),
    ],
    ids=["a-providers-agent", "sampler-weights", "a-second-trained-model", "beside-on-together"],
)
async def test_what_the_run_cannot_train_or_open_is_refused_before_anything_opens(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    held: list[tuple[str, Model]],
    replacements: dict[str, str],
    refused: str,
) -> None:
    proxy, service = fakes
    with pytest.raises(ValueError, match=refused):
        _connected(tmp_path, held, **replacements)
    assert service.calls == [] and proxy.placed == []
