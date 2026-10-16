"""Test the vocabulary: what it defines and re-exports, the group's rule, no loop imported."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from shipyard import admit, types
from shipyard.config import Model
from shipyard.harnesses import PROFILES, Profile
from shipyard.training import trainer
from shipyard.training.client import TinkerClient
from shipyard.types import (
    SERVE_TTL,
    Agent,
    Connection,
    Endpoint,
    Group,
    Harness,
    Member,
    Policy,
    Records,
    RolloutEngine,
    Sequence,
    Settings,
    TrainingClient,
    Trial,
    Verdict,
    harness,
)
from tests.proxies import MODEL
from tests.trainers import FakeService
from tests.trials import write_result


class _Provider(Endpoint):
    async def open(self) -> None: ...

    def close(self) -> None: ...

    def connection(self, trial: Trial) -> Connection:
        return Connection(model="m")

    async def serve(self, path: str | None) -> None:
        raise RuntimeError("a provider serves no weights")

    async def record(self, job: str, trials: list[Trial]) -> Records | None:
        return None


class _Served(_Provider):
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")

    async def serve(self, path: str | None) -> None:
        self.calls.append(("serve", path))


def _agent(*names: str) -> Agent:
    return Agent(Policy(_Provider()), [harness(name) for name in names])


def _client(service: FakeService, **model: Any) -> TinkerClient:
    return TinkerClient(service, Model(name=MODEL, **model))


def _trial(tmp_path: Path, task: str = "t", attempt: int = 0) -> Trial:
    return Trial("j", Path(task), attempt, harness("pi"), tmp_path / f"{task}__{attempt}")


def _member(tmp_path: Path, reward: float | None, *, mask: str | None = None) -> Member:
    found = () if mask else (Sequence((1, 2, 3, 10), ((3, 4),), (-0.5,)),)
    return Member(_trial(tmp_path), Verdict(reward, mask, None), found, 1 if found else 0)


def test_the_unchanged_dataclasses_are_the_same_objects_under_their_old_names() -> None:
    assert trainer.Trainer.ADAM == types.Adam(beta1=0.9, beta2=0.95, eps=1e-8)
    assert types.Datum is __import__("tinker").Datum
    assert types.Task is Path


def test_the_module_imports_no_part_of_the_loop() -> None:
    tree = ast.parse(Path(types.__file__).read_text(encoding="utf-8"))
    imported = sorted(
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("shipyard")
    )
    assert imported == ["shipyard.harnesses", "shipyard.modules", "shipyard.proxy.wire"]


def test_a_part_missing_a_method_cannot_be_built() -> None:
    class Engine(RolloutEngine):
        pass

    class Client(TrainingClient):
        pass

    for incomplete in (Engine, Client):
        with pytest.raises(TypeError, match="abstract"):
            incomplete()  # type: ignore[abstract]
    assert _Provider().budget is None


def test_a_built_in_harness_carries_its_profile_and_a_custom_one_the_generic() -> None:
    found = harness("pi@0.85.1")
    assert (found.name, found.agent) == ("pi", "pi@0.85.1")
    assert found.profile is PROFILES["pi"]
    custom = harness("acp:x.y:Z@1")
    assert custom.name == "acp:x.y:Z@1" and custom.profile == Profile()
    assert Harness("mine", "mine@2", Profile(dialect="anthropic")).profile.dialect == "anthropic"


def test_an_agent_is_a_policy_under_harnesses_with_modules_and_varies_by_replace() -> None:
    agent = _agent("claude-code", "pi")
    assert [one.name for one in agent.harness] == ["claude-code", "pi"] and agent.modules is None
    assert agent.policy.client is None and agent.policy.start is None
    other = replace(agent, harness=[harness("pi")])
    assert other.policy is agent.policy and len(agent.harness) == 2
    with pytest.raises(FrozenInstanceError):
        agent.modules = None  # type: ignore[misc]


async def test_a_provider_endpoint_records_nothing_and_serves_nothing() -> None:
    endpoint = _Provider()
    assert await endpoint.record("j", []) is None and endpoint.check("j", {}) is None
    with pytest.raises(RuntimeError, match="serves no weights"):
        await endpoint.serve("tinker://x")
    assert Connection(model=None).env == {} and Connection("m", "http://p/v1", "t").token == "t"
    assert "token" not in repr(Connection("m", "http://p/v1", "secret"))


def test_a_trial_names_itself_after_its_directory(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "alpha", 2)
    assert trial.name == "alpha__2" and admit.result_of(trial.dir) is None
    write_result(trial.dir, reward=1.0)
    assert admit.result_of(trial.dir)["verifier_result"] == {"rewards": {"reward": 1.0}}
    assert Settings().verify is True and Settings(verify=False, env={"A": "1"}).env == {"A": "1"}


def test_a_group_reports_the_variance_of_its_shaped_rewards_over_the_measured(
    tmp_path: Path,
) -> None:
    agent = _agent("pi")
    members = [
        _member(tmp_path, 1.0),
        _member(tmp_path, 0.0),
        _member(tmp_path, None, mask="timeout"),
    ]
    group = Group(Path("t"), agent, agent.harness[0], members)
    assert [one.measured for one in members] == [True, True, False]
    assert len(group.measured) == 2 and group.shaped == [] and group.reward_variance == 0.0
    group.shaped = [1.0, 0.0, None]
    assert group.reward_variance == 0.25
    group.shaped = [1.0, 1.0, None]
    assert group.reward_variance == 0.0, "all equal"
    group.shaped = [0.5, None, None]
    assert group.reward_variance == 0.0, "a lone member"
    group.shaped = [None, None, None]
    assert group.reward_variance == 0.0
    assert members[0].sequences[0].targets == (3,) and members[2].sequences == ()


async def test_a_policy_opens_its_endpoint_serving_the_start_and_closes_both() -> None:
    endpoint, service = _Served(), FakeService()
    async with Policy(endpoint, _client(service)) as policy:
        assert endpoint.calls == ["open", ("serve", "tinker://fake/sampler_weights/start")]
        assert service.client.calls == [("save_weights_for_sampler", "start", SERVE_TTL)]
        assert await policy.sync("sample-1") == "tinker://fake/sampler_weights/sample-1"
    assert endpoint.calls[2:] == [("serve", "tinker://fake/sampler_weights/sample-1"), "close"]
    assert service.closed == [("success", None)]


async def test_a_resumed_policy_serves_its_checkpoint_and_a_providers_serves_nothing() -> None:
    endpoint, service = _Served(), FakeService()
    async with Policy(endpoint, _client(service), start="tinker://w"):
        pass
    assert endpoint.calls == ["open", ("serve", "tinker://w"), "close"]
    assert service.client.calls == [], "the checkpoint's sampler path is served as it is"
    endpoint = _Served()
    async with Policy(endpoint) as policy:
        with pytest.raises(RuntimeError, match="no weights to sync"):
            await policy.sync("sample-1")
    assert endpoint.calls == ["open", "close"]


@pytest.mark.parametrize(
    ("failed", "closed"),
    [
        (RuntimeError("the backend went away"), ("errored", "RuntimeError: the backend went away")),
        (KeyboardInterrupt(), ("interrupted", "KeyboardInterrupt: ")),
    ],
)
async def test_a_policy_closes_its_client_with_how_the_run_ended(
    failed: BaseException, closed: tuple[str, str]
) -> None:
    endpoint, service = _Served(), FakeService()
    with pytest.raises(type(failed)):
        async with Policy(endpoint, _client(service)):
            raise failed
    assert endpoint.calls[-1] == "close" and service.closed == [closed]


async def test_an_anchored_resumed_policy_takes_the_anchor_before_loading_its_state() -> None:
    endpoint, service = _Served(), FakeService()
    client = _client(service, from_checkpoint="tinker://c")
    policy = Policy(endpoint, client, start="tinker://w", state="tinker://s", anchored=True)
    assert policy.anchor is None and client.client is None
    async with policy:
        assert policy.anchor is service.anchor and client.client is service.client
    published = "tinker://fake/sampler_weights/anchor"
    assert service.calls == [("from_state", "tinker://c"), ("sampling", MODEL, published)]
    assert service.client.calls == [
        ("save_weights_for_sampler", "anchor", SERVE_TTL),
        ("load_state_with_optimizer", "tinker://s"),
        ("consumed", "load"),
    ], "the anchor is the weights as opened, before the checkpoint's state replaces them"
    assert endpoint.calls == ["open", ("serve", "tinker://w"), "close"]
    assert service.closed == [("success", None)]


async def test_a_policy_that_cannot_enter_closes_the_client_it_opened() -> None:
    class Down(_Served):
        async def open(self) -> None:
            raise RuntimeError("the proxy never came up")

    endpoint, service = Down(), FakeService()
    with pytest.raises(RuntimeError, match="never came up"):
        async with Policy(endpoint, _client(service)):
            raise AssertionError("not entered")
    assert endpoint.calls == ["close"] and service.calls == [("lora", MODEL, 32)]
    assert service.closed == [("errored", "RuntimeError: the proxy never came up")]


def test_an_evaluation_carries_its_mean() -> None:
    assert types.Evaluation(0, 0, 4, 8, 7, 1, 0.5).reward_mean == 0.5
