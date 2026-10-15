"""Test the vocabulary: what it defines and re-exports, the group's rule, no loop imported."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from shipyard import admit, pack, trainer, types
from shipyard.proxy.profiles import PROFILES, Profile
from shipyard.types import (
    Agent,
    Checkpoint,
    Connection,
    Cursor,
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
from tests.trials import write_result


class _Provider(Endpoint):
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    def connection(self, trial: Trial) -> Connection:
        return Connection(model="m")

    def serve(self, path: str) -> None:
        raise RuntimeError("a provider serves no weights")

    async def record(self, job: str, trials: list[Trial]) -> Records | None:
        return None


def _agent(*names: str) -> Agent:
    return Agent(Policy(_Provider()), [harness(name) for name in names])


def _trial(tmp_path: Path, task: str = "t", attempt: int = 0) -> Trial:
    return Trial("j", Path(task), attempt, harness("pi"), tmp_path / f"{task}__{attempt}")


def _member(tmp_path: Path, reward: float | None, *, mask: str | None = None) -> Member:
    found = () if mask else (Sequence((1, 2, 3, 10), ((3, 4),), (-0.5,)),)
    return Member(_trial(tmp_path), Verdict(reward, mask, None), found, 1 if found else 0)


def test_the_unchanged_dataclasses_are_the_same_objects_under_their_old_names() -> None:
    assert pack.Sequence is types.Sequence
    assert admit.Verdict is types.Verdict
    assert trainer.Adam is types.Adam and trainer.Update is types.Update
    assert trainer.COOKBOOK == types.Adam(beta1=0.9, beta2=0.95, eps=1e-8)
    assert types.Datum is __import__("tinker").Datum
    assert types.Task is Path


def test_the_module_imports_no_part_of_the_loop() -> None:
    tree = ast.parse(Path(types.__file__).read_text(encoding="utf-8"))
    imported = sorted(
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("shipyard")
    )
    assert imported == ["shipyard.modules", "shipyard.proxy.profiles", "shipyard.proxy.wire"]


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
    assert (found.name, found.agent, found.effort) == ("pi", "pi@0.85.1", None)
    assert found.profile is PROFILES["pi"]
    assert harness("claude-code", effort="high").effort == "high"
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
        endpoint.serve("tinker://x")
    assert Connection(model=None).env == {} and Connection("m", "http://p/v1", "t").token == "t"
    assert "token" not in repr(Connection("m", "http://p/v1", "secret"))


def test_a_trial_names_itself_after_its_directory_and_reads_its_result(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "alpha", 2)
    assert trial.name == "alpha__2" and trial.result is None
    write_result(trial.dir, reward=1.0)
    assert trial.result is not None and trial.result["verifier_result"] == {
        "rewards": {"reward": 1.0}
    }
    assert Settings().verify is None and Settings(verify=False, concurrency=1).concurrency == 1


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


def test_a_checkpoint_is_the_agent_written_down_with_the_cursor() -> None:
    cursor = Cursor(step=7, updates=5, epoch=1, drawn=3)
    found = Checkpoint("step-7", "tinker://s", "tinker://w", 168.0, cursor, "abcd", ("pi",))
    assert found.cursor.drawn == 3 and found.harness == ("pi",) and found.modules == "abcd"
    final = Checkpoint("final", "tinker://s", "tinker://w", None, cursor)
    assert final.ttl_hours is None and final.modules is None and final.harness == ()
    assert types.Evaluation(0, 0, 4, 8, 7, 1, 0.5).mean == 0.5
