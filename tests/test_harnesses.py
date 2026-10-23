"""Tests for `shipyard.jobs.harnesses.Harness`: how a harness is given, what each one dials to reach
the model, whether it can run with no network, its name and version, and whether its log says its
last call failed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shipyard.config import ConfigError, load
from shipyard.groups import Connection, Where
from shipyard.jobs.harnesses import Harness
from tests.runs import GRPO, run_toml

#: The proxy as a trial's container reaches it, and as this machine does.
WHERE = Where(
    "https://x.trycloudflare.com/r/trial/t/v1",
    local="http://127.0.0.1:8000/r/trial/t/v1",
    token="k",
    host="x.trycloudflare.com",
)
MODEL = "Qwen/Qwen3-8B"


def given(tmp_path: Path, harness: str) -> Harness:
    """Return the harness a recipe's `[recipe] harness = <harness>` reads as."""
    replaced = {'harness = "pi@0.85.1"': f"harness = {harness}"}
    return load(run_toml(tmp_path, **replaced), GRPO.Recipe).recipe.harness


def test_a_harness_reads_from_its_name_alone_or_from_a_table(tmp_path: Path) -> None:
    assert Harness.model_validate("pi@0.85.1") == Harness(name="pi@0.85.1")
    assert Harness.model_validate(" pi@0.85.1 ").name == "pi@0.85.1", "its name is trimmed"
    assert given(tmp_path, '"pi@0.85.1"') == Harness(name="pi@0.85.1")
    table = given(
        tmp_path, '{ name = "pi@0.85.1", env = { A = "1" }, kwargs = { thinking = "high" } }'
    )
    assert table == Harness(name="pi@0.85.1", env={"A": "1"}, kwargs={"thinking": "high"})
    assert Harness(name="pi").env == {} and Harness(name="pi").kwargs == {}, "neither, unless given"


def test_a_harness_refuses_a_blank_name_and_a_key_it_does_not_know(tmp_path: Path) -> None:
    """Harbor reads an unnamed agent as its oracle, which runs the task's reference solution."""
    for blank in ("", "   "):
        with pytest.raises(ValidationError, match="a harness needs its name"):
            Harness.model_validate(blank)
        with pytest.raises(ValidationError, match="a harness needs its name"):
            Harness(name=blank)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Harness.model_validate({"name": "pi", "version": "0.85.1"})
    for written, problem in (
        ('""', "[recipe.harness] name: a harness needs its name, such as `pi@0.85.1`"),
        ('{ name = "pi", version = "1" }', "[recipe.harness] version: unknown key"),
        ("{ env = {} }", "[recipe.harness] name: field required"),
    ):
        with pytest.raises(ConfigError) as refused:
            given(tmp_path, written)
        assert refused.value.problems == [problem]


def test_each_harness_dials_the_proxy_on_its_own_wire() -> None:
    assert Harness(name="pi@0.85.1").dial(WHERE, MODEL, fill_context=True) == Connection(
        model=f"openai/{MODEL}",
        env={"OPENAI_BASE_URL": WHERE.address, "OPENAI_API_KEY": "k"},
        kwargs={"model_api": "openai-completions"},
        host=WHERE.host,
    )
    claude = Harness(name="claude-code").dial(WHERE, MODEL, fill_context=True)
    assert claude.model == f"anthropic/{MODEL}"
    assert claude.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": "https://x.trycloudflare.com/r/trial/t",
        "ANTHROPIC_API_KEY": "k",
    }, "it appends /v1 itself, and leaves its context uncompacted when the run fills it"
    plain = Harness(name="claude-code").dial(WHERE, MODEL, fill_context=False)
    assert "DISABLE_COMPACT" not in plain.env
    opencode = Harness(name="opencode@1.18.35").dial(WHERE, MODEL, fill_context=False)
    assert opencode.model == f"shipyard/{MODEL}"
    assert set(opencode.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    unknown = Harness(name="my.agents:Agent").dial(WHERE, MODEL, fill_context=False)
    assert (unknown.model, unknown.kwargs) == (f"openai/{MODEL}", {}), "the generic profile"


def test_a_harness_whose_calls_leave_from_here_dials_this_machines_address() -> None:
    terminus = Harness(name="terminus-2")
    assert terminus.runs_without_network and not Harness(name="pi").runs_without_network
    dialled = terminus.dial(WHERE, MODEL, fill_context=False)
    assert dialled.env["OPENAI_BASE_URL"] == WHERE.local
    assert dialled.kwargs == {"api_base": WHERE.local, "llm_call_kwargs": {"api_key": "k"}}


def test_a_harness_names_its_agent_and_the_version_it_pins() -> None:
    pinned = Harness(name="pi@0.85.1")
    assert (pinned.name, pinned.bare, pinned.version) == ("pi@0.85.1", "pi", "0.85.1")
    assert Harness(name="pi").version is None and Harness(name="pi@").version is None
    path = Harness(name="my.agents:Agent@x")
    assert (path.bare, path.version) == ("my.agents:Agent@x", None), "an import path stays whole"


def test_a_harness_reads_its_own_log_for_a_last_call_that_failed(tmp_path: Path) -> None:
    failed, settled, unlogged = (tmp_path / name for name in ("failed", "settled", "unlogged"))
    for trial, stop in ((failed, "error"), (settled, "stop")):
        (trial / "agent").mkdir(parents=True)
        message = {"role": "assistant", "stopReason": stop}
        (trial / "agent" / "pi.txt").write_text(
            json.dumps({"type": "message_end", "message": message})
        )
        (trial / "agent" / "claude-code.txt").write_text("anything")
    pi = Harness(name="pi@0.85.1")
    assert pi.last_call_failed(failed) and not pi.last_call_failed(settled)
    assert not pi.last_call_failed(unlogged), "no log, nothing to read"
    assert not Harness(name="claude-code").last_call_failed(failed), (
        "a harness whose log is not read"
    )
