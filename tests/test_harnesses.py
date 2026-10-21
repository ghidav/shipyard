"""Tests for `shipyard.jobs.harnesses.Harness`: what each harness dials to reach the model, whether
it can run with no network, its name and version, and whether its log says its last call failed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipyard.groups import Connection, Where
from shipyard.jobs.harnesses import harness, resolved
from tests.fakes import FakeTrials, Harbor
from tests.test_modules import CARRYING, _batch, _carrying, _roll

#: The proxy as a trial's container reaches it, and as this machine does.
WHERE = Where(
    "https://x.trycloudflare.com/r/trial/t/v1",
    local="http://127.0.0.1:8000/r/trial/t/v1",
    token="k",
    host="x.trycloudflare.com",
)
MODEL = "Qwen/Qwen3-8B"


def test_each_harness_dials_the_proxy_on_its_own_wire() -> None:
    assert harness("pi@0.85.1").dial(WHERE, MODEL, fill_context=True) == Connection(
        model=f"openai/{MODEL}",
        env={"OPENAI_BASE_URL": WHERE.address, "OPENAI_API_KEY": "k"},
        kwargs={"model_api": "openai-completions"},
        host=WHERE.host,
    )
    claude = harness("claude-code").dial(WHERE, MODEL, fill_context=True)
    assert claude.model == f"anthropic/{MODEL}"
    assert claude.env == {
        "DISABLE_COMPACT": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "ANTHROPIC_BASE_URL": "https://x.trycloudflare.com/r/trial/t",
        "ANTHROPIC_API_KEY": "k",
    }, "it appends /v1 itself, and leaves its context uncompacted when the run fills it"
    plain = harness("claude-code").dial(WHERE, MODEL, fill_context=False)
    assert "DISABLE_COMPACT" not in plain.env
    opencode = harness("opencode@1.18.35").dial(WHERE, MODEL, fill_context=False)
    assert opencode.model == f"shipyard/{MODEL}"
    assert set(opencode.env) == {"SHIPYARD_BASE_URL", "SHIPYARD_API_KEY"}
    unknown = harness("my.agents:Agent").dial(WHERE, MODEL, fill_context=False)
    assert (unknown.model, unknown.kwargs) == (f"openai/{MODEL}", {}), "the generic profile"


def test_a_harness_whose_calls_leave_from_here_dials_this_machines_address() -> None:
    terminus = harness("terminus-2")
    assert terminus.runs_without_network and not harness("pi").runs_without_network
    dialled = terminus.dial(WHERE, MODEL, fill_context=False)
    assert dialled.env["OPENAI_BASE_URL"] == WHERE.local
    assert dialled.kwargs == {"api_base": WHERE.local, "llm_call_kwargs": {"api_key": "k"}}


def test_a_harness_names_its_agent_and_the_version_it_pins() -> None:
    assert (harness("pi@0.85.1").name, harness("pi@0.85.1").version) == ("pi", "0.85.1")
    assert harness("pi").version is None and harness("pi@").version is None
    path = harness("my.agents:Agent@x")
    assert (path.name, path.version) == ("my.agents:Agent@x", None), "an import path stays whole"


def test_a_harness_reads_its_own_log_for_a_last_call_that_failed(tmp_path: Path) -> None:
    failed, settled, unlogged = (tmp_path / name for name in ("failed", "settled", "unlogged"))
    for trial, stop in ((failed, "error"), (settled, "stop")):
        (trial / "agent").mkdir(parents=True)
        message = {"role": "assistant", "stopReason": stop}
        (trial / "agent" / "pi.txt").write_text(
            json.dumps({"type": "message_end", "message": message})
        )
        (trial / "agent" / "claude-code.txt").write_text("anything")
    pi = harness("pi@0.85.1")
    assert pi.last_call_failed(failed) and not pi.last_call_failed(settled)
    assert not pi.last_call_failed(unlogged), "no log, nothing to read"
    assert not harness("claude-code").last_call_failed(failed), "a harness whose log is not read"


def test_a_bare_name_of_a_listed_harness_runs_it_at_its_version_and_any_other_as_named() -> None:
    listed = ["pi@0.85.1", "claude-code@2.1.0"]
    assert resolved("pi", listed) == "pi@0.85.1"
    assert resolved("claude-code@2.1.0", listed) == "claude-code@2.1.0"
    assert resolved("pi@0.86.0", listed) == "pi@0.86.0", "another version, as named"
    assert resolved("opencode", listed) == "opencode", "a harness the blueprint lists not"
    with pytest.raises(ValueError, match="names pi@0.85.1 and pi@0.86.0 .* give its version"):
        resolved("pi", ["pi@0.85.1", "pi@0.86.0"])


async def test_a_rollout_of_the_agent_through_a_bare_name_installs_the_pinned_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, CARRYING.replace('modules = "modules"\n', ""), modules=None)
    runner = harbor(FakeTrials())
    await _roll(home, tmp_path / "runs", (_batch(tmp_path, "alpha"), {"harness": "pi"}))
    assert {(one.agent.name, one.agent.kwargs.get("version")) for one in runner.configs} == {
        ("pi", "0.85.1")
    }, "[agent] harness pins pi@0.85.1"
