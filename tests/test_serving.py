"""The run's serving half: where the proxy goes for each sandbox, what it is started with,
what each job's records cost and leave in `requests.jsonl`, the served-path refusal, and
the run stopping its proxy on the way out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shipyard import record, serving
from shipyard.config import Blueprint
from shipyard.proxy.profiles import PROFILES, bare_name
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard.rollout import Rollouts
from shipyard.run import Run
from shipyard.serving import (
    Serving,
    answered_by_ours,
    backend,
    endpoint_settings,
    placement,
    tinker_base_url,
    turns_asked,
    volatile_for,
)
from tests.records import FakeProxy, made
from tests.trials import write_result

DAPO = Path(__file__).parent / "blueprints" / "dapo"


def blueprint(**rollout: Any) -> Blueprint:
    model = {"name": "Qwen/Qwen3-8B"}
    if "from_checkpoint" in rollout:
        model["from_checkpoint"] = rollout.pop("from_checkpoint")
    return Blueprint.model_validate(
        {
            "model": model,
            "data": {"dataset": "d", "batch_size": 1},
            "rollout": {"harness": "pi@0.85.1", **rollout},
            "recipe": {"kind": "evaluate"},
        }
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    proxy = FakeProxy()
    monkeypatch.setattr(serving, "Proxy", proxy)
    return proxy


# ------------------------------------------------------------------------ placement


@pytest.mark.parametrize("sandbox", ["docker", "podman", "apple-container"])
def test_a_sandbox_on_this_machine_reaches_a_proxy_here(sandbox: str) -> None:
    assert placement(blueprint(sandbox=sandbox).rollout) == "local"


@pytest.mark.parametrize("sandbox", ["modal", "e2b", "daytona"])
def test_a_sandbox_elsewhere_needs_a_tunnel_unless_a_url_is_named(sandbox: str) -> None:
    assert placement(blueprint(sandbox=sandbox).rollout) == "tunnel"
    assert (
        placement(blueprint(sandbox=sandbox, endpoint_url="https://p.example").rollout) == "remote"
    )
    assert (
        placement(blueprint(sandbox="docker", endpoint_url="https://p.example").rollout) == "remote"
    )


def test_the_party_is_tinker_or_tinker_at_the_backends_host() -> None:
    assert backend({}) == "tinker"
    assert (
        backend({"TINKER_BASE_URL": "https://api.fireworks.ai/inference/v1"})
        == "tinker@api.fireworks.ai"
    )
    assert backend({"TINKER_BASE_URL": ""}) == "tinker"
    assert (
        tinker_base_url({"TINKER_BASE_URL": "https://api.fireworks.ai/"})
        == "https://api.fireworks.ai"
    )
    assert tinker_base_url({}).startswith("https://tinker.thinkingmachines.dev")
    assert bare_name("pi@0.85.1") == "pi" and bare_name("acp:x.y:Z@1") == "acp:x.y:Z@1"


def test_the_endpoint_settings_are_the_rollout_knobs_and_the_checkpoint() -> None:
    cfg = blueprint(
        from_checkpoint="tinker://w/step-9",
        renderer="tml_v0",
        bind="127.0.0.1",
        bind_port=8000,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=4096,
        max_context=32768,
        fill_context=True,
    )
    assert endpoint_settings(cfg) == {
        "model": "Qwen/Qwen3-8B",
        "weights": "tinker://w/step-9",
        "renderer": "tml_v0",
        "bind": "127.0.0.1",
        "bind_port": 8000,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 4096,
        "max_context": 32768,
        "fill_context": True,
        "volatile": [],
    }
    bare = endpoint_settings(blueprint())
    assert bare["weights"] is None and bare["renderer"] is None and bare["max_context"] is None
    assert (bare["bind"], bare["bind_port"], bare["max_tokens"]) == ("0.0.0.0", 0, 8192)
    assert "metadata" not in bare


def test_the_proxys_tinker_session_is_tagged_with_the_run_and_its_recipe() -> None:
    cfg = blueprint()
    tagged = endpoint_settings(cfg, run="dapo-docker__abc")
    assert tagged["metadata"] == {
        "shipyard_run": "dapo-docker__abc",
        "shipyard_recipe": cfg.recipe.kind,
    }


def test_volatile_lines_are_cut_by_default_and_the_profiles_say_which() -> None:
    claude = PROFILES["claude-code"]
    assert volatile_for(blueprint(harness="claude-code").rollout, claude) == claude.volatile
    assert volatile_for(blueprint(harness="claude-code", cut_volatile=False).rollout, claude) == ()
    assert endpoint_settings(blueprint(harness="claude-code", cut_volatile=True))["volatile"] == [
        *claude.volatile
    ]


# ------------------------------------------------------------------- start and stop


async def test_a_local_sandbox_starts_serve_once_with_the_checkpoint_on_its_command_line(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = Serving(blueprint(from_checkpoint="tinker://w/step-2", host="proxy.lan"), tmp_path)
    assert await held.start() is fake and await held.start() is fake
    [(how, settings, named)] = fake.placed
    assert (
        how == "local"
        and named["host"] == "proxy.lan"
        and settings["weights"] == "tinker://w/step-2"
    )
    assert named["log"].name == str(tmp_path / serving.PROXY_LOG)
    assert held.told == "tinker://w/step-2" and held.pointed is True
    assert fake.swapped == [], "serve was started on the path; nothing to swap"
    assert held.model_name() == "openai/Qwen/Qwen3-8B"
    assert held.served().proxy is fake and held.served().fill_context is False
    held.close()
    assert fake.closed == 1 and held.proxy is None
    assert (tmp_path / serving.PROXY_LOG).is_file()
    held.close()
    assert fake.closed == 1


def test_opencode_is_named_under_its_own_provider(tmp_path: Path) -> None:
    named = Serving(blueprint(harness="opencode@1.18.35"), tmp_path).model_name()
    assert named == "shipyard/Qwen/Qwen3-8B"


async def test_a_sandbox_elsewhere_gets_a_tunnel_and_claude_code_an_anthropic_name(
    fake: FakeProxy, tmp_path: Path
) -> None:
    held = Serving(blueprint(sandbox="modal", harness="claude-code", fill_context=True), tmp_path)
    await held.start()
    [(how, settings, named)] = fake.placed
    assert how == "tunnel" and settings["weights"] is None and "tunnel_log" in named
    assert held.told is None and held.pointed is True
    assert held.model_name() == "anthropic/Qwen/Qwen3-8B"
    assert held.served().profile is PROFILES["claude-code"] and held.served().fill_context
    held.close()
    assert fake.closed == 1


async def test_a_remote_proxy_is_pointed_through_its_control_route(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t", CONTROL_TOKEN_ENV: "c"}
    cfg = blueprint(endpoint_url="https://p.example/", from_checkpoint="tinker://w/step-4")
    held = Serving(cfg, tmp_path, environ=environ)
    await held.start()
    assert fake.placed == [("remote", "https://p.example/", {"token": "t", "control_token": "c"})]
    assert fake.probed == 1, "probed before any sandbox opens"
    assert fake.swapped == ["tinker://w/step-4"]
    assert held.told == "tinker://w/step-4" and held.pointed is True
    assert not (tmp_path / serving.PROXY_LOG).exists()
    await held.point("tinker://w/step-5")
    assert fake.swapped[-1] == "tinker://w/step-5" and held.told == "tinker://w/step-5"
    await held.point(None)
    assert held.told is None and held.pointed is True


async def test_a_remote_proxy_without_a_control_token_serves_only_what_it_was_started_on(
    fake: FakeProxy, tmp_path: Path
) -> None:
    environ = {PROXY_TOKEN_ENV: "t"}
    held = Serving(blueprint(endpoint_url="https://p.example"), tmp_path, environ=environ)
    await held.start()
    assert fake.swapped == [] and held.pointed is True and held.told is None, "base, checked"
    with pytest.raises(RuntimeError, match="not started"):
        await Serving(blueprint(endpoint_url="https://p.example"), tmp_path).point(None)
    cfg = blueprint(endpoint_url="https://p.example", from_checkpoint="tinker://w/step-4")
    with pytest.raises(RuntimeError, match=CONTROL_TOKEN_ENV):
        await Serving(cfg, tmp_path, environ=environ).start()
    with pytest.raises(ValueError, match=PROXY_TOKEN_ENV):
        await Serving(blueprint(endpoint_url="https://p.example"), tmp_path, environ={}).start()


# ------------------------------------------------------------------------ harvest


def rolled(tmp_path: Path, *outcomes: tuple[str, dict[str, Any] | None]) -> Rollouts:
    """A job's trials under `jobs/j`, each with the result given, None for none."""
    trials = []
    for n, (task, outcome) in enumerate(outcomes):
        trial = tmp_path / "jobs" / "j" / f"{task}__{n:07d}"
        if outcome is None:
            trials.append(trial)
        else:
            trials.append(write_result(trial, **outcome))
    return Rollouts(job="j", trials=trials, plan=[])


async def test_harvest_writes_a_row_per_record_and_counts_the_tokens(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {
        "alpha": [
            made("one", "ok", seq=1, request_id="r-1"),
            made("oneoktwo", "fine", seq=2, cached=5, bridged=True),
        ],
        "beta": [made("go", error="context", seq=1)],
    }
    fake.counts = {"beta": {"turned_away": 1, "cut": 3, "spoke": 0}}
    held = Serving(blueprint(), tmp_path)
    await held.start()
    batch = rolled(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 0.0}), ("gamma", None))
    harvested = await held.harvest(batch)
    assert fake.fetched == [trial.name for trial in batch.trials]
    assert harvested.rollouts.records is not None
    assert [len(harvested.rollouts.records[t.name]) for t in batch.trials] == [2, 1, 1]
    assert harvested.rollouts.asked == {} and harvested.rollouts.trials == batch.trials
    # prefill = 3 + (8 - 5) + 2 (the refused turn's prompt) + 2 (gamma's default record),
    # cached = 5, sampled = 2 + 4 + 0 + 2.
    assert harvested.counted == {
        "trials": 2,
        "input_tokens": 15,
        "cache_tokens": 5,
        "output_tokens": 8,
    }
    assert harvested.noted == {"turned_away": 1, "cut": 3, "bridged": 1}
    rows = list(record.read(tmp_path / record.REQUESTS))
    assert [row["trial"] for row in rows] == [batch.trials[0].name] * 2 + [
        t.name for t in batch.trials[1:]
    ]
    first = rows[0]
    assert set(first) == {
        "at",
        "job",
        "trial",
        "seq",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "stop_reason",
        "sample_ms",
        "served",
        "request_id",
        "bridged",
        "error",
    }
    assert (first["job"], first["seq"], first["prompt_tokens"], first["completion_tokens"]) == (
        "j",
        1,
        3,
        2,
    )
    assert (first["request_id"], first["served"], first["error"], first["bridged"]) == (
        "r-1",
        None,
        None,
        False,
    )
    assert (rows[1]["cached_tokens"], rows[1]["bridged"], rows[1]["sample_ms"]) == (5, True, 12.0)
    assert (rows[2]["error"], rows[2]["completion_tokens"], rows[2]["stop_reason"]) == (
        "context",
        0,
        "",
    )
    assert fake.counters == {}, "the counters were taken with the records"


async def test_a_batch_answered_by_other_weights_is_refused(
    fake: FakeProxy, tmp_path: Path
) -> None:
    fake.answers = {
        "alpha": [made(served="tinker://w/step-2")],
        "beta": [made(served="tinker://intruder/step-9")],
    }
    held = Serving(blueprint(from_checkpoint="tinker://w/step-2"), tmp_path)
    await held.start()
    with pytest.raises(ValueError, match="intruder"):
        await held.harvest(rolled(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 1.0})))
    fake.answers = {
        "alpha": [made(served="tinker://w/step-2")],
        "beta": [made(served="tinker://w/step-2")],
    }
    assert (await held.harvest(rolled(tmp_path, ("alpha", {"reward": 1.0})))).counted["trials"] == 1
    answered_by_ours("j", {"t": [made(served=None)]}, None)
    with pytest.raises(ValueError, match="pointed its proxy at None"):
        answered_by_ours("j", {"t": [made(served="tinker://x/step-1")]}, None)


async def test_a_remote_proxy_this_run_did_not_point_must_serve_the_base_model(
    fake: FakeProxy, tmp_path: Path
) -> None:
    """With no control route the run cannot point it, so it measures the base model, and a
    batch answered by any checkpoint is refused rather than filed as the base's."""
    fake.answers = {"alpha": [made(served="tinker://theirs/step-5")]}
    environ = {PROXY_TOKEN_ENV: "t"}
    held = Serving(blueprint(endpoint_url="https://p.example"), tmp_path, environ=environ)
    await held.start()
    with pytest.raises(ValueError, match="re-pointed"):
        await held.harvest(rolled(tmp_path, ("alpha", {"reward": 1.0})))
    fake.answers = {"beta": [made()]}
    harvested = await held.harvest(rolled(tmp_path, ("beta", {"reward": 1.0})))
    assert harvested.counted["output_tokens"] == 2


async def test_a_fetch_that_fails_fails_the_job_rather_than_masking_a_trial(
    fake: FakeProxy, tmp_path: Path
) -> None:
    import httpx

    fake.failing = {"beta"}
    held = Serving(blueprint(), tmp_path)
    await held.start()
    with pytest.raises(httpx.ConnectError):
        await held.harvest(rolled(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 1.0})))


async def test_turns_are_counted_off_the_harness_log_only_when_asked(
    fake: FakeProxy, tmp_path: Path
) -> None:
    batch = rolled(tmp_path, ("alpha", {"reward": 1.0}), ("beta", {"reward": 1.0}))
    log = batch.trials[0] / "agent" / "pi.txt"
    log.parent.mkdir()
    log.write_text('{"type":"step-start"}\n{"type":"step-start"}\n')
    counted = Serving(blueprint(harness="opencode", check_turns=True), tmp_path)
    await counted.start()
    assert (await counted.harvest(batch)).rollouts.asked == {}
    (batch.trials[0] / "agent" / "opencode.txt").write_text(log.read_text())
    assert (await counted.harvest(batch)).rollouts.asked == {batch.trials[0].name: 2}
    quiet = Serving(blueprint(harness="opencode"), tmp_path)
    await quiet.start()
    assert (await quiet.harvest(batch)).rollouts.asked == {}
    unprofiled = Serving(blueprint(harness="terminus-2", check_turns=True), tmp_path)
    await unprofiled.start()
    assert (await unprofiled.harvest(batch)).rollouts.asked == {}
    assert turns_asked(batch.trials[1], "opencode", PROFILES["opencode"].turns) is None
    assert (await counted.harvest(batch)).rollouts.failed == set()
    assert turns_asked(batch.trials[0], "opencode@1", PROFILES["opencode"].turns) == 2


async def test_a_pi_log_ending_on_a_failed_call_is_read_whatever_check_turns_says(
    fake: FakeProxy, tmp_path: Path
) -> None:
    batch = rolled(tmp_path, ("alpha", {"reward": 0.0}), ("beta", {"reward": 1.0}))
    for trial, stop in zip(batch.trials, ("error", "stop"), strict=True):
        (trial / "agent").mkdir()
        message = {"role": "assistant", "stopReason": stop}
        (trial / "agent" / "pi.txt").write_text(
            json.dumps({"type": "message_end", "message": message})
        )
    held = Serving(blueprint(), tmp_path)
    await held.start()
    assert (await held.harvest(batch)).rollouts.failed == {batch.trials[0].name}


# --------------------------------------------------------------------- the run


def test_a_served_run_notes_the_backend_and_stops_its_proxy_on_exit(
    fake: FakeProxy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    with Run.open(DAPO, root=tmp_path) as opened:
        assert opened.serving is not None and opened.serving.party == "tinker@api.fireworks.ai"
        note = Run.read(opened.directory)
        assert note["tinker_base_url"] == "https://api.fireworks.ai/inference/v1"
        assert fake.closed == 0
    assert fake.closed == 0, "nothing was started, so nothing was stopped"


async def test_a_started_proxy_is_stopped_when_the_run_closes(
    fake: FakeProxy, tmp_path: Path
) -> None:
    with Run.open(DAPO, root=tmp_path) as opened:
        assert opened.serving is not None
        await opened.serving.start()
        assert fake.placed and fake.closed == 0
    assert fake.closed == 1
