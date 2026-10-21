"""Tests for the proxy server over HTTP: auth, both wires, weight swaps, and sampling parameters."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import tinker

from harness_proxy.backends.tinker import context_of
from harness_proxy.chat import Caller
from harness_proxy.recorder import Recorder
from harness_proxy.rendering import Prompt
from harness_proxy.server import DEFAULT_MAX_TOKENS, Lane, Server
from proxies import FakeSampler, address, ask, ask_anthropic, by_path, control, ids, only, server

# ------------------------------------------------------------------ before it starts


def test_nothing_is_recorded_before_start() -> None:
    with pytest.raises(RuntimeError):
        only(server()).records_for("task-1")


def test_each_endpoint_has_its_own_token_and_an_empty_one_is_replaced() -> None:
    assert server().token != server().token
    assert server(token="").token


def test_neither_token_is_in_the_endpoints_repr() -> None:
    """As with the run's `Proxy`, tokens must not leak into a traceback's locals or a log."""
    made = server(token="harness-secret", control_token="control-secret")
    shown = repr(made)
    assert "harness-secret" not in shown and "control-secret" not in shown
    assert "Qwen/Qwen3-8B" in shown


# ------------------------------------------------------------------------ the address


async def test_the_port_is_ephemeral_and_two_endpoints_do_not_collide() -> None:
    async with server() as first, server() as second:
        assert first.port and second.port and first.port != second.port
        assert address(first, "task-1") == f"http://127.0.0.1:{first.port}/r/trial/task-1/v1"


async def test_the_port_is_given_back_on_stop() -> None:
    started = server()
    await started.start()
    await started.stop()
    assert started.port is None


# ------------------------------------------------------------------ serving a trial


async def test_a_request_is_served_and_recorded_under_its_trial() -> None:
    async with server(FakeSampler("ok")) as started:
        answered = await ask(started, "task-1")
        assert answered.status_code == 200
        assert answered.json()["choices"][0]["message"]["content"] == "ok"
        (record,) = only(started).records_for("task-1")
    assert record.completion_token_ids == ids("ok")
    assert record.prompt_token_ids == ids("go")
    assert record.requested_model == "gpt-oss-20b"
    assert len(record.inference_logprobs) == len(record.completion_token_ids)
    assert record.weights is None and record.error is None and record.bridged is False
    assert record.seq == 1 and record.sampling_params["temperature"] == 1.0


async def test_a_request_over_aiohttps_default_body_size_is_served_and_recorded() -> None:
    async with server(FakeSampler("ok")) as started:
        answered = await ask(started, "task-1", metadata={"pad": "x" * (2 * 1024**2)})
        assert answered.status_code == 200
        assert len(only(started).records_for("task-1")) == 1


async def test_the_anthropic_wire_is_served_and_recorded_too() -> None:
    async with server(FakeSampler("ok")) as started:
        answered = await ask_anthropic(started, "task-1")
        assert answered.status_code == 200
        assert answered.json()["content"][0]["text"] == "ok"
        assert only(started).records_for("task-1")[0].requested_model == "claude-sonnet-4-5"


async def test_two_trials_keep_separate_records_and_seq_counts_per_trial() -> None:
    async with server(FakeSampler("ok")) as started:
        await ask(started, "task-1")
        await ask(started, "task-2")
        await ask(started, "task-2")
        assert [r.seq for r in only(started).records_for("task-1")] == [1]
        assert [r.seq for r in only(started).records_for("task-2")] == [1, 2]
        assert only(started).records_for("task-3") == []


async def test_a_request_with_no_trial_in_its_address_is_recorded_under_nothing() -> None:
    async with server(FakeSampler("ok")) as started:
        await ask(started, "task-1")
        async with httpx.AsyncClient() as client:
            answered = await client.post(
                f"http://127.0.0.1:{started.port}/v1/chat/completions",
                headers={"x-api-key": started.token},
                json={"model": "m", "messages": [{"role": "user", "content": "go"}]},
            )
        assert answered.status_code == 200
        assert len(only(started).records_for("")) == 1
        assert len(only(started).records_for("task-1")) == 1


async def test_the_token_is_the_gate() -> None:
    async with server(FakeSampler("ok")) as started:
        assert (await ask(started, "task-1", key="not-the-token")).status_code == 401
        assert only(started).records_for("task-1") == []


# ------------------------------------------------------------------------- swapping


async def test_swap_changes_what_is_served_and_every_record_says_who_answered() -> None:
    async with server(by_path=by_path) as started:
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "base"
        await only(started).swap("tinker://run/sampler_weights/step-1")
        answered = await ask(started, "t")
        assert answered.json()["choices"][0]["message"]["content"] == "step-1"
        assert answered.json()["model"] == "tinker://run/sampler_weights/step-1"
        await only(started).swap(None)
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "base"
        assert [r.weights for r in only(started).records_for("t")] == [
            None,
            "tinker://run/sampler_weights/step-1",
            None,
        ]
        assert address(started, "t").startswith(f"http://127.0.0.1:{started.port}/")


async def test_a_path_given_before_start_is_what_start_serves() -> None:
    async with server(by_path=by_path, weights="tinker://w/step-7") as started:
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "step-7"
        assert only(started).records_for("t")[0].weights == "tinker://w/step-7"


async def test_a_bare_name_is_not_a_path() -> None:
    async with server(by_path=by_path) as started:
        with pytest.raises(ValueError, match="tinker://"):
            await only(started).swap("step-1")


async def test_a_record_is_stamped_with_the_weights_that_answered_it() -> None:
    """The weights are read before the sample is awaited, so a swap mid-flight does not
    change the record's stamp."""

    class Swapping:
        def __init__(self) -> None:
            self.recorder: Recorder | None = None

        async def sample_async(self, **_: Any) -> Any:
            assert self.recorder is not None
            self.recorder.client, self.recorder.weights = object(), "tinker://intruder/step-9"
            return SimpleNamespace(
                sequences=[SimpleNamespace(tokens=[3], logprobs=[-0.1], stop_reason="stop")],
                prompt_cache_hit_tokens=0,
            )

    swapping = Swapping()
    recorder = Recorder(swapping, weights="tinker://mine/step-1")
    swapping.recorder = recorder
    prompt = Prompt(tinker.ModelInput.from_ints([1, 2]))
    await recorder.sample(Caller(trial="t-1"), prompt, tinker.SamplingParams())
    (record,) = recorder.take("t-1")
    assert record.weights == "tinker://mine/step-1"


# --------------------------------------------------------------- the run's parameters


async def test_the_runs_temperature_overrides_the_harnesses() -> None:
    sampler = FakeSampler("ok")
    async with server(sampler, temperature=0.7) as started:
        await ask(started, "t", temperature=0.0)
        assert sampler.asked[0].temperature == 0.7
        assert only(started).records_for("t")[0].sampling_params["temperature"] == 0.7


async def test_what_the_run_did_not_name_is_left_to_the_harness() -> None:
    sampler = FakeSampler("ok")
    async with server(sampler, temperature=None, top_p=None, top_k=None) as started:
        await ask(started, "t", temperature=0.25, top_p=0.9, max_tokens=64)
        assert (sampler.asked[0].temperature, sampler.asked[0].top_p) == (0.25, 0.9)
        assert sampler.asked[0].max_tokens == 64


async def test_max_tokens_caps_a_turn_and_is_the_default_for_one_naming_none() -> None:
    sampler = FakeSampler("ok")
    async with server(sampler, max_tokens=16) as started:
        await ask(started, "t", max_tokens=99)
        await ask(started, "t", max_tokens=8)
        await ask(started, "t")
    assert [asked.max_tokens for asked in sampler.asked] == [16, 8, 16]


async def test_a_request_naming_no_limit_is_not_cut_at_the_cookbooks_default() -> None:
    sampler = FakeSampler("ok")
    async with server(sampler) as started:
        await ask(started, "t")
    assert sampler.asked[0].max_tokens == DEFAULT_MAX_TOKENS


def test_every_knob_that_shapes_the_distribution_is_pinned_and_the_object_left_alone() -> None:
    recorder = Recorder(object(), temperature=1.0, top_p=1.0, top_k=-1)
    asked = tinker.SamplingParams(max_tokens=64, temperature=0.2, top_p=0.9, top_k=40)
    pinned = recorder.pin(asked)
    assert (pinned.temperature, pinned.top_p, pinned.top_k, pinned.max_tokens) == (1, 1, -1, 64)
    assert (asked.temperature, asked.top_p, asked.top_k) == (0.2, 0.9, 40)
    untouched = Recorder(object()).pin(asked)
    assert (untouched.temperature, untouched.top_p, untouched.top_k) == (0.2, 0.9, 40)


def test_the_endpoint_defaults_to_truncating_nothing() -> None:
    lane = Lane(name="agent", model="a/b")
    assert (lane.temperature, lane.top_p, lane.top_k, lane.max_tokens) == (1.0, 1.0, -1, 8192)
    assert Server(lanes=[lane]).control_token is None


def test_a_server_serves_lanes_each_named_once_by_a_name_an_address_can_hold() -> None:
    with pytest.raises(ValueError, match="at least one lane"):
        Server(lanes=[])
    with pytest.raises(ValueError, match="two lanes are named 'agent'"):
        Server(lanes=[Lane(name="agent", model="a/b"), Lane(name="agent", model="c/d")])
    for name in ("", "a/b"):
        with pytest.raises(ValueError, match="cannot be"):
            Lane(name=name, model="a/b")


# ------------------------------------------------------------------ the control route


async def test_the_control_route_is_closed_without_a_control_token() -> None:
    async with server(by_path=by_path, control_token=None) as started:
        assert (await control(started, "tinker://w/step-1", token="x")).status_code == 404


async def test_the_control_route_opens_to_the_control_token_and_not_the_harnesses() -> None:
    async with server(by_path=by_path, control_token="ctl") as started:
        assert (await control(started, "tinker://w/step-1", token="wrong")).status_code == 401
        assert (await control(started, "tinker://w/step-1", token=started.token)).status_code == 401
        swapped = await control(started, "tinker://w/step-1")
        assert swapped.status_code == 200 and swapped.json() == {"serving": "tinker://w/step-1"}
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "step-1"
        back = await control(started, None)
        assert back.status_code == 200 and back.json() == {"serving": None}
        assert (await ask(started, "t")).json()["choices"][0]["message"]["content"] == "base"
        assert [r.weights for r in only(started).records_for("t")] == ["tinker://w/step-1", None]


async def test_the_control_route_refuses_what_is_not_a_path() -> None:
    async with server(by_path=by_path, control_token="ctl") as started:
        assert (await control(started, "step-1")).status_code == 400
        assert (await control(started, 7)).status_code == 400
        assert (await control(started, None, raw=b"{not json")).status_code == 400
        assert (await control(started, None, raw=b'{"path": "tinker://x"}')).status_code == 400
        assert only(started).weights is None


# ------------------------------------------------------------------ the context length


async def test_the_context_length_is_read_off_the_capabilities_or_is_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Service:
        async def get_server_capabilities_async(self) -> Any:
            return SimpleNamespace(
                supported_models=[
                    SimpleNamespace(model_name="a/b", max_context_length=4096),
                    SimpleNamespace(model_name="c/d", max_context_length=None),
                ]
            )

    class Broken:
        async def get_server_capabilities_async(self) -> Any:
            raise RuntimeError("no network")

    assert await context_of(Service(), "a/b") == 4096
    assert await context_of(Service(), "c/d") is None
    assert await context_of(Service(), "e/f") is None
    with caplog.at_level(logging.WARNING, logger="harness_proxy.server"):
        assert await context_of(Broken(), "a/b") is None
    assert "could not read the context length of a/b" in caplog.text
    assert "not enforced" in caplog.text and "RuntimeError: no network" in caplog.text


# ------------------------------------------------------------------ the Tinker session


class Service:
    """Fake Tinker service that records each `close` status and can raise on close."""

    def __init__(self, fails: bool = False) -> None:
        self.closed: list[str] = []
        self.fails = fails

    async def get_server_capabilities_async(self) -> Any:
        return SimpleNamespace(supported_models=[])

    async def close(self, status: str) -> None:
        self.closed.append(status)
        if self.fails:
            raise RuntimeError("the session is already gone")


async def test_the_proxy_closes_its_tinker_session_once_on_the_way_out() -> None:
    """The status is `success` even when the caller failed, because the proxy's own
    shutdown was clean. The run closes its own session with the failure reason."""
    service = Service()
    with pytest.raises(KeyError):
        async with server(_service=service, max_context=4096) as started:
            assert started.port and service.closed == []
            raise KeyError("the run failed")
    assert service.closed == ["success"]
    assert only(started)._service is None and started.port is None


async def test_a_close_that_fails_still_stops_the_proxy() -> None:
    service = Service(fails=True)
    async with server(_service=service, max_context=4096) as started:
        pass
    assert service.closed == ["success"] and started.port is None
