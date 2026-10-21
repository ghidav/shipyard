"""Tests for the messages between the run and the proxy, each read back as it was written:
a trial's address, `/healthz`, the weights switch, a records fetch, and serve's startup lines."""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness_proxy.wire import (
    BANNER,
    TUNNEL_LINE,
    Address,
    Fetch,
    Served,
    Swap,
    banner,
    health,
    lanes_in,
    records_answer,
    records_in,
    records_path,
    tunnel_line,
)
from proxies import made


def sent(message: Any) -> Any:
    """Return `message` as the other side reads it, off JSON."""
    return json.loads(json.dumps(message))


@pytest.mark.parametrize(
    "address",
    [
        Address("t-1"),
        Address("t-1", lane="teacher"),
        Address("t-1", renderer="low"),
        Address("a b#c?d%2F__x", lane="v1", renderer="trial"),
    ],
)
def test_an_address_reads_back_as_written_whatever_its_names_hold(address: Address) -> None:
    assert Address.of(address.path) == address
    assert Address.of(f"{address.path}/chat/completions") == address, "under a wire's route"


def test_an_address_names_its_lane_trial_and_renderer_in_order_each_encoded() -> None:
    assert Address("t").path == "/r/trial/t/v1"
    assert Address("a b", "teacher", "low").path == "/r/lane/teacher/trial/a%20b/renderer/low/v1"


@pytest.mark.parametrize(
    "path", ["/r/lane/x/v1", "/r/trial/t/model/m/v1", "/r/trial/t/trial/u/v1", "/v1/x", "/r/trial"]
)
def test_a_path_that_is_no_trials_address_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="is no trial's address"):
        Address.of(path)


@pytest.mark.parametrize("named", [{"trial": ""}, {"trial": "a/b"}, {"trial": "t", "lane": ""}])
def test_a_name_the_path_cannot_hold_is_refused_on_either_side(named: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="goes in the URL path"):
        Address(**named)
    with pytest.raises(ValueError, match="goes in the URL path"):
        Address.of("/r/trial/a%2Fb/v1")


def test_each_lane_reads_back_from_healthz_as_it_was_reported() -> None:
    lanes = {
        "agent": Served("m", "tinker://w/step-1", 32768, ("low", "high")),
        "teacher": Served("t", "t", None, ()),
    }
    assert lanes_in(sent(health(lanes))) == lanes
    assert lanes_in({"status": "ok"}) == {} and lanes_in({"lanes": [1]}) == {}


def test_a_swap_reads_back_as_written_and_anything_else_is_refused() -> None:
    for swap in (Swap("tinker://w/1", "teacher"), Swap(None), Swap("together://s/2")):
        assert Swap.of(sent(swap.body)) == swap
    assert Swap(None).body == {"weights": None}, "a proxy's only lane is named by none"
    for body in (None, [], {"lane": "a"}, {"weights": 3}, {"weights": None, "lane": 2}):
        with pytest.raises(ValueError):
            Swap.of(body)


def test_a_fetch_and_its_answer_read_back_as_written() -> None:
    for fetch in (Fetch(), Fetch(delta=True), Fetch(delta=True, again=True)):
        assert Fetch.of(fetch.query) == fetch
    assert Fetch(delta=True).query == {"delta": "1"}
    records = [made("go", "ok"), made("gook", "more", seq=2)]
    for delta in (True, False):
        assert records_in(sent(records_answer(records, delta=delta))) == records
    assert records_path("a b/c") == "/records/a%20b%2Fc"


def test_serves_startup_lines_read_back_as_written() -> None:
    line = BANNER.match(banner(["agent", "teacher"], "host.docker.internal", 8765, control=True))
    assert line is not None
    assert line.group("lanes", "host", "port", "control") == (
        "agent,teacher",
        "host.docker.internal",
        "8765",
        "on",
    )
    opened = TUNNEL_LINE.match(tunnel_line("https://fake-words.trycloudflare.com"))
    assert opened is not None and opened.group("origin") == "https://fake-words.trycloudflare.com"
