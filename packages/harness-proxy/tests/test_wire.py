"""Tests for the messages between the run and the proxy, each read back as it was written:
opening a trial, `/healthz`, the records a close answers, and serve's startup lines."""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness_proxy.wire import (
    BANNER,
    TUNNEL_LINE,
    Opening,
    Served,
    banner,
    closing_path,
    health,
    lanes_in,
    records_answer,
    records_in,
    tunnel_line,
)
from proxies import made


def sent(message: Any) -> Any:
    """Return `message` as the other side reads it, off JSON."""
    return json.loads(json.dumps(message))


@pytest.mark.parametrize(
    "opening",
    [
        Opening("t-1"),
        Opening("t-1", lane="teacher", renderer="low", weights="tinker://w/1"),
        Opening("a b#c?d%2F__x", weights="together://s/2"),
    ],
)
def test_an_opening_reads_back_as_written(opening: Opening) -> None:
    assert Opening.of(sent(opening.body)) == opening


@pytest.mark.parametrize(
    "body",
    [None, [], {}, {"trial": 3}, {"trial": "t", "lane": 2}, {"trial": "t", "graph": "g"}],
)
def test_anything_but_an_opening_is_refused(body: Any) -> None:
    with pytest.raises(ValueError):
        Opening.of(body)


@pytest.mark.parametrize("name", ["", "a/b"])
def test_a_trial_the_close_route_cannot_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="cannot be a trial's name"):
        Opening(name)
    with pytest.raises(ValueError, match="cannot be a trial's name"):
        Opening.of({"trial": name})
    assert closing_path("a b%2F") == "/trials/a%20b%252F/close"


def test_each_lane_reads_back_from_healthz_as_it_was_reported() -> None:
    lanes = {"agent": Served("m", 32768, ("low", "high")), "teacher": Served("t", None, ())}
    assert lanes_in(sent(health(lanes))) == lanes
    assert lanes_in({"status": "ok"}) == {} and lanes_in({"lanes": [1]}) == {}


def test_the_records_a_close_answers_read_back_as_written() -> None:
    records = [made("go", "ok"), made("gook", "more", seq=2)]
    answered = sent(records_answer(records))
    assert records_in(answered) == records
    assert answered["records"][1]["prompt_from"] == 0, "a prompt that extends one goes as a delta"


def test_serves_startup_lines_read_back_as_written() -> None:
    line = BANNER.match(banner(["agent", "teacher"], "host.docker.internal", 8765))
    assert line is not None
    assert line.group("lanes", "host", "port") == ("agent,teacher", "host.docker.internal", "8765")
    opened = TUNNEL_LINE.match(tunnel_line("https://fake-words.trycloudflare.com"))
    assert opened is not None and opened.group("origin") == "https://fake-words.trycloudflare.com"
