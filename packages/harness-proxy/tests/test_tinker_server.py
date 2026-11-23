"""Tests for a lane on a server that speaks Tinker's API and is not Thinking Machines' own: its
`base_url` and `api_key_env`, which the `tinker` backend reaches it by."""

from __future__ import annotations

from typing import Any

import pytest

from harness_proxy.backends import tinker as backend
from harness_proxy.serve import lanes_of
from harness_proxy.server import Lane
from proxies import MODEL, FakeRenderer

SERVER = "https://spindle.example.modal.run"
KEY = "SPINDLE_API_KEY"


def test_a_server_is_reached_by_its_url_and_the_key_its_variable_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY, "tml-own")
    assert backend.reaching(SERVER, KEY) == {"base_url": SERVER, "api_key": "tml-own"}
    assert backend.reaching(None, KEY) == {"api_key": "tml-own"}, "another key on Tinker itself"
    assert backend.reaching(None, None) == {}, "left to TINKER_BASE_URL and TINKER_API_KEY"


def test_a_variable_that_is_not_set_is_refused_so_tinkers_own_key_goes_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-tinkers-own")
    with pytest.raises(ValueError, match=f"{KEY} is not set: it holds the key of .*{SERVER}"):
        backend.reaching(SERVER, KEY)


async def test_connect_opens_the_session_on_the_server_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []

    class Service:
        def __init__(self, **given: Any) -> None:
            made.append(given)

    monkeypatch.setattr("tinker.ServiceClient", Service)
    monkeypatch.setenv(KEY, "tml-own")
    tags = {"shipyard_run": "r"}
    await backend.connect(metadata=tags, session=None, base_url=SERVER, api_key_env=KEY)
    await backend.connect(metadata=None, session=None)
    assert made == [
        {"user_metadata": tags, "base_url": SERVER, "api_key": "tml-own"},
        {"user_metadata": None},
    ]


async def test_a_lane_samples_the_server_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[dict[str, Any]] = []

    async def connect(**given: Any) -> object:
        asked.append(given)
        return object()

    async def context_of(service: Any, base_model: str) -> int:
        return 4096

    monkeypatch.setattr(backend, "connect", connect)
    monkeypatch.setattr(backend, "context_of", context_of)
    own = Lane(name="agent", model=MODEL, renderer=FakeRenderer(), base_url=SERVER, api_key_env=KEY)
    await own.open()
    await Lane(name="teacher", model=MODEL, renderer=FakeRenderer()).open()
    assert asked == [
        {"metadata": None, "session": None, "base_url": SERVER, "api_key_env": KEY},
        {"metadata": None, "session": None},
    ]


def test_a_runs_document_names_a_lanes_server() -> None:
    (lane,) = lanes_of([{"name": "agent", "model": MODEL, "base_url": SERVER, "api_key_env": KEY}])
    assert (lane.base_url, lane.api_key_env) == (SERVER, KEY)


def test_only_a_lane_on_the_tinker_backend_names_a_server() -> None:
    with pytest.raises(ValueError, match="speaks Tinker's API.*the together backend"):
        Lane(name="agent", model=MODEL, backend="together", session="s", base_url=SERVER)
