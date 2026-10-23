"""Fixtures for the tests. The autouse fixture unsets the Tinker, proxy and Modal settings a
shell may carry, so no test reaches a backend or depends on this machine."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard.jobs.engine import MODAL_IMAGE_BUILDER
from shipyard.models.clients import together
from tests.fakes import FakeProxy, Harbor
from tests.service import FakeService, capable

SHELL_VARS = (
    "TINKER_API_KEY",
    "TINKER_BASE_URL",
    "HARNESS_PROXY_TOKEN",
    "HARNESS_PROXY_CONTROL_TOKEN",
    MODAL_IMAGE_BUILDER,
)


@pytest.fixture(autouse=True)
def _shell_vars_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SHELL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def together_exit(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep a Together client's exit hook, and the SIGTERM handler it may set, to the test that
    opened it; return the stops the hook would run at exit."""
    before = signal.getsignal(signal.SIGTERM)
    hooks: list[Any] = []
    monkeypatch.setattr(together, "at_exit", hooks.append)
    monkeypatch.setattr(together, "RUNNING", set())
    monkeypatch.setattr(together, "_hooked", False)
    yield hooks
    signal.signal(signal.SIGTERM, before)


@pytest.fixture
def harbor(monkeypatch: pytest.MonkeyPatch) -> Harbor:
    """Return a function that installs a fake Harbor trial runner in the engine."""

    def standing(fake: Any) -> Any:
        monkeypatch.setattr("shipyard.jobs.engine._run_trial", fake)
        return fake

    return standing


#: A teacher the fake Tinker service serves beside the agent's model, for sampling alone.
TEACHER = "Qwen/Qwen3-32B"


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProxy, FakeService]:
    """Install a fake proxy and a fake Tinker service, which every Tinker session the run opens
    gets, and change into `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True), (TEACHER, False)))
    service.client.trained = -0.3
    monkeypatch.setattr(
        "shipyard.models.clients.tinker.service_client", lambda metadata=None: service
    )
    return proxy, service
