"""Fixtures for the tests. The autouse fixture unsets the Tinker, proxy and Modal settings a
shell may carry, so no test reaches a backend or depends on this machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from proxies import MODEL
from shipyard.jobs.engine import MODAL_IMAGE_BUILDER
from tests.fakes import FakeProxy, Harbor
from tests.trainers import FakeService, capable

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


@pytest.fixture
def harbor(monkeypatch: pytest.MonkeyPatch) -> Harbor:
    """Return a function that installs a fake Harbor trial runner in the engine."""

    def standing(fake: Any) -> Any:
        monkeypatch.setattr("shipyard.jobs.engine._run_trial", fake)
        return fake

    return standing


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProxy, FakeService]:
    """Install a fake proxy and a fake Tinker service in the proxy wrapper and the training
    client, and change into `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.models.endpoints.Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True)))
    service.client.trained = -0.3
    monkeypatch.setattr("shipyard.models.client.service_client", lambda metadata=None: service)
    return proxy, service
