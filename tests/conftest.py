"""Fixtures every test runs under. They restore the Modal builder default a rollout sets,
remove the shell's Tinker and proxy variables so no test reaches a backend, and pin the
machine facts `check` reads (Harbor's extras, logins, docker, Modal), so no finding depends on
this machine. `fakes` and `harbor` are for the tests that ask for them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shipyard.sampling.engine import MODAL_IMAGE_BUILDER
from tests.proxies import MODEL
from tests.records import FakeProxy
from tests.trainers import FakeService, capable
from tests.trials import Harbor

SHELL_VARS = (
    "TINKER_API_KEY",
    "TINKER_BASE_URL",
    "SHIPYARD_PROXY_TOKEN",
    "SHIPYARD_CONTROL_TOKEN",
)


@pytest.fixture(autouse=True)
def _modal_builder_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the builder version that a rollout on `modal` sets in the process environment."""
    monkeypatch.delenv(MODAL_IMAGE_BUILDER, raising=False)


@pytest.fixture(autouse=True)
def _shell_vars_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SHELL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _machine_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the machine: every provider's extra installed and logged in, Modal not asked,
    docker running with no leftover networks. The tests of those findings undo the pins."""
    from shipyard import preflight

    monkeypatch.setattr(preflight, "extra_installed", lambda name: True)
    monkeypatch.setattr(preflight, "extra_missing", lambda name: [])
    monkeypatch.setattr(preflight, "logged_in", lambda name, environ: True)
    monkeypatch.setattr(preflight, "modal_finding", lambda environ: None)
    monkeypatch.setattr(preflight, "docker_running", lambda: True)
    monkeypatch.setattr(preflight, "leftover_networks", lambda: 0)


@pytest.fixture
def harbor(monkeypatch: pytest.MonkeyPatch) -> Harbor:
    """Stand a fake in for Harbor's trial in every engine the test builds; returns the fake."""

    def standing(fake: Any) -> Any:
        monkeypatch.setattr("shipyard.sampling.engine._run_trial", fake)
        return fake

    return standing


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProxy, FakeService]:
    """The proxy and the Tinker service, faked; the working directory is `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True)))
    service.client.trained = -0.3
    monkeypatch.setattr("shipyard.training.client.service_client", lambda metadata=None: service)
    return proxy, service
