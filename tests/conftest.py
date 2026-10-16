"""Shared fixtures. The autouse fixtures unset Tinker and proxy environment variables and pin
the host checks that `check` reads, so no test reaches a backend or depends on this machine."""

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
    """Unset the Modal image builder version, which a `modal` rollout sets in the environment."""
    monkeypatch.delenv(MODAL_IMAGE_BUILDER, raising=False)


@pytest.fixture(autouse=True)
def _shell_vars_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SHELL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _machine_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the host checks in `preflight` to report a ready machine: every provider extra
    installed and logged in, no Modal finding, and docker running with no leftover networks.
    Tests of those checks undo the patches."""
    from shipyard.cli import preflight

    monkeypatch.setattr(preflight, "extra_installed", lambda name: True)
    monkeypatch.setattr(preflight, "extra_missing", lambda name: [])
    monkeypatch.setattr(preflight, "logged_in", lambda name, environ: True)
    monkeypatch.setattr(preflight, "modal_finding", lambda environ: None)
    monkeypatch.setattr(preflight, "docker_running", lambda: True)
    monkeypatch.setattr(preflight, "leftover_networks", lambda: 0)


@pytest.fixture
def harbor(monkeypatch: pytest.MonkeyPatch) -> Harbor:
    """Return a function that installs a fake Harbor trial runner and returns that fake."""

    def standing(fake: Any) -> Any:
        monkeypatch.setattr("shipyard.sampling.engine._run_trial", fake)
        return fake

    return standing


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProxy, FakeService]:
    """Install a fake proxy and a fake Tinker service, and change into `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    proxy = FakeProxy()
    monkeypatch.setattr("shipyard.sampling.endpoint.Proxy", proxy)
    service = FakeService(capabilities=capable((MODEL, True)))
    service.client.trained = -0.3
    monkeypatch.setattr("shipyard.training.client.service_client", lambda metadata=None: service)
    return proxy, service
