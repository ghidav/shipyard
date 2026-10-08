"""Fixtures every test runs under. They restore the Modal builder default a rollout sets,
remove the shell's Tinker and proxy variables so no test reaches a backend, and pin the
machine facts `check` reads (Harbor's extras, logins, docker, Modal), so no finding depends on
this machine."""

from __future__ import annotations

import pytest

from shipyard.rollout import MODAL_IMAGE_BUILDER

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
