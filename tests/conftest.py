"""What every test runs under: the Modal builder default a rollout sets is restored, the
shell's Tinker and proxy variables are out, so no test reaches a backend, and the machine
`check` reads (Harbor's extras, logins, docker, Modal) is pinned, so no finding depends on
this one."""

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
    """A rollout on `modal` sets the builder version in the process environment; a test
    that did so must not leave it for the next."""
    monkeypatch.delenv(MODAL_IMAGE_BUILDER, raising=False)


@pytest.fixture(autouse=True)
def _shell_vars_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SHELL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _machine_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every provider's extra installed and logged in, Modal never asked, docker running
    with no leftover networks: the tests of those findings take the pins off themselves."""
    from shipyard import preflight

    monkeypatch.setattr(preflight, "extra_installed", lambda name: True)
    monkeypatch.setattr(preflight, "extra_missing", lambda name: [])
    monkeypatch.setattr(preflight, "logged_in", lambda name, environ: True)
    monkeypatch.setattr(preflight, "modal_finding", lambda environ: None)
    monkeypatch.setattr(preflight, "docker_running", lambda: True)
    monkeypatch.setattr(preflight, "leftover_networks", lambda: 0)
