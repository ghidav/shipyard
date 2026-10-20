"""Shared fixtures. The autouse fixture unsets the Tinker and proxy environment variables, so no
test reaches a backend or picks up this machine's tokens."""

from __future__ import annotations

import pytest

SHELL_VARS = (
    "TINKER_API_KEY",
    "TINKER_BASE_URL",
    "HARNESS_PROXY_TOKEN",
    "HARNESS_PROXY_CONTROL_TOKEN",
)


@pytest.fixture(autouse=True)
def _shell_vars_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SHELL_VARS:
        monkeypatch.delenv(name, raising=False)
