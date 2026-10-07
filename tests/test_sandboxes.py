"""`check`'s sandbox findings: Harbor's extra and the credentials for a provider elsewhere
(the table confirmed against the installed Harbor), a short setup time, a run secret in
`[rollout] env`, a key `.env` assigns twice, and docker for a docker sandbox or the tunnel's
image. The conftest pins this machine's facts; each test here takes off the pin it tests."""

from __future__ import annotations

import importlib
import os
import re
import sys
import textwrap
from importlib.metadata import metadata
from itertools import count
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from shipyard import preflight
from shipyard.cli import app
from shipyard.config import Finding, check
from shipyard.preflight import (
    INSTEAD,
    LOGINS,
    PROVIDERS,
    docker_running,
    extra_installed,
    logged_in,
)
from shipyard.proxy import tunnel
from tests.trials import fixture_tasks, write_blueprint

runner = CliRunner()
BLUEPRINT = """
[model]
name = "Qwen/Qwen3-8B"
provider = "{provider}"
[data]
dataset = "fixture"
batch_size = 1
[rollout]
harness = "pi@0.85.1"
sandbox = "{sandbox}"
{rollout}
[recipe]
kind = "evaluate"
"""


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with the fixtures, a home with no logins, no provider variables."""
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for _, variables in PROVIDERS.values():
        for variable in variables:
            monkeypatch.delenv(variable, raising=False)
    for variables in INSTEAD.values():
        for variable in variables:
            monkeypatch.delenv(variable, raising=False)
    return tmp_path


MADE = count()


def blueprint(
    tmp_path: Path, sandbox: str, *, provider: str = "openrouter", rollout: str = ""
) -> Path:
    """An evaluate blueprint over the fixture dataset, in a directory of its own."""
    text = BLUEPRINT.format(provider=provider, sandbox=sandbox, rollout=rollout)
    return write_blueprint(tmp_path / "made" / str(next(MADE)), text)


def problems(found: list[Finding]) -> list[str]:
    return [f"{f.level}  {f.text}" for f in found if f.level != "ok"]


# ------------------------------------------------------------- the table, in Harbor


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_module_and_variable_is_harbors_own(name: str) -> None:
    module_path, variables = PROVIDERS[name]
    module = importlib.import_module(module_path)
    assert hasattr(module, f"_HAS_{name.upper()}"), "the flag that says the SDK was found"
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    named = set(re.findall(r"\b[A-Z][A-Z0-9_]+\b", source))
    assert set(variables) <= named and set(INSTEAD.get(name, ())) <= named
    login = LOGINS.get(name)
    if login is not None:
        assert all(part in source for part in Path(login).parts)
    assert name in (metadata("harbor").get_all("Provides-Extra") or [])


def test_the_extra_is_read_off_harbors_flag_not_the_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("harbor.environments.modal")
    fake._HAS_MODAL = False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "harbor.environments.modal", fake)
    assert extra_installed("modal") is False
    fake._HAS_MODAL = True  # type: ignore[attr-defined]
    assert extra_installed("modal") is True

    def broken(name: str) -> ModuleType:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", broken)
    assert extra_installed("e2b") is False


def test_a_login_harbor_takes_stands_in_for_the_variables(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not logged_in("modal", {}) and not logged_in("e2b", {})
    (Path.home() / ".modal.toml").write_text("[default]\n", encoding="utf-8")
    assert logged_in("modal", {})
    (Path.home() / ".blaxel").mkdir()
    (Path.home() / ".blaxel" / "config.yaml").write_text("{}", encoding="utf-8")
    assert logged_in("blaxel", {})
    pair = dict.fromkeys(INSTEAD["daytona"], "x")
    assert logged_in("daytona", pair) and not logged_in("daytona", {"DAYTONA_JWT_TOKEN": "x"})
    elsewhere = here / "beam.ini"
    elsewhere.write_text("[default]\n", encoding="utf-8")
    assert not logged_in("beam", {}) and logged_in("beam", {"CONFIG_PATH": str(elsewhere)})


# ------------------------------------------------------------------- the findings


def test_check_on_modal_without_credentials_blocks_each_variable(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance check, through the CLI: no login file, no variables."""
    monkeypatch.setattr(preflight, "logged_in", logged_in)
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["check", str(blueprint(here, "modal"))])
    assert result.exit_code == 1
    lines = [line for line in result.output.splitlines() if line.startswith("blocked")]
    assert lines == [
        "blocked  sandbox modal: MODAL_TOKEN_ID is unset",
        "blocked  sandbox modal: MODAL_TOKEN_SECRET is unset",
    ]
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-1")
    assert runner.invoke(app, ["check", str(blueprint(here, "modal"))]).exit_code == 0


def test_a_missing_extra_names_the_command_that_installs_it(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "extra_installed", lambda name: name != "e2b")
    monkeypatch.setattr(preflight, "logged_in", logged_in)
    monkeypatch.setenv("E2B_API_KEY", "k")
    assert problems(check(blueprint(here, "e2b"))) == [
        'blocked  sandbox e2b: Harbor\'s e2b extra is not installed; run `uv add "harbor[e2b]"`'
    ]
    monkeypatch.setenv("COLUMNS", "200")
    shown = runner.invoke(app, ["check", str(blueprint(here, "e2b"))]).output
    assert 'run `uv add "harbor[e2b]"`' in shown


def test_a_short_setup_time_is_a_warning_for_a_sandbox_elsewhere_only(here: Path) -> None:
    short = "setup_timeout = 300"
    assert problems(check(blueprint(here, "modal", rollout=short))) == [
        "warning  sandbox modal: setup_timeout 300s is short for a remote image pull"
    ]
    assert problems(check(blueprint(here, "modal", rollout="setup_timeout = 900"))) == []
    assert problems(check(blueprint(here, "modal"))) == [], "Harbor's own default"
    assert problems(check(blueprint(here, "docker", rollout=short))) == []


def test_a_run_secret_in_the_rollout_env_is_blocked(here: Path) -> None:
    env = 'env = { TINKER_API_KEY = "a", SHIPYARD_CONTROL_TOKEN = "b", OPENAI_API_KEY = "c" }'
    assert problems(check(blueprint(here, "docker", rollout=env))) == [
        "blocked  [rollout] env carries TINKER_API_KEY, which must never enter a sandbox",
        "blocked  [rollout] env carries SHIPYARD_CONTROL_TOKEN, which must never enter a sandbox",
    ]


def test_a_run_secret_a_rollout_env_value_names_for_harbor_to_fill_is_blocked(
    here: Path,
) -> None:
    """Harbor fills `${VAR}` and `${VAR:-default}` from this process (utils/env.py)."""
    env = (
        'env = { KEY = "${TINKER_API_KEY}", GATE = "${SHIPYARD_CONTROL_TOKEN:-x}", '
        'OPENAI_API_KEY = "${OPENAI_API_KEY}", NOTE = "a ${TINKER_API_KEY} b" }'
    )
    assert problems(check(blueprint(here, "docker", rollout=env))) == [
        "blocked  [rollout] env carries TINKER_API_KEY (in KEY), which must never enter a sandbox",
        "blocked  [rollout] env carries SHIPYARD_CONTROL_TOKEN (in GATE), which must never "
        "enter a sandbox",
    ]


def test_a_key_dotenv_assigns_twice_is_named_and_its_values_never_are(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (here / ".env").write_text(
        textwrap.dedent(
            """\
            # TINKER_API_KEY=commented-out
            TINKER_API_KEY=first-secret-value
            ANTHROPIC_API_KEY=only-once
            export TINKER_API_KEY="second-secret-value"
            """
        ),
        encoding="utf-8",
    )
    assert problems(check(blueprint(here, "docker"))) == [
        "warning  .env assigns TINKER_API_KEY twice; the last one wins"
    ]
    # The CLI loads `.env` under the shell, so the shell's are set and nothing leaks out.
    for name in ("TINKER_API_KEY", "ANTHROPIC_API_KEY", "COLUMNS"):
        monkeypatch.setenv(name, "200" if name == "COLUMNS" else "from-the-shell")
    shown = runner.invoke(app, ["check", str(blueprint(here, "docker"))]).output
    assert "assigns TINKER_API_KEY twice" in shown and "secret-value" not in shown


def test_docker_is_needed_for_a_docker_sandbox_and_for_the_tunnels_image(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "docker_running", lambda: False)
    assert problems(check(blueprint(here, "docker"))) == [
        "blocked  docker is not running (needed for the docker sandbox)"
    ]
    image = ["/usr/local/bin/docker", "run", "--rm", tunnel.IMAGE, "tunnel"]
    monkeypatch.setattr(tunnel, "found", lambda: (image[0], image))
    served = check(blueprint(here, "modal", provider="tinker"))
    assert "blocked  docker is not running (needed for the tunnel image)" in problems(served)
    monkeypatch.setattr(tunnel, "found", lambda: ("/opt/bin/cloudflared", []))
    served = check(blueprint(here, "modal", provider="tinker", rollout="max_tokens = 100"))
    assert not any("docker" in line for line in problems(served))
    assert Finding("warning", preflight.NOT_KEPT_ALIVE) in served
    assert not any("docker" in line for line in problems(check(blueprint(here, "podman"))))


DOCKER_STUB = """\
#!/bin/sh
case "$DOCKER_STUB" in
  down) echo "Cannot connect to the Docker daemon" >&2; exit 1 ;;
esac
echo "28.3.2"
"""


def test_docker_running_asks_the_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(DOCKER_STUB, encoding="utf-8")
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert docker_running() is True
    monkeypatch.setenv("DOCKER_STUB", "down")
    assert docker_running() is False
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert docker_running() is False


def test_the_env_template_ships_every_value_commented_out() -> None:
    template = Path(__file__).resolve().parents[1] / "docs" / ".env.example"
    lines = [line for line in template.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines and all(line.startswith("#") for line in lines)
    named = {line[2:].split("=")[0] for line in lines if line.startswith("# ") and "=" in line}
    wanted = {"TINKER_API_KEY", "SHIPYARD_PROXY_TOKEN", "SHIPYARD_CONTROL_TOKEN"}
    assert wanted | {v for _, variables in PROVIDERS.values() for v in variables} <= named
    text = template.read_text(encoding="utf-8")
    page = template.parent / "getting-started" / "installation.md"
    body = text[text.index("# A model served from Tinker") :]
    assert f'```sh title=".env"\n{body}```' in page.read_text(encoding="utf-8"), "one template"
