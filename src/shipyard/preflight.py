"""What `check` asks before anything is spent: whether the sandbox can open here (Harbor's
extra, its credentials, docker) and what must never enter it; for a served model, where its
proxy will stand, how its harness is wired, and whether the backend lists the model."""

from __future__ import annotations

import importlib
import io
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from dotenv.parser import parse_stream

from shipyard.config import Blueprint, Finding
from shipyard.proxy import tunnel
from shipyard.proxy.profiles import PROFILES, bare_name
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard.serving import LOCAL_SANDBOXES, backend, placement

#: Over this, a reply to a sandbox elsewhere takes long enough for a harness to give up on it.
REMOTE_MAX_TOKENS = 16384
#: Under this, a sandbox elsewhere can spend the harness's whole setup pulling its image.
REMOTE_SETUP_SECONDS = 600.0
DOCKER_SECONDS = 15.0
NOT_KEPT_ALIVE = (
    "streamed replies are kept alive; long non-streaming replies may be cut by the tunnel "
    "after ~100 s"
)

#: Harbor's module per provider and the variables its `preflight` reads, the same paths in
#: 0.23 and 0.24. Each module imports without the SDK and says in `_HAS_<PROVIDER>` whether
#: it found it (0.24: modal.py:88-97, daytona/environment.py:85-106, e2b.py:30-60,
#: runloop.py:38-56, beam.py:88, blaxel.py:62-75). The variables: modal.py:876-877,
#: daytona/environment.py:150, e2b.py:74, runloop.py:65, beam.py:807, blaxel.py:131-136.
PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "modal": ("harbor.environments.modal", ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")),
    "daytona": ("harbor.environments.daytona.environment", ("DAYTONA_API_KEY",)),
    "e2b": ("harbor.environments.e2b", ("E2B_API_KEY",)),
    "runloop": ("harbor.environments.runloop", ("RUNLOOP_API_KEY",)),
    "beam": ("harbor.environments.beam", ("BEAM_TOKEN",)),
    "blaxel": ("harbor.environments.blaxel", ("BL_WORKSPACE", "BL_API_KEY")),
}
#: What those checks take in place of the variables: a login file under the home directory
#: (modal.py:875, beam.py:141, blaxel.py:134-135), or daytona's other pair (:151-154).
LOGINS = {"modal": ".modal.toml", "beam": ".beam/config.ini", "blaxel": ".blaxel/config.yaml"}
#: A variable that moves the login file elsewhere: beam's `CONFIG_PATH` (beam.py:139-140).
LOGIN_PATHS = {"beam": "CONFIG_PATH"}
INSTEAD = {"daytona": ("DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID")}
#: A value Harbor replaces with the host's variable before the agent sees it, as
#: `${VAR}` or `${VAR:-default}` (harbor/utils/env.py:4, resolved at agents/factory.py:147).
TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def findings(cfg: Blueprint, environ: Mapping[str, str] | None = None) -> list[Finding]:
    """The sandbox's findings, then a served model's: problems only for the first, since
    the facts `check` states already name the sandbox."""
    environ = os.environ if environ is None else environ
    return sandbox_findings(cfg, environ) + served_findings(cfg, environ)


def sandbox_findings(cfg: Blueprint, environ: Mapping[str, str]) -> list[Finding]:
    """For a sandbox elsewhere, Harbor's extra, the credentials and the setup time; docker
    for a docker sandbox or the tunnel's image; a run secret in `[rollout] env`; and a key
    `.env` in the working directory assigns twice."""
    rollout, found = cfg.rollout, []
    name = rollout.sandbox
    if name in PROVIDERS:
        found += provider_findings(name, environ)
    seconds = rollout.setup_timeout
    if name not in LOCAL_SANDBOXES and seconds is not None and seconds < REMOTE_SETUP_SECONDS:
        found.append(
            Finding(
                "warning",
                f"sandbox {name}: setup_timeout {seconds:g}s is short for a remote image pull",
            )
        )
    if name == "docker":
        found += docker_findings("the docker sandbox")
    found += [
        Finding("blocked", f"[rollout] env carries {carried}, which must never enter a sandbox")
        for key, value in rollout.env.items()
        for carried in secrets_in(key, value)
    ]
    return found + twice_in(Path.cwd() / ".env")


def secrets_in(key: str, value: str) -> list[str]:
    """The run secrets an env entry hands a sandbox, by name: its key, or the variable its
    value names as a template Harbor resolves from this process's environment."""
    named = TEMPLATE.fullmatch(value)
    variable = named.group(1).strip() if named else None
    said = [key] if secret(key) else []
    if variable and variable != key and secret(variable):
        said.append(f"{variable} (in {key})")
    return said


def secret(name: str) -> bool:
    return name.upper().startswith("TINKER_") or name == CONTROL_TOKEN_ENV


def provider_findings(name: str, environ: Mapping[str, str]) -> list[Finding]:
    """Harbor's extra for the provider, and each credential variable that is unset when
    no login Harbor takes instead is there."""
    found = []
    if not extra_installed(name):
        found.append(
            Finding(
                "blocked",
                f"sandbox {name}: Harbor's {name} extra is not installed; "
                f'run `uv add "harbor[{name}]"`',
            )
        )
    if not logged_in(name, environ):
        found += [
            Finding("blocked", f"sandbox {name}: {variable} is unset")
            for variable in PROVIDERS[name][1]
            if not environ.get(variable)
        ]
    return found


def extra_installed(name: str) -> bool:
    """Whether Harbor found the provider's SDK. Its module imports without it, so the import
    proves nothing; the module's own `_HAS_<PROVIDER>` says."""
    try:
        module = importlib.import_module(PROVIDERS[name][0])
    except ImportError:
        return False
    return bool(getattr(module, f"_HAS_{name.upper()}", False))


def logged_in(name: str, environ: Mapping[str, str]) -> bool:
    """Whether the provider's check passes without its variables: a login file under the
    home directory, or daytona's other pair of variables."""
    login, moved = LOGINS.get(name), environ.get(LOGIN_PATHS.get(name) or "")
    path = Path(moved).expanduser() if moved else Path.home() / login if login else None
    if path is not None and path.is_file():
        return True
    other = INSTEAD.get(name, ())
    return bool(other) and all(environ.get(variable) for variable in other)


def docker_findings(needed_for: str) -> list[Finding]:
    if docker_running():
        return []
    return [Finding("blocked", f"docker is not running (needed for {needed_for})")]


def docker_running() -> bool:
    """Whether a docker daemon answers here: `docker info`, given up on after a while."""
    binary = shutil.which("docker")
    if binary is None:
        return False
    try:
        done = subprocess.run(
            [binary, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def twice_in(dotenv: Path) -> list[Finding]:
    """A key `.env` assigns more than once, by name: python-dotenv keeps the last one
    (dotenv/main.py:75-84), and no value is ever read out of the file."""
    try:
        text = dotenv.read_text(encoding="utf-8")
    except OSError:
        return []
    counts: dict[str, int] = {}
    for binding in parse_stream(io.StringIO(text)):
        if binding.key:
            counts[binding.key] = counts.get(binding.key, 0) + 1
    return [
        Finding("warning", f".env assigns {key} twice; the last one wins")
        for key, count in counts.items()
        if count > 1
    ]


def served_findings(cfg: Blueprint, environ: Mapping[str, str]) -> list[Finding]:
    """What `check` says of a served model: where its proxy will stand and whether that
    can work here, the harness's wiring, and whether the backend serves the model."""
    model, rollout = cfg.model, cfg.rollout
    if not model.served:
        return []
    found: list[Finding] = []
    where = placement(rollout)
    if where == "remote":
        found.append(Finding("ok", f"rollouts reach the proxy at {rollout.endpoint_url}"))
        if not environ.get(PROXY_TOKEN_ENV):
            found.append(
                Finding(
                    "blocked",
                    f"[rollout] endpoint_url names a proxy this run does not start, and "
                    f"{PROXY_TOKEN_ENV} is unset; export the token its harnesses must present",
                )
            )
    elif where == "local":
        found.append(
            Finding("ok", f"serving {model.name} on this machine for a {rollout.sandbox} sandbox")
        )
    elif (located := tunnel.found()) is not None:
        binary, command = located
        in_docker = tunnel.IMAGE in command
        how = f"docker at {binary}, image {tunnel.IMAGE}" if in_docker else binary
        found.append(
            Finding("ok", f"serving {model.name} behind a cloudflared tunnel (found at {how})")
        )
        found += docker_findings("the tunnel image") if in_docker else []
    else:
        found.append(
            Finding(
                "blocked",
                f"sandbox {rollout.sandbox} cannot reach a proxy in this process and "
                "cloudflared is not installed; install it or set [rollout] endpoint_url",
            )
        )
    if rollout.sandbox not in LOCAL_SANDBOXES:
        if rollout.max_tokens > REMOTE_MAX_TOKENS:
            found.append(
                Finding(
                    "warning",
                    f"[rollout] max_tokens = {rollout.max_tokens}: a reply that long reaches a "
                    f"sandbox elsewhere slowly enough for a harness to give up on it; "
                    f"{REMOTE_MAX_TOKENS} or less is safer",
                )
            )
        found.append(Finding("warning", NOT_KEPT_ALIVE))
    if bare_name(rollout.harness) not in PROFILES:
        found.append(
            Finding("warning", f"no profile for harness {rollout.harness}; generic OpenAI wiring")
        )
    found.append(probe(model.name, environ))
    return found


def probe(model: str, environ: Mapping[str, str]) -> Finding:
    """Whether the backend lists the model: a network call, so only under a key."""
    name = backend(environ)
    if not environ.get("TINKER_API_KEY"):
        return Finding(
            "warning", f"could not ask {name} whether it serves {model}: TINKER_API_KEY is unset"
        )
    try:
        models = supported_models()
    except Exception as failed:  # noqa: BLE001 - a probe that fails is a warning, not a block
        return Finding("warning", f"could not ask {name} whether it serves {model}: {failed}")
    if model in models:
        return Finding("ok", f"{name} serves {model}")
    return Finding("blocked", f"{name} does not serve {model}; it serves {', '.join(models)}")


def supported_models() -> list[str]:
    """The backend's model names off `get_server_capabilities`; the one Tinker call `check`
    makes, replaced in tests."""
    import tinker

    service = tinker.ServiceClient()
    try:
        caps = service.get_server_capabilities()
    finally:
        service.close("success").result()
    return [str(found.model_name) for found in caps.supported_models]
