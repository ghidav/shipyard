"""What `check` asks before anything is spent: whether the sandbox can open here (Harbor's
extra, its credentials, docker and its leftover networks) and what must never enter it; for
a served model, where its proxy will stand, how its harness is wired, and whether the
backend lists the model."""

from __future__ import annotations

import importlib
import io
import os
import re
import shutil
import subprocess
import threading
import tomllib
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path

from dotenv.parser import parse_stream

from shipyard.config import Blueprint, Finding
from shipyard.proxy import tunnel
from shipyard.proxy.profiles import PROFILES, bare_name, profile_for
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard.resolved import PROMPT_EFFORT, served_renderer
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
#: The app `check` looks up to learn whether Modal takes the token: a lookup without
#: `create_if_missing` creates nothing (modal/app.py, `App.lookup`).
MODAL_APP = "shipyard"
#: Modal reads each half from the environment before its profile file (modal/config.py).
MODAL_TOKEN = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
#: How long `check` waits for Modal to answer that lookup before it gives up and warns.
MODAL_SECONDS = 15.0
#: At this many leftover trial networks `check` warns: Docker's default address pools hold
#: about 30 user networks, and past them no trial's containers start.
LEFTOVER_NETWORKS = 20
#: Removes those networks and no other: Harbor names a trial's Compose projects
#: `<trial>__env` and `<trial>__verifier__<key>` (harbor/trial/trial.py), and a network with
#: a container on it is not dangling.
REMOVE_NETWORKS = (
    "docker network ls -q --filter dangling=true --filter name=__env_default "
    "--filter name=__verifier__ | xargs docker network rm"
)


def findings(cfg: Blueprint, environ: Mapping[str, str] | None = None) -> list[Finding]:
    """The sandbox's findings, then a served model's. The sandbox's are problems only, plus
    Modal's verdict on its token, since the facts `check` states already name the
    sandbox."""
    environ = os.environ if environ is None else environ
    found = sandbox_findings(cfg, environ)
    if (reflector := getattr(cfg.recipe, "reflection_harness", "").strip()) and (
        unkeyed := key_finding(reflector, cfg.recipe.reflection_model, environ)
    ) is not None:
        found.append(unkeyed)
    return found + served_findings(cfg, environ)


def key_finding(harness: str, model: str | None, environ: Mapping[str, str]) -> Finding | None:
    """A warning when none of the keys Harbor hands `harness` for `model` is set: the
    harness's own names in its `MODEL_CONNECTION`, then the provider's. Quiet for a harness
    Harbor does not name, or one that declares no connection or no key."""
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.agents.model_connection import PROVIDERS, resolve_model_connection
        from harbor.models.agent.name import AgentName

        spec = AgentFactory.get_agent_class(AgentName(bare_name(harness))).MODEL_CONNECTION
    except Exception:  # noqa: BLE001 - an unknown harness is the run's refusal to make
        return None
    if spec is None:
        return None

    def set_one(*names: str) -> tuple[str, str] | None:
        return next(((name, environ[name]) for name in names if environ.get(name)), None)

    connection = resolve_model_connection(model, spec, set_one)
    provider = PROVIDERS.get(connection.provider or "")
    names = (*spec.api_key_envs, *(provider.api_key_envs if provider else ()))
    if not names or connection.api_key:
        return None
    return Finding(
        "warning",
        f"reflector {harness}: none of {', '.join(dict.fromkeys(names))} is set, the "
        "keys Harbor hands it; its first call fails unless it signs in another way",
    )


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
        down = docker_findings("the docker sandbox")
        found += down if down else network_findings()
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
    """Harbor's extra for the provider, each credential variable that is unset when no
    login Harbor takes instead is there, and, for Modal with credentials, its verdict."""
    found = []
    missing = extra_missing(name)
    if missing or not extra_installed(name):
        lacks = f"lacks {', '.join(missing)}" if missing else "is not installed"
        found.append(
            Finding(
                "blocked",
                f'sandbox {name}: Harbor\'s {name} extra {lacks}; run `uv add "harbor[{name}]"`',
            )
        )
    unset = [variable for variable in PROVIDERS[name][1] if not environ.get(variable)]
    if unset and not logged_in(name, environ):
        found += [Finding("blocked", f"sandbox {name}: {variable} is unset") for variable in unset]
    elif name == "modal" and extra_installed(name) and (asked := modal_finding(environ)):
        found.append(asked)
    return found


def modal_finding(environ: Mapping[str, str]) -> Finding | None:
    """One read against Modal that creates nothing: an app looked up by name, which a good
    token answers found or not found. A refused token blocks, naming where it came from;
    anything else, or no answer in `MODAL_SECONDS`, is a warning. None when the SDK does
    not import."""
    source, shadowed = modal_source(environ)
    try:
        modal = importlib.import_module("modal")
    except ImportError:
        return None
    except Exception as failed:  # noqa: BLE001 - modal reads its profile file at import
        return Finding("warning", f"sandbox modal: could not ask Modal about its token: {failed}")
    try:
        within(MODAL_SECONDS, lambda: modal.App.lookup(MODAL_APP))
    except modal.exception.NotFoundError:
        pass
    except modal.exception.AuthError as refused:
        fix = f"; {shadowed}" if shadowed else "; `modal token new` makes a new one"
        return Finding(
            "blocked", f"sandbox modal: Modal refuses the token from {source}: {refused}{fix}"
        )
    except Exception as failed:  # noqa: BLE001 - not a verdict on the token, as for the probe
        return Finding(
            "warning",
            f"sandbox modal: could not ask Modal whether it takes the token from {source}: "
            f"{failed}",
        )
    return Finding("ok", f"sandbox modal: Modal takes the token from {source}")


def within(seconds: float, call: Callable[[], object]) -> None:
    """`call` on a daemon thread, raising here what it raised; TimeoutError when it has not
    returned after `seconds`, its thread left to end with the process."""
    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as failed:  # noqa: BLE001 - raised again on the caller's thread
            raised.append(failed)

    worker = threading.Thread(target=run, name="shipyard-check", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no answer in {seconds:g} s")
    if raised:
        raise raised[0]


def modal_source(environ: Mapping[str, str]) -> tuple[str, str]:
    """Where Modal takes its token from, as its config reads it: each half from the
    environment before the profile file (`MODAL_CONFIG_PATH`, else ~/.modal.toml), whose
    profile is `MODAL_PROFILE`, else the active one, else `default`. And, when
    `MODAL_TOKEN_ID` differs from the token id of that profile, a sentence saying the
    environment shadows it; ids only, no secret is read."""
    moved = environ.get("MODAL_CONFIG_PATH")
    path = Path(moved).expanduser() if moved else Path.home() / ".modal.toml"
    try:
        profiles = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        profiles = {}
    active = [
        name for name, body in profiles.items() if isinstance(body, dict) and body.get("active")
    ]
    profile = environ.get("MODAL_PROFILE") or (active[0] if active else "default")
    named = [variable for variable in MODAL_TOKEN if variable in environ]
    if not named:
        return f"profile {profile} in {path}", ""
    source = f"{' and '.join(named)} in the environment"
    if set(named) & set(assigned(Path.cwd() / ".env")):
        source += " (assigned in the working directory's .env)"
    body = profiles.get(profile)
    held = body.get("token_id") if isinstance(body, dict) else None
    shadowed = (
        f"they shadow profile {profile} in {path}, which Modal uses once they are unset"
        if held and environ.get("MODAL_TOKEN_ID", held) != held
        else ""
    )
    return source, shadowed


def extra_installed(name: str) -> bool:
    """Whether Harbor found the provider's SDK. Its module imports without it, so the import
    proves nothing; the module's own `_HAS_<PROVIDER>` says."""
    try:
        module = importlib.import_module(PROVIDERS[name][0])
    except ImportError:
        return False
    return bool(getattr(module, f"_HAS_{name.upper()}", False))


def extra_missing(name: str) -> list[str]:
    """The packages Harbor's `name` extra requires that are not installed, off Harbor's own
    metadata: the SDK is not the whole extra, and Harbor's environment imports the rest
    (modal's `dockerfile-parse`) only when the first sandbox opens."""
    try:
        listed = metadata.requires("harbor") or []
    except metadata.PackageNotFoundError:
        return []
    for_extra = re.compile(rf"""extra\s*==\s*['"]{re.escape(name)}['"]""")
    missing = []
    for line in listed:
        spec, _, marker = line.partition(";")
        package = re.match(r"[A-Za-z0-9._-]+", spec.strip())
        if package is None or package[0] == "harbor" or not for_extra.search(marker):
            continue
        try:
            metadata.distribution(package[0])
        except metadata.PackageNotFoundError:
            missing.append(package[0])
    return missing


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


def network_findings() -> list[Finding]:
    """A warning when hard-killed runs left enough trial networks to near Docker's pools."""
    left = leftover_networks()
    if left < LEFTOVER_NETWORKS:
        return []
    return [
        Finding(
            "warning",
            f"{left} trial networks are left over from earlier runs, with no container on "
            "them; Docker's default address pools hold about 30 networks, and past them no "
            f"trial starts. `{REMOVE_NETWORKS}` removes them",
        )
    ]


def leftover_networks() -> int:
    """How many of Harbor's trial networks have no container on them: `<trial>__env` and
    `<trial>__verifier__<key>` Compose projects a run killed hard did not take down."""
    binary = shutil.which("docker")
    if binary is None:
        return 0
    try:
        done = subprocess.run(
            [binary, "network", "ls", "--filter", "dangling=true", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if done.returncode != 0:
        return 0
    names = done.stdout.split()
    return sum(1 for name in names if name.endswith("__env_default") or "__verifier__" in name)


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
    (dotenv/main.py:75-84)."""
    counts: dict[str, int] = {}
    for key in assigned(dotenv):
        counts[key] = counts.get(key, 0) + 1
    return [
        Finding("warning", f".env assigns {key} twice; the last one wins")
        for key, count in counts.items()
        if count > 1
    ]


def assigned(dotenv: Path) -> list[str]:
    """Every key `.env` assigns, once per assignment; no value is ever read out of it."""
    try:
        text = dotenv.read_text(encoding="utf-8")
    except OSError:
        return []
    return [binding.key for binding in parse_stream(io.StringIO(text)) if binding.key]


def served_findings(cfg: Blueprint, environ: Mapping[str, str]) -> list[Finding]:
    """What `check` says of a served model: where its proxy will stand and whether that
    can work here, the harness's wiring, whether its renderer takes the effort named, and
    whether the backend serves the model."""
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
    profiled = profile_for(rollout.harness).env
    by_hand = [key for key, value in profiled.items() if rollout.env.get(key) == value]
    if by_hand and not rollout.fill_context:
        found.append(
            Finding(
                "warning",
                f"[rollout] env sets {', '.join(by_hand)}, which switches the harness's "
                "compaction off, but the proxy still refuses a call that overflows the "
                "context, as a harness that compacts expects; set [rollout] fill_context = "
                "true, which sets the same keys and gives such a call what is left",
            )
        )
    if rollout.effort is not None and (renderer := served_renderer(cfg)) not in PROMPT_EFFORT:
        found.append(
            Finding(
                "blocked",
                f"[rollout] effort: the renderer {renderer or '(none known for the model)'} "
                "takes no thinking effort when it builds a prompt; where a model's effort "
                "levels are renderers, [rollout] renderer picks one",
            )
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
