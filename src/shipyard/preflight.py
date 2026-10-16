"""Every finding `check` reports: the blueprint's, then the environment's. For the sandbox:
Harbor's extra, its credentials, docker, leftover networks, and secrets that must not enter
it. For a served model: where its proxy will stand, how its harness is wired, and whether
the backend lists the model."""

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
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from dotenv.parser import parse_stream
from harbor.models.environment_type import EnvironmentType

from shipyard.config import (
    LENGTH_CAP,
    Blueprint,
    DrGrpoRecipe,
    FstRecipe,
    GepaRecipe,
    Gradient,
    Search,
    modules_dir,
)
from shipyard.data import NoSuchDataset, home_of, search_sets, tasks
from shipyard.harnesses import PROFILES, bare_name, profile_for
from shipyard.modules import Inadmissible, seed
from shipyard.proxy import tunnel
from shipyard.proxy.wire import CONTROL_TOKEN_ENV, PROXY_TOKEN_ENV
from shipyard.sampling.endpoint import LOCAL_SANDBOXES, backend, placement

#: Harbor's environments, the values `[rollout] sandbox` takes.
SANDBOXES = tuple(kind.value for kind in EnvironmentType)
#: What `check` says of a blueprint that cannot run yet.
NO_HARNESS = (
    "[rollout] harness: name the harness; Harbor reads an unnamed agent as its oracle, "
    "which runs the task's reference solution"
)
NO_REFLECTOR = (
    "[recipe] reflection_harness: name the harness the reflector runs as; Harbor reads an "
    "unnamed agent as its oracle, and the reflection task has no reference solution"
)
#: Over this, a reply to a sandbox elsewhere takes long enough for a harness to give up on it.
#: The reflector's container when a gepa blueprint names none: a shell and a Python.
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
#: How long `check` waits for Modal to answer that lookup before it gives up and warns.
MODAL_SECONDS = 15.0
#: At this many leftover trial networks `check` warns: Docker's default address pools hold
#: about 30 user networks, and past them no trial's containers start.
LEFTOVER_NETWORKS = 20
#: Removes the leftover trial networks only. Harbor names a trial's Compose projects
#: `<trial>__env` and `<trial>__verifier__<key>` (harbor/trial/trial.py), and a network with
#: a container on it is not dangling.
REMOVE_NETWORKS = (
    "docker network ls -q --filter dangling=true --filter name=__env_default "
    "--filter name=__verifier__ | xargs docker network rm"
)


@dataclass(frozen=True)
class Finding:
    level: Literal["ok", "warning", "blocked"]
    text: str


def findings(loaded: Blueprint, environ: Mapping[str, str] | None = None) -> list[Finding]:
    """The findings of a blueprint that loaded: one ok line per fact, then every problem.
    Datasets are looked up under `tasks/` in the working directory, as `run` does."""
    environ = os.environ if environ is None else environ
    model, rollout = loaded.model, loaded.rollout
    served = " (served by this run)" if model.served else ""
    found = [
        Finding("ok", f"recipe {loaded.recipe.kind}"),
        Finding("ok", f"model {model.name}, provider {model.provider}{served}"),
    ]
    for label, names in (("train", loaded.datasets), ("test", loaded.test_sets)):
        if names:
            found.append(Finding("ok", f"{label} {', '.join(names)}"))
    found.append(Finding("ok", f"harness {rollout.harness}, sandbox {rollout.sandbox}"))
    found += [_dataset_finding(name) for name in [*loaded.datasets, *loaded.test_sets]]
    found += _data_findings(loaded)
    for finding in (_provider_finding(loaded), _checkpoint_finding(loaded)):
        if finding is not None:
            found.append(finding)
    if (directory := modules_dir(loaded)) is not None:
        found.append(_modules_finding(directory))
        if (template := _template_finding(loaded, directory)) is not None:
            found.append(template)
    if isinstance(loaded.recipe, Search):
        found.append(_reflector_finding(loaded.recipe))
    if isinstance(loaded.recipe, GepaRecipe):
        found += _pareto_findings(loaded)
    if isinstance(loaded.recipe, FstRecipe) and loaded.data.group_size % loaded.recipe.population:
        k, g = loaded.recipe.population, loaded.data.group_size
        found.append(
            Finding(
                "blocked",
                f"[recipe] population: {k} does not divide [data] group_size {g}; each "
                "candidate takes group_size / population rollouts of every task's group",
            )
        )
    if (cap := _length_cap(loaded.recipe)) is not None and cap >= 1:
        found.append(
            Finding(
                "warning",
                f"[recipe] length_cap: {cap} lets the length rule take a solved answer's "
                "whole reward, so a long solved answer can score as low as a failure",
            )
        )
    if not rollout.harness.strip():
        found.append(Finding("blocked", NO_HARNESS))
    if rollout.sandbox not in SANDBOXES:
        named = f"no Harbor environment called {rollout.sandbox!r}; one of {', '.join(SANDBOXES)}"
        found.append(Finding("blocked", f"[rollout] sandbox: {named}"))
    found += environment_findings(loaded, environ)
    return found


def environment_findings(cfg: Blueprint, environ: Mapping[str, str]) -> list[Finding]:
    """The sandbox findings, then the served-model findings. Sandbox findings are problems
    only, plus Modal's verdict on its token, because the blueprint findings state the
    sandbox facts."""
    found = sandbox_findings(cfg, environ)
    if (reflector := getattr(cfg.recipe, "reflection_harness", "").strip()) and (
        unkeyed := key_finding(reflector, cfg.recipe.reflection_model, environ)
    ) is not None:
        found.append(unkeyed)
    return found + served_findings(cfg, environ)


def _data_findings(loaded: Blueprint) -> list[Finding]:
    """The older spelling, a task in both sets, an evaluation cadence past the steps."""
    found: list[Finding] = []
    if "dataset" in loaded.raw.get("data", {}):
        key = "test" if loaded.recipe.kind == "evaluate" else "train"
        found.append(Finding("warning", f"[data] dataset: the older spelling of {key}; still read"))
    if loaded.datasets and loaded.test_sets:
        try:
            train = {task for name in loaded.datasets for task in tasks(name)}
            both = [task for name in loaded.test_sets for task in tasks(name) if task in train]
        except NoSuchDataset:
            both = []
        if both:
            found.append(
                Finding(
                    "warning",
                    f"[data] test: {_tasks(len(both))} also in train, so the measurement is "
                    "not held out",
                )
            )
    recipe = loaded.recipe
    if isinstance(recipe, Gradient) and recipe.evaluate_every > recipe.steps:
        found.append(
            Finding(
                "warning",
                f"[recipe] evaluate_every: {recipe.evaluate_every} is past the {recipe.steps} "
                "steps, so the run evaluates at its end only",
            )
        )
    return found


def _length_cap(recipe: Any) -> float | None:
    """The cap on dr-grpo's length rule, as set or defaulted. None for a recipe without the
    rule or with `length_penalty` at 0."""
    if isinstance(recipe, DrGrpoRecipe):
        penalty, cap = recipe.length_penalty, recipe.length_cap
    elif isinstance(recipe, FstRecipe) and recipe.slow == "dr-grpo":
        penalty = recipe.length_penalty or 0.0
        cap = LENGTH_CAP if recipe.length_cap is None else recipe.length_cap
    else:
        return None
    return cap if penalty > 0 else None


def _provider_finding(loaded: Blueprint) -> Finding | None:
    """`blocked` when the recipe trains weights this run does not serve."""
    if loaded.trains and not loaded.model.served:
        return Finding(
            "blocked",
            f"[recipe] kind = {loaded.recipe.kind!r} trains the weights this run serves, and "
            f"[model] provider = {loaded.model.provider!r} serves them elsewhere",
        )
    return None


def _checkpoint_finding(loaded: Blueprint) -> Finding | None:
    """`blocked` when `from_checkpoint` names the wrong column of `checkpoints.jsonl`. A
    training run resumes from a `state_path`. A run that only samples needs a `sampler_path`."""
    named = loaded.model.from_checkpoint or ""
    if loaded.trains and "/sampler_weights/" in named:
        return Finding(
            "blocked",
            "[model] from_checkpoint: a training run resumes from a checkpoint's state_path "
            "(tinker://.../weights/...), not its sampler_path",
        )
    if not loaded.trains and "/weights/" in named and "/sampler_weights/" not in named:
        return Finding(
            "blocked",
            f"[model] from_checkpoint: {loaded.recipe.kind} samples a checkpoint's sampler_path "
            "(tinker://.../sampler_weights/...), not its state_path",
        )
    return None


def _modules_finding(directory: Path) -> Finding:
    """`ok` with the component count and digest, or `blocked` with why `seed` refused."""
    try:
        candidate = seed(directory)
    except Inadmissible as refused:
        return Finding("blocked", f"[recipe] modules: {refused}")
    return Finding(
        "ok", f"modules: {len(candidate)} component(s) under {directory} ({candidate.digest})"
    )


def _template_finding(loaded: Blueprint, directory: Path) -> Finding | None:
    """`warning` when the modules hold a prompt and `[rollout.kwargs]` names a prompt
    template: the prompt's template replaces it in every rollout that carries the prompt."""
    if "prompt_template_path" not in loaded.rollout.kwargs:
        return None
    try:
        kinds = {one.kind for one in seed(directory).components.values()}
    except Inadmissible:
        return None  # the modules finding says why
    if "prompt" not in kinds:
        return None
    return Finding(
        "warning",
        "[rollout.kwargs] prompt_template_path: the prompt in the modules is delivered as the "
        "prompt template, so this one is not used where the prompt is carried",
    )


def _reflector_finding(recipe: Search) -> Finding:
    """`ok` naming the reflector's harness, model and image, or `blocked` for a blank
    harness. The image is not checked: Harbor pulls it at the first round."""
    if not recipe.reflection_harness.strip():
        return Finding("blocked", NO_REFLECTOR)
    model = recipe.reflection_model or "the harness's own default"
    return Finding(
        "ok",
        f"reflector {recipe.reflection_harness}, model {model}, image {recipe.reflection_image}",
    )


def _pareto_findings(loaded: Blueprint) -> list[Finding]:
    """The finding for a named held-out dataset, then `ok` with how the tasks split or
    `blocked` with why they cannot. No split line when a dataset is missing, which that
    dataset's finding already says."""
    named = loaded.recipe.pareto
    found = [_dataset_finding(named)] if isinstance(named, str) else []
    try:
        listed = [task for name in loaded.datasets for task in tasks(name)]
        feedback, pareto = search_sets(listed, named, seed=loaded.data.seed)
    except NoSuchDataset:
        return found
    except ValueError as refused:
        return [*found, Finding("blocked", f"[recipe] pareto: {refused}")]
    if isinstance(named, str):
        said = f"dataset {named}'s {_tasks(len(pareto))} to select on"
        said += f", the run's {_tasks(len(feedback))} to reflect on"
    elif named == 0:
        said = f"none held out; {_tasks(len(feedback))} both reflected and selected on"
    else:
        said = f"{len(pareto)} of {_tasks(len(feedback) + len(pareto))} held out with seed "
        said += f"{loaded.data.seed} to select on, {len(feedback)} to reflect on"
    return [*found, Finding("ok", f"pareto: {said}")]


def _tasks(count: int) -> str:
    return f"{count} task" if count == 1 else f"{count} tasks"


def _dataset_finding(name: str) -> Finding:
    """`ok` with the task count under `tasks/<name>`, or `blocked` with the path."""
    try:
        listed = tasks(name)
    except NoSuchDataset as missing:
        return Finding("blocked", str(missing))
    plural = "" if len(listed) == 1 else "s"
    return Finding("ok", f"dataset {name}: {len(listed)} task{plural} under {home_of(name)}")


def key_finding(harness: str, model: str | None, environ: Mapping[str, str]) -> Finding | None:
    """A warning when none of the keys Harbor passes `harness` for `model` is set: the names
    in the harness's `MODEL_CONNECTION`, then the provider's. None for a harness Harbor does
    not name or one that declares no connection or no key."""
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.agents.model_connection import PROVIDERS, resolve_model_connection
        from harbor.models.agent.name import AgentName

        spec = AgentFactory.get_agent_class(AgentName(bare_name(harness))).MODEL_CONNECTION
    except Exception:  # noqa: BLE001 - the run refuses an unknown harness
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
    """For a sandbox elsewhere: Harbor's extra, the credentials and the setup time. For a
    docker sandbox: docker and leftover networks. Also a run secret in `[rollout] env` and a
    key the working directory's `.env` assigns twice."""
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
    """The run secrets an env entry passes to a sandbox, by name: its key, or the variable
    its value references as a template that Harbor resolves from this process's
    environment."""
    named = TEMPLATE.fullmatch(value)
    variable = named.group(1).strip() if named else None
    said = [key] if secret(key) else []
    if variable and variable != key and secret(variable):
        said.append(f"{variable} (in {key})")
    return said


def secret(name: str) -> bool:
    return name.upper().startswith("TINKER_") or name == CONTROL_TOKEN_ENV


def provider_findings(name: str, environ: Mapping[str, str]) -> list[Finding]:
    """Findings for a provider sandbox: Harbor's extra, each unset credential variable
    (unless a login that Harbor accepts instead is there), and, for Modal with credentials,
    its verdict on the token."""
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
    """Look up an app by name, a read against Modal that creates nothing. A good token gets
    a found or not-found answer. A refused token blocks and names where it came from. Any
    other failure, or no answer in `MODAL_SECONDS`, is a warning. None when the SDK does not
    import."""
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
    """Run `call` on a daemon thread and raise here what it raised. Raises TimeoutError if
    it has not returned after `seconds`; its thread ends with the process."""
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
    """Where Modal takes its token from, as its config reads it. Each half comes from the
    environment first, then from the profile file (`MODAL_CONFIG_PATH`, else
    ~/.modal.toml). The profile is `MODAL_PROFILE`, else the active one, else `default`.
    When `MODAL_TOKEN_ID` differs from that profile's token id, the second value says the
    environment shadows the profile. Only ids are read, no secret."""
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
    named = [variable for variable in PROVIDERS["modal"][1] if variable in environ]
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
    """Whether Harbor found the provider's SDK, as the module's `_HAS_<PROVIDER>` says. The
    module imports without the SDK, so a successful import proves nothing."""
    try:
        module = importlib.import_module(PROVIDERS[name][0])
    except ImportError:
        return False
    return bool(getattr(module, f"_HAS_{name.upper()}", False))


def extra_missing(name: str) -> list[str]:
    """The packages Harbor's `name` extra requires that are not installed, read from Harbor's
    metadata. The SDK is not the whole extra: Harbor's environment imports the rest (modal's
    `dockerfile-parse`) only when the first sandbox opens."""
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
    """Whether the provider's check passes without its variables, through a login file under
    the home directory or daytona's other pair of variables."""
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
    """A warning when killed runs left enough trial networks to approach Docker's pool limit."""
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
    """How many of Harbor's trial networks have no container on them. These are the
    `<trial>__env` and `<trial>__verifier__<key>` Compose projects that a run killed hard
    did not take down."""
    listed = _docker("network", "ls", "--filter", "dangling=true", "--format", "{{.Name}}")
    names = (listed or "").split()
    return sum(1 for name in names if name.endswith("__env_default") or "__verifier__" in name)


def docker_running() -> bool:
    """Whether a docker daemon answers `docker info` within `DOCKER_SECONDS`."""
    return bool((_docker("info", "--format", "{{.ServerVersion}}") or "").strip())


def _docker(*args: str) -> str | None:
    """`docker <args>`'s stdout, or None when there is no docker, it fails, or it hangs."""
    binary = shutil.which("docker")
    if binary is None:
        return None
    try:
        done = subprocess.run(
            [binary, *args], capture_output=True, text=True, timeout=DOCKER_SECONDS
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def twice_in(dotenv: Path) -> list[Finding]:
    """A warning for each key `.env` assigns more than once. python-dotenv keeps the last
    one (dotenv/main.py:75-84)."""
    counts: dict[str, int] = {}
    for key in assigned(dotenv):
        counts[key] = counts.get(key, 0) + 1
    return [
        Finding("warning", f".env assigns {key} twice; the last one wins")
        for key, count in counts.items()
        if count > 1
    ]


def assigned(dotenv: Path) -> list[str]:
    """Every key `.env` assigns, once per assignment. Returns keys only, no values."""
    try:
        text = dotenv.read_text(encoding="utf-8")
    except OSError:
        return []
    return [binding.key for binding in parse_stream(io.StringIO(text)) if binding.key]


def served_findings(cfg: Blueprint, environ: Mapping[str, str]) -> list[Finding]:
    """Findings for a served model: where its proxy will stand and whether that works here,
    the harness's wiring, and whether the backend serves the model."""
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
                "compaction off. The proxy still refuses a call that overflows the "
                "context, as a harness that compacts expects. Set [rollout] fill_context = "
                "true instead. It sets the same keys and gives such a call what is left",
            )
        )
    found.append(probe(model.name, environ))
    return found


def probe(model: str, environ: Mapping[str, str]) -> Finding:
    """Whether the backend lists the model. This is a network call, made only with a key."""
    name = backend(environ)
    if not environ.get("TINKER_API_KEY"):
        return Finding(
            "warning", f"could not ask {name} whether it serves {model}: TINKER_API_KEY is unset"
        )
    try:
        models = supported_models()
    except Exception as failed:  # noqa: BLE001 - a failed probe is a warning
        return Finding("warning", f"could not ask {name} whether it serves {model}: {failed}")
    if model in models:
        return Finding("ok", f"{name} serves {model}")
    return Finding("blocked", f"{name} does not serve {model}; it serves {', '.join(models)}")


def supported_models() -> list[str]:
    """The backend's model names from `get_server_capabilities`. This is the only Tinker
    call `check` makes. Tests replace it."""
    import tinker

    service = tinker.ServiceClient()
    try:
        caps = service.get_server_capabilities()
    finally:
        service.close("success").result()
    return [str(found.model_name) for found in caps.supported_models]
