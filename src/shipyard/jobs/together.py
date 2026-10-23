"""Harbor's environment for a Together sandbox: `[rollout] sandbox = "together"`.

A trial's sandbox is a VM booted from a snapshot of its task's image. Together's image builder
builds the snapshot from the task's Dockerfile, or pulls its `docker_image`, once for each
content: the snapshot's alias is a hash of what it was built from, so every trial, and every
task, whose environment matches boots from it, and the trials of a process that need one
snapshot wait for a single build. A sandbox is ephemeral: its trial terminates it, and Together
does after `ttl_sec` if nothing does, as when a run is killed.

Commands run as root unless Harbor names a user, as `docker exec` does, rather than as the
sandbox's default user 1000. Files cross as tar archives that root packs and unpacks, which
keep modes and links, and are read back with their encoding, which keeps binary files whole.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import os
import random
import shlex
import tarfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, override
from urllib.parse import quote

import httpx
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import (
    COMPOSE_FILE_NAME,
    DOCKERFILE_NAME,
    effective_exec_cwd,
    parse_dockerfile_workdir,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.environments.tar_transfer import extract_dir, pack_dir_to_bytes

#: The alias namespace of the snapshots this module builds, and the tag its sandboxes carry.
NAMESPACE = "shipyard"
#: How long a sandbox lives if nothing terminates it: a day, as Modal's sandboxes.
TTL_SEC = 24 * 60 * 60
#: Together's bounds on a sandbox: at most 16 CPUs, and from 1 GB to 8 GB of memory per CPU.
#: The floor is taken in GiB and the ceiling in GB, so either reading of "GB" holds.
MAX_CPUS = 16
MIN_MEMORY_PER_CPU = 1024**3
MAX_MEMORY_PER_CPU = 8 * 10**9
#: A task that names no resources gets the SDK's defaults: 1 CPU and 2 GiB.
DEFAULT_CPUS = 1
DEFAULT_MEMORY_MB = 2048
#: How long stopping may take before the trial goes on and leaves the sandbox to its TTL.
STOP_TIMEOUT_SEC = 120.0
#: How long killing a command that outlived its caller may take.
KILL_TIMEOUT_SEC = 10.0
#: How many times a call Together failed in passing is made, in all.
ATTEMPTS = 4
#: Statuses that say the request was refused before it ran, or may not have run.
TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
#: Transport failures before any byte of the request was sent: safe to repeat any call.
UNSENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
#: Management calls that are not safe to repeat once they may have run: a repeat would start
#: a second sandbox, or register a second snapshot.
UNREPEATABLE = frozenset({"api.create_sandbox", "api.create_snapshot"})
#: The in-VM API's timeouts. A command's output is polled, so no call waits on the command.
AGENT_TIMEOUT = httpx.Timeout(120.0, connect=15.0)
#: How often a running command's output is polled: from the first interval, doubling, to the last.
FIRST_POLL_SEC = 0.05
LAST_POLL_SEC = 1.0

#: The snapshot builds in flight in this process, by alias.
BUILDS: dict[str, asyncio.Future[str]] = {}


class TogetherEnvironment(BaseEnvironment):
    """A trial's sandbox on Together, for the run's `[rollout] sandbox = "together"`.

    `sdk` makes the management client, and `agent` connects to a running sandbox's own API;
    both default to Together's, and tests pass stand-ins."""

    def __init__(
        self,
        *,
        ttl_sec: int = TTL_SEC,
        sdk: Callable[[], Any] | None = None,
        agent: Callable[[Any], Agent] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ttl = ttl_sec
        self._new_sdk = sdk if sdk is not None else management
        self._connect = agent if agent is not None else Agent.of
        self._workdir = parse_dockerfile_workdir(self.environment_dir / DOCKERFILE_NAME)
        self._sdk: Any = None
        self._sandbox: Any = None
        self._agent: Agent | None = None

    @staticmethod
    @override
    def type() -> str:
        return "together"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # No GPUs, and no egress control: Harbor refuses a task that needs either.
        return EnvironmentCapabilities()

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # A VM's CPUs and memory are reserved for it, and all it has.
        return EnvironmentResourceCapabilities(
            cpu_limit=True, cpu_request=True, memory_limit=True, memory_request=True
        )

    @classmethod
    @override
    def preflight(cls) -> None:
        if not os.environ.get("TOGETHER_API_KEY"):
            raise SystemExit("Together sandboxes need TOGETHER_API_KEY")

    @override
    def _validate_definition(self) -> None:
        require_agent_environment_definition(
            self.environment_dir, docker_image=self.task_env_config.docker_image
        )
        if (self.environment_dir / COMPOSE_FILE_NAME).exists():
            raise ValueError(
                f"{self.environment_name} defines its environment with {COMPOSE_FILE_NAME}, "
                "and a Together sandbox runs one container: a Dockerfile or a docker_image"
            )

    @override
    async def start(self, force_build: bool) -> None:
        cpus, memory = self._resources()  # before a build, which may take minutes
        self._sdk = self._new_sdk()
        snapshot = await self._snapshot(force_build)
        self._sandbox = await self._sdk.sandboxes.create(
            snapshot_id=snapshot,
            cpu=cpus,
            memory_bytes=memory,
            ttl=self._ttl,
            tags={
                "client": NAMESPACE,
                f"{NAMESPACE}.task": self.environment_name,
                f"{NAMESPACE}.session": self.session_id,
            },
        )
        self.logger.debug(f"Together sandbox {self._sandbox.id} runs {self.session_id}")
        self._agent = self._connect(self._sandbox.vm_info)
        workdir = self.task_env_config.workdir
        if workdir:  # commands run in it, so it is made from the root
            command = self._ensure_dirs_command([workdir], chmod=False)
            await self.exec(command, cwd="/", user="root")
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    @override
    async def stop(self, delete: bool) -> None:
        """Terminate the sandbox, whatever `delete` says: a Together sandbox keeps nothing.
        One that will not terminate is left to its TTL."""
        sandbox, agent, sdk = self._sandbox, self._agent, self._sdk
        self._sandbox = self._agent = self._sdk = None
        try:
            if sandbox is not None:
                try:
                    async with asyncio.timeout(STOP_TIMEOUT_SEC):
                        await sdk.sandboxes.terminate(sandbox.id, snapshot=None)
                except Exception:
                    self.logger.warning(
                        f"could not terminate Together sandbox {sandbox.id}: it ends at its "
                        f"TTL, {self._ttl} s after it started",
                        exc_info=True,
                    )
        finally:
            for one in (agent, sdk):
                if one is not None:
                    try:
                        await one.close()
                    except Exception:
                        self.logger.debug("could not close a Together client", exc_info=True)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        agent = self._connected()
        user = self._resolve_user(user)
        started = await agent.start(
            ["bash", "-c", command],
            cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
            env=self._merge_env(env),
            user="root" if user is None else str(user),
        )
        finished = False
        clock = asyncio.timeout(timeout_sec)
        try:
            async with clock:
                result = await exited(agent, started)
            finished = True
            return result
        except TimeoutError:
            if clock.expired():
                raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
            raise
        finally:
            if not finished:  # timed out, failed or cancelled: the command must not run on
                try:
                    async with asyncio.timeout(KILL_TIMEOUT_SEC):
                        await asyncio.shield(agent.kill(started))
                except Exception:
                    self.logger.debug(f"could not kill command {started}", exc_info=True)

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source, target = Path(source_path), PurePosixPath(target_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file {source} does not exist")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            tar.add(source, arcname=target.name)
        await self._unpack(archive.getvalue(), str(target.parent))

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._unpack(pack_dir_to_bytes(source_dir, compress=True).getvalue(), target_dir)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        staged = _staged()
        copy = f"cp -- {shlex.quote(source_path)} {staged} && chmod 644 {staged}"
        data = await self._fetch(copy, staged, what=f"copy {source_path}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        staged = _staged()
        # GNU tar exits 1 when a file changed as it read it, and still writes the archive.
        pack = (
            f"tar -czf {staged} -C {shlex.quote(source_dir)} . ; "
            f'[ "$?" -le 1 ] && chmod 644 {staged}'
        )
        data = await self._fetch(pack, staged, what=f"pack {source_dir}")
        extract_dir(io.BytesIO(data), target_dir)

    def _connected(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("the Together sandbox is not running: start it first")
        return self._agent

    async def _unpack(self, archive: bytes, into: str) -> None:
        """Unpack a gzipped tar into `into`, as root, which creates it if need be."""
        staged = _staged()
        await self._connected().write(staged, archive)
        result = await self.exec(
            f"mkdir -p {shlex.quote(into)} && "
            f"tar --no-same-owner -xzf {staged} -C {shlex.quote(into)}; "
            f"code=$?; rm -f {staged}; exit $code",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(f"could not unpack into {into}: {_said(result)}")

    async def _fetch(self, command: str, staged: str, *, what: str) -> bytes:
        """Run `command` as root, which writes `staged`, and return what it wrote."""
        result = await self.exec(command, user="root")
        if result.return_code != 0:
            raise RuntimeError(f"could not {what}: {_said(result)}")
        data = await self._connected().read(staged)
        await self.exec(f"rm -f {staged}", user="root")
        return data

    async def _snapshot(self, force_build: bool) -> str:
        """Return the id of the snapshot this environment boots from, built if need be.
        Concurrent trials that need one snapshot wait for a single build."""
        image = should_use_prebuilt_docker_image(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            force_build=force_build,
        )
        source = (
            f"image:{self.task_env_config.docker_image}"
            if image
            else f"context:{self.environment_id}"
        )
        architecture = os.environ.get("TOGETHER_REMOTE_ARCHITECTURE", "amd64")
        digest = hashlib.sha256(f"{architecture}\n{source}".encode()).hexdigest()[:16]
        alias = f"{NAMESPACE}@{digest}"
        if not force_build:
            found = await built(self._sdk, alias)
            if found is not None:
                return found
        building = BUILDS.get(alias)
        if building is None:
            building = asyncio.ensure_future(self._build(alias, image, force_build))
            BUILDS[alias] = building
            building.add_done_callback(lambda done: _settled(alias, done))
        # A trial that stops waiting leaves the build to the others.
        return await asyncio.shield(building)

    async def _build(self, alias: str, image: bool, force_build: bool) -> str:
        """Build the snapshot `alias` names with a client of its own, which outlives the
        trial that started the build."""
        from together_sandbox import CreateContextSnapshotParams, CreateImageSnapshotParams

        sdk = self._new_sdk()
        try:
            if not force_build:
                found = await built(sdk, alias)  # built since this process last looked
                if found is not None:
                    return found
            self.logger.info(f"Together builds snapshot {alias} for {self.environment_name}")
            common: dict[str, Any] = {
                "alias": alias,
                "tags": {"client": NAMESPACE},
                "on_progress": lambda step: self.logger.debug(f"[{step.step}] {step.output}"),
            }
            if image:
                params: Any = CreateImageSnapshotParams(
                    image=str(self.task_env_config.docker_image), **common
                )
            else:
                params = CreateContextSnapshotParams(
                    context=str(self.environment_dir),
                    dockerfile=str(self.environment_dir / DOCKERFILE_NAME),
                    **common,
                )
            made = await sdk.snapshots.create(params)
            return str(made.snapshot_id)
        finally:
            await sdk.close()

    def _resources(self) -> tuple[float, int]:
        """Return the CPUs and memory bytes the sandbox gets: the task's, raised to what
        Together takes, which is no fewer CPUs than memory needs and no less memory than
        the CPUs need."""
        asked = self._effective_cpus or DEFAULT_CPUS
        memory = (self._effective_memory_mb or DEFAULT_MEMORY_MB) * 1024**2
        cpus = max(asked, math.ceil(memory / MAX_MEMORY_PER_CPU))
        if cpus > MAX_CPUS:
            raise ValueError(
                f"{self.environment_name} asks for {asked} CPUs and {memory // 1024**2} MB, "
                f"which take {cpus} CPUs at 8 GB each, and a Together sandbox has at most "
                f"{MAX_CPUS}"
            )
        return float(cpus), max(memory, cpus * MIN_MEMORY_PER_CPU)


@dataclass(frozen=True)
class Output:
    """A chunk of a command's output, in the order the sandbox numbered it."""

    text: str
    sequence: int
    stderr: bool = False
    #: Set on the chunk that says the command exited.
    exit_code: int | None = None


class Agent:
    """A running sandbox's own API, which starts commands, reads their output and moves
    files: plain HTTP, whose status is read before its body, so an error page from a gateway
    is retried as any failure is. A call Together failed in passing is made again only if
    repeating it cannot do twice what it did once: starting a command is repeated only when
    it never left."""

    def __init__(
        self, url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.http = httpx.AsyncClient(
            base_url=url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=AGENT_TIMEOUT,
            transport=transport,
        )

    @classmethod
    def of(cls, vm_info: Any) -> Agent:
        """Connect to the sandbox Together describes as `vm_info`."""
        agent = vm_info.agent
        if not agent or not agent.url or not agent.token:
            raise RuntimeError(f"Together gave sandbox {vm_info.id} no address to reach it at")
        return cls(agent.url, agent.token)

    async def start(
        self, argv: list[str], *, cwd: str | None, env: dict[str, str] | None, user: str
    ) -> str:
        """Start `argv` and return the id of the command."""
        body: dict[str, Any] = {"command": argv[0], "args": argv[1:], "autostart": True}
        if cwd:
            body["cwd"] = cwd
        if env is not None:
            body["env"] = env
        body["user"] = user
        started = await self._call(
            "start a command", "POST", "/api/v1/execs", json=body, repeatable=False
        )
        return str(started["id"])

    async def output(self, command: str, after: int | None) -> list[Output]:
        """Return the command's output after chunk `after`, or all of it."""
        chunks = await self._call(
            f"read the output of command {command}",
            "GET",
            f"/api/v1/execs/{quote(command, safe='')}/io",
            params={} if after is None else {"lastSequence": after},
        )
        return [
            Output(
                one.get("output", ""),
                int(one["sequence"]),
                stderr=one.get("type") == "stderr",
                exit_code=one["exitCode"] if isinstance(one.get("exitCode"), int) else None,
            )
            for one in chunks
        ]

    async def kill(self, command: str) -> None:
        """Kill the command, if it has not ended."""
        await self._call(
            f"kill command {command}",
            "DELETE",
            f"/api/v1/execs/{quote(command, safe='')}",
            missing_ok=True,
        )

    async def read(self, path: str) -> bytes:
        """Return the file's bytes, decoded from the encoding the sandbox sent them in."""
        read = await self._call(f"read {path}", "GET", f"/api/v1/files/{quote(path, safe='')}")
        if read.get("encoding") == "base64":
            return base64.b64decode(read["content"])
        return str(read["content"]).encode("utf-8")

    async def write(self, path: str, data: bytes) -> None:
        """Write `data` to the file at `path`, replacing it."""
        await self._call(
            f"write {path}",
            "POST",
            f"/api/v1/files/{quote(path, safe='')}",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(
        self,
        what: str,
        method: str,
        url: str,
        *,
        repeatable: bool = True,
        missing_ok: bool = False,
        **request: Any,
    ) -> Any:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = await self.http.request(method, url, **request)
            except httpx.TransportError as failed:
                if attempt == ATTEMPTS or not (repeatable or isinstance(failed, UNSENT)):
                    raise RuntimeError(
                        f"could not {what}: {type(failed).__name__}: {failed}"
                    ) from failed
            else:
                if response.is_success:
                    return response.json() if response.content else None
                status = response.status_code
                if status == 404 and missing_ok:
                    return None
                if attempt == ATTEMPTS or not (
                    status == 429 or (repeatable and status in TRANSIENT)
                ):
                    raise RuntimeError(f"could not {what}: HTTP {status}: {response.text[:500]}")
            await asyncio.sleep(_backoff(attempt))
        raise AssertionError("unreachable")


async def exited(agent: Agent, command: str) -> ExecResult:
    """Wait for the command to exit, and return its output and exit code."""
    stdout: list[str] = []
    stderr: list[str] = []
    after: int | None = None
    code: int | None = None
    wait = FIRST_POLL_SEC
    while True:
        for chunk in await agent.output(command, after):
            after = chunk.sequence if after is None else max(after, chunk.sequence)
            (stderr if chunk.stderr else stdout).append(chunk.text)
            if chunk.exit_code is not None:
                code = chunk.exit_code
        if code is not None:
            return ExecResult(stdout="".join(stdout), stderr="".join(stderr), return_code=code)
        await asyncio.sleep(wait)
        wait = min(wait * 2, LAST_POLL_SEC)


async def built(sdk: Any, alias: str) -> str | None:
    """Return the id of the live snapshot `alias` names, or None if there is none."""
    from together_sandbox import HttpError

    try:
        snapshot = await sdk.snapshots.get_by_alias(alias)
    except HttpError as missing:
        if missing.status == 404:
            return None
        raise
    return None if snapshot.retired_at else str(snapshot.id)


def management() -> Any:
    """Return a client of Together's management API, which repeats a failed call unless
    repeating could start a second sandbox or register a second snapshot."""
    from together_sandbox import RetryConfig, TogetherSandbox

    return TogetherSandbox(retry=RetryConfig(max_attempts=ATTEMPTS, should_retry=may_repeat))


def may_repeat(context: Any) -> bool:
    """Whether a failed management call may be made again: `should_retry` for the SDK."""
    status = context.status
    if context.operation in UNREPEATABLE:
        return status == 429 or isinstance(context.error.__cause__, UNSENT)
    return status == 0 or status in TRANSIENT


def _settled(alias: str, build: asyncio.Future[str]) -> None:
    if BUILDS.get(alias) is build:
        del BUILDS[alias]
    if not build.cancelled():
        build.exception()  # retrieved, so a failure no trial waited for is not reported


def _staged() -> str:
    """Return a fresh path under /tmp to stage an archive at, which needs no shell quoting."""
    return f"/tmp/{NAMESPACE}-{uuid.uuid4().hex}"


def _said(result: ExecResult) -> str:
    said = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return said[-500:] or f"exit code {result.return_code}"


def _backoff(attempt: int) -> float:
    return min(0.5 * 2 ** (attempt - 1), 8.0) + random.uniform(0, 0.25)
