"""Tests for `shipyard.jobs.together`: Harbor's environment on a Together sandbox, against
stand-ins for Together's management API and a sandbox's own API. A trial boots from a snapshot
built once for its environment's content, its commands run as root and report their output
apart, its files cross whole, and its sandbox is terminated, or left to its TTL."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("together_sandbox")

from harbor.environments.base import ExecResult  # noqa: E402
from harbor.environments.factory import EnvironmentFactory  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.task.config import EnvironmentConfig as TaskEnvironment  # noqa: E402
from harbor.models.trial.config import AgentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402
from together_sandbox import (  # noqa: E402
    CreateContextSnapshotParams,
    CreateImageSnapshotParams,
    HttpError,
)

from shipyard.jobs import engine, together  # noqa: E402
from shipyard.jobs.together import (  # noqa: E402
    ATTEMPTS,
    MIN_MEMORY_PER_CPU,
    TTL_SEC,
    Agent,
    Output,
    TogetherEnvironment,
    may_repeat,
)

DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"


@pytest.fixture(autouse=True)
def _no_builds_or_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no build in flight, and retries without waiting."""
    monkeypatch.setattr(together, "BUILDS", {})
    monkeypatch.setattr(together, "_backoff", lambda attempt: 0.0)


@dataclass
class Cloud:
    """Together's management API: snapshots by alias, the sandboxes made and terminated,
    and the clients opened and closed."""

    aliases: dict[str, str] = field(default_factory=dict)
    retired: set[str] = field(default_factory=set)
    builds: list[Any] = field(default_factory=list)
    #: Holds each build open until set, so trials can meet at one.
    gate: asyncio.Event | None = None
    #: Raised by the next builds, until cleared.
    failure: Exception | None = None
    made: list[dict[str, Any]] = field(default_factory=list)
    terminated: list[tuple[str, Any]] = field(default_factory=list)
    refuse_to_terminate: bool = False
    clients: int = 0
    closed: int = 0

    def client(self) -> Any:
        self.clients += 1
        return SimpleNamespace(
            snapshots=Snapshots(self), sandboxes=Sandboxes(self), close=self._close
        )

    async def _close(self) -> None:
        self.closed += 1


@dataclass
class Snapshots:
    cloud: Cloud

    async def get_by_alias(self, alias: str) -> Any:
        if alias not in self.cloud.aliases:
            raise HttpError(f"no snapshot {alias}", 404)
        snapshot = self.cloud.aliases[alias]
        retired = "2026-10-01T00:00:00Z" if snapshot in self.cloud.retired else None
        return SimpleNamespace(id=snapshot, retired_at=retired)

    async def create(self, params: Any) -> Any:
        self.cloud.builds.append(params)
        if self.cloud.gate is not None:
            await self.cloud.gate.wait()
        if self.cloud.failure is not None:
            raise self.cloud.failure
        snapshot = f"snapshot-{len(self.cloud.builds)}"
        self.cloud.aliases[params.alias] = snapshot
        return SimpleNamespace(snapshot_id=snapshot, alias=params.alias)


@dataclass
class Sandboxes:
    cloud: Cloud

    async def create(self, **kwargs: Any) -> Any:
        self.cloud.made.append(kwargs)
        name = f"sandbox-{len(self.cloud.made)}"
        agent = SimpleNamespace(url=f"https://{name}.example", token="token")
        return SimpleNamespace(id=name, vm_info=SimpleNamespace(id=name, agent=agent))

    async def terminate(self, sandbox_id: str, *, snapshot: Any) -> None:
        if self.cloud.refuse_to_terminate:
            raise HttpError("terminate failed", 500)
        self.cloud.terminated.append((sandbox_id, snapshot))


Run = Callable[[str, "Box"], tuple[int, str, str]]


class Box:
    """A sandbox's own API: each command runs through `run`, a function of the shell
    command and the box that returns its exit code, stdout and stderr, as chunks numbered
    from 0; `None` never exits. Files are kept by path."""

    def __init__(self, run: Run | None = None, *, chunked: bool = False) -> None:
        self.run = run or (lambda command, box: (0, "", ""))
        self.chunked = chunked
        self.started: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.killed: list[str] = []
        self.closed = False
        self._outputs: dict[str, list[Output]] = {}

    @property
    def commands(self) -> list[str]:
        return [one["argv"][2] for one in self.started]

    async def start(
        self, argv: list[str], *, cwd: str | None, env: dict[str, str] | None, user: str
    ) -> str:
        assert argv[:2] == ["bash", "-c"]
        command = f"command-{len(self.started)}"
        self.started.append({"argv": argv, "cwd": cwd, "env": env, "user": user})
        ran = self.run(argv[2], self)
        if ran is None:
            self._outputs[command] = [Output("started\n", 0)]
            return command
        code, stdout, stderr = ran
        chunks = [Output(stdout, 0), Output(stderr, 1, stderr=True)]
        self._outputs[command] = [*chunks, Output("", 2, exit_code=code)]
        return command

    async def output(self, command: str, after: int | None) -> list[Output]:
        chunks = [one for one in self._outputs[command] if after is None or one.sequence > after]
        return chunks[:1] if self.chunked else chunks

    async def kill(self, command: str) -> None:
        self.killed.append(command)

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def close(self) -> None:
        self.closed = True


def _environment(
    tmp_path: Path,
    cloud: Cloud,
    box: Box | None = None,
    *,
    name: str = "task",
    dockerfile: str | None = DOCKERFILE,
    **task: Any,
) -> TogetherEnvironment:
    environment = tmp_path / name / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    if dockerfile is not None:
        (environment / "Dockerfile").write_text(dockerfile)
    connected = box if box is not None else Box()
    return TogetherEnvironment(
        environment_dir=environment,
        environment_name=name,
        session_id=f"{name}__abc__env",
        trial_paths=TrialPaths(tmp_path / name / "trial"),
        task_env_config=TaskEnvironment(**task),
        sdk=cloud.client,
        agent=lambda vm_info: connected,
    )


# ---------------------------------------------------------------------------------- snapshots


async def test_trials_whose_environments_match_share_one_snapshot_built_once(
    tmp_path: Path,
) -> None:
    cloud = Cloud(gate=asyncio.Event())
    first = _environment(tmp_path, cloud, name="one")
    second = _environment(tmp_path, cloud, name="two")  # another task, the same Dockerfile
    starting = [asyncio.ensure_future(env.start(force_build=False)) for env in (first, second)]
    await asyncio.sleep(0.01)
    assert len(cloud.builds) == 1  # the second waits for the first's build
    cloud.gate.set()
    await asyncio.gather(*starting)
    assert [one["snapshot_id"] for one in cloud.made] == ["snapshot-1", "snapshot-1"]
    third = _environment(tmp_path, cloud, name="three")
    await third.start(force_build=False)
    assert len(cloud.builds) == 1 and cloud.made[-1]["snapshot_id"] == "snapshot-1"
    params = cloud.builds[0]
    assert isinstance(params, CreateContextSnapshotParams)
    assert params.context == str(tmp_path / "one" / "environment")
    assert params.dockerfile == str(tmp_path / "one" / "environment" / "Dockerfile")
    assert re.fullmatch(r"shipyard@[0-9a-f]{16}", params.alias)


async def test_a_changed_environment_builds_its_own_snapshot(tmp_path: Path) -> None:
    cloud = Cloud()
    await _environment(tmp_path, cloud, name="one").start(force_build=False)
    changed = _environment(tmp_path, cloud, name="two", dockerfile=DOCKERFILE + "RUN true\n")
    await changed.start(force_build=False)
    assert len(cloud.builds) == 2
    assert cloud.builds[0].alias != cloud.builds[1].alias


async def test_a_retired_snapshot_is_built_again(tmp_path: Path) -> None:
    cloud = Cloud()
    await _environment(tmp_path, cloud, name="one").start(force_build=False)
    cloud.retired.add("snapshot-1")
    await _environment(tmp_path, cloud, name="two").start(force_build=False)
    assert [one["snapshot_id"] for one in cloud.made] == ["snapshot-1", "snapshot-2"]


async def test_a_failed_build_fails_the_trials_waiting_on_it_and_the_next_builds_again(
    tmp_path: Path,
) -> None:
    cloud = Cloud(gate=asyncio.Event(), failure=RuntimeError("the build ran out of memory"))
    waiting = [_environment(tmp_path, cloud, name=name) for name in ("one", "two")]
    starting = [asyncio.ensure_future(env.start(force_build=False)) for env in waiting]
    await asyncio.sleep(0.01)
    cloud.gate.set()
    outcomes = await asyncio.gather(*starting, return_exceptions=True)
    assert [str(one) for one in outcomes] == ["the build ran out of memory"] * 2
    assert len(cloud.builds) == 1 and together.BUILDS == {}
    cloud.failure = None
    await _environment(tmp_path, cloud, name="three").start(force_build=False)
    assert len(cloud.builds) == 2 and cloud.made[-1]["snapshot_id"] == "snapshot-2"


async def test_a_trial_that_stops_waiting_leaves_the_build_to_the_others(tmp_path: Path) -> None:
    cloud = Cloud(gate=asyncio.Event())
    first = asyncio.ensure_future(_environment(tmp_path, cloud, name="one").start(False))
    second = asyncio.ensure_future(_environment(tmp_path, cloud, name="two").start(False))
    await asyncio.sleep(0.01)
    first.cancel()
    await asyncio.sleep(0)
    cloud.gate.set()
    await second
    assert first.cancelled()
    assert len(cloud.builds) == 1 and cloud.made[-1]["snapshot_id"] == "snapshot-1"


async def test_a_docker_image_is_pulled_into_a_snapshot_of_its_own(tmp_path: Path) -> None:
    cloud = Cloud()
    image = "ghcr.io/example/task:1.0"
    await _environment(tmp_path, cloud, name="one", dockerfile=None, docker_image=image).start(
        force_build=False
    )
    await _environment(tmp_path, cloud, name="two").start(force_build=False)
    pulled, built = cloud.builds
    assert isinstance(pulled, CreateImageSnapshotParams) and pulled.image == image
    assert isinstance(built, CreateContextSnapshotParams)
    assert pulled.alias != built.alias


def test_an_environment_with_no_image_or_with_compose_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _environment(tmp_path, Cloud(), name="bare", dockerfile=None)
    composed = tmp_path / "composed" / "environment"
    composed.mkdir(parents=True)
    (composed / "docker-compose.yaml").write_text("services: {}\n")
    with pytest.raises(ValueError, match="runs one container"):
        _environment(tmp_path, Cloud(), name="composed")


# ---------------------------------------------------------------------------------- sandboxes


async def test_a_sandbox_starts_with_its_ttl_and_tags_and_the_dirs_harbor_mounts(
    tmp_path: Path,
) -> None:
    cloud, box = Cloud(), Box()
    env = _environment(tmp_path, cloud, box, workdir="/srv/task")
    env._mounts = [{"type": "bind", "source": str(tmp_path), "target": "/logs/agent"}]
    await env.start(force_build=False)
    made = cloud.made[0]
    assert made["ttl"] == TTL_SEC
    assert made["tags"] == {
        "client": "shipyard",
        "shipyard.task": "task",
        "shipyard.session": "task__abc__env",
    }
    workdir, mounted = box.started
    assert workdir["cwd"] == "/" and workdir["user"] == "root"
    assert "mkdir -p /srv/task" in workdir["argv"][2]
    assert mounted["user"] == "root"
    assert "mkdir -p /logs/agent" in mounted["argv"][2] and "chmod 777" in mounted["argv"][2]


@pytest.mark.parametrize(
    ("cpus", "memory_mb", "expected"),
    [
        (None, None, (1.0, 2048 * 1024**2)),
        (1, 512, (1.0, MIN_MEMORY_PER_CPU)),  # raised to Together's floor
        (2, 4096, (2.0, 4096 * 1024**2)),
        (1, 16384, (3.0, 16384 * 1024**2)),  # 8 GB a CPU at most
    ],
)
async def test_a_sandbox_gets_the_tasks_resources_raised_to_what_together_takes(
    tmp_path: Path, cpus: int | None, memory_mb: int | None, expected: tuple[float, int]
) -> None:
    cloud = Cloud()
    await _environment(tmp_path, cloud, cpus=cpus, memory_mb=memory_mb).start(False)
    assert (cloud.made[0]["cpu"], cloud.made[0]["memory_bytes"]) == expected


async def test_a_task_asking_for_more_than_a_sandbox_has_is_refused_before_a_build(
    tmp_path: Path,
) -> None:
    cloud = Cloud()
    with pytest.raises(ValueError, match="at most 16"):
        await _environment(tmp_path, cloud, cpus=8, memory_mb=200_000).start(False)
    assert cloud.builds == [] and cloud.made == []


async def test_stopping_terminates_the_sandbox_with_no_snapshot_and_closes_its_clients(
    tmp_path: Path,
) -> None:
    cloud, box = Cloud(), Box()
    env = _environment(tmp_path, cloud, box)
    await env.start(force_build=False)
    await env.stop(delete=False)  # a Together sandbox keeps nothing either way
    assert cloud.terminated == [("sandbox-1", None)]
    assert box.closed and cloud.closed == cloud.clients
    await env.stop(delete=True)  # once stopped, nothing is left to stop
    assert len(cloud.terminated) == 1


async def test_a_sandbox_that_will_not_terminate_is_left_to_its_ttl(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cloud = Cloud(refuse_to_terminate=True)
    env = _environment(tmp_path, cloud)
    await env.start(force_build=False)
    with caplog.at_level(logging.WARNING):
        await env.stop(delete=True)
    assert "could not terminate Together sandbox sandbox-1" in caplog.text
    assert cloud.closed == cloud.clients


async def test_a_trial_whose_sandbox_never_started_stops_cleanly(tmp_path: Path) -> None:
    cloud = Cloud(failure=RuntimeError("no capacity"))
    env = _environment(tmp_path, cloud)
    with pytest.raises(RuntimeError, match="no capacity"):
        await env.start(force_build=False)
    await env.stop(delete=True)
    assert cloud.terminated == [] and cloud.closed == cloud.clients


# ----------------------------------------------------------------------------------- commands


async def test_commands_run_as_root_unless_harbor_names_a_user(tmp_path: Path) -> None:
    box = Box(lambda command, box: (0, "ok\n", ""))
    env = _environment(tmp_path, Cloud(), box, env={"TASK": "1"})
    await env.start(force_build=False)
    await env.exec("whoami")
    with env.with_default_user("agent"):
        await env.exec("whoami", env={"CALL": "2"})
    await env.exec("whoami", user=1000, cwd="/tmp")
    root, agent, uid = box.started
    assert (root["user"], agent["user"], uid["user"]) == ("root", "agent", "1000")
    assert root["cwd"] == "/app" and uid["cwd"] == "/tmp"  # the Dockerfile's WORKDIR
    assert root["env"] == {"TASK": "1"} and agent["env"] == {"TASK": "1", "CALL": "2"}


async def test_a_command_reports_its_stdout_and_stderr_apart(tmp_path: Path) -> None:
    box = Box(lambda command, box: (3, "out\n", "err\n"), chunked=True)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    assert await env.exec("false") == ExecResult(stdout="out\n", stderr="err\n", return_code=3)


async def test_a_command_past_its_timeout_is_killed(tmp_path: Path) -> None:
    box = Box(lambda command, box: None if command == "sleep 600" else (0, "", ""))
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    with pytest.raises(RuntimeError, match="timed out after 0.05 seconds"):
        await env.exec("sleep 600", timeout_sec=0.05)  # type: ignore[arg-type]
    assert box.killed == ["command-0"]


async def test_a_cancelled_command_is_killed(tmp_path: Path) -> None:
    box = Box(lambda command, box: None)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    running = asyncio.ensure_future(env.exec("sleep 600"))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert box.killed == ["command-0"]


async def test_a_command_that_ran_to_its_end_is_not_killed(tmp_path: Path) -> None:
    box = Box(lambda command, box: (0, "", ""))
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    await env.exec("true", timeout_sec=5)
    assert box.killed == []


# -------------------------------------------------------------------------------------- files


def _members(archive: bytes) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {one.name: one for one in tar.getmembers()}


def _unpacked(command: str, box: Box) -> tuple[int, str, str]:
    """Check an upload's command, which unpacks a staged archive as root, and leave the
    archive in the box for the test to read."""
    match = re.fullmatch(
        r"mkdir -p (\S+) && tar --no-same-owner -xzf (\S+) -C \1; code=\$\?; rm -f \2; "
        r"exit \$code",
        command,
    )
    assert match is not None, command
    box.files["unpacked into " + match[1]] = box.files[match[2]]
    return 0, "", ""


async def test_a_file_goes_up_as_an_archive_root_unpacks_keeping_its_mode(
    tmp_path: Path,
) -> None:
    script = tmp_path / "solve.sh"
    script.write_text("#!/bin/sh\necho solved\n")
    script.chmod(0o755)
    box = Box(_unpacked)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    await env.upload_file(script, "/solution/run.sh")
    (member,) = _members(box.files["unpacked into /solution"]).values()
    assert member.name == "run.sh" and member.mode & 0o777 == 0o755
    assert box.started[-1]["user"] == "root"


async def test_a_directory_goes_up_whole(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    (tests / "data").mkdir(parents=True)
    (tests / "test.sh").write_text("exit 0\n")
    (tests / "data" / "blob.bin").write_bytes(bytes(range(256)))
    box = Box(_unpacked)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    await env.upload_dir(tests, "/tests")
    members = _members(box.files["unpacked into /tests"])
    assert {"./test.sh", "./data", "./data/blob.bin"} <= set(members)


async def test_a_binary_file_comes_back_whole(tmp_path: Path) -> None:
    blob = bytes(range(256)) * 4

    def run(command: str, box: Box) -> tuple[int, str, str]:
        copied = re.fullmatch(r"cp -- /logs/agent/shot\.png (\S+) && chmod 644 \1", command)
        if copied:
            box.files[copied[1]] = blob
        return 0, "", ""

    box = Box(run)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    target = tmp_path / "out" / "shot.png"
    await env.download_file("/logs/agent/shot.png", target)
    assert target.read_bytes() == blob
    assert box.commands[-1].startswith("rm -f /tmp/shipyard-")  # the staged copy is removed


async def test_a_directory_comes_back_whole(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    (logs / "sub").mkdir(parents=True)
    (logs / "trajectory.json").write_text("{}")
    (logs / "sub" / "blob.bin").write_bytes(bytes(range(256)))

    def run(command: str, box: Box) -> tuple[int, str, str]:
        packed = re.fullmatch(
            r'tar -czf (\S+) -C /logs/agent \. ; \[ "\$\?" -le 1 \] && chmod 644 \1', command
        )
        if packed:
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                tar.add(logs, arcname=".")
            box.files[packed[1]] = archive.getvalue()
        return 0, "", ""

    box = Box(run)
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    await env.download_dir("/logs/agent", tmp_path / "copy")
    assert (tmp_path / "copy" / "trajectory.json").read_text() == "{}"
    assert (tmp_path / "copy" / "sub" / "blob.bin").read_bytes() == bytes(range(256))


async def test_a_download_that_fails_says_why(tmp_path: Path) -> None:
    box = Box(lambda command, box: (2, "", "tar: /logs/agent: Cannot open"))
    env = _environment(tmp_path, Cloud(), box)
    await env.start(force_build=False)
    with pytest.raises(RuntimeError, match="could not pack /logs/agent: tar: /logs/agent"):
        await env.download_dir("/logs/agent", tmp_path / "copy")


# ------------------------------------------------------------------------- the wire, and Harbor


class Server:
    """A sandbox's own API over HTTP: each request is answered by the next of `replies` for
    its method and path, an `httpx` exception to raise or a status and JSON body."""

    def __init__(self, replies: dict[str, list[Any]]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.raw_path.split(b"?")[0].decode()  # still percent-encoded
        reply = self.replies[f"{request.method} {path}"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        if isinstance(body, str):  # a page, such as a gateway's
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    def agent(self) -> Agent:
        return Agent("https://agent.example", "token", transport=httpx.MockTransport(self))


EXEC = {
    "id": "exec-1",
    "command": "bash",
    "args": ["-c", "true"],
    "status": "running",
    "pid": 7,
    "pty": False,
    "exitCode": 0,
}


async def test_a_command_is_started_as_the_sandbox_api_expects() -> None:
    server = Server({"POST /api/v1/execs": [(201, EXEC)]})
    started = await server.agent().start(
        ["bash", "-c", "ls"], cwd="/app", env={"A": "1"}, user="root"
    )
    assert started == "exec-1"
    (request,) = server.requests
    assert json.loads(request.content) == {
        "command": "bash",
        "args": ["-c", "ls"],
        "autostart": True,
        "cwd": "/app",
        "env": {"A": "1"},
        "user": "root",
    }
    assert request.headers["authorization"] == "Bearer token"


@pytest.mark.parametrize(
    ("first", "repeated"),
    [
        (httpx.ConnectError("refused"), True),  # never left: safe to send again
        ((429, {"message": "slow down"}), True),  # refused before it ran
        (httpx.ReadTimeout("no answer"), False),  # may have started: never twice
        ((503, {"message": "unavailable"}), False),
    ],
)
async def test_starting_a_command_is_repeated_only_when_it_cannot_have_run(
    first: Any, repeated: bool
) -> None:
    server = Server({"POST /api/v1/execs": [first, (201, EXEC)]})
    if repeated:
        assert await server.agent().start(["true"], cwd=None, env=None, user="root") == "exec-1"
    else:
        with pytest.raises(RuntimeError, match="could not start a command"):
            await server.agent().start(["true"], cwd=None, env=None, user="root")
    assert len(server.requests) == 1 + repeated


async def test_output_is_read_after_the_last_chunk_and_repeated_in_passing() -> None:
    chunks = [
        {"type": "stdout", "output": "out", "sequence": 4},
        {"type": "stderr", "output": "err", "sequence": 5},
        {"type": "stdout", "output": "", "sequence": 6, "exitCode": 1},
    ]
    failures = [(502, "<html>Bad Gateway</html>"), httpx.ReadTimeout("no answer")]
    server = Server({"GET /api/v1/execs/exec-1/io": [*failures, (200, chunks)]})
    assert await server.agent().output("exec-1", 3) == [
        Output("out", 4),
        Output("err", 5, stderr=True),
        Output("", 6, exit_code=1),
    ]
    assert [one.url.params.get("lastSequence") for one in server.requests] == ["3"] * 3


async def test_a_failure_that_lasts_is_raised_after_the_last_attempt() -> None:
    server = Server({"GET /api/v1/execs/exec-1/io": [(503, {})] * ATTEMPTS})
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await server.agent().output("exec-1", None)
    assert len(server.requests) == ATTEMPTS


@pytest.mark.parametrize(
    ("encoding", "content", "expected"),
    [
        ("base64", base64.b64encode(b"\x89PNG\x00\xff").decode(), b"\x89PNG\x00\xff"),
        ("utf-8", "plain text é", "plain text é".encode()),
    ],
)
async def test_a_file_is_read_in_the_encoding_the_sandbox_sent(
    encoding: str, content: str, expected: bytes
) -> None:
    path = "/api/v1/files/%2Ftmp%2Fx"
    body = {"path": "/tmp/x", "content": content, "encoding": encoding}
    server = Server({f"GET {path}": [(200, body)]})
    assert await server.agent().read("/tmp/x") == expected


async def test_a_file_is_written_as_raw_bytes_and_a_gone_command_is_killed_quietly() -> None:
    written = {"path": "/tmp/x", "content": "", "encoding": "utf-8"}
    server = Server(
        {
            "POST /api/v1/files/%2Ftmp%2Fx": [(201, written)],
            "DELETE /api/v1/execs/exec-1": [(404, {"code": 404, "message": "no exec"})],
        }
    )
    agent = server.agent()
    await agent.write("/tmp/x", b"\x00\x01")
    await agent.kill("exec-1")
    upload, _ = server.requests
    assert upload.content == b"\x00\x01"
    assert upload.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize(
    ("operation", "status", "cause", "repeated"),
    [
        ("api.create_sandbox", 0, httpx.ConnectError("refused"), True),
        ("api.create_sandbox", 429, None, True),
        ("api.create_sandbox", 0, httpx.ReadTimeout("no answer"), False),
        ("api.create_sandbox", 503, None, False),
        ("api.create_snapshot", 502, None, False),
        ("api.get_snapshot_by_alias", 503, None, True),
        ("api.terminate_sandbox", 0, httpx.ReadTimeout("no answer"), True),
        ("api.get_snapshot_by_alias", 404, None, False),
    ],
)
def test_a_management_call_is_repeated_unless_that_could_make_a_second_one(
    operation: str, status: int, cause: Exception | None, repeated: bool
) -> None:
    error = HttpError("failed", status)
    error.__cause__ = cause
    context = SimpleNamespace(operation=operation, status=status, error=error)
    assert may_repeat(context) is repeated


def test_a_trial_runs_in_shipyards_together_environment_through_harbor(tmp_path: Path) -> None:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    agent = AgentConfig(name="oracle")
    config = engine._trial_config(task, trials_dir=tmp_path, agent=agent, sandbox="together")
    assert config.environment.import_path == "shipyard.jobs.together:TogetherEnvironment"
    assert config.environment.type is None
    made = EnvironmentFactory.create_environment_from_config(
        config.environment,
        environment_dir=task / "environment",
        environment_name="task",
        session_id="task__abc__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=TaskEnvironment(),
    )
    assert isinstance(made, TogetherEnvironment) and made.type() == "together"
    modal = engine._trial_config(task, trials_dir=tmp_path, agent=agent, sandbox="modal")
    assert modal.environment.type == EnvironmentType.MODAL


def test_a_together_run_without_the_package_is_refused_with_its_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "find_spec", lambda name: None)
    needs = r"needs the together_sandbox package: .*shipyard\[together\]"
    with pytest.raises(ValueError, match=needs):
        engine.installed("together")
    engine.installed("docker")  # Harbor's own, with nothing to install
