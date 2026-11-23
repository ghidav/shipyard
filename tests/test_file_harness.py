"""Tests for a harness given as a Python file: how a config names and finds it, what Harbor is told
to run, the adapter on this machine (`shipyard.jobs.file`), the runner in the sandbox
(`shipyard.jobs.runner`), and the trajectory made from the runner's record."""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from shipyard.config import ConfigError, load
from shipyard.groups import Where
from shipyard.jobs import runner
from shipyard.jobs.file import HOME, MARGIN, RUNNER, FileHarness, given, trajectory
from shipyard.jobs.harnesses import FILE_AGENT, Harness, Profile, unrunnable
from shipyard.jobs.modules import Module, Modules, deliver
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.directory import continued, open_run, reopened
from tests.fakes import FakeTrials
from tests.runs import GRPO, GRPO_FILE, run_toml
from tests.test_engine import _batch, _engine, _run

#: A harness file: it asks the model once, and writes what it was given beside the task.
HARNESS = """import json
import os


def run(task, model):
    reply = model([{"role": "user", "content": task}], max_tokens=16)
    seen = {name: os.environ.get(name) for name in ("OPENAI_API_KEY", "HARNESS_SECONDS", "MINE")}
    with open("seen.json", "w") as out:
        json.dump({"task": task, "said": reply["content"], "finish": reply.finish, **seen}, out)
"""
#: `run_toml`'s config, its harness given as the file beside it.
FILE = {'harness = "pi@0.85.1"': 'harness = "harness.py"'}
WHERE = Where(
    "https://x.trycloudflare.com/r/trial/t/v1",
    local="http://127.0.0.1:8000/r/trial/t/v1",
    token="k",
    host="x.trycloudflare.com",
)
BASH = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}


def written(home: Path, text: str = HARNESS, name: str = "harness.py") -> Path:
    """Write a harness file in `home`, and return it."""
    file = home / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file


# How it is named and found.


def test_a_name_ending_in_py_is_a_harness_given_as_a_file() -> None:
    harness = Harness(name="runs/a@b/harnesses/0002.py", env={"MINE": "1"})
    assert harness.file == Path("runs/a@b/harnesses/0002.py")
    assert harness.agent == FILE_AGENT == FileHarness.import_path(), "what Harbor runs it as"
    assert harness.bare == harness.name and harness.version is None, "a path is left whole"
    assert harness.profile == Profile() and not harness.runs_without_network
    dialled = harness.dial(WHERE, "Qwen/Qwen3-8B", fill_context=False)
    assert dialled.model == "openai/Qwen/Qwen3-8B"
    assert dialled.env == {"OPENAI_BASE_URL": WHERE.address, "OPENAI_API_KEY": "k"}
    for other in ("pi@0.85.1", "some.module:Agent"):
        assert Harness(name=other).file is None and Harness(name=other).agent != FILE_AGENT


def test_a_config_finds_its_harness_file_from_its_own_folder(tmp_path: Path) -> None:
    home = run_toml(tmp_path, **FILE)
    file = written(home)
    assert load(home, GRPO.Recipe).recipe.harness == Harness(name=str(file))
    below = written(home, name="harnesses/seed.py")
    table = {'harness = "pi@0.85.1"': 'harness = { name = "harnesses/seed.py", env = { A = "1" } }'}
    assert load(run_toml(tmp_path, **table), GRPO.Recipe).recipe.harness == Harness(
        name=str(below), env={"A": "1"}
    )
    elsewhere = written(tmp_path / "elsewhere")
    absolute = {'harness = "pi@0.85.1"': f'harness = "{elsewhere}"'}
    assert load(run_toml(tmp_path, **absolute), GRPO.Recipe).recipe.harness.file == elsewhere
    assert Harness(name="harness.py").file == Path("harness.py"), "one made in code is as given"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (None, "no harness file to read at {file}: No such file or directory"),
        ("def run(task, model:\n", "{file} is not Python: '(' was never closed, at line 1"),
        ("def main(task, model):\n    pass\n", "{file} defines no `run(task, model)`"),
        ("async def run(task, model):\n    pass\n", "{file} defines no `run(task, model)`"),
    ],
)
def test_a_file_that_cannot_run_is_refused_before_any_trial(
    tmp_path: Path, text: str | None, problem: str
) -> None:
    """Read and parsed, never run: a trial of such a file would end at once and be graded."""
    home = run_toml(tmp_path, **FILE)
    file = home / "harness.py"
    if text is not None:
        written(home, text)
    said = problem.format(file=file)
    assert (unrunnable(file) or "").startswith(said)
    with pytest.raises(ConfigError) as refused:
        load(home, GRPO.Recipe)
    assert refused.value.problems[0].startswith(f"[recipe.harness] name: {said}")
    assert (
        unrunnable(written(tmp_path, "import nothing_here\n\ndef run(task, model): ...\n")) is None
    )


def test_a_run_that_goes_on_finds_the_file_where_its_config_was_written(tmp_path: Path) -> None:
    """The run holds a copy of run.toml alone, so it notes the folder the config came from."""
    home = run_toml(tmp_path, **FILE)
    file = written(home).resolve()
    directory, config = open_run(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    assert config.recipe.harness.file == file
    assert files.read_json(directory / files.RUN)["home"] == str(home.resolve())
    assert not (directory / "harness.py").exists()
    with pytest.raises(ConfigError, match="no harness file to read at"):
        load(directory, GRPO.Recipe)  # as the run's folder alone would read it
    (directory / files.CHECKPOINTS).write_text('{"tag": "step-1", "step": 1}\n', encoding="utf-8")
    again, checkpoint = reopened(directory)
    assert again.recipe.harness.file == file and checkpoint == Checkpoint("step-1", 1)
    resumed, last = continued(directory, checkpoint)
    assert last.recipe.harness.file == file
    assert files.read_json(resumed / files.RUN)["home"] == str(home.resolve()), "handed on"


# What Harbor is told to run.


async def test_the_agent_config_of_a_file_names_the_adapter_the_file_and_the_runs_limit(
    tmp_path: Path,
) -> None:
    batch, file = _batch(tmp_path, "alpha"), written(tmp_path)
    harbor = FakeTrials()
    deployed = Harness(name=str(file), env={"MINE": "1"})
    await _run(_engine(tmp_path, harbor, timeout=900.0), "j", batch, 1, deployed=deployed)
    agent = harbor.configs[0].agent
    assert (agent.name, agent.import_path) == (FILE_AGENT, None)
    assert agent.kwargs == {"file": str(file), "timeout": 900.0}
    assert agent.env["MINE"] == "1" and agent.model_name == "anthropic/some-model"
    made = AgentFactory.create_agent_from_config(  # as a Harbor trial makes its agent
        agent, logs_dir=tmp_path, skills_dir="/harbor/skills"
    )
    assert isinstance(made, FileHarness) and made.capabilities.skills
    assert (made.file, made.timeout, made.skills_dir) == (file, 900.0, "/harbor/skills")
    assert made.extra_env["MINE"] == "1", "its own `env`, which the trial gives every command"

    harbor = FakeTrials()
    carried = Modules({"prompt": Module("prompt", "prompt", {"PROMPT.md": "Check your work."})})
    await _run(_engine(tmp_path, harbor), "k", batch, 1, deployed=deployed, modules=carried)
    kwargs = harbor.configs[0].agent.kwargs
    assert kwargs["timeout"] is None, "each task's own limit"
    assert Path(kwargs["prompt_template_path"]).is_file(), "its prompt, which the adapter renders"


async def test_a_job_refuses_a_file_that_cannot_run(tmp_path: Path) -> None:
    """A recipe that writes its harnesses makes them in code, past the config's check."""
    batch, harbor = _batch(tmp_path, "alpha"), FakeTrials()
    missing = Harness(name=str(tmp_path / "0003.py"))
    with pytest.raises(ValueError, match="job j: no harness file to read at"):
        await _run(_engine(tmp_path, harbor), "j", batch, 1, deployed=missing)
    empty = Harness(name=str(written(tmp_path, "x = 1\n")))
    with pytest.raises(ValueError, match=r"job j: .* defines no `run\(task, model\)`"):
        await _run(_engine(tmp_path, harbor), "j", batch, 1, deployed=empty)
    assert harbor.configs == []


# The adapter.


class Sandbox:
    """Stands in for a trial's environment: it keeps the commands it is asked to run and the
    files it is given, and answers each command with the next of `codes`, then 0."""

    def __init__(self, task: Path, *codes: int) -> None:
        self.environment_dir = task / "environment"
        self.codes = list(codes)
        self.ran: list[dict[str, Any]] = []
        self.files: dict[str, str] = {}

    async def exec(
        self, command: str, env: dict[str, str] | None = None, user: Any = None, **_: Any
    ) -> ExecResult:
        self.ran.append({"command": command, "env": env, "user": user})
        return ExecResult(return_code=self.codes.pop(0) if self.codes else 0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.files[target_path] = Path(source_path).read_text(encoding="utf-8")


def task_at(tmp_path: Path, limit: float | None = 600.0) -> Path:
    """Write a task whose agent has `limit` seconds, and return its folder."""
    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True, exist_ok=True)
    said = f"[agent]\ntimeout_sec = {limit}\n" if limit else '[metadata]\ndifficulty = "easy"\n'
    (task / "task.toml").write_text(said, encoding="utf-8")
    return task


def adapter(tmp_path: Path, **kwargs: Any) -> FileHarness:
    logs = tmp_path / "agent"
    logs.mkdir(exist_ok=True)
    return FileHarness(
        logs, model_name="openai/Qwen/Qwen3-8B", file=str(written(tmp_path)), **kwargs
    )


async def test_setup_finds_a_python_and_uploads_the_runner_and_the_file(tmp_path: Path) -> None:
    sandbox, agent = Sandbox(task_at(tmp_path)), adapter(tmp_path)
    await agent.setup(sandbox)  # type: ignore[arg-type]
    assert [one["user"] for one in sandbox.ran] == ["root"] and "apt-get" in sandbox.ran[0][
        "command"
    ]
    assert sandbox.files == {
        f"{HOME}/runner.py": RUNNER.read_text(encoding="utf-8"),
        f"{HOME}/harness.py": HARNESS,
    }
    other = FileHarness(tmp_path / "agent", file=str(written(tmp_path, "x = 1\n", "other.py")))
    assert len(agent.version()) == 12 and agent.version() != other.version(), "the file's digest"
    with pytest.raises(RuntimeError, match="no Python to run the harness on"):
        await agent.setup(Sandbox(task_at(tmp_path), 127))  # type: ignore[arg-type]


async def test_a_trial_runs_the_runner_with_its_task_its_limit_and_its_skills(
    tmp_path: Path,
) -> None:
    sandbox = Sandbox(task_at(tmp_path, 600.0))
    agent = adapter(tmp_path)
    await agent.run("Fix the bug.", sandbox, AgentContext())  # type: ignore[arg-type]
    assert sandbox.files[f"{HOME}/task.md"] == "Fix the bug."
    (ran,) = sandbox.ran
    assert f"{HOME}/runner.py > /logs/agent/harness.txt" in ran["command"]
    assert ran["env"] == {
        "HARNESS_FILE": f"{HOME}/harness.py",
        "HARNESS_TASK": f"{HOME}/task.md",
        "HARNESS_RECORD": "/logs/agent/calls.jsonl",
        "HARNESS_MODEL": "Qwen/Qwen3-8B",
        "HARNESS_SECONDS": str(600 - MARGIN),
    }, "the task's own limit less the margin, and no skills"
    assert (tmp_path / "agent" / "seconds.txt").read_text(encoding="utf-8") == "570\n"

    template = deliver(
        Modules({"prompt": Module("prompt", "prompt", {"PROMPT.md": "Check your work."})}),
        tmp_path / "delivered",
    )["kwargs"]
    carrying = adapter(tmp_path, timeout=900.0, skills_dir="/harbor/skills", **template)
    sandbox = Sandbox(task_at(tmp_path, 600.0))
    await carrying.run("Fix the bug.", sandbox, AgentContext())  # type: ignore[arg-type]
    assert sandbox.files[f"{HOME}/task.md"] == "Check your work.\n\nFix the bug.", "its prompt"
    env = sandbox.ran[0]["env"]
    assert env["HARNESS_SKILLS"] == "/harbor/skills", "where Harbor uploaded them"
    assert env["HARNESS_SECONDS"] == "870", "the run's limit, over the task's own"


def test_the_seconds_a_trial_gives_are_its_limit_less_the_margin(tmp_path: Path) -> None:
    assert given(task_at(tmp_path, 3000.0)) == 2970
    assert given(task_at(tmp_path, 3000.0), 900.0) == 870, "`[rollout] timeout` wins"
    assert given(task_at(tmp_path, 20.0)) == 1, "never none"
    assert given(task_at(tmp_path, None)) == 0, "a task with no limit: the runner sets none"
    assert given(tmp_path / "missing") == 0


@pytest.mark.parametrize(
    ("code", "raised"),
    [(127, "no python3 or python"), (75, "could not reach the model"), (1, None), (0, None)],
)
async def test_a_trial_fails_only_for_what_says_nothing_about_the_harness(
    tmp_path: Path, code: int, raised: str | None
) -> None:
    """A harness that raised ended its own trial, and the tests grade what it left."""
    sandbox, agent = Sandbox(task_at(tmp_path), code), adapter(tmp_path)
    if raised is None:
        await agent.run("Do it.", sandbox, AgentContext())  # type: ignore[arg-type]
        return
    with pytest.raises(RuntimeError, match=raised):
        await agent.run("Do it.", sandbox, AgentContext())  # type: ignore[arg-type]


# The trajectory.

CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
}
SYSTEM = {"role": "system", "content": "You are in a shell."}
USER = {"role": "user", "content": [{"type": "text", "text": "Fix the bug."}]}
RECORD = [
    {
        "at": 100.0,
        "kept": 0,
        "messages": [SYSTEM, USER],
        "tools": [BASH],
        "seconds": 2.0,
        "reply": {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning": "Look."},
        "usage": {"prompt_tokens": 50, "completion_tokens": 7},
        "finish": "tool_calls",
    },
    {
        "at": 103.0,
        "kept": 2,
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
        ],
        "seconds": 1.0,
        "error": "prompt is too long",
    },
    {
        "at": 105.0,
        "kept": 4,
        "messages": [],
        "seconds": 1.0,
        "reply": {"role": "assistant", "content": "Done."},
        "usage": {"prompt_tokens": 60, "completion_tokens": 3},
        "finish": "stop",
    },
    {"end": "raised", "error": "Traceback: KeyError"},
]


def test_the_trajectory_is_made_from_the_runners_record() -> None:
    found = trajectory(RECORD, session="s", version="abc", model="openai/m")
    assert [step.source for step in found.steps] == [
        "system",
        "user",
        "agent",
        "system",
        "agent",
        "system",
    ]
    system, user, called, refused, answered, ended = found.steps
    assert (system.message, user.message) == ("You are in a shell.", "Fix the bug.")
    assert called.reasoning_content == "Look." and called.metrics.prompt_tokens == 50
    (call,) = called.tool_calls
    assert (call.function_name, call.arguments) == ("bash", {"command": "ls"})
    (result,) = called.observation.results
    assert (result.source_call_id, result.content) == ("c1", "a.txt"), "as the next call sent it"
    assert refused.message == "[the call was not answered] prompt is too long"
    assert answered.message == "Done." and answered.tool_calls is None
    assert ended.message == "[the harness: raised] Traceback: KeyError"
    assert found.agent.name == "file-harness" and found.agent.version == "abc"
    assert found.agent.tool_definitions == [BASH]
    totals = found.final_metrics
    assert (totals.total_prompt_tokens, totals.total_completion_tokens) == (110, 10)
    assert totals.total_steps == 6


def test_a_call_whose_arguments_are_not_json_is_kept_as_written() -> None:
    broken = {**CALL, "function": {"name": "bash", "arguments": '{"command": "ls"]}'}}
    line = {**RECORD[0], "reply": {"role": "assistant", "content": "", "tool_calls": [broken]}}
    (call,) = trajectory([line], session=None, version="v", model=None).steps[-1].tool_calls
    assert call.arguments == {"raw": '{"command": "ls"]}'}


def test_the_adapter_writes_the_trajectory_beside_the_record(tmp_path: Path) -> None:
    agent, context = adapter(tmp_path), AgentContext()
    agent.populate_context_post_run(context)
    assert context.n_input_tokens is None, "no record: a trial that never ran its harness"
    record = "".join(json.dumps(line) + "\n" for line in RECORD)
    (tmp_path / "agent" / "calls.jsonl").write_text(record, encoding="utf-8")
    agent.populate_context_post_run(context)
    kept = json.loads((tmp_path / "agent" / "trajectory.json").read_text(encoding="utf-8"))
    assert kept["schema_version"] == "ATIF-v1.7" and len(kept["steps"]) == 6
    assert kept["agent"]["version"] == agent.version()
    assert (context.n_input_tokens, context.n_output_tokens) == (110, 10)


# The runner.


def test_the_runner_keeps_to_python_38_and_the_standard_library() -> None:
    """It runs on the sandbox's own Python, with nothing installed for it."""
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 8))
    assert "from __future__" not in source
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


class Proxy(ThreadingHTTPServer):
    """Stands in for the run's proxy: it answers each completion with the next of `answers`,
    a list of stream events or a status and its error message, and keeps what it was asked."""

    def __init__(self, *answers: Any) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answers = list(answers)
        self.asked: list[dict[str, Any]] = []
        self.keys: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - the name `http.server` calls
        server: Proxy = self.server  # type: ignore[assignment]
        body = self.rfile.read(int(self.headers["Content-Length"]))
        server.asked.append(json.loads(body))
        server.keys.append(self.headers["Authorization"])
        answer = server.answers.pop(0)
        if isinstance(answer, tuple):
            status, message = answer
            sent = json.dumps({"error": {"message": message}}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(sent)))
            self.end_headers()
            self.wfile.write(sent)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in answer:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: Any) -> None:
        pass


def stream(text: str = "", *, calls: tuple = (), reasoning: str = "", finish: str = "stop") -> list:
    """Return the events of one streamed reply, its usage last, as the proxy sends them."""
    half = len(text) // 2
    deltas: list[dict[str, Any]] = [{"reasoning": reasoning}] if reasoning else []
    deltas += [{"content": part} for part in (text[:half], text[half:]) if part]
    deltas += [{"tool_calls": [{"index": at, **call}]} for at, call in enumerate(calls)]
    events: list[dict[str, Any]] = [{"choices": [{"delta": delta}]} for delta in deltas]
    events.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    events.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}})
    return events


@pytest.fixture
def proxy() -> Iterator[Any]:
    """Start a `Proxy` with the given answers, and stop each one started when the test ends."""
    started: list[Proxy] = []

    def start(*answers: Any) -> Proxy:
        server = Proxy(*answers)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def lines(record: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in record.getvalue().splitlines()]


def test_a_call_returns_the_reply_as_a_message_and_records_it(proxy: Any) -> None:
    server = proxy(stream(calls=(CALL,), reasoning="Look.", finish="tool_calls"), stream("Done."))
    record = io.StringIO()
    model = runner.Model(server.url, "key", "Qwen/Qwen3-8B", record)
    messages: list[dict[str, Any]] = [SYSTEM, {"role": "user", "content": "Fix the bug."}]

    reply = model(messages, tools=[BASH], max_tokens=64)
    assert reply == {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning": "Look."}
    assert reply.finish == "tool_calls" and reply.usage["prompt_tokens"] == 12
    asked = server.asked[0]
    assert (asked["model"], asked["max_tokens"], asked["tools"]) == ("Qwen/Qwen3-8B", 64, [BASH])
    assert asked["messages"] == messages and asked["stream"] is True
    assert server.keys == ["Bearer key"]

    messages += [reply, {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}]
    answer = model(messages, tools=[BASH], max_tokens=64)
    assert (answer["content"], answer.finish) == ("Done.", "stop") and "tool_calls" not in answer
    first, second = lines(record)
    assert (first["kept"], first["messages"], first["tools"]) == (0, messages[:2], [BASH])
    assert first["reply"] == reply and first["finish"] == "tool_calls"
    assert first["params"] == {"max_tokens": 64} and first["seconds"] >= 0
    assert (second["kept"], second["messages"]) == (2, messages[2:]), "only what is new"
    assert "tools" not in second, "the tools, when they are not the last call's"


def test_a_call_the_proxy_refuses_raises_refused_with_its_reason(proxy: Any) -> None:
    server = proxy((400, "prompt is too long: 70000 tokens"))
    record = io.StringIO()
    model = runner.Model(server.url, "key", "m", record)
    with pytest.raises(model.Refused, match="prompt is too long: 70000 tokens"):
        model([{"role": "user", "content": "x"}])
    assert len(server.asked) == 1, "never asked again"
    assert lines(record)[0]["error"] == "prompt is too long: 70000 tokens"


def test_a_call_that_fails_is_tried_again_and_then_is_unreachable(
    proxy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "WAITS", (0, 0))
    cut = [{"choices": [{"delta": {"content": "Do"}}]}]  # a stream that ends before the model stops
    server = proxy((500, "the sampler is restarting"), cut, stream("Done."))
    record = io.StringIO()
    model = runner.Model(server.url, "key", "m", record)
    assert model([{"role": "user", "content": "x"}])["content"] == "Done."
    assert len(server.asked) == 3

    server = proxy(*[(503, "down")] * 3)
    model = runner.Model(server.url, "key", "m", record)
    with pytest.raises(runner.Unreachable, match="503: down"):
        model([{"role": "user", "content": "x"}])
    assert len(server.asked) == 3 and lines(record)[-1]["error"] == "503: down"


def ran(tmp_path: Path, harness: str, url: str, seconds: str = "0") -> tuple[int, list[dict]]:
    """Run the runner on `harness` as a trial does, in its own process, and return its exit
    status and its record."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (tmp_path / "task.md").write_text("Fix the bug.", encoding="utf-8")
    env = {
        "PATH": os.environ["PATH"],
        "OPENAI_BASE_URL": url,
        "OPENAI_API_KEY": "key",
        "MINE": "1",
        "HARNESS_FILE": str(written(tmp_path, harness)),
        "HARNESS_TASK": str(tmp_path / "task.md"),
        "HARNESS_RECORD": str(tmp_path / "calls.jsonl"),
        "HARNESS_MODEL": "m",
        "HARNESS_SECONDS": seconds,
    }
    ended = subprocess.run([sys.executable, str(RUNNER)], env=env, cwd=work, timeout=60)
    record = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return ended.returncode, [json.loads(line) for line in record]


def test_the_runner_calls_run_in_the_tasks_directory_without_the_proxys_key(
    tmp_path: Path, proxy: Any
) -> None:
    code, record = ran(tmp_path, HARNESS, proxy(stream("Done.")).url, seconds="870")
    assert code == 0 and [line.get("end") for line in record] == [None, "returned"]
    seen = json.loads((tmp_path / "work" / "seen.json").read_text(encoding="utf-8"))
    assert seen == {
        "task": "Fix the bug.",
        "said": "Done.",
        "finish": "stop",
        "OPENAI_API_KEY": None,  # taken out, so the commands a harness runs do not hold it
        "HARNESS_SECONDS": "870",
        "MINE": "1",  # the harness's own `env`
    }


def test_a_harness_that_raises_ends_its_trial_with_what_it_raised(tmp_path: Path) -> None:
    code, record = ran(tmp_path, "def run(task, model):\n    raise KeyError('lost')\n", "http://x")
    assert code == 1 and record[-1]["end"] == "raised" and "KeyError: 'lost'" in record[-1]["error"]


def test_a_harness_is_ended_once_its_seconds_have_passed(tmp_path: Path) -> None:
    slow = "import time\n\n\ndef run(task, model):\n    time.sleep(30)\n"
    code, record = ran(tmp_path, slow, "http://x", seconds="0.2")
    assert code == 0 and record == [{"end": "out of time"}]


def test_a_model_that_cannot_be_reached_says_nothing_about_the_harness(
    tmp_path: Path, proxy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner's exit status tells the adapter, which fails the trial rather than grade it."""
    server = proxy(*[(503, "down")] * 2)
    monkeypatch.setattr(runner, "WAITS", (0,))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.md").write_text("Fix the bug.", encoding="utf-8")
    for name, value in {
        "OPENAI_BASE_URL": server.url,
        "OPENAI_API_KEY": "key",
        "HARNESS_FILE": str(written(tmp_path)),
        "HARNESS_TASK": str(tmp_path / "task.md"),
        "HARNESS_RECORD": str(tmp_path / "calls.jsonl"),
        "HARNESS_MODEL": "m",
        "HARNESS_SECONDS": "0",
    }.items():
        monkeypatch.setenv(name, value)
    assert runner.main() == runner.UNREACHABLE == 75
    record = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(record[-1]) == {"end": "unreachable", "error": "503: down"}
