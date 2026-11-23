"""A harness given as one Python file, `harness = "harness.py"`, as a Harbor agent. The file is
uploaded into the trial's sandbox with `runner.py` and run there, on the sandbox's own Python, so
it never runs on this machine; a sandbox with no Python is given its distribution's at setup.
`runner.py` states what a harness file is written against.

A trial gives the harness its time limit less a margin, so the harness ends as the runner's own,
its record whole. The prompt a trial carries is put before its instruction here, as Harbor's
installed agents do it, and its skills are where Harbor uploaded them, which the harness reads
from `HARNESS_SKILLS`. The runner's record of the harness's calls comes back with the trial's
logs, and the trial's trajectory, which `trial.transcript` reads, is made from it."""

from __future__ import annotations

import json
import tempfile
import tomllib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.templating import render_prompt_template

RUNNER = Path(__file__).with_name("runner.py")
#: Where the files sit in the sandbox.
HOME = "/installed-agent"
#: The runner's record of the harness's calls, and what the harness printed, in the trial's
#: `agent/`.
RECORD = "calls.jsonl"
PRINTED = "harness.txt"
#: The runner's exit status when the model could not be reached (`runner.UNREACHABLE`).
UNREACHABLE = 75
#: A shell's exit status for a command it cannot find.
NOT_FOUND = 127
#: Seconds before the trial's limit at which the runner ends a harness, so that the trial ends
#: as the runner's own, its record whole, and not cut by Harbor.
MARGIN = 30
#: The seconds a trial gave its harness, in the trial's `agent/`.
SECONDS = "seconds.txt"
#: Notes the sandbox's Python of 3.8 or later in `{home}/python`. An image that has none is
#: given `python3` and the certificates an HTTPS call needs, on Debian's and Ubuntu's images,
#: as root.
READY = r"""
found() {{
  for p in python3 python; do
    command -v "$p" >/dev/null 2>&1 \
      && "$p" -c 'import ssl, sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null \
      && {{ command -v "$p"; return 0; }}
  done
  return 1
}}
mkdir -p {home}
PYTHON=$(found) || {{
  export DEBIAN_FRONTEND=noninteractive
  (apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates) \
    > {home}/setup.txt 2>&1
  PYTHON=$(found)
}} || exit 127
echo "$PYTHON" > {home}/python
"""


def given(task: Path, timeout: float | None = None) -> int:
    """Return the seconds a trial of the task at `task` gives a harness: `timeout`, or with none
    the task's own limit, less `MARGIN`; 0, for no limit of the runner's, when the task states
    none that can be read."""
    limit = timeout
    if not limit:
        try:
            config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
            limit = float(config["agent"]["timeout_sec"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0
    return max(int(limit) - MARGIN, 1)


class FileHarness(BaseAgent):
    capabilities = AgentCapabilities(atif=True, skills=True)

    def __init__(
        self,
        logs_dir: Path,
        *args: Any,
        file: str,
        timeout: float | None = None,
        prompt_template_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, *args, **kwargs)
        self.file = Path(file)
        #: The limit `[rollout] timeout` gives every trial, which Harbor does not tell an
        #: agent; None for each task's own.
        self.timeout = timeout
        #: The template the trial's prompt is delivered as; None for none.
        self.template = Path(prompt_template_path) if prompt_template_path else None

    @staticmethod
    def name() -> str:
        return "file-harness"

    def version(self) -> str:
        """The file's digest: what tells one file's trials from another's."""
        return sha256(self.file.read_bytes()).hexdigest()[:12]

    async def setup(self, environment: BaseEnvironment) -> None:
        ready = await environment.exec(command=READY.format(home=HOME), user="root")
        if ready.return_code == NOT_FOUND:
            raise RuntimeError("the sandbox has no Python to run the harness on, and got none")
        await environment.upload_file(RUNNER, f"{HOME}/runner.py")
        await environment.upload_file(self.file, f"{HOME}/harness.py")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if self.template is not None:
            instruction = render_prompt_template(self.template, instruction)
        with tempfile.TemporaryDirectory(prefix="harness-task-") as scratch:
            task = Path(scratch) / "task.md"
            task.write_text(instruction, encoding="utf-8")
            await environment.upload_file(task, f"{HOME}/task.md")
        logs = self.environment_logs_dir.as_posix()
        seconds = given(environment.environment_dir.parent, self.timeout)
        (self.logs_dir / SECONDS).write_text(f"{seconds}\n", encoding="utf-8")
        env = {
            "HARNESS_FILE": f"{HOME}/harness.py",
            "HARNESS_TASK": f"{HOME}/task.md",
            "HARNESS_RECORD": f"{logs}/{RECORD}",
            "HARNESS_MODEL": (self.model_name or "").partition("/")[2] or "model",
            "HARNESS_SECONDS": str(seconds),
        }
        if self.skills_dir:
            env["HARNESS_SKILLS"] = str(self.skills_dir)
        # The proxy's address, the trial's token and the harness's own `env` are in the env the
        # trial gives every command.
        ended = await environment.exec(
            command=(
                f"PYTHON=$(cat {HOME}/python 2>/dev/null); "
                '[ -n "$PYTHON" ] || exit 127; '
                f'"$PYTHON" {HOME}/runner.py > {logs}/{PRINTED} 2>&1'
            ),
            env=env,
        )
        if ended.return_code == NOT_FOUND:
            raise RuntimeError("the sandbox has no python3 or python to run the harness on")
        if ended.return_code == UNREACHABLE:
            raise RuntimeError("the harness could not reach the model")
        # A harness that raised ended its own trial: the tests grade what it left.

    def populate_context_post_run(self, context: AgentContext) -> None:
        try:
            text = (self.logs_dir / RECORD).read_text(encoding="utf-8")
        except OSError:
            return
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        found = trajectory(
            lines, session=self.session_id, version=self.version(), model=self.model_name
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(found.to_json_dict(), indent=2), encoding="utf-8"
        )
        totals = found.final_metrics
        if totals is not None:
            context.n_input_tokens = totals.total_prompt_tokens
            context.n_output_tokens = totals.total_completion_tokens


def trajectory(
    lines: list[dict[str, Any]], *, session: str | None, version: str, model: str | None
) -> Trajectory:
    """Return the trajectory of a trial from the runner's record of it: a step for each message
    the harness gave the model, a step for each reply, with what its tool calls gave back as the
    next call sent it, and a step for each call the proxy refused and for a harness that
    raised."""
    steps: list[Step] = []
    told: set[str] = set()  # the system and user messages already steps
    waiting: dict[str, Step] = {}  # the step whose call awaits its result, by the call's id
    tools: list[dict[str, Any]] | None = None

    def add(source: str, at: float, **fields: Any) -> Step:
        step = Step(step_id=len(steps) + 1, timestamp=_stamp(at), source=source, **fields)
        steps.append(step)
        return step

    at = 0.0
    for line in lines:
        at = float(line.get("at") or at)
        tools = tools or line.get("tools")
        for message in line.get("messages") or []:
            role, content = message.get("role"), _text(message.get("content"))
            if role == "tool":
                step = waiting.pop(str(message.get("tool_call_id")), None)
                if step is not None:
                    result = ObservationResult(
                        source_call_id=str(message.get("tool_call_id")), content=content
                    )
                    held = step.observation.results if step.observation else []
                    step.observation = Observation(results=[*held, result])
            elif role in ("system", "user"):
                key = json.dumps([role, content])
                if key not in told:
                    told.add(key)
                    add(role, at, message=content)
        at += float(line.get("seconds") or 0.0)
        reply = line.get("reply")
        if reply is not None:
            calls = [_call(one) for one in reply.get("tool_calls") or []]
            usage = line.get("usage") or {}
            step = add(
                "agent",
                at,
                model_name=model,
                message=_text(reply.get("content")),
                reasoning_content=reply.get("reasoning") or None,
                tool_calls=calls or None,
                metrics=Metrics(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                ),
                llm_call_count=1,
                extra={"finish": line.get("finish"), "kept": line.get("kept")},
            )
            waiting.update({call.tool_call_id: step for call in calls})
        elif "end" not in line:
            add("system", at, message=f"[the call was not answered] {line.get('error')}")
        elif line["end"] != "returned":
            add("system", at, message=f"[the harness: {line['end']}] {line.get('error') or ''}")
    replies = [step for step in steps if step.source == "agent"]
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session,
        agent=Agent(
            name=FileHarness.name(), version=version, model_name=model, tool_definitions=tools
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=sum(one.metrics.prompt_tokens or 0 for one in replies),
            total_completion_tokens=sum(one.metrics.completion_tokens or 0 for one in replies),
            total_steps=len(steps),
        ),
    )


def _call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except ValueError:
        arguments = None
    if not isinstance(arguments, dict):
        arguments = {"raw": function.get("arguments")}
    return ToolCall(
        tool_call_id=str(raw.get("id")),
        function_name=str(function.get("name")),
        arguments=arguments,
    )


def _text(content: Any) -> str:
    """Return a message's content as text: a string as it is, a list as its text parts."""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _stamp(at: float) -> str:
    return datetime.fromtimestamp(at, UTC).isoformat()
