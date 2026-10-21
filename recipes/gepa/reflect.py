"""GEPA's reflection as a Harbor task: the reflector is rolled out through its harness at a
task written for each module it rewrites. The task holds the module's text and the agent's
attempts as files, which the reflector reads with its tools, and asks it to rewrite the
module in place; its tests keep what it wrote where the run reads it back."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shipyard import Trial

#: The file the module's text is in, in the task's workspace and, once its tests ran, in the
#: trial's `verifier/`.
MODULE = "module.md"
#: The image the reflector's harness is installed in: a plain Python one, as Harbor's tasks use.
IMAGE = "ghcr.io/laude-institute/t-bench/python-3-13:20250620"
#: Seconds the reflector may take to rewrite a module.
REFLECTION_SECONDS = 900.0

INSTRUCTION = """\
You are improving `{name}`, one of the modules an agent carries into its tasks. Its text is \
in /workspace/{module}.

/workspace/attempts/ holds the agent's attempts at some of its tasks while carrying it: each \
file has the task, what the agent did, the reward it earned (1 is solved, 0 failed), and what \
the task's tests printed.

Read the attempts and work out what made the agent succeed or fail. Then rewrite \
/workspace/{module} so the agent does better on tasks like these: keep what works, fix what \
did not, and keep the module's format (a skill keeps its frontmatter). Write the whole new \
text to /workspace/{module}, in place of the old.
"""

TASK = f"""\
version = "1.0"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = {REFLECTION_SECONDS}

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 1024
"""

DOCKERFILE = f"""\
FROM {IMAGE}
COPY . /workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]
"""

TESTS = f"""\
#!/bin/bash
# Keep what the reflector wrote where the run reads it, and reward a module that is there.
mkdir -p /logs/verifier
if [ -s /workspace/{MODULE} ]; then
  cp /workspace/{MODULE} /logs/verifier/{MODULE}
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


class Reflector(BaseModel):
    """The `[recipe.reflector]` table: the model that rewrites the modules, and the harness
    it runs through, with that harness's `env` and `kwargs`."""

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str
    harness: str
    env: dict[str, str] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)


def reflection(into: Path, name: str, text: str, attempts: Sequence[Mapping[str, Any]]) -> Path:
    """Write the task that asks the reflector to rewrite module `name`, whose text is `text`,
    from the agent's `attempts`, each a task, its transcript, reward and test output."""
    into.mkdir(parents=True)
    (into / "instruction.md").write_text(
        INSTRUCTION.format(name=name, module=MODULE), encoding="utf-8"
    )
    (into / "task.toml").write_text(TASK, encoding="utf-8")
    environment = into / "environment"
    (environment / "attempts").mkdir(parents=True)
    (environment / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (environment / MODULE).write_text(text, encoding="utf-8")
    for at, attempt in enumerate(attempts, 1):
        (environment / "attempts" / f"{at:02d}.md").write_text(written(attempt), encoding="utf-8")
    tests = into / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(TESTS, encoding="utf-8")
    (tests / "test.sh").chmod(0o755)
    return into


def written(attempt: Mapping[str, Any]) -> str:
    """Return one attempt as the file the reflector reads."""
    return (
        f"# The task\n\n{attempt['task']}\n\n"
        f"# The reward\n\n{attempt['reward']:.3f}\n\n"
        f"# What the agent did\n\n{attempt['transcript']}\n\n"
        f"# What the tests printed\n\n{attempt['tests']}\n"
    )


def rewritten(trial: Trial) -> str | None:
    """Return the module the reflector wrote in `trial`, or None if it wrote none."""
    try:
        text = (trial.dir / "verifier" / MODULE).read_text(encoding="utf-8")
    except OSError:
        return None
    return text if text.strip() else None
