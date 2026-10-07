"""The reflector as a Harbor trial: a task Harbor itself loads, the traces and the
component as files, the instruction naming the artifact, the artifact read back and
checked, and the job recorded as a reflection under the provider's party."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from harbor.environments.definition import should_upload_environment_dir
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.gepa.fitness import Outcome
from shipyard.gepa.propose import Reflection
from shipyard.gepa.reflect import (
    COLLECTED,
    PUBLISH,
    REFERENCE,
    SETUP_TIMEOUT,
    instruction,
    reflect,
    task_for,
    unwrapped,
    written,
)
from shipyard.modules import KINDS, SKILL_FILE, Candidate, Module
from shipyard.rollout import Rollouts
from shipyard.run import Run
from tests.trials import write_blueprint, write_result

SKILL = "---\nname: solving\ndescription: how to solve.\n---\n\nset up the algebra first\n"
REWRITTEN = "---\nname: solving\ndescription: how to solve.\n---\n\ncheck the constraints\n"
BLUEPRINT = """
[model]
name = "some-hosted-model"
provider = "openrouter"
[data]
dataset = "fixture"
batch_size = 1
[rollout]
harness = "pi@0.85.1"
sandbox = "modal"
kwargs = { turns = 10 }
env = { POLICY_ONLY = "1" }
timeout = 900
[recipe]
kind = "gepa"
reflection_harness = "claude-code"
"""
OUTCOMES = (
    Outcome("tasks/aime/63", 0.0, feedback="Expected: 385\nGot: 12", transcript="I answered 12"),
    Outcome("tasks/aime/70", None, feedback="the container never answered"),
)


def _module(text: str = SKILL, **beside: str) -> Module:
    return Module("solving", "skill", {SKILL_FILE: text, **beside})


def _reflection(
    module: Module | None = None, outcomes: tuple[Outcome, ...] = OUTCOMES
) -> Reflection:
    module = module or _module()
    return Reflection(Candidate({module.name: module}), module.name, outcomes)


def _rollouts(*trials: Path) -> Rollouts:
    return Rollouts(job="r-0001", trials=list(trials), plan=[])


def _published(trial: Path, *parts: str) -> Path:
    found = trial.joinpath(*COLLECTED, *parts)
    found.parent.mkdir(parents=True, exist_ok=True)
    return found


# ---------------------------------------------------------------- the task it writes


def test_harbor_loads_the_synthesized_task_and_uploads_its_environment(tmp_path: Path) -> None:
    task = task_for(_reflection(), tmp_path, image="python:3.12-slim", incremental=False)
    loaded = Task(task_dir=task, disable_verification=True)
    assert loaded.config.task.name == "shipyard/reflection"
    assert loaded.config.environment.docker_image == "python:3.12-slim"
    assert loaded.config.environment.workdir == "/app"
    assert "/app/current/solving/SKILL.md" in loaded.instruction
    assert should_upload_environment_dir(task / "environment", docker_image="python:3.12-slim")
    assert Task.is_valid_dir(task, disable_verification=True)


def test_the_traces_and_the_component_reach_the_container_as_files(tmp_path: Path) -> None:
    module = _module(**{"examples/worked.md": "# a worked example"})
    task = task_for(_reflection(module), tmp_path, image="i", incremental=False)
    traces = sorted((task / "environment" / "traces").iterdir())
    assert [p.name for p in traces] == ["00-tasks-aime-63.txt", "01-tasks-aime-70.txt"]
    first = traces[0].read_text()
    assert first.startswith("task: tasks/aime/63\nscore: 0.000\n")
    assert "--- asked ---" not in first, "no inputs, no heading to read"
    assert first.index("--- transcript ---\nI answered 12") < first.index(
        "--- observed ---\nExpected: 385"
    )
    second = traces[1].read_text()
    assert "score: unmeasured" in second and "--- transcript ---" not in second
    home = task / "environment" / "current" / "solving"
    assert (home / SKILL_FILE).read_text() == SKILL
    assert (home / "examples" / "worked.md").read_text() == "# a worked example"
    text = (task / "instruction.md").read_text()
    assert "`examples/worked.md`" in text and "not yours to rewrite" in text
    for stray in ("task.toml", "reference.md", str(tmp_path), "environment/"):
        assert stray not in text


def test_a_trace_carries_what_was_asked_before_what_was_observed(tmp_path: Path) -> None:
    asked = "Every morning Aya goes for a 9-kilometer walk"
    outcome = Outcome("aime/63", 0.0, feedback="Expected: 385", inputs=asked)
    task = task_for(_reflection(outcomes=(outcome,)), tmp_path, image="i", incremental=False)
    (trace,) = (task / "environment" / "traces").iterdir()
    text = trace.read_text()
    assert text.index("--- asked ---\n" + asked) < text.index("--- observed ---\nExpected: 385")


def test_the_reference_is_written_only_when_the_kind_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = task_for(_reflection(), tmp_path / "skill", image="i", incremental=False)
    assert not (task / "environment" / REFERENCE).exists()
    assert REFERENCE not in (task / "instruction.md").read_text()

    class Contracted:
        marker = "AGENT.md"

        def check(self, module: Module) -> str | None:
            return None

        def deliver(self, modules: Any, into: Path) -> dict[str, Any]:
            return {}

        def reference(self) -> str | None:
            return "what a loop may call; do not redefine `run`"

    monkeypatch.setitem(KINDS, "contracted", Contracted())
    module = Module("loop", "contracted", {"AGENT.md": "the loop"})
    task = task_for(_reflection(module), tmp_path / "other", image="i", incremental=False)
    assert (task / "environment" / REFERENCE).read_text().startswith("what a loop may call")
    text = (task / "instruction.md").read_text()
    assert f"Read `/app/{REFERENCE}` first" in text
    assert f"Write the new `AGENT.md` to `{PUBLISH}/AGENT.md`" in text
    assert (task / "environment" / "current" / "loop" / "AGENT.md").read_text() == "the loop"


def test_the_instruction_names_the_artifact_and_asks_for_small_edits_only_when_told() -> None:
    plain = instruction(
        name="solving", marker=SKILL_FILE, beside=[], reference=False, incremental=False
    )
    assert f"Write the new `{SKILL_FILE}` to `{PUBLISH}/{SKILL_FILE}`" in plain
    assert "Improve `/app/current/solving/SKILL.md`" in plain
    assert "the tail of the assistant's own transcript" in plain
    assert "smallest change" not in plain and "also holds" not in plain
    small = instruction(
        name="solving", marker=SKILL_FILE, beside=[], reference=False, incremental=True
    )
    assert "smallest change" in small and "do not rewrite it from scratch" in small


# ---------------------------------------------------------------- reading the reply back


def test_the_rewrite_is_read_back_as_a_module_of_the_same_name_and_kind(tmp_path: Path) -> None:
    trial = tmp_path / "reflection__abc"
    _published(trial, SKILL_FILE).write_text(REWRITTEN)
    _published(trial, "examples", "worked.md").write_text("the corrected example")
    _published(trial, ".hidden").write_text("not a file of the module")
    found = written(_rollouts(trial), _module())
    assert found is not None and (found.name, found.kind) == ("solving", "skill")
    assert found.files == {SKILL_FILE: REWRITTEN, "examples/worked.md": "the corrected example"}
    assert found.digest != _module().digest


def test_a_rewrite_that_fails_the_kinds_check_is_declined_and_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trial = tmp_path / "reflection__abc"
    _published(trial, SKILL_FILE).write_text("# Solving\n\nno frontmatter any more\n")
    with caplog.at_level(logging.WARNING, logger="shipyard.gepa.reflect"):
        assert written(_rollouts(trial), _module()) is None
    assert "rewrite of 'solving' declined" in caplog.text
    assert "does not open with a `---` frontmatter block" in caplog.text


def test_a_wrapper_around_the_file_is_not_the_file(tmp_path: Path) -> None:
    trial = tmp_path / "reflection__abc"
    _published(trial, SKILL_FILE).write_text("```markdown\n" + REWRITTEN + "```\n")
    found = written(_rollouts(trial), _module())
    assert found is not None and found.text == REWRITTEN
    body = "from x import y\n\n\nclass Harness(y):\n    pass\n"
    assert unwrapped(body + "</content>\n") == body
    assert unwrapped("```python\n" + body + "```\n") == body
    assert unwrapped(body) == body
    inside = "a = '```'\nb = '</content>'\n"
    assert unwrapped(inside) == inside
    skill = "the rewritten skill\n```sh\nls\n```\n"
    assert unwrapped(skill) == skill


def test_a_rewrite_left_one_directory_down_is_read_and_two_are_not(tmp_path: Path) -> None:
    trial = tmp_path / "reflection__1"
    _published(trial, "solving", SKILL_FILE).write_text(REWRITTEN)
    _published(trial, "solving", "worked.md").write_text("beside it\n")
    found = written(_rollouts(trial), _module())
    assert found is not None and found.text == REWRITTEN and "worked.md" in found.files
    _published(trial, "other", SKILL_FILE).write_text(REWRITTEN)
    assert written(_rollouts(trial), _module()) is None


def test_a_trial_that_wrote_nothing_declines_and_one_that_died_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert written(_rollouts(), _module()) is None
    quiet = tmp_path / "quiet"
    _published(quiet, "x")  # the directory, and nothing in it
    write_result(quiet)
    with caplog.at_level(logging.INFO, logger="shipyard.gepa.reflect"):
        assert written(_rollouts(quiet), _module()) is None
    assert "WARNING" not in caplog.text and "wrote no 'SKILL.md'" in caplog.text
    caplog.clear()
    wrong = tmp_path / "wrong"
    _published(wrong, "notes.md").write_text("prose instead of the file")
    with caplog.at_level(logging.WARNING, logger="shipyard.gepa.reflect"):
        assert written(_rollouts(wrong), _module()) is None
    assert "it left ['notes.md']" in caplog.text
    caplog.clear()
    dead = tmp_path / "dead"
    write_result(dead, exception="AgentSetupTimeoutError")
    with caplog.at_level(logging.WARNING, logger="shipyard.gepa.reflect"):
        assert written(_rollouts(dead), _module()) is None
    assert "the reflection trial failed: AgentSetupTimeoutError" in caplog.text


# -------------------------------------------------------------------- the job it runs


class Reflector:
    """Harbor's trial replaced: notes the config and publishes `answer` as the artifact,
    with a result naming the provider and the tokens the harness reported."""

    def __init__(self, answer: str | None = REWRITTEN) -> None:
        self.configs: list[TrialConfig] = []
        self.answer = answer

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        trial = Path(config.trials_dir) / config.trial_name
        if self.answer is not None:
            _published(trial, SKILL_FILE).write_text(self.answer)
        write_result(trial, tokens=(2000, 500, 30), provider="anthropic", seconds=12.0)


def _run(tmp_path: Path, text: str = BLUEPRINT) -> Run:
    opened = Run.open(write_blueprint(tmp_path, text), root=tmp_path / "runs")
    assert opened.serving is None, "provider-served: no proxy to point a reflection at"
    return opened


async def test_the_reflector_runs_the_trial_our_way_and_records_it_as_a_reflection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = Reflector()
    with _run(tmp_path) as opened:
        opened.run_trial = runner
        write = reflect(opened, harness="claude-code", model="anthropic/claude-sonnet-5", image="i")
        found = await write(_reflection())
    assert found is not None and found.text == REWRITTEN and found.name == "solving"
    (config,) = runner.configs
    agent = config.agent
    assert agent.name == "claude-code" and agent.model_name == "anthropic/claude-sonnet-5"
    assert config.verifier.disable is True
    assert config.environment.type == "modal", "the machine's sandbox, from [rollout]"
    assert agent.override_setup_timeout_sec == SETUP_TIMEOUT > 366.0
    assert agent.override_timeout_sec is None and agent.kwargs == {} and agent.env == {}
    assert agent.skills == [], "nothing of the policy's: the candidate is not the reflector's"
    assert Path(str(config.task.path)).name == "reflection"
    assert not Path(str(config.task.path)).exists(), "the temp task is gone once read"
    assert Path(config.trials_dir) == Path("jobs") / f"{opened.id}-0000"
    (row,) = record.read(opened.directory / record.JOBS)
    assert [key for key in row if key != "at"] == [
        "job",
        "purpose",
        "trials",
        "component",
        "candidate",
        "party",
        "input_tokens",
        "cache_tokens",
        "output_tokens",
        "sandbox",
        "sandbox_seconds",
    ]
    assert row["job"] == f"{opened.id}-0000" and row["purpose"] == "reflection"
    assert row["component"] == "solving" and row["candidate"] == _reflection().candidate.digest
    assert (row["trials"], row["party"]) == (1, "anthropic")
    assert (row["input_tokens"], row["cache_tokens"], row["output_tokens"]) == (2000, 500, 30)
    assert row["sandbox"] == "modal" and row["sandbox_seconds"] == 12.0
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs["parties"] == {
        "anthropic": {"trials": 1, "input_tokens": 2000, "cache_tokens": 500, "output_tokens": 30}
    }
    assert costs["sandbox"] == {"modal": {"trials": 1, "seconds": 12.0}}


async def test_a_reflection_that_published_nothing_or_crashed_is_a_decline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with _run(tmp_path) as opened:
        opened.run_trial = Reflector(answer=None)
        write = reflect(opened, harness="claude-code", model=None, image="i")
        assert await write(_reflection()) is None

        async def crashes(config: TrialConfig) -> None:
            raise RuntimeError("the daemon went away")

        opened.run_trial = crashes
        assert await write(_reflection()) is None
    rows = list(record.read(opened.directory / record.JOBS))
    assert [row["purpose"] for row in rows] == ["reflection", "reflection"]
    assert [row["party"] for row in rows] == ["anthropic", "unknown"]
    assert rows[1]["input_tokens"] == 0 and rows[1]["sandbox_seconds"] == 0.0


async def test_the_party_falls_back_to_the_models_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    async def wrote_no_result(config: TrialConfig) -> None:
        _published(Path(config.trials_dir) / config.trial_name, SKILL_FILE).write_text(REWRITTEN)

    with _run(tmp_path) as opened:
        opened.run_trial = wrote_no_result
        write = reflect(opened, harness="pi", model="openai/gpt-5", image="i")
        assert (await write(_reflection())) is not None
    (row,) = record.read(opened.directory / record.JOBS)
    assert row["party"] == "openai" and row["trials"] == 1


def test_a_reflector_without_a_harness_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a harness"):
        reflect(object(), harness=" ", model=None, image="i")  # type: ignore[arg-type]
