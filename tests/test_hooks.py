"""Tests for `shipyard.jobs.hooks`: a task's hooks.py read once into hooks by event, refused
when wrong before anything opens, each hook's run logged, and every hook attached by the engine
to each Harbor trial of the task."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.job.lock import TrialLock
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent

from shipyard.jobs import hooks
from shipyard.jobs.engine import _run_trial, _trial_config
from shipyard.jobs.hooks import hooks_of
from shipyard.run.directory import open_run
from tests.runs import GRPO, run_toml

SETUP_AND_TEARDOWN = """
from os.path import join
from pathlib import Path

LIMIT = 3


def _helper():
    return join("a", "b")


async def environment_start(event):
    event.config.verifier.env["SEEN"] = "environment_start"


async def end(event):
    raise RuntimeError("teardown broke")
"""


@pytest.fixture(autouse=True)
def _read_afresh() -> Iterator[None]:
    """Each test reads its tasks' hooks.py anew."""
    hooks._loaded.cache_clear()
    yield
    hooks._loaded.cache_clear()


def _task(tmp_path: Path, source: str | None) -> Path:
    task = tmp_path / "tasks" / "task"
    (task / "environment").mkdir(parents=True)
    if source is not None:
        (task / "hooks.py").write_text(source, encoding="utf-8")
    return task


def _config(tmp_path: Path, task: Path) -> TrialConfig:
    return _trial_config(
        task, trials_dir=tmp_path / "jobs", agent=AgentConfig(name="pi"), sandbox="docker"
    )


def test_a_task_without_hooks_py_has_no_hooks(tmp_path: Path) -> None:
    assert dict(hooks_of(_task(tmp_path, None))) == {}


def test_hooks_are_read_by_event_leaving_helpers_and_imports_out(tmp_path: Path) -> None:
    found = hooks_of(_task(tmp_path, SETUP_AND_TEARDOWN))
    assert set(found) == {TrialEvent.ENVIRONMENT_START, TrialEvent.END}


def test_a_tasks_hooks_py_is_read_once(tmp_path: Path) -> None:
    reading = 'from pathlib import Path\nPath(__file__).with_name("reads").open("a").write("x")\n'
    task = _task(tmp_path, reading)
    hooks_of(task)
    hooks_of(task / ".." / task.name)
    assert (task / "reads").read_text() == "x"


def test_a_public_function_named_after_no_event_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, "async def enviroment_start(event):\n    pass\n")
    with pytest.raises(ValueError, match="enviroment_start is named after no event of a trial"):
        hooks_of(task)


def test_a_hook_that_is_not_async_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end is a hook, so it is written `async def`"):
        hooks_of(_task(tmp_path, "def end(event):\n    pass\n"))


def test_a_hooks_py_that_cannot_be_loaded_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not be loaded"):
        hooks_of(_task(tmp_path, "async def end(event)\n"))


async def test_run_trial_attaches_every_hook_and_logs_each_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attached: list[TrialEvent] = []

    class Harbor:
        """Stands in for Harbor's trial: it calls the hooks it was given, as Harbor's run does."""

        def __init__(self, config: TrialConfig) -> None:
            self.config = config
            self.hooks: dict[TrialEvent, Any] = {}

        @classmethod
        async def create(cls, config: TrialConfig) -> Harbor:
            return cls(config)

        def add_hook(self, event: TrialEvent, hook: Any) -> None:
            attached.append(event)
            self.hooks[event] = hook

        async def run(self) -> str:
            event = SimpleNamespace(trial_name=self.config.trial_name, config=self.config)
            await self.hooks[TrialEvent.ENVIRONMENT_START](event)
            with pytest.raises(RuntimeError, match="teardown broke"):
                await self.hooks[TrialEvent.END](event)
            return "ran"

    monkeypatch.setattr("harbor.trial.trial.Trial", Harbor)
    config = _config(tmp_path, _task(tmp_path, SETUP_AND_TEARDOWN))
    with caplog.at_level(logging.INFO, logger="shipyard.jobs.hooks"):
        assert await _run_trial(config) == "ran"
    assert attached == [TrialEvent.ENVIRONMENT_START, TrialEvent.END]
    assert config.verifier.env["SEEN"] == "environment_start"
    said = [record.getMessage() for record in caplog.records]
    assert any(f"hook environment_start of {config.trial_name} done in" in one for one in said)
    assert any(
        f"hook end of {config.trial_name} failed after" in one and "teardown broke" in one
        for one in said
    )


def test_a_hook_is_handed_the_trials_own_config(tmp_path: Path) -> None:
    """A hook hands the verifier values through `event.config.verifier.env`, as Bauplan's do
    with their baseline, which holds only while Harbor's event carries the trial's own config
    rather than a copy of it."""
    config = _config(tmp_path, _task(tmp_path, None))
    event = TrialHookEvent(
        event=TrialEvent.ENVIRONMENT_START,
        task_name="task",
        config=config,
        result=TrialResult.model_construct(),
        lock=TrialLock.model_construct(),
    )
    assert event.config is config
    event.config.verifier.env["BAUPLAN_BASELINE"] = "{}"
    assert config.verifier.env["BAUPLAN_BASELINE"] == "{}"


def test_a_dataset_task_with_a_wrong_hooks_py_stops_the_run_before_anything_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = run_toml(tmp_path)
    (tmp_path / "tasks" / "four" / "a" / "hooks.py").write_text("def end(event):\n    pass\n")
    directory, config = open_run(home, root=tmp_path / "runs", trainer=GRPO)
    with pytest.raises(ValueError, match="end is a hook"):
        GRPO(config, directory)
