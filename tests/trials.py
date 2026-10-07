"""Trials and blueprints without Harbor: `result.json` files in the shape `TrialResult`
dumps, a runner that stands in for `Trial.create` + `run` and writes them, and a `run.toml`
written where a test asks."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig

FIXTURES = Path(__file__).parent / "tasks"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def fixture_tasks(tmp_path: Path) -> Path:
    """The fixture datasets copied to `<tmp>/tasks`, where a working directory reads them."""
    return shutil.copytree(FIXTURES, tmp_path / "tasks")


def write_blueprint(tmp_path: Path, text: str) -> Path:
    """`text` as `<tmp>/bp/run.toml`; the blueprint directory, as `load` and `check` take it."""
    home = tmp_path / "bp"
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


def write_result(
    trial_dir: Path,
    *,
    reward: float | None = None,
    rewards: Mapping[str, Any] | None = None,
    exception: str | None = None,
    tokens: tuple[int | None, int | None, int | None] | None = None,
    steps: list[tuple[int | None, int | None, int | None]] | None = None,
    provider: str | None = None,
    seconds: float | None = None,
    raw: str | None = None,
) -> Path:
    """A trial directory holding a `result.json` built from the parts given."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "result.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return trial_dir
    payload: dict[str, Any] = {
        "task_name": trial_dir.name.split("__")[0],
        "trial_name": trial_dir.name,
        "agent_info": {"name": "pi", "version": "0.85.1", "model_info": None},
        "agent_result": None,
        "verifier_result": None,
        "exception_info": None,
    }
    if rewards is not None:
        payload["verifier_result"] = {"rewards": dict(rewards)}
    elif reward is not None:
        payload["verifier_result"] = {"rewards": {"reward": reward}}
    if exception is not None:
        payload["exception_info"] = {
            "exception_type": exception,
            "exception_message": "",
            "exception_traceback": "",
            "occurred_at": T0.isoformat(),
        }
    if tokens is not None:
        payload["agent_result"] = _context(tokens)
    if steps is not None:
        payload["step_results"] = [
            {"step_name": f"step-{n}", "agent_result": _context(each)}
            for n, each in enumerate(steps)
        ]
    if provider is not None:
        payload["agent_info"]["model_info"] = {"name": "m", "provider": provider}
    if seconds is not None:
        payload["environment_setup"] = _phase(T0, 1.0)
        payload["agent_execution"] = _phase(T0 + timedelta(seconds=1), seconds - 2.0)
        payload["verifier"] = _phase(T0 + timedelta(seconds=seconds - 1), 1.0)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return trial_dir


def _context(tokens: tuple[int | None, int | None, int | None]) -> dict[str, Any]:
    n_input, n_cache, n_output = tokens
    return {"n_input_tokens": n_input, "n_cache_tokens": n_cache, "n_output_tokens": n_output}


def _phase(start: datetime, seconds: float) -> dict[str, str]:
    return {
        "started_at": start.isoformat(),
        "finished_at": (start + timedelta(seconds=seconds)).isoformat(),
    }


class FakeTrials:
    """Harbor's `Trial.create` + `run`, replaced: records every config and how many ran
    at once, and writes each trial's directory and result as `outcomes` says by task name.
    A task in `breaks` raises instead; an outcome of None leaves the directory empty."""

    def __init__(
        self,
        *,
        dwell: float = 0.0,
        breaks: tuple[str, ...] = (),
        outcomes: Mapping[str, dict[str, Any] | None] | None = None,
        default: dict[str, Any] | None = None,
    ) -> None:
        self.configs: list[TrialConfig] = []
        self.dwell = dwell
        self.breaks = breaks
        self.outcomes = dict(outcomes or {})
        self.default = {"reward": 1.0} if default is None else default
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.dwell)
            task = Path(str(config.task.path)).name
            if task in self.breaks:
                raise RuntimeError("the container never came up")
            trial_dir = Path(config.trials_dir) / config.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            outcome = self.outcomes.get(task, self.default)
            if outcome is not None:
                write_result(trial_dir, **outcome)
        finally:
            self.in_flight -= 1
