"""Test helpers that replace Harbor and the proxy, and need nothing of the framework: `result.json`
writers, fake trial runners, fixture datasets, and a fake proxy client."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
from harbor.models.trial.config import TrialConfig

from harness_proxy.wire import Record
from proxies import MODEL, made

FIXTURES = Path(__file__).parent / "tasks"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
#: Token counts of the fake proxy's single record per trial: prompt "go" plus completion "ok"
#: is four tokens, three of which are model input.
SEQUENCE_TOKENS = 4
INPUT_TOKENS = 3
START = "tinker://fake/sampler_weights/start"


def dataset(root: Path, name: str, *tasks: str) -> Path:
    """Create a dataset of copies of the fixture `alpha` task under the given names."""
    home = root / name
    for task in tasks:
        shutil.copytree(FIXTURES / "fixture" / "alpha", home / task)
    return home


def write_result(
    trial_dir: Path,
    *,
    reward: float | None = None,
    rewards: Mapping[str, Any] | None = None,
    exception: str | None = None,
    tokens: tuple[int | None, int | None, int | None] | None = None,
    steps: list[tuple[int | None, int | None, int | None]] | None = None,
    usage: Mapping[str, tuple[int, int, int]] | None = None,
    provider: str | None = None,
    seconds: float | None = None,
    raw: str | None = None,
) -> Path:
    """Create `trial_dir` with a `result.json` in the shape Harbor's `TrialResult` dumps,
    built from the given parts."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "result.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return trial_dir
    task = f"tasks/{trial_dir.name.split('__')[0]}"
    payload: dict[str, Any] = {
        "task_name": trial_dir.name.split("__")[0],
        "trial_name": trial_dir.name,
        "trial_uri": trial_dir.resolve().as_uri(),
        "task_id": {"path": task},
        "task_checksum": "0",
        "config": {"task": {"path": task}},
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
    if usage is not None:
        payload["agent_result"] = payload["agent_result"] or {}
        payload["agent_result"]["model_usage"] = {
            name: dict(_context(counts)) for name, counts in usage.items()
        }
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
    """Fake for Harbor's `Trial.create` and `run`. It records every config and the peak
    concurrency, and writes each trial's result from `default`. A task in `breaks` raises
    instead."""

    def __init__(
        self,
        *,
        dwell: float = 0.0,
        breaks: tuple[str, ...] = (),
        default: dict[str, Any] | None = None,
    ) -> None:
        self.configs: list[TrialConfig] = []
        self.dwell = dwell
        self.breaks = breaks
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
            write_result(trial_dir, **self.default)
        finally:
            self.in_flight -= 1


class Alternating:
    """Fake trial runner that rewards each task 1, 0, 1, 0 in call order, so every group of
    two has one success and one failure."""

    def __init__(self) -> None:
        self.configs: list[TrialConfig] = []
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        write_result(Path(config.trials_dir) / config.trial_name, reward=float(self.seen[task] % 2))


class Spread:
    """Fake Harbor trial runner: tasks in `spread` score 1, 0, 1, 0 in call order, others 1."""

    def __init__(self, spread: set[str]) -> None:
        self.spread = spread
        self.seen: dict[str, int] = {}

    async def __call__(self, config: TrialConfig) -> None:
        task = Path(str(config.task.path)).name
        self.seen[task] = self.seen.get(task, 0) + 1
        reward = float(self.seen[task] % 2) if task in self.spread else 1.0
        write_result(Path(config.trials_dir) / config.trial_name, reward=reward)


Fake = TypeVar("Fake")


class Harbor(Protocol):
    """Type of the `harbor` fixture, which installs a fake trial runner and returns it."""

    def __call__(self, fake: Fake, /) -> Fake: ...


@dataclass
class FakeProxy:
    """Fake `Proxy` driven by `ProxyEndpoint` that logs each call and answers by task name."""

    origin: str = "http://host.docker.internal:8000"
    loopback: str = "http://127.0.0.1:8000"
    token: str = "harness-token"
    control_token: str | None = "control-token"
    answers: dict[str, list[Record]] = field(default_factory=dict)
    default: list[Record] | None = None
    failing: set[str] = field(default_factory=set)
    placed: list[tuple[str, Any, dict[str, Any]]] = field(default_factory=list)
    addressed: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    swapped: list[str | None] = field(default_factory=list)
    #: Token budget reported by `/healthz`; None means no budget is enforced.
    budget: int | None = None
    closed: int = 0
    #: Number of times `/healthz` was probed.
    probed: int = 0
    #: Reason the proxy exited, or None while it is alive.
    exited: str | None = None

    @property
    def host(self) -> str:
        return urlsplit(self.origin).hostname or ""

    async def local(self, settings: Any, *, host: str, **named: Any) -> FakeProxy:
        self.placed.append(("local", dict(settings), {"host": host, **named}))
        return self

    async def tunnelled(self, settings: Any, **named: Any) -> FakeProxy:
        self.placed.append(("tunnel", dict(settings), named))
        self.origin = "https://fake-words.trycloudflare.com"
        return self

    def remote(self, url: str, token: str, control_token: str | None = None) -> FakeProxy:
        if not token:
            raise ValueError("HARNESS_PROXY_TOKEN is unset")
        self.placed.append(("remote", url, {"token": token, "control_token": control_token}))
        self.origin, self.token, self.control_token = url.rstrip("/"), token, control_token
        return self

    def address_for(self, trial: str) -> str:
        self.addressed.append(trial)
        return f"{self.origin}/r/trial/{trial}/v1"

    async def swap(self, model_path: str | None) -> None:
        if self.control_token is None:
            raise RuntimeError("no control route")
        self.swapped.append(model_path)

    async def records(self, trial: str) -> list[Record]:
        if self.control_token is None:
            raise RuntimeError("no records route")
        self.fetched.append(trial)
        task = trial.split("__")[0]
        if task in self.failing:
            raise httpx.ConnectError("the proxy went away")
        found = self.answers.get(task, self.default)
        return list(found if found is not None else [made()])

    async def ready(self, seconds: float = 30.0) -> str:
        self.probed += 1
        return MODEL

    def gone(self) -> str | None:
        return self.exited

    def close(self) -> None:
        self.closed += 1
