"""Tests for `shipyard.jobs.secrets`: the pool a trial leases an identity from, the secrets
it holds for its duration and that are scrubbed from what it leaves, and the trainer's pool from
`[rollout] secrets`."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig
from harbor.utils.env import redact_sensitive_value

from shipyard import Config, Trainer, Trial
from shipyard.config import Rollout
from shipyard.jobs.engine import HarborEngine, Job
from shipyard.jobs.harnesses import Harness
from shipyard.jobs.secrets import Pool, Secrets, scrub
from shipyard.models.endpoints import ProviderEndpoint
from shipyard.run.directory import open_run
from tests.fakes import FakeProxy, Harbor, dataset
from tests.runs import GRPO
from tests.service import FakeService

NAME = "BAUPLAN_API_KEY"
#: Pool keys longer than eight characters, so Harbor's redacted form of each is distinct.
KEYS = ("key-0000-aaaaaaaa", "key-0001-bbbbbbbb")
RUN_TOML = """[models.agent]
model = "claude-x"
provider = "anthropic"

[data]
train = "four"
batch_size = 4

[rollout]
concurrency = 4
secrets = {{ {secrets} }}

[recipe]
steps = 1
harness = "pi@0.85.1"
group_size = 2
"""


@pytest.fixture(autouse=True)
def _placeholder() -> Iterator[None]:
    """Unset the pooled variable around each test, since the engine sets it in this process."""
    os.environ.pop(NAME, None)
    yield
    os.environ.pop(NAME, None)


class Holding:
    """Fake Harbor trial runner that tracks held secrets and concurrency, and writes the
    secret where Harbor and an agent's log would. Each trial ends by raising `raises`, if set."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.configs: list[TrialConfig] = []
        self.holding: dict[str, str] = {}
        self.most = 0
        self.shared = False
        self.raises = raises

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        key = config.environment.env.get(NAME, "")
        self.shared |= key in self.holding.values()
        self.holding[config.trial_name] = key
        self.most = max(self.most, len(self.holding))
        trial = Path(config.trials_dir) / config.trial_name
        trial.mkdir(parents=True, exist_ok=True)
        shown = redact_sensitive_value(key)
        (trial / "config.json").write_text(f'{{"env": {{"{NAME}": "{shown}"}}}}\n')
        (trial / "agent.log").write_text(f"$ env\n{NAME}={key}\n")
        await asyncio.sleep(0.01)
        del self.holding[config.trial_name]
        if self.raises is not None:
            raise self.raises


class Minted(Secrets):
    """Fake `Secrets` that mints a key for each trial and records each release."""

    def __init__(self) -> None:
        self.acquired: list[Trial] = []
        self.released: list[tuple[Trial, Mapping[str, str]]] = []

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        self.acquired.append(trial)
        return {NAME: f"minted-for-{trial.name}"}

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        self.released.append((trial, held))


def _engine(tmp_path: Path, runner: Any, secrets: Secrets, concurrency: int = 4) -> HarborEngine:
    rollout = Rollout(concurrency=concurrency, jobs_dir=str(tmp_path / "jobs"))
    return HarborEngine(rollout, secrets=secrets, run_trial=runner)


async def _run(engine: HarborEngine, name: str, tasks: list[Path], attempts: int) -> list[Trial]:
    endpoint = ProviderEndpoint("m", "openrouter")
    job = await engine.run(
        Job(name, harness=Harness(name="pi")), tasks, attempts, endpoint=endpoint
    )
    return job.trials


def _batch(tmp_path: Path, count: int) -> list[Path]:
    tasks = [tmp_path / "tasks" / f"t{n}" for n in range(count)]
    for task in tasks:
        task.mkdir(parents=True, exist_ok=True)
    return tasks


def test_a_pool_pairs_its_variables_by_position_and_refuses_what_it_cannot_pair() -> None:
    environ = {"KEYS": "a, b", "USERS": "u1\nu2", "ONE": "x"}
    pool = Pool.of({"KEY": "KEYS", "USER": "USERS"}, environ)
    assert pool.identities == [{"KEY": "a", "USER": "u1"}, {"KEY": "b", "USER": "u2"}]
    with pytest.raises(ValueError, match="MISSING unset or empty"):
        Pool.of({"KEY": "MISSING"}, environ)
    with pytest.raises(ValueError, match=r"differ in size \(KEYS 2, ONE 1\)"):
        Pool.of({"KEY": "KEYS", "USER": "ONE"}, environ)


async def test_no_two_trials_in_flight_hold_one_identity_nor_more_run_than_there_are(
    tmp_path: Path,
) -> None:
    runner = Holding()
    engine = _engine(tmp_path, runner, Pool([{NAME: key} for key in KEYS]), concurrency=4)
    await _run(engine, "job-0000", _batch(tmp_path, 6), 1)
    assert len(runner.configs) == 6 and runner.most == 2 and not runner.shared
    for config in runner.configs:
        key = config.environment.env[NAME]
        assert key in KEYS and config.verifier.env == {NAME: key}, "the verifier holds it too"
    assert os.environ[NAME] == "", "the task's ${BAUPLAN_API_KEY} resolves, to no identity"


async def test_what_a_trial_held_is_scrubbed_from_its_directory(tmp_path: Path) -> None:
    engine = _engine(tmp_path, Holding(), Pool([{NAME: KEYS[0]}]))
    (trial,) = await _run(engine, "job-0000", _batch(tmp_path, 1), 1)
    left = {path.name: path.read_text() for path in trial.dir.iterdir()}
    assert left == {
        "config.json": '{"env": {"BAUPLAN_API_KEY": "${BAUPLAN_API_KEY}"}}\n',
        "agent.log": "$ env\nBAUPLAN_API_KEY=${BAUPLAN_API_KEY}\n",
    }


def test_the_scrub_leaves_binary_files_and_harbors_bare_redaction_alone(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\0" + KEYS[0].encode())
    (tmp_path / "short.txt").write_text("****")
    scrub(tmp_path, {NAME: KEYS[0], "PIN": "1234"})
    assert (tmp_path / "blob.bin").read_bytes() == b"\0" + KEYS[0].encode()
    assert (tmp_path / "short.txt").read_text() == "****"


def test_a_file_the_scrub_cannot_rewrite_is_removed_and_the_rest_still_scrubbed(
    tmp_path: Path,
) -> None:
    locked, plain = tmp_path / "locked.txt", tmp_path / "plain.txt"
    for one in (locked, plain):
        one.write_text(f"key={KEYS[0]}")
    locked.chmod(0o444)
    scrub(tmp_path, {NAME: KEYS[0]})
    assert not locked.exists(), "it held the secret and could not be rewritten"
    assert plain.read_text() == f"key=${{{NAME}}}"


@pytest.mark.parametrize("raises", [RuntimeError("the sandbox died"), asyncio.CancelledError()])
async def test_an_identity_comes_back_however_its_trial_ended(
    tmp_path: Path, raises: BaseException
) -> None:
    """A trial that raises, or is cancelled by something other than the run, still releases
    its identity. With a single identity, the second job would otherwise hang."""
    pool = Pool([{NAME: KEYS[0]}])
    await _run(_engine(tmp_path, Holding(raises), pool), "job-0000", _batch(tmp_path, 2), 1)
    runner = Holding()
    job = _run(_engine(tmp_path, runner, pool), "job-0001", _batch(tmp_path, 2), 1)
    await asyncio.wait_for(job, timeout=5)
    assert len(runner.configs) == 2


async def test_secrets_of_ones_own_are_acquired_and_released_with_each_trial(
    tmp_path: Path,
) -> None:
    runner, minted = Holding(), Minted()
    found = await _run(_engine(tmp_path, runner, minted), "job-0000", _batch(tmp_path, 2), 2)
    assert sorted(t.dir for t in minted.acquired) == sorted(t.dir for t in found)
    assert {trial.dir: held[NAME] for trial, held in minted.released} == {
        trial.dir: f"minted-for-{trial.name}" for trial in found
    }
    assert not runner.shared


def _home(tmp_path: Path, secrets: str = "") -> Path:
    """Write a config of a provider's agent over a four-task dataset, with `secrets` as the
    entries of `[rollout] secrets`."""
    if not (tmp_path / "tasks" / "four").is_dir():
        dataset(tmp_path / "tasks", "four", *"abcd")
    home = tmp_path / "bp" / "leased"
    home.mkdir(parents=True, exist_ok=True)
    (home / "run.toml").write_text(RUN_TOML.format(secrets=secrets), encoding="utf-8")
    return home


class RollingOut(GRPO):  # type: ignore[misc, valid-type]
    """GRPO whose step only rolls the agent out, as a provider's agent cannot update."""

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent.training = False

    async def step(self) -> None:
        await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=None,
        )


async def _rolled_out(home: Path, root: Path) -> None:
    directory, config = open_run(home, root=root, trainer=RollingOut)
    with RollingOut(config, directory) as trainer:
        await trainer.run()


async def test_a_trainer_leases_the_configs_pool_to_its_trials_or_gives_them_none(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BAUPLAN_API_KEYS", ",".join(KEYS))
    runner = harbor(Holding())
    await _rolled_out(_home(tmp_path, f'{NAME} = "BAUPLAN_API_KEYS"'), tmp_path / "runs")
    assert len(runner.configs) == 8 and runner.most == 2 and not runner.shared
    assert {config.environment.env[NAME] for config in runner.configs} <= set(KEYS)
    plain = harbor(Holding())
    await _rolled_out(_home(tmp_path), tmp_path / "runs")
    assert [config.environment.env for config in plain.configs] == [{}] * 8
    assert plain.most == 4, "without a pool, only [rollout] concurrency limits the trials"


def test_a_trainer_refuses_a_pool_it_cannot_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NOBODY", raising=False)
    monkeypatch.setenv("BAUPLAN_API_KEYS", ",".join(KEYS))
    monkeypatch.setenv("BAUPLAN_USERS", "one")
    directory, config = open_run(
        _home(tmp_path, f'{NAME} = "NOBODY"'), root=tmp_path / "runs", trainer=GRPO
    )
    with pytest.raises(ValueError, match=r"^\[rollout\.secrets\]: NOBODY unset or empty$"):
        Trainer(config, directory)
    paired = f'{NAME} = "BAUPLAN_API_KEYS", BAUPLAN_USER = "BAUPLAN_USERS"'
    directory, config = open_run(_home(tmp_path, paired), root=tmp_path / "runs", trainer=GRPO)
    with pytest.raises(ValueError, match=r"differ in size \(BAUPLAN_API_KEYS 2, BAUPLAN_USERS 1\)"):
        Trainer(config, directory)
