"""Test the secrets a trial holds: a pool leased one identity per trial in flight, its value in
the trial's environment and verifier, scrubbed from what the trial leaves, handed back however
the trial ends; a trainer's own; and what `check` says of them."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard.cli import resolved
from shipyard.cli.run import Run
from shipyard.config import Rollout
from shipyard.sampling import Pool, Secrets
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.sampling.engine import HarborEngine
from shipyard.sampling.secrets import redacted, scrub
from shipyard.training import DAPOTrainer, Trainer
from shipyard.types import Agent, Policy, Settings, Trial, harness
from tests.trials import blueprint, check, fixture_tasks, write_blueprint

NAME = "BAUPLAN_API_KEY"
#: Longer than eight characters, so Harbor's redacted form of each is its own.
KEYS = ("key-0000-aaaaaaaa", "key-0001-bbbbbbbb")


@pytest.fixture(autouse=True)
def _placeholder() -> Iterator[None]:
    """The engine sets the pooled name in this process; each test starts and ends without it."""
    os.environ.pop(NAME, None)
    yield
    os.environ.pop(NAME, None)


class Holding:
    """Stands in for Harbor's trial: notes what each trial holds and how many hold at once,
    writes the value as Harbor's saved config and an agent's log would, and raises when told."""

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
        (trial / "config.json").write_text(f'{{"env": {{"{NAME}": "{redacted(key)}"}}}}\n')
        (trial / "agent.log").write_text(f"$ env\n{NAME}={key}\n")
        await asyncio.sleep(0.01)
        del self.holding[config.trial_name]
        if self.raises is not None:
            raise self.raises


class Minted(Secrets):
    """A key made for each trial and revoked after it."""

    def __init__(self) -> None:
        self.acquired: list[Trial] = []
        self.released: list[tuple[Trial, Mapping[str, str]]] = []

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        self.acquired.append(trial)
        return {NAME: f"minted-for-{trial.task.name}-{trial.attempt}"}

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        self.released.append((trial, held))


def _engine(tmp_path: Path, runner: Any, secrets: Secrets, concurrency: int = 4) -> HarborEngine:
    table = Rollout(sandbox="docker", concurrency=concurrency, jobs_dir=str(tmp_path / "jobs"))
    return HarborEngine(table, secrets=secrets, run_trial=runner)


def _batch(tmp_path: Path, count: int) -> list[Path]:
    tasks = [tmp_path / "tasks" / f"t{n}" for n in range(count)]
    for task in tasks:
        task.mkdir(parents=True, exist_ok=True)
    return tasks


AGENT = Agent(Policy(ProviderEndpoint("m")), [harness("pi")])


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
    await engine.run("job-0000", _batch(tmp_path, 6), 1, agent=AGENT)
    assert len(runner.configs) == 6 and runner.most == 2 and not runner.shared
    for config in runner.configs:
        key = config.environment.env[NAME]
        assert key in KEYS and config.verifier.env == {NAME: key}, "the verifier holds it too"
    assert os.environ[NAME] == "", "the task's ${BAUPLAN_API_KEY} resolves, to no identity"


async def test_what_a_trial_held_is_scrubbed_from_its_directory(tmp_path: Path) -> None:
    runner = Holding()
    engine = _engine(tmp_path, runner, Pool([{NAME: KEYS[0]}]))
    (trial,) = await engine.run("job-0000", _batch(tmp_path, 1), 1, agent=AGENT)
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


@pytest.mark.parametrize("raises", [RuntimeError("the sandbox died"), asyncio.CancelledError()])
async def test_an_identity_comes_back_however_its_trial_ended(
    tmp_path: Path, raises: BaseException
) -> None:
    """A trial that raised, or met a cancellation not the run's, frees its identity: with one
    identity, a second job would otherwise wait forever."""
    pool = Pool([{NAME: KEYS[0]}])
    await _engine(tmp_path, Holding(raises), pool).run(
        "job-0000", _batch(tmp_path, 2), 1, agent=AGENT
    )
    runner = Holding()
    job = _engine(tmp_path, runner, pool).run("job-0001", _batch(tmp_path, 2), 1, agent=AGENT)
    await asyncio.wait_for(job, timeout=5)
    assert len(runner.configs) == 2


async def test_a_trial_that_runs_no_task_acquires_nothing(tmp_path: Path) -> None:
    runner, minted = Holding(), Minted()
    engine = _engine(tmp_path, runner, minted)
    await engine.run(
        "job-0000", _batch(tmp_path, 1), 1, agent=AGENT, settings=Settings(secrets=False)
    )
    assert minted.acquired == [] and runner.configs[0].environment.env == {}


async def test_secrets_of_ones_own_are_acquired_and_released_with_each_trial(
    tmp_path: Path,
) -> None:
    runner, minted = Holding(), Minted()
    found = await _engine(tmp_path, runner, minted).run(
        "job-0000", _batch(tmp_path, 2), 2, agent=AGENT
    )
    assert sorted(t.dir for t in minted.acquired) == sorted(t.dir for t in found)
    assert {trial.dir: held[NAME] for trial, held in minted.released} == {
        trial.dir: f"minted-for-{trial.task.name}-{trial.attempt}" for trial in found
    }
    assert not runner.shared


def test_a_trainer_leases_the_blueprints_pool_or_returns_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BAUPLAN_API_KEYS", ",".join(KEYS))
    home = blueprint(
        tmp_path, **{"[rollout]\n": f'[rollout]\nsecrets = {{ {NAME} = "BAUPLAN_API_KEYS" }}\n'}
    )
    opened = Run.open(home, root=tmp_path / "runs")
    pool = DAPOTrainer(opened.config, opened.directory).engine.secrets
    assert isinstance(pool, Pool) and pool.identities == [{NAME: key} for key in KEYS]

    class Mine(DAPOTrainer):
        def secrets(self) -> Secrets:
            return Minted()

    assert isinstance(Mine(opened.config, opened.directory).engine.secrets, Minted)
    plain = Run.open(blueprint(tmp_path / "plain"), root=tmp_path / "runs")
    assert type(Trainer(plain.config, plain.directory).engine.secrets) is Secrets


SECRETS = """
[agent]
model = "m"
provider = "openrouter"
harness = "pi@0.85.1"
{agent}
[data]
test = "fixture"
batch_size = 1
[rollout]
concurrency = {concurrency}
[rollout.secrets]
{secrets}
[recipe]
kind = "evaluate"
"""


def _check(
    tmp_path: Path, secrets: str, *, agent: str = "", concurrency: int = 2
) -> list[tuple[str, str]]:
    text = SECRETS.format(secrets=secrets, agent=agent, concurrency=concurrency)
    home = write_blueprint(tmp_path / f"bp{len(list(tmp_path.glob('bp*')))}", text)
    return [(one.level, one.text) for one in check(home) if "secrets" in one.text]


def test_check_says_how_many_identities_and_warns_when_more_trials_may_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BAUPLAN_API_KEYS", ",".join(KEYS))
    pooled = f'{NAME} = "BAUPLAN_API_KEYS"'
    assert _check(tmp_path, pooled) == [
        ("ok", f"secrets {NAME}: 2 identities from BAUPLAN_API_KEYS")
    ]
    assert _check(tmp_path, pooled, concurrency=8)[1] == (
        "warning",
        "[rollout] concurrency = 8 with 2 identities in [rollout.secrets]: at most 2 trials "
        "run at once",
    )


def test_check_blocks_a_pool_it_cannot_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NOBODY", raising=False)
    monkeypatch.setenv("BAUPLAN_API_KEYS", ",".join(KEYS))
    monkeypatch.setenv("BAUPLAN_USERS", "one")
    monkeypatch.setenv("TINKER_API_KEY", "t")
    assert _check(tmp_path, f'{NAME} = "NOBODY"') == [
        (
            "blocked",
            f"[rollout.secrets] {NAME}: NOBODY is unset or empty; it holds the values, "
            "separated by commas, one per trial in flight",
        )
    ]
    paired = f'{NAME} = "BAUPLAN_API_KEYS"\nBAUPLAN_USER = "BAUPLAN_USERS"'
    assert _check(tmp_path, paired) == [
        (
            "blocked",
            "[rollout.secrets]: the pools differ in size (BAUPLAN_API_KEYS 2, BAUPLAN_USERS 1); "
            "an identity pairs their values by position",
        )
    ]
    shared = _check(tmp_path, f'{NAME} = "BAUPLAN_API_KEYS"', agent=f'env = {{ {NAME} = "x" }}')
    assert shared[0] == (
        "blocked",
        f"[rollout.secrets] {NAME} is also in [agent] env, which gives every trial the same value",
    )
    leaked = _check(tmp_path, 'KEY = "TINKER_API_KEY"')
    assert leaked[0] == (
        "blocked",
        "[rollout.secrets] carries TINKER_API_KEY (in KEY), which must never enter a sandbox",
    )


def test_check_names_a_trainer_files_own_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    home = write_blueprint(
        tmp_path,
        SECRETS.format(secrets="", agent="", concurrency=2).replace("[rollout.secrets]\n", ""),
    )

    class Mine(Trainer):
        def secrets(self) -> Secrets:
            return Minted()

    texts = [one.text for one in resolved.report(home, Mine).findings]
    assert "secrets from Mine.secrets" in texts
    assert "secrets from Trainer.secrets" not in [
        one.text for one in resolved.report(home, Trainer).findings
    ]
