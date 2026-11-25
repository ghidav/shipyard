"""Tests for a model's seed: the seed of a new adapter's first weights, on Tinker and on
Together, and, as a list, a run for each seed, which `shipyard run` trains together in one
process and `shipyard resume` brings back together."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proxies import MODEL
from shipyard import Config
from shipyard.cli import Going, app, together
from shipyard.config import ConfigError, ModelTable, load, seeded
from shipyard.model import Model
from shipyard.run import files
from shipyard.run.checkpoint import Checkpoint
from shipyard.run.directory import open_runs, resumed_together
from shipyard.run.dry import checked
from tests.fakes import Alternating, FakeProxy, Harbor
from tests.runs import GRPO, GRPO_FILE, ONE_STEP, run_toml, train
from tests.service import FakeService
from tests.test_connect import agent_keys, connected, table
from tests.test_together import FakeResources, provisioned  # noqa: F401 - the fixture

#: The config's agent, trained from two seeds: two runs.
TWO_SEEDS = agent_keys(seed=[0, 1])


def test_a_list_of_seeds_is_a_run_for_each_and_lists_pair_by_position() -> None:
    assert seeded({"models": {"agent": {"model": "m"}}}) == [{}], "one run, which sets nothing"
    assert seeded({"models": {"agent": {"seed": 3}}}) == [{}], "one seed is the run's own"
    two = {"agent": {"seed": [0, 1]}, "critic": {"seed": [5, 7]}, "base": {"model": "m"}}
    assert seeded({"models": two}) == [
        {"models.agent.seed": "0", "models.critic.seed": "5"},
        {"models.agent.seed": "1", "models.critic.seed": "7"},
    ], "run k takes the k-th seed of each, and a model with no list is as it is in each"
    with pytest.raises(ConfigError, match=r"\[models.agent\] names 2, \[models.critic\] names 3"):
        seeded({"models": {"agent": {"seed": [0, 1]}, "critic": {"seed": [1, 2, 3]}}})


@pytest.mark.parametrize("seed", ["[]", '"a"', "true", "[0, 0.5]"])
def test_a_seed_is_a_whole_number_or_a_list_of_them(tmp_path: Path, seed: str) -> None:
    home = run_toml(tmp_path, **{"lora_rank = 32\n": f"lora_rank = 32\nseed = {seed}\n"})
    with pytest.raises(ConfigError, match=r"\[models.agent\] seed: a whole number, or a list"):
        load(home, GRPO.Recipe)


def test_a_seed_is_of_a_model_the_run_trains_one_for_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"seed: a run has one seed for a model, and this names 2"):
        Model("agent", ModelTable(model=MODEL, seed=[0, 1]))
    with pytest.raises(ValueError, match=r"seed: it seeds the adapter .* anthropic serves"):
        Model("agent", ModelTable(model="claude-sonnet-5-5", provider="anthropic", seed=1))
    with pytest.raises(ValueError, match=r"\[models.teacher\] seed: .* does not train"):
        connected(tmp_path, table("teacher", "Qwen/Qwen3-32B", seed=1))
    assert Model("base", ModelTable(model=MODEL, provider="together", sampler_replicas=0))


async def test_a_new_adapter_on_tinker_starts_from_its_models_seed(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    _, service = fakes
    harbor(Alternating())
    await train(run_toml(tmp_path, **ONE_STEP), tmp_path / "runs")
    await train(run_toml(tmp_path, **ONE_STEP, **agent_keys(seed=7)), tmp_path / "runs")
    assert service.seeded == [{}, {"seed": 7}], "no seed is sent unless the model names one"


def test_a_run_is_opened_for_each_seed_and_each_notes_the_others(tmp_path: Path) -> None:
    (one,) = open_runs(run_toml(tmp_path), root=tmp_path / "runs", trainer=GRPO_FILE)
    assert "together" not in files.read_json(one[0] / files.RUN), "a run alone notes no other"
    opened = open_runs(run_toml(tmp_path, **TWO_SEEDS), root=tmp_path / "runs", trainer=GRPO_FILE)
    names = [directory.name for directory, _ in opened]
    assert len(set(names)) == 2
    for at, (directory, config) in enumerate(opened):
        assert config.models["agent"].seed == at, "its own seed, in the config it runs"
        assert load(directory, GRPO.Recipe).models["agent"].seed == at, "and in its copy of it"
        assert files.read_json(directory / files.RUN)["together"] == names
    with pytest.raises(ConfigError, match="seed: a whole number"):
        open_runs(run_toml(tmp_path, **agent_keys(seed=[])), root=tmp_path / "runs", trainer=GRPO)
    assert len(list((tmp_path / "runs").iterdir())) == 3, "a config refused makes no run"


async def test_the_seeds_of_a_model_on_together_train_together_on_one_model_resource(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],  # noqa: F811
) -> None:
    proxy, _ = fakes
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **agent_keys(provider="together", seed=[0, 1]))
    opened = open_runs(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    assert await together([Going(directory, config) for directory, config in opened]) == [None] * 2
    (resources,) = provisioned
    assert resources.sampler_replicas == 2, "each run's agent brings its sampler replica"
    assert sorted(asked["lora_config"]["seed"] for asked, _ in resources.sessions) == [0, 1]
    assert resources.stopped and len(proxy.placed) == 2, "a proxy each, and the GPUs stop"
    for directory, _ in opened:
        assert files.read_json(directory / files.RUN)["process"]["failed"] is False
        assert [row["tag"] for row in files.read(directory / files.CHECKPOINTS)][-1] == "final"


async def test_a_run_that_fails_leaves_the_runs_beside_it_going(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Failing(GRPO):  # type: ignore[misc, valid-type]
        """The shipped recipe, whose second seed fails in its first step."""

        async def step(self) -> None:
            if self.config.models["agent"].seed == 1:
                raise RuntimeError("the second seed fails")
            await super().step()

    monkeypatch.setattr("shipyard.cli.trainer_class", lambda directory: Failing)
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **agent_keys(provider="together", seed=[0, 1]))
    opened = open_runs(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    ended, failed = await together([Going(directory, config) for directory, config in opened])
    assert ended is None and str(failed) == "the second seed fails"
    (resources,) = provisioned
    assert resources.stopped, "the resource they shared stops once both have ended"
    noted = [files.read_json(directory / files.RUN)["process"] for directory, _ in opened]
    assert [one["failed"] for one in noted] == [False, True]
    assert noted[1]["error"] == "RuntimeError: the second seed fails"

    class Unread(GRPO):  # type: ignore[misc, valid-type]
        """A recipe the run refuses before anything opens, for its second seed alone."""

        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            if config.models["agent"].seed == 1:
                raise ValueError("refused as it is made")

    monkeypatch.setattr("shipyard.cli.trainer_class", lambda directory: Unread)
    opened = open_runs(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    ended, failed = await together([Going(directory, config) for directory, config in opened])
    assert ended is None and str(failed) == "refused as it is made", "the first is not held up"
    assert provisioned[1].sampler_replicas == 1 and provisioned[1].stopped, "its own alone"


def test_a_resume_brings_back_the_runs_started_together_that_have_steps_left(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    home = run_toml(tmp_path, **agent_keys(seed=[0, 1, 2]))
    (a, _), (b, _), (c, _) = open_runs(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    for run, step in ((a, 1), (b, 1)):
        Checkpoint.save(run, f"step-{step}", step, {"steps": step})
    found = resumed_together(b)
    assert [source for *_, source in found] == [a, b], "in the order they started; c has none"
    assert "has no checkpoint to go on from" in caplog.text and c.name in caplog.text
    new = [directory.name for directory, *_ in found]
    for (directory, config, checkpoint, source), seed in zip(found, (0, 1), strict=True):
        assert config.models["agent"].seed == seed and checkpoint.tag == "step-1"
        noted = files.read_json(directory / files.RUN)
        assert noted["origin"]["run"] == str(source.resolve()) and noted["together"] == new
    Checkpoint.save(a, "final", 2, {"steps": 2})
    assert [source for *_, source in resumed_together(b)] == [b], "a has nothing left to run"
    assert [source for *_, source in resumed_together(a)] == [b], "asked of a, b still goes on"
    with pytest.raises(ValueError, match="has no checkpoint 'final'; it has step-1"):
        resumed_together(a, tag="final")
    note = files.read_json(b / files.RUN)
    files.write_json(b / files.RUN, {**note, "process": {"pid": os.getpid()}})
    assert resumed_together(a) == [], "b's process still runs: it is not started twice"
    assert "still runs" in caplog.text


def test_dry_says_the_runs_a_list_of_seeds_starts(tmp_path: Path, fakes: Any) -> None:
    home = run_toml(tmp_path, **TWO_SEEDS)
    result = CliRunner().invoke(app, ["run", str(home), str(GRPO_FILE), "--dry"])
    assert "seed = [0, 1]\n" in result.stdout, "the config as the file has it"
    assert "2 runs start together, in one process" in result.stderr
    assert not (tmp_path / "runs").exists()
    result = CliRunner().invoke(
        app, ["run", str(home), str(GRPO_FILE), "--models.agent.seed", "4", "--dry"]
    )
    assert "seed = 4\n" in result.stdout and "runs start together" not in result.stderr


def test_dry_says_the_model_resources_the_runs_would_start_on_together(
    tmp_path: Path, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def context_of(client: Any, model: str) -> int:
        return 32768

    monkeypatch.setattr("harness_proxy.backends.together.context_of", context_of)

    class Penalized(GRPO):  # type: ignore[misc, valid-type]
        """The shipped recipe, reading a model it measures a penalty against."""

        def __init__(self, config: Config, directory: Path) -> None:
            super().__init__(config, directory)
            self.base = self.models["base"]

    base = table("base", MODEL, provider="together", sampler_replicas=0)
    home = run_toml(tmp_path, base, **agent_keys(provider="together"))
    runs = [load(home, Penalized.Recipe, {"models.agent.seed": str(seed)}) for seed in range(3)]
    assert checked(Penalized, *runs).starts == [
        "3 runs start together, in one process",
        f"together: a model resource for {MODEL}, with 3 sampler replicas and 4 sessions: "
        "agent of run 1 (seed 0); agent of run 2 (seed 1); agent of run 3 (seed 2); "
        "one untrained, for base of run 1, base of run 2, base of run 3",
    ]
    assert checked(Penalized, load(home, Penalized.Recipe)).starts == [
        f"together: a model resource for {MODEL}, with 1 sampler replica and 2 sessions: "
        "agent; one untrained, for base"
    ]


def test_the_command_trains_a_run_for_each_seed_and_names_each_in_its_lines(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    provisioned: list[FakeResources],  # noqa: F811
) -> None:
    harbor(Alternating())
    home = run_toml(tmp_path, **ONE_STEP, **agent_keys(provider="together", seed=[0, 1]))
    root = tmp_path / "runs"
    result = CliRunner().invoke(app, ["run", str(home), str(GRPO_FILE), "--root", str(root)])
    assert result.exit_code == 0, result.output
    names = sorted(one.name for one in root.iterdir())
    said = [line for line in result.output.splitlines() if "trained by GRPOTrainer" in line]
    assert sorted(line.split()[1] for line in said) == names and len(names) == 2
    (resources,) = provisioned
    assert resources.stopped and len(resources.sessions) == 2


def test_runs_that_would_share_one_proxy_are_refused_before_any_is_made(tmp_path: Path) -> None:
    for key, why in (
        ("bind_port = 9000", r"\[rollout\] bind_port: the 2 runs .* cannot all bind port 9000"),
        ('endpoint_url = "https://proxy.example"', r"\[rollout\] endpoint_url: the 2 runs"),
    ):
        home = run_toml(tmp_path, **TWO_SEEDS, **{"concurrency = 2": f"concurrency = 2\n{key}"})
        with pytest.raises(ConfigError, match=why):
            open_runs(home, root=tmp_path / "runs", trainer=GRPO_FILE)
    assert not (tmp_path / "runs").exists()
