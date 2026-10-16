"""Where a trainer writes: its directory, wandb beside it, and both closed with the trainer."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from shipyard import record
from shipyard.config import load
from shipyard.cost import Costs
from shipyard.logs import Folder, Logger, Loggers, Wandb
from shipyard.recipes import trainer_for
from shipyard.run import Run
from shipyard.training.trainer import Trainer
from tests.records import FakeProxy
from tests.trainers import FakeService
from tests.trials import Alternating, Harbor, blueprint, check, dataset

WANDB = '\n[wandb]\nproject = "shipyard"\nentity = "team"\n'


class FakeWandbRun:
    """What the trainer asked of its wandb run."""

    def __init__(self, **init: Any) -> None:
        self.init = init
        self.logged: list[dict[str, Any]] = []
        self.metrics: list[tuple[str, str | None]] = []
        self.summary: dict[str, Any] = {}
        self.finished: list[int | None] = []

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        self.metrics.append((name, step_metric))

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data)

    def finish(self, exit_code: int | None = None) -> None:
        self.finished.append(exit_code)


@pytest.fixture(autouse=True)
def _here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Datasets are looked up under the working directory's `tasks/`."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def wandb(monkeypatch: pytest.MonkeyPatch) -> list[FakeWandbRun]:
    """A stand-in `wandb` module; the list holds every run it started."""
    started: list[FakeWandbRun] = []

    def init(**kwargs: Any) -> FakeWandbRun:
        started.append(FakeWandbRun(**kwargs))
        return started[-1]

    module = ModuleType("wandb")
    module.__spec__ = ModuleSpec("wandb", None)
    module.init = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    return started


def _wandb_home(tmp_path: Path, **replacements: str) -> Path:
    """The dapo fixture with a `[wandb]` table."""
    home = blueprint(tmp_path, **replacements)
    (home / "run.toml").write_text((home / "run.toml").read_text() + WANDB)
    return home


def _wandb_run(tmp_path: Path) -> Run:
    return Run.open(_wandb_home(tmp_path), root=tmp_path / "runs")


def test_the_folder_numbers_metrics_on_from_the_rows_already_there(tmp_path: Path) -> None:
    Folder(tmp_path).log("train", step=0, reward=0.5)
    again = Folder(tmp_path)
    again.log("eval", step=1, evaluation=True)
    again.job({"job": "j-0000", "trials": 2})
    again.checkpoint({"tag": "final", "step": 1})
    costs = Costs()
    costs.add_party("tinker", input_tokens=3)
    again.costs(costs)
    metrics = list(record.read(tmp_path / record.METRICS))
    assert [(row["seq"], row["step"]) for row in metrics] == [(1, 0), (2, 1)]
    assert "section" not in metrics[1] and metrics[1]["evaluation"] is True
    assert [row["job"] for row in record.read(tmp_path / record.JOBS)] == ["j-0000"]
    assert [row["tag"] for row in record.read(tmp_path / record.CHECKPOINTS)] == ["final"]
    assert record.read_json(tmp_path / record.COSTS) == costs.to_dict()


def test_loggers_hand_every_row_to_each_in_order() -> None:
    told: list[tuple[str, Any]] = []

    class Told(Logger):
        def __init__(self, name: str) -> None:
            self.name = name

        def log(self, section: str, /, **row: Any) -> None:
            told.append((self.name, (section, row)))

        def close(self, error: BaseException | None) -> None:
            told.append((self.name, error))

    both = Loggers([Told("a"), Told("b")])
    both.log("train", step=0)
    both.job({"job": "j"})
    both.close(None)
    assert told == [
        ("a", ("train", {"step": 0})),
        ("b", ("train", {"step": 0})),
        ("a", None),
        ("b", None),
    ]


def test_wandb_takes_the_runs_id_and_writes_each_rows_numbers_under_its_section(
    tmp_path: Path, wandb: list[FakeWandbRun]
) -> None:
    opened = _wandb_run(tmp_path)
    logs = Wandb(opened.config, opened.directory)
    (started,) = wandb
    assert started.init == {
        "project": "shipyard",
        "entity": "team",
        "id": opened.id,
        "name": opened.id,
        "resume": "allow",
        "reinit": "create_new",
        "dir": opened.directory,
        "config": opened.config.model_dump(mode="json"),
    }
    logs.log("train", step=0, trained=True, reward_mean=0.5, jobs=["j"], loss_fn="ppo", kl=None)
    logs.log("train", step=1, reward_mean=0.75)
    logs.log("eval", step=1, evaluation=True, reward_mean=0.25, checkpoint="final")
    logs.log("search", round=1, parent="abc", accepted=False)
    logs.log("other", value=1.0)
    assert started.metrics == [
        ("train/step", None),
        ("train/*", "train/step"),
        ("eval/step", None),
        ("eval/*", "eval/step"),
        ("search/round", None),
        ("search/*", "search/round"),
    ]
    assert started.logged == [
        {"train/step": 0, "train/trained": True, "train/reward_mean": 0.5},
        {"train/step": 1, "train/reward_mean": 0.75},
        {"eval/step": 1, "eval/evaluation": True, "eval/reward_mean": 0.25},
        {"search/round": 1, "search/accepted": False},
        {"other/value": 1.0},
    ]
    costs = Costs()
    costs.add_party("tinker", train_tokens=7)
    logs.costs(costs)
    assert started.summary == {"costs": costs.to_dict()}
    logs.close(RuntimeError("the loop broke"))
    assert started.finished == [1]


def test_wandb_missing_is_said_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _wandb_run(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"install shipyard\[wandb\]"):
        Wandb(opened.config, opened.directory)


def test_check_names_the_wandb_project_or_blocks_without_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wandb: list[FakeWandbRun]
) -> None:
    home = _wandb_home(tmp_path)
    found = [(one.level, one.text) for one in check(home)]
    assert ("ok", "wandb team/shipyard") in found
    monkeypatch.setitem(sys.modules, "wandb", None)
    found = [(one.level, one.text) for one in check(home)]
    assert ("blocked", "[wandb]: wandb is not installed; install shipyard[wandb]") in found


async def test_a_trainer_without_a_run_writes_to_its_directory_and_wandb_and_closes_them(
    tmp_path: Path,
    fakes: tuple[FakeProxy, FakeService],
    harbor: Harbor,
    wandb: list[FakeWandbRun],
) -> None:
    dataset(tmp_path / "tasks", "two", "x", "y")
    home = _wandb_home(tmp_path, **{"batch_size = 2": 'batch_size = 2\ntest = "two"'})
    harbor(Alternating())
    config, out = load(home), tmp_path / "out" / "dapo-1"
    with trainer_for(config.recipe)(config, out) as trainer:
        await trainer.train()
    (started,) = wandb
    assert started.init["id"] == "dapo-1" and started.init["dir"] == out
    metrics = list(record.read(out / record.METRICS))
    assert [row["step"] for row in metrics] == [0, 1, 2]
    assert [row["job"] for row in record.read(out / record.JOBS)][0] == "dapo-1-0000"
    noted = record.read_json(out / record.PROCESS)
    assert list(noted) == ["proxy"], "where the proxy stood; who ran it is a run's to note"
    assert [sorted({key.split("/")[0] for key in one}) for one in started.logged] == [
        ["train"],
        ["train"],
        ["eval"],
    ]
    assert [one["train/step"] for one in started.logged[:2]] == [0, 1]
    assert started.logged[2]["eval/reward_mean"] == 0.5
    assert started.summary["costs"] == record.read_json(out / record.COSTS)
    assert started.finished == [0]


def test_a_trainer_that_raises_closes_wandb_as_failed(
    tmp_path: Path, wandb: list[FakeWandbRun]
) -> None:
    opened = _wandb_run(tmp_path)
    with pytest.raises(RuntimeError), Trainer(opened.config, opened.directory):
        raise RuntimeError("the loop broke")
    (started,) = wandb
    assert started.finished == [1]


def test_a_trainer_without_wandb_writes_to_its_folder_alone(tmp_path: Path) -> None:
    opened = Run.open(blueprint(tmp_path), root=tmp_path / "runs")
    trainer = Trainer(opened.config, opened.directory)
    assert [type(one) for one in trainer.logs.each] == [Folder]


def test_real_wandb_offline_writes_its_run_inside_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    opened = _wandb_run(tmp_path)
    logs = Wandb(opened.config, opened.directory)
    logs.log("train", step=0, reward_mean=0.5, jobs=["j"])
    logs.costs(Costs())
    logs.close(None)
    (kept,) = (opened.directory / "wandb").glob(f"offline-run-*-{opened.id}")
    assert kept.is_dir()
