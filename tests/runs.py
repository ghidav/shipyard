"""A run as the tests open one: a blueprint over the four-task dataset, a trainer trained on it
with the `fakes` and `harbor` fixtures standing in for Tinker, the proxy and Harbor, and the
step rows it wrote."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shipyard import Trainer
from shipyard.run import files
from shipyard.run.directory import open_run, running, trainer_class
from tests.fakes import dataset

BLUEPRINT = """[agent]
model = "Qwen/Qwen3-8B"
harness = "pi@0.85.1"

[data]
train = "four"
batch_size = 2
group_size = 2

[rollout]
concurrency = 2

[recipe]
steps = 2
learning_rate = 2e-5
lora_rank = 32
"""
ONE_STEP = {"steps = 2": "steps = 1"}


def blueprint(tmp_path: Path, *keys: str, **replacements: str) -> Path:
    """Write a blueprint over the four-task dataset, with `keys` added to `[recipe]`."""
    if not (tmp_path / "tasks" / "four").is_dir():
        dataset(tmp_path / "tasks", "four", *"abcd")
    text = BLUEPRINT + "".join(f"{one}\n" for one in keys)
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp" / "four"
    home.mkdir(parents=True, exist_ok=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


async def train(home: Path, root: Path, made_by: type[Trainer] | Path = Trainer) -> Path:
    """Open a run of `home` and train it with `made_by`: a trainer class, or a recipe's trainer
    file, which the run copies and reads its `[recipe]` for, as `shipyard run` does."""
    if isinstance(made_by, Path):
        directory, config = open_run(home, root=root, trainer=made_by)
        made_by = trainer_class(directory)
    else:
        directory, config = open_run(home, root=root)
    trainer = made_by(config, directory)
    with running(directory), trainer:
        await trainer.run()
    return directory


def rows(directory: Path) -> list[dict[str, Any]]:
    """Return the run's step rows, one per step."""
    return list(files.read(directory / files.METRICS))
