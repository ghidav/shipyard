"""A run as the tests open one: a config over the four-task dataset, a trainer trained on it
with the `fakes` and `harbor` fixtures standing in for Tinker, the proxy and Harbor, and the
step rows it wrote. The trainer is the shipped GRPO recipe's unless a test gives its own."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shipyard import Trainer
from shipyard.run import files
from shipyard.run.directory import open_run, running, trainer_class, trainer_in
from tests.fakes import dataset

#: The shipped GRPO recipe's trainer file, and its trainer, which the tests run by default.
GRPO_FILE = Path(__file__).parents[1] / "recipes" / "grpo" / "trainer.py"
GRPO = trainer_in(GRPO_FILE)

RUN_TOML = """[models.agent]
model = "Qwen/Qwen3-8B"
learning_rate = 2e-5
lora_rank = 32

[data]
train = "four"
batch_size = 2

[rollout]
concurrency = 2

[recipe]
steps = 2
harness = "pi@0.85.1"
group_size = 2
"""
ONE_STEP = {"steps = 2": "steps = 1"}


def run_toml(tmp_path: Path, *keys: str, **replacements: str) -> Path:
    """Write a config over the four-task dataset, with `keys` added to `[recipe]`."""
    if not (tmp_path / "tasks" / "four").is_dir():
        dataset(tmp_path / "tasks", "four", *"abcd")
    text = RUN_TOML + "".join(f"{one}\n" for one in keys)
    for old, new in replacements.items():
        text = text.replace(old, new)
    home = tmp_path / "bp" / "four"
    home.mkdir(parents=True, exist_ok=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    return home


async def train(home: Path, root: Path, made_by: type[Trainer] | Path = GRPO) -> Path:
    """Open a run of `home` and train it with `made_by`: a trainer class, or a recipe's trainer
    file, which the run reads its `[recipe]` for and a resume finds again, as `shipyard run`
    does."""
    directory, config = open_run(home, root=root, trainer=made_by)
    if isinstance(made_by, Path):
        made_by = trainer_class(directory)
    trainer = made_by(config, directory)
    with running(directory), trainer:
        await trainer.run()
    return directory


def rows(directory: Path) -> list[dict[str, Any]]:
    """Return the run's step rows, one per step."""
    return list(files.read(directory / files.METRICS))
