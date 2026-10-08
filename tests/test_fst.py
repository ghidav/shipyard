"""Fast-slow training: the paper's defaults, the slow recipe fst resolves to, the jobs of a
slow step merged into one group per task, and one cycle end to end on fakes."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.config import ConfigError, check, load
from shipyard.data import batches
from shipyard.gepa.reflect import COLLECTED
from shipyard.modules import SKILL_FILE, seed
from shipyard.recipes import resolution, run_recipe
from shipyard.recipes.fst import ADAM, BUDGET_PASSES, merged, preset
from shipyard.rollout import Rollouts
from shipyard.run import Run
from tests.records import made
from tests.test_cycle import drawn
from tests.test_loop import fakes  # noqa: F401 - the proxy and the Tinker service, faked
from tests.trials import FIXTURES, write_blueprint, write_result

FST = Path(__file__).parent / "blueprints" / "fst"
VALID = Path(__file__).parent / "modules" / "valid"


def _text(**changes: str) -> str:
    text = (FST / "run.toml").read_text(encoding="utf-8")
    for old, new in changes.items():
        text = text.replace(old, new)
    return text


def test_the_fixture_takes_the_papers_defaults() -> None:
    recipe = load(FST).recipe
    assert (recipe.slow, recipe.kl_coef, recipe.edits) == ("cispo", 0.001, "incremental")
    assert (recipe.cycle, recipe.population, recipe.anchor) == (2, 2, None)
    found = preset(recipe)
    assert (found.name, found.loss_fn) == ("fst", "cispo")
    assert found.loss_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    assert (found.substeps, found.aggregation, found.refill) == (1, "prompt", 0), "App. D, Eq. 4"
    assert resolution(recipe).startswith("# fst: cycles of 2 cispo steps")
    assert (found.overlong_penalty, found.adam) == (0.0, ADAM), "App. D; ScaleRL A.10"
    assert (ADAM.beta1, ADAM.beta2, ADAM.eps, ADAM.weight_decay, ADAM.warmup) == (
        0.9,
        0.999,
        1e-8,
        0.0,
        10,
    ), "App. D"
    assert ADAM.grad_clip_norm == 0.0, "App. D states no gradient clipping"
    assert resolution(recipe).endswith(
        "no lower bound, averaged per prompt; 1 substep; adamw betas 0.9 / 0.999, eps 1e-08, "
        "learning rate warmed up over 10 steps; degenerate groups dropped; "
        "kl 0.001 to the starting weights"
    )


def test_another_slow_recipe_takes_its_own_knobs(tmp_path: Path) -> None:
    text = _text(**{'kind = "fst"': 'kind = "fst"\nslow = "dapo"\nclip_high = 0.3'})
    found = preset(load(write_blueprint(tmp_path, text)).recipe)
    assert found.loss_fn == "ppo"
    assert found.loss_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.3}
    assert (found.substeps, found.refill) == (1, 0), "fst's own step, not dapo's"


def test_a_knob_of_another_slow_recipe_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refused:
        load(write_blueprint(tmp_path, _text(**{'kind = "fst"': 'kind = "fst"\nclip = 0.2'})))
    assert refused.value.problems == [
        "[recipe] clip: not a knob of slow = 'cispo', which takes clip_high"
    ]


def test_a_population_that_does_not_divide_the_group_is_blocked(tmp_path: Path) -> None:
    home = write_blueprint(tmp_path, _text(**{"population = 2": "population = 3"}))
    texts = [found.text for found in check(home) if found.level == "blocked"]
    assert any(text.startswith("[recipe] population: 3 does not divide") for text in texts)


def _part(job: str, suffix: str, updates: int = 3) -> Rollouts:
    trials = [Path(f"jobs/{job}/{task}__{suffix}") for task in ("a", "b")]
    return Rollouts(
        job=job,
        trials=trials,
        plan=[(Path(task), 0) for task in ("a", "b")],
        records={trial.name: [made()] for trial in trials},
        updates=updates,
    )


def test_the_jobs_of_a_step_merge_into_one_group_per_task() -> None:
    found = merged([_part("j1", "one"), _part("j2", "two")], share=1)
    assert [trial.name for trial in found.trials] == ["a__one", "a__two", "b__one", "b__two"]
    assert [task.name for task, _ in found.plan] == ["a", "a", "b", "b"]
    assert set(found.records or {}) == {"a__one", "a__two", "b__one", "b__two"}
    assert (found.job, found.updates) == ("j1+j2", 3)
    with pytest.raises(ValueError, match="different weights"):
        merged([_part("j1", "one"), _part("j2", "two", updates=4)], share=1)


class Trials:
    """A policy trial scores 1 when its skill names its task; a reflection publishes a
    skill naming the task of its first trace."""

    def __init__(self) -> None:
        self.configs: list[TrialConfig] = []

    async def __call__(self, config: TrialConfig) -> None:
        self.configs.append(config)
        task = Path(str(config.task.path))
        trial = Path(config.trials_dir) / config.trial_name
        if task.name == "reflection":
            traces = sorted((task / "environment" / "traces").iterdir())
            named = traces[0].read_text().splitlines()[0].removeprefix("task: ")
            published = trial.joinpath(*COLLECTED)
            published.mkdir(parents=True)
            skill = (
                f"---\nname: solving\ndescription: what to do.\n---\n\nmention {Path(named).name}\n"
            )
            (published / SKILL_FILE).write_text(skill)
            write_result(trial, tokens=(1000, 100, 50), provider="anthropic")
            return
        text = (Path(config.agent.skills[0]) / "solving" / SKILL_FILE).read_text()
        trial.mkdir(parents=True)
        (trial / "agent").mkdir()
        (trial / "agent" / "pi.txt").write_text(f"working on {task.name}")
        write_result(trial, reward=1.0 if f"mention {task.name}\n" in text else 0.0)


@pytest.mark.usefixtures("fakes")
async def test_a_cycle_evolves_two_texts_then_splits_every_group_across_them(
    tmp_path: Path,
) -> None:
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = Trials()
    with opened:
        await run_recipe(opened)
    rows = list(record.read(opened.directory / record.METRICS))
    (evolution,) = [row for row in rows if row.get("evolution")]
    population = evolution["population"]
    assert evolution["cycle"] == 0 and len(population) == 2 and population[0] != population[1]
    assert all(row["cycle"] == 0 for row in rows if "round" in row), "gepa's rounds, by cycle"
    steps = [row for row in rows if "step" in row]
    assert [(row["step"], row["cycle"]) for row in steps] == [(0, 0), (1, 0)]
    assert any(row["trained"] for row in steps), "a task the two texts split has spread"
    jobs = [
        job for job in record.read(opened.directory / record.JOBS) if job["purpose"] == "rollout"
    ]
    assert [job["modules"] for job in jobs[-4:]] == population * 2, "one job per text per step"
    # The anchor set is both what the search learns from and what it keeps texts by: the
    # seed's job covers it, and each round's minibatch is drawn from it.
    assert jobs[0]["tasks"] == 4 and all(job["tasks"] == 3 for job in jobs[1:3])
    kept = opened.directory / record.MODULES
    assert [seed(kept / "cycle-0" / rank).digest for rank in ("0", "1")] == population
    assert seed(kept / "best").digest == population[0]
    assert Run.read(opened.directory)["kind"] == "fst"


@pytest.mark.usefixtures("fakes")
async def test_the_fast_phase_scores_each_cell_with_one_rollout_on_five_passes(
    tmp_path: Path,
) -> None:
    """FST App. D: 960 metric calls over 192 examples, one rollout each, whatever share of
    the group a text takes in the slow steps (here two of four)."""
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    home = write_blueprint(tmp_path, _text(**{'"aime-train"': '"four"'}))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = Trials()
    with opened:
        await run_recipe(opened)
    rows = list(record.read(opened.directory / record.METRICS))
    (evolution,) = [row for row in rows if row.get("evolution")]
    assert 0 < evolution["spent"] <= BUDGET_PASSES * 4, "five passes over four tasks"
    jobs = [
        job for job in record.read(opened.directory / record.JOBS) if job["purpose"] == "rollout"
    ]
    fast, slow = jobs[:-4], jobs[-4:]
    assert fast and all(job["trials"] == job["tasks"] for job in fast), "one rollout a cell"
    assert all(job["trials"] == 2 * job["tasks"] for job in slow), "group_size / population"


@pytest.mark.usefixtures("fakes")
async def test_cycle_c_draws_its_minibatches_with_data_seed_plus_c(tmp_path: Path) -> None:
    """Two cycles over two epochs of four tasks: each fast phase's parents run on the
    minibatches `[data] seed + cycle` predicts over that cycle's tasks."""
    for name in ("a", "b", "c", "d"):
        shutil.copytree(FIXTURES / "fixture" / "alpha", tmp_path / "tasks" / "four" / name)
    changes = {
        '"aime-train"': '"four"',
        "group_size = 4": "group_size = 2\nepochs = 2\nseed = 3",
    }
    home = write_blueprint(tmp_path, _text(**changes))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.run_trial = trials = Trials()
    with opened:
        await run_recipe(opened)
    names = [Path(str(config.task.path)).name for config in trials.configs]
    runs = [names[at - 3 : at] for at, name in enumerate(names) if name == "reflection"]
    cycles = [
        row["cycle"] for row in record.read(opened.directory / record.METRICS) if "round" in row
    ]
    planned = list(batches(["four"], size=2, seed=3, epochs=2))
    for cycle in (0, 1):
        tasks = list(
            dict.fromkeys(task for batch in planned[2 * cycle : 2 * cycle + 2] for task in batch)
        )
        ran = [batch for batch, at in zip(runs, cycles, strict=True) if at == cycle]
        assert ran and ran == drawn(tasks, 3, random.Random(3 + cycle), len(ran))


def test_fst_averages_per_prompt_whatever_its_slow_loss(tmp_path: Path) -> None:
    """FST aggregates at the prompt level (Eq. 4); a dr-grpo step on its own sums."""
    text = _text(**{'kind = "fst"': 'kind = "fst"\nslow = "dr-grpo"'})
    found = preset(load(write_blueprint(tmp_path, text)).recipe)
    assert (found.aggregation, found.refill) == ("prompt", 0)
    assert (found.length_cap, found.adam) == (0.5, ADAM), "dr-grpo's cap, FST's optimizer"
