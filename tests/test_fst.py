"""Fast-slow training: the paper's defaults, the slow recipe fst resolves to, the jobs of a
slow step merged into one group per task, and one cycle end to end on fakes."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.config import (
    SLOW_KNOBS,
    CispoRecipe,
    ConfigError,
    DapoRecipe,
    DrGrpoRecipe,
    FstRecipe,
    Gradient,
    load,
)
from shipyard.data import Generator, tasks
from shipyard.modules import seed
from shipyard.recipes import resolution, trainer_for
from shipyard.recipes.cispo import CISPOTrainer
from shipyard.recipes.fst import ADAM, BUDGET_PASSES, fst_trainer, merged, slow_recipe
from shipyard.run import Run
from shipyard.runner import Runner
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.types import Agent, Group, Member, Policy, Sequence, Trial, Verdict, harness
from tests.trials import (
    BLUEPRINTS,
    VALID,
    Fakes,
    Stopped,
    check,
    dataset,
    drawn,
    write_blueprint,
    write_result,
)

FST = BLUEPRINTS / "fst"
AGENT = Agent(Policy(ProviderEndpoint("m")), [harness("pi")])


def _text(**changes: str) -> str:
    text = (FST / "run.toml").read_text(encoding="utf-8")
    for old, new in changes.items():
        text = text.replace(old, new)
    return text


def test_the_fixture_takes_the_papers_defaults(tmp_path: Path) -> None:
    recipe = load(FST).recipe
    assert (recipe.slow, recipe.kl_coef, recipe.edits) == ("cispo", 0.001, "incremental")
    assert (recipe.cycle, recipe.population, recipe.anchor) == (2, 2, None)
    assert recipe.patience is None, "gepa's: the budget alone ends a fast phase"
    trainer = trainer_for(recipe)(Run.open(FST, root=tmp_path))
    assert type(trainer).__name__ == "FSTCISPOTrainer" and trainer.loss_fn == "cispo"
    assert trainer.bounds() == {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    assert (trainer.recipe.substeps, trainer.recipe.overlong_penalty) == (1, 0.0), "App. D"
    assert trainer.weights([1, 3]) == [2.0, 2 / 3], "Eq. 4: averaged per prompt"
    assert trainer.ADAM == ADAM, "App. D; ScaleRL A.10"
    assert (ADAM.beta1, ADAM.beta2, ADAM.eps, ADAM.weight_decay) == (
        0.9,
        0.999,
        1e-8,
        0.0,
    ), "App. D; its 10-step warm-up is the warmup key's"
    assert ADAM.grad_clip_norm == 0.0, "App. D states no gradient clipping"
    assert resolution(recipe).startswith("# fst: cycles of 2 cispo steps")
    assert resolution(recipe).endswith(
        "no lower bound, averaged per prompt; 1 substep; adamw betas 0.9 / 0.999, eps 1e-08, "
        "no warm-up (the paper's is 10 steps); degenerate groups dropped; "
        "kl 0.001 to the starting weights"
    )


def test_another_slow_recipe_takes_its_own_knobs(tmp_path: Path) -> None:
    text = _text(**{'kind = "fst"': 'kind = "fst"\nslow = "dapo"\nclip_high = 0.3'})
    home = write_blueprint(tmp_path, text)
    trainer = trainer_for(load(home).recipe)(Run.open(home, root=tmp_path / "runs"))
    assert trainer.loss_fn == "ppo"
    assert trainer.bounds() == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.3}
    assert trainer.recipe.substeps == 1, "fst's own step, not dapo's"


@pytest.mark.parametrize(
    ("slow", "recipe"), [("dapo", DapoRecipe), ("dr-grpo", DrGrpoRecipe), ("cispo", CispoRecipe)]
)
def test_slow_knobs_are_each_slow_recipes_own_fields_and_fsts(slow: str, recipe: type) -> None:
    """A knob a slow recipe gains must reach fst's list and its fields, or fst cannot set
    it. The overlong term stays out: a slow step trains on the batch it sampled."""
    outside = {"overlong_penalty", "overlong_buffer"}
    own = set(recipe.model_fields) - set(Gradient.model_fields) - {"kind"} - outside
    assert own == set(SLOW_KNOBS[slow])
    assert own <= set(FstRecipe.model_fields) and not outside & set(FstRecipe.model_fields)


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


def _group(task: str, job: str, *rewards: float) -> Group:
    members = [
        Member(
            Trial(job, Path(task), at, harness("pi"), Path(f"jobs/{job}/{task}__{at}")),
            Verdict(reward, None, None),
            (Sequence((1, 2, 10 + at), ((2, 3),), (-0.1,)),),
            1,
        )
        for at, reward in enumerate(rewards)
    ]
    return Group(Path(task), AGENT, harness("pi"), members)


async def test_the_jobs_of_a_step_merge_into_one_group_per_task(tmp_path: Path) -> None:
    trainer = fst_trainer(CISPOTrainer)(Run.open(FST, root=tmp_path))
    parts = [
        [_group("a", "j1", 1.0), _group("b", "j1", 0.0)],
        [_group("a", "j2", 0.0), _group("b", "j2", 0.0)],
    ]
    found = trainer.credited(merged(parts))
    assert [group.task.name for group in found] == ["a", "b"]
    assert [[one.trial.job for one in group.members] for group in found] == [["j1", "j2"]] * 2
    assert found[0].shaped == [1.0, 0.0] and found[0].reward_variance == 0.25
    assert found[1].shaped == [0.0, 0.0] and found[1].reward_variance == 0.0
    assert found[0].advantages == pytest.approx([1.0, -1.0], abs=1e-5), "credited as one group"
    with pytest.raises(ValueError, match="different numbers of groups"):
        merged([parts[0], parts[1][:1]])


@pytest.mark.usefixtures("fakes")
async def test_the_slow_steps_run_the_slow_classs_own_train(tmp_path: Path) -> None:
    passes: list[int] = []

    class Marked(CISPOTrainer):
        async def train(self, agent: Agent, tasks: Any) -> Agent:
            passes.append(self.steps)
            return await super().train(agent, tasks)

    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = Fakes()
    started = Runner.from_run(opened, fst_trainer(Marked))
    with opened:
        await started.run()
    assert type(started.trainer).__name__ == "FSTMarked"
    assert passes and passes[0] == 0 and len(passes) == started.trainer.cycles, (
        "the slow class's train runs once per cycle, for its slow steps"
    )


@pytest.mark.usefixtures("fakes")
async def test_a_cycle_evolves_two_texts_then_splits_every_group_across_them(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = Fakes()
    with opened:
        await Runner.from_run(opened).run()
    rows = list(record.read(opened.directory / record.METRICS))
    (evolution,) = [row for row in rows if row.get("evolution")]
    population = evolution["population"]
    assert evolution["cycle"] == 0 and len(population) == 2 and population[0] != population[1]
    assert all(row["cycle"] == 0 for row in rows if "round" in row), "gepa's rounds, by cycle"
    steps = [row for row in rows if "step" in row]
    assert [(row["step"], row["cycle"]) for row in steps] == [(0, 0), (1, 0)]
    assert any(row["trained"] for row in steps), "a task the two texts split has spread"
    jobs = [
        job
        for job in record.read(opened.directory / record.JOBS)
        if job["purpose"] in ("training", "search")
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
    """FST App. D: 960 metric calls over 192 examples, one rollout each, at any share of
    the group a text takes in the slow steps (here two of four)."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(tmp_path, _text(**{'"aime-train"': '"four"'}))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = Fakes()
    with opened:
        await Runner.from_run(opened).run()
    rows = list(record.read(opened.directory / record.METRICS))
    (evolution,) = [row for row in rows if row.get("evolution")]
    assert 0 < evolution["spent"] <= BUDGET_PASSES * 4, "five passes over four tasks"
    jobs = [
        job
        for job in record.read(opened.directory / record.JOBS)
        if job["purpose"] in ("training", "search")
    ]
    fast, slow = jobs[:-4], jobs[-4:]
    assert fast and all(job["trials"] == job["tasks"] for job in fast), "one rollout a cell"
    assert all(job["trials"] == 2 * job["tasks"] for job in slow), "group_size / population"


@pytest.mark.usefixtures("fakes")
async def test_patience_ends_a_fast_phase_as_it_ends_gepa(tmp_path: Path) -> None:
    """`patience = 1` under a budget that would run on: the fast phase stops at its first
    round that does not raise the best anchor-set mean. Each text here names one task, so
    the first child's rise is the only one."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    changes = {'"aime-train"': '"four"', 'kind = "fst"': 'kind = "fst"\npatience = 1\nbudget = 100'}
    home = write_blueprint(tmp_path, _text(**changes))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = Fakes()
    with opened:
        await Runner.from_run(opened).run()
    rows = list(record.read(opened.directory / record.METRICS))
    rounds = [row for row in rows if "round" in row]
    assert len(rounds) == 2 and rounds[0]["accepted"] is True
    (evolution,) = [row for row in rows if row.get("evolution")]
    assert evolution["rounds"] == 2 and evolution["spent"] < 100


@pytest.mark.usefixtures("fakes")
async def test_cycle_c_draws_its_minibatches_with_data_seed_plus_c(tmp_path: Path) -> None:
    """Two cycles over two epochs of four tasks: each fast phase's parents run on the
    minibatches `[data] seed + cycle` predicts over that cycle's tasks."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    changes = {
        '"aime-train"': '"four"',
        "group_size = 4": "group_size = 2\nepochs = 2\nseed = 3",
    }
    home = write_blueprint(tmp_path, _text(**changes))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = trials = Fakes()
    with opened:
        await Runner.from_run(opened).run()
    names = [Path(str(config.task.path)).name for config in trials.configs]
    runs = [names[at - 3 : at] for at, name in enumerate(names) if name == "reflection"]
    cycles = [
        row["cycle"] for row in record.read(opened.directory / record.METRICS) if "round" in row
    ]
    made = Generator(tasks("four"), 2, 3)
    planned = [*made.batches(0), *made.batches(1)]
    for cycle in (0, 1):
        asked = list(
            dict.fromkeys(task for batch in planned[2 * cycle : 2 * cycle + 2] for task in batch)
        )
        ran = [batch for batch, at in zip(runs, cycles, strict=True) if at == cycle]
        assert ran and ran == drawn(asked, 3, random.Random(3 + cycle), len(ran))


def test_fst_averages_per_prompt_whatever_its_slow_loss(tmp_path: Path) -> None:
    """FST aggregates at the prompt level (Eq. 4); a dr-grpo step on its own sums."""
    text = _text(**{'kind = "fst"': 'kind = "fst"\nslow = "dr-grpo"'})
    home = write_blueprint(tmp_path, text)
    recipe = load(home).recipe
    trainer = trainer_for(recipe)(Run.open(home, root=tmp_path / "runs"))
    assert trainer.weights([1, 3]) == [2.0, 2 / 3] and trainer.loss_fn == "ppo"
    assert (slow_recipe(recipe).length_cap, trainer.ADAM) == (0.5, ADAM), (
        "dr-grpo's cap, FST's optimizer"
    )
    assert "summed over tokens" not in resolution(recipe) and "averaged per prompt" in resolution(
        recipe
    )


def test_fsts_warmup_key_reaches_its_slow_step(tmp_path: Path) -> None:
    text = _text(**{'kind = "fst"': 'kind = "fst"\nwarmup = 10'})
    recipe = load(write_blueprint(tmp_path, text)).recipe
    assert slow_recipe(recipe).warmup == 10 and trainer_for(recipe).ADAM.beta2 == 0.999
    assert "learning rate warmed up over 10 steps" in resolution(recipe)


class AlternatingFakes(Fakes):
    """A reflection as `Fakes` does it; a policy trial scores its job's parity."""

    async def __call__(self, config: TrialConfig) -> None:
        if Path(str(config.task.path)).name == "reflection":
            await super().__call__(config)
            return
        self.configs.append(config)
        number = int(Path(config.trials_dir).name.rsplit("-", 1)[1])
        write_result(Path(config.trials_dir) / config.trial_name, reward=float(number % 2 == 0))


@pytest.mark.usefixtures("fakes")
async def test_a_cycle_interrupted_after_a_slow_step_resumes_without_another_fast_phase(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    opened.engine.run_trial = AlternatingFakes()
    started = Runner.from_run(opened)
    trainer = started.trainer
    checkpointed, job = trainer.checkpoint, trainer.job
    stopping = False

    async def stop_after_the_checkpoint(agent: Any, tag: str, **more: Any) -> Any:
        nonlocal stopping
        saved = await checkpointed(agent, tag, **more)
        stopping = True
        return saved

    async def stopped_job(agent: Any, tasks: Any, *rest: Any, **more: Any) -> Any:
        if stopping:
            raise Stopped("after step-1")
        return await job(agent, tasks, *rest, **more)

    trainer.checkpoint = stop_after_the_checkpoint  # type: ignore[method-assign]
    trainer.job = stopped_job  # type: ignore[method-assign]
    with pytest.raises(Stopped), opened:
        await started.run()
    (saved,) = list(record.read(opened.directory / record.CHECKPOINTS))
    assert (saved["tag"], saved["cycle"]) == ("step-1", 0)
    before = list(record.read(opened.directory / record.JOBS))

    resumed = Runner.from_checkpoint(opened.directory)
    assert resumed.consumed == 1 and resumed.trainer.offset == 1
    assert resumed.trainer.cycles == 0
    population = [one.digest for one in resumed.trainer.population]
    assert len(population) == 2 and resumed.modules is not None
    assert resumed.modules.digest == population[0]
    resumed.trainer.run.engine.run_trial = AlternatingFakes()
    with resumed.trainer.run:
        await resumed.run()
    after = list(record.read(resumed.trainer.run.directory / record.JOBS))[len(before) :]
    assert [job["purpose"] for job in after] == ["training", "training"], (
        "the cycle's last step only"
    )
    assert [job["modules"] for job in after] == population
    rows = [
        row for row in record.read(resumed.trainer.run.directory / record.METRICS) if "step" in row
    ]
    assert [(row["step"], row["cycle"]) for row in rows] == [(0, 0), (1, 0)]
    saved = list(record.read(resumed.trainer.run.directory / record.CHECKPOINTS))
    assert [(c["tag"], c["cycle"]) for c in saved] == [("step-1", 0), ("step-2", 0), ("final", 1)]
    assert seed(resumed.trainer.run.directory / record.MODULES / "best").digest == population[0]
