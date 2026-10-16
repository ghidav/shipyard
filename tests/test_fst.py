"""Tests for the fast-slow training (FST) recipe."""

from __future__ import annotations

import random
import shutil
from itertools import islice
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import TrialConfig

from shipyard import record
from shipyard.cli.run import Run, reopened
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
from shipyard.sampling.endpoint import ProviderEndpoint
from shipyard.training.recipes import resolution, trainer_for
from shipyard.training.recipes.cispo import CISPOTrainer
from shipyard.training.recipes.fst import (
    ADAM,
    BUDGET_PASSES,
    FSTTrainer,
    fst_trainer,
    merged,
    slow_recipe,
)
from shipyard.types import Agent, Group, Member, Policy, Sequence, Trial, Verdict, harness
from tests.trials import (
    BLUEPRINTS,
    VALID,
    Fakes,
    Harbor,
    Stopped,
    check,
    dataset,
    drawn,
    write_blueprint,
    write_result,
)

FST = BLUEPRINTS / "fst"
AGENT = Agent(Policy(ProviderEndpoint("m")), [harness("pi")])


@pytest.fixture(autouse=True)
def _tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Change into `tmp_path` and create a two-task `aime-train` dataset there."""
    monkeypatch.chdir(tmp_path)
    dataset(tmp_path / "tasks", "aime-train", "a", "b")


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
    opened = Run.open(FST, root=tmp_path)
    trainer = trainer_for(recipe)(opened.config, opened.directory)
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
    opened = Run.open(home, root=tmp_path / "runs")
    trainer = trainer_for(load(home).recipe)(opened.config, opened.directory)
    assert trainer.loss_fn == "ppo"
    assert trainer.bounds() == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.3}
    assert trainer.recipe.substeps == 1, "fst's own step, not dapo's"


@pytest.mark.parametrize(
    ("slow", "recipe"), [("dapo", DapoRecipe), ("dr-grpo", DrGrpoRecipe), ("cispo", CispoRecipe)]
)
def test_slow_knobs_are_each_slow_recipes_own_fields_and_fsts(slow: str, recipe: type) -> None:
    """Every knob a slow recipe adds must appear in `SLOW_KNOBS` and in fst's fields, or fst
    cannot set it. The overlong term is excluded because a slow step trains on the batch it
    sampled."""
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
    opened = Run.open(FST, root=tmp_path)
    trainer = fst_trainer(CISPOTrainer)(opened.config, opened.directory)
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
async def test_the_slow_steps_are_the_slow_classs_own(tmp_path: Path, harbor: Harbor) -> None:
    passes: list[int] = []

    class Marked(CISPOTrainer):
        async def advance(self, agent: Agent) -> None:
            passes.append(self.steps)
            await super().advance(agent)

    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Fakes())
    started = fst_trainer(Marked)(opened.config, opened.directory)
    with opened:
        await started.train()
    assert type(started).__name__ == "FSTMarked"
    assert passes == [0, 1] and started.cycles == 1, "the slow class takes each slow step"


@pytest.mark.usefixtures("fakes")
async def test_a_cycle_evolves_two_texts_then_splits_every_group_across_them(
    tmp_path: Path,
    harbor: Harbor,
) -> None:
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
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
    harbor: Harbor,
) -> None:
    """FST Appendix D: 960 metric calls over 192 examples, one rollout each, whatever share
    of the group each text takes in the slow steps (here two of four)."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(tmp_path, _text(**{'"aime-train"': '"four"'}))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
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
async def test_patience_ends_a_fast_phase_as_it_ends_gepa(tmp_path: Path, harbor: Harbor) -> None:
    """With `patience = 1` and a budget that would allow more rounds, the fast phase stops at
    the first round that does not raise the best anchor-set mean. Each text here names one
    task, so only the first child raises it."""
    dataset(tmp_path / "tasks", "four", *"abcd")
    changes = {'"aime-train"': '"four"', 'kind = "fst"': 'kind = "fst"\npatience = 1\nbudget = 100'}
    home = write_blueprint(tmp_path, _text(**changes))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    rows = list(record.read(opened.directory / record.METRICS))
    rounds = [row for row in rows if "round" in row]
    assert len(rounds) == 2 and rounds[0]["accepted"] is True
    (evolution,) = [row for row in rows if row.get("evolution")]
    assert evolution["rounds"] == 2 and evolution["spent"] < 100


@pytest.mark.usefixtures("fakes")
async def test_cycle_c_draws_its_minibatches_with_data_seed_plus_c(
    tmp_path: Path, harbor: Harbor
) -> None:
    dataset(tmp_path / "tasks", "four", *"abcd")
    changes = {
        '"aime-train"': '"four"',
        "group_size = 4": "group_size = 2\nseed = 3",
        "steps = 2": "steps = 4",
    }
    home = write_blueprint(tmp_path, _text(**changes))
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    trials = harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    names = [Path(str(config.task.path)).name for config in trials.configs]
    runs = [names[at - 3 : at] for at, name in enumerate(names) if name == "reflection"]
    cycles = [
        row["cycle"] for row in record.read(opened.directory / record.METRICS) if "round" in row
    ]
    planned = list(islice(Generator(tasks("four"), 2, 3).batches(), 4))
    for cycle in (0, 1):
        asked = list(
            dict.fromkeys(task for batch in planned[2 * cycle : 2 * cycle + 2] for task in batch)
        )
        ran = [batch for batch, at in zip(runs, cycles, strict=True) if at == cycle]
        assert ran and ran == drawn(asked, 3, random.Random(3 + cycle), len(ran))


def test_fst_averages_per_prompt_whatever_its_slow_loss(tmp_path: Path) -> None:
    """FST aggregates at the prompt level (Eq. 4), whereas a standalone dr-grpo step sums."""
    text = _text(**{'kind = "fst"': 'kind = "fst"\nslow = "dr-grpo"'})
    home = write_blueprint(tmp_path, text)
    recipe = load(home).recipe
    opened = Run.open(home, root=tmp_path / "runs")
    trainer = trainer_for(recipe)(opened.config, opened.directory)
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


def test_fsts_teacher_reaches_its_slow_step(tmp_path: Path) -> None:
    taught = 'kind = "fst"\ndistill = 0.5\nteacher = "Qwen/Qwen3-32B"'
    recipe = load(write_blueprint(tmp_path, _text(**{'kind = "fst"': taught}))).recipe
    slow = slow_recipe(recipe)
    assert (slow.distill, slow.teacher, slow.kl_coef) == (0.5, "Qwen/Qwen3-32B", 0.001)
    assert "distill 0.5 toward Qwen/Qwen3-32B" in resolution(recipe)


class AlternatingFakes(Fakes):
    """Fake trial runner that handles reflections like `Fakes` and scores a policy trial by
    the parity of its job number."""

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
    harbor: Harbor,
) -> None:
    dataset(tmp_path / "tasks", "four", *"abcd")
    home = write_blueprint(
        tmp_path, _text(**{'"aime-train"': '"four"', "group_size = 4": "group_size = 2"})
    )
    shutil.copytree(VALID, home / "modules")
    opened = Run.open(home, root=tmp_path / "runs")
    harbor(AlternatingFakes())
    trainer = trainer_for(opened.config.recipe)(opened.config, opened.directory)
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
        await trainer.train()
    (saved,) = list(record.read(opened.directory / record.CHECKPOINTS))
    assert (saved["tag"], saved["cycle"], saved["drawn"], saved["pending"]) == ("step-1", 0, 2, 1)
    before = list(record.read(opened.directory / record.JOBS))

    again, resumed = reopened(opened.directory)
    assert isinstance(resumed, FSTTrainer)
    assert (resumed.cycles, resumed.drawn, len(resumed.pending)) == (0, 2, 1), (
        "the cycle's second batch, drawn again"
    )
    population = [one.digest for one in resumed.population]
    assert len(population) == 2 and resumed.modules is not None
    assert resumed.modules.digest == population[0]
    harbor(AlternatingFakes())
    with again:
        await resumed.train()
    after = list(record.read(again.directory / record.JOBS))[len(before) :]
    assert [job["purpose"] for job in after] == ["training", "training"], (
        "the cycle's last step only"
    )
    assert [job["modules"] for job in after] == population
    rows = [row for row in record.read(again.directory / record.METRICS) if "step" in row]
    assert [(row["step"], row["cycle"]) for row in rows] == [(0, 0), (1, 0)]
    saved = list(record.read(again.directory / record.CHECKPOINTS))
    assert [(c["tag"], c["cycle"]) for c in saved] == [("step-1", 0), ("step-2", 0), ("final", 1)]
    assert seed(again.directory / record.MODULES / "best").digest == population[0]
