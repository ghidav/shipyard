"""The gepa recipe end to end on fakes: the seed kept, every job on the record, a row
per round and a final row, the winner under `modules/best/` as a seed; and what the
blueprint may say, as `load` and `check` read it."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig
from typer.testing import CliRunner

from shipyard import record
from shipyard.cli.main import app
from shipyard.cli.preflight import NO_REFLECTOR, Finding
from shipyard.config import (
    REFLECTION_IMAGE,
    ConfigError,
    load,
)
from shipyard.data import Generator, held_out, search_sets, tasks
from shipyard.modules import SKILL_FILE, seed
from shipyard.recipes import trainer_for
from shipyard.recipes.gepa import BEST, SEED, GEPATrainer
from shipyard.run import Run
from tests.trials import (
    BLUEPRINTS,
    GEPA,
    VALID,
    Fakes,
    Harbor,
    check,
    dataset,
    drawn,
    fixture_tasks,
    write_blueprint,
    write_result,
)


def _blueprint(tmp_path: Path, text: str = GEPA) -> Path:
    fixture_tasks(tmp_path)
    home = write_blueprint(tmp_path, text)
    shutil.copytree(VALID, home / "modules")
    return home


def _names(configs: list[TrialConfig]) -> list[str]:
    return [Path(str(config.task.path)).name for config in configs]


async def test_the_recipe_runs_end_to_end_on_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    """`pareto = 0`: both tasks are reflected on and selected on (GEPA section 6)."""
    monkeypatch.chdir(tmp_path)
    home = _blueprint(tmp_path, GEPA + "pareto = 0\n")
    opened = Run.open(home, root=tmp_path / "runs")
    assert opened.config.recipe.budget is None
    fakes = harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    seeded = seed(home / "modules")
    kept = opened.directory / record.MODULES
    assert seed(kept / SEED) == seeded, "the blueprint's modules, as the search started"
    # The round's minibatch, one task drawn with [data] seed: the parent's job runs it first.
    names = _names(fakes.configs)
    drawn = names[2]
    assert names == ["alpha", "beta", drawn, "reflection", drawn, "alpha", "beta"]
    winner = seed(kept / BEST)
    assert f"mention {drawn}" in winner.components["solving"].text
    assert list(winner.components["solving"].files) == [SKILL_FILE], "the reflector wrote one"
    assert winner.digest != seeded.digest
    # Five jobs: the seed over both tasks, the parent on its minibatch, the reflection, the
    # child on the same minibatch, the child over both tasks; a rollout's batch is the round
    # it served, the seed's being round 0.
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["purpose"] for job in jobs] == [
        "search",
        "search",
        "reflection",
        "search",
        "search",
    ]
    assert [job.get("round") for job in jobs] == [0, 1, None, 1, 1]
    assert [job["tasks"] for job in jobs] == [2, 1, 1, 1, 2]
    assert [job["modules"] for job in jobs if "modules" in job] == [
        seeded.digest,
        seeded.digest,
        winner.digest,
        winner.digest,
    ]
    assert jobs[2]["component"] == "solving" and jobs[2]["candidate"] == seeded.digest
    assert jobs[2]["party"] == "anthropic" and jobs[2]["input_tokens"] == 1000
    assert [job["party"] for job in jobs if job["purpose"] == "search"] == ["openrouter"] * 4
    assert fakes.reflected == [[f"00-tasks-fixture-{drawn}.txt"]]
    reflection = fakes.configs[3]
    assert reflection.agent.name == "claude-code" and reflection.verifier.disable is True
    assert reflection.agent.model_name == "anthropic/claude-sonnet-5"
    assert reflection.environment.type == "modal" and reflection.agent.skills == []
    # One row per round, then the final row.
    *rounds, final = list(record.read(opened.directory / record.METRICS))
    (row,) = rounds
    assert row["seq"] == 1 and row["round"] == 1 and row["component"] == "solving"
    assert row["skipped"] is None
    assert (row["parent"], row["child"]) == (seeded.digest, winner.digest)
    assert (row["parent_mean"], row["child_mean"], row["accepted"]) == (0.0, 1.0, True)
    # The winner ties the seed on one task and beats it on the other: the seed is dominated.
    assert (row["pool"], row["frontier"], row["spent"]) == (2, 1, 6)
    assert final["seq"] == 2 and final["evolution"] is True
    assert (final["rounds"], final["spent"], final["best"]) == (1, 6, winner.digest)
    assert final["best_mean"] == 0.5 and final["frontier"] == 1 and final["pool"] == 2
    assert final["moved"] is True
    costs = record.read_json(opened.directory / record.COSTS)
    assert costs["parties"]["anthropic"]["trials"] == 1
    assert costs["parties"]["openrouter"]["trials"] == 6
    note = Run.read(opened.directory)
    assert note["kind"] == "gepa" and note["failed"] is False and note["finished_at"]


async def test_by_default_two_thirds_are_held_out_and_the_winner_is_scored_on_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    """GEPA 4.1: minibatches from the training split, the rest scores candidates. The child
    wins its minibatch task, and its Pareto score is the held-out task's alone."""
    monkeypatch.chdir(tmp_path)
    home = _blueprint(tmp_path)
    listed = [Path("tasks/fixture") / n for n in ("alpha", "beta")]
    feedback, pareto = search_sets(listed, None, seed=0)
    assert (feedback, pareto) == held_out(listed, 1, seed=0)
    opened = Run.open(home, root=tmp_path / "runs")
    fakes = harbor(Fakes())
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    taught, held = feedback[0].name, pareto[0].name
    assert _names(fakes.configs) == [held, taught, "reflection", taught, held]
    assert fakes.reflected == [[f"00-tasks-fixture-{taught}.txt"]]
    *rounds, final = list(record.read(opened.directory / record.METRICS))
    (row,) = rounds
    assert (row["parent_mean"], row["child_mean"], row["accepted"]) == (0.0, 1.0, True)
    assert (final["rounds"], final["spent"], final["best_mean"]) == (1, 4, 0.0)


async def test_data_seed_draws_the_minibatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    """Six tasks, two a minibatch: each round the parent runs on the minibatch `[data] seed`
    predicts, and another seed draws others."""
    monkeypatch.chdir(tmp_path)
    fixture_tasks(tmp_path)
    dataset(tmp_path / "tasks", "six", "t1", "t2", "t3", "t4", "t5", "t6")
    found = {}
    for value in (0, 7):
        text = GEPA.replace('"fixture"', '"six"').replace("minibatch = 1", "minibatch = 2")
        text = text.replace("batch_size = 1", f"batch_size = 1\nseed = {value}")
        home = write_blueprint(tmp_path / f"seed{value}", text + "pareto = 0\nbudget = 30\n")
        shutil.copytree(VALID, home / "modules")
        opened = Run.open(home, root=tmp_path / "runs")
        fakes = harbor(Fakes())
        with opened:
            await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
        names = _names(fakes.configs)
        runs = [names[at - 2 : at] for at, name in enumerate(names) if name == "reflection"]
        feedback, _ = search_sets(tasks("six"), 0, seed=value)
        assert runs == drawn(feedback, 2, random.Random(value), 3), (
            "three rounds, ten rollouts each"
        )
        found[value] = runs
    assert found[0] != found[7]


async def test_a_quiet_reflector_leaves_the_seed_as_the_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")

    async def quiet(config: TrialConfig) -> None:
        trial = Path(config.trials_dir) / config.trial_name
        if Path(str(config.task.path)).name == "reflection":
            trial.mkdir(parents=True)
            return
        write_result(trial, reward=0.5)

    harbor(quiet)
    with opened:
        agent = await GEPATrainer(opened.config, opened.directory).train()
    seeded = seed(opened.directory / record.MODULES / SEED)
    assert seed(opened.directory / record.MODULES / BEST) == seeded
    assert agent.modules == seeded, "handed back, for a caller"
    *rounds, final = list(record.read(opened.directory / record.METRICS))
    assert len(rounds) == 2, "patience = 2"
    assert all(row["child"] is None and row["accepted"] is False for row in rounds)
    assert (final["rounds"], final["spent"], final["best"]) == (2, 3, seeded.digest)
    assert final["best_mean"] == 0.5 and final["moved"] is False
    jobs = list(record.read(opened.directory / record.JOBS))
    assert [job["purpose"] for job in jobs] == [
        "search",
        "search",
        "reflection",
        "search",
        "reflection",
    ], "the seed on the held-out task, then each round the parent before the reflector"


async def test_a_seed_perfect_on_its_minibatches_is_never_reflected_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
) -> None:
    """Every rollout scores 1.0: each round runs the parent and asks for no rewrite, and
    with the best Pareto mean never rising, `patience = 2` ends it after two rounds."""
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_blueprint(tmp_path), root=tmp_path / "runs")
    ran: list[str] = []

    async def solved(config: TrialConfig) -> None:
        ran.append(Path(str(config.task.path)).name)
        write_result(Path(config.trials_dir) / config.trial_name, reward=1.0)

    harbor(solved)
    with opened:
        await trainer_for(opened.config.recipe)(opened.config, opened.directory).train()
    assert "reflection" not in ran and len(ran) == 3, "the seed's scoring, then two parents"
    *rounds, final = list(record.read(opened.directory / record.METRICS))
    assert [(row["skipped"], row["component"], row["child"]) for row in rounds] == [
        ("perfect", None, None)
    ] * 2
    assert (final["rounds"], final["spent"], final["moved"]) == (2, 3, False)


# -------------------------------------------------------------------- the blueprint


def test_the_fixture_loads_with_the_searchs_knobs() -> None:
    loaded = load(BLUEPRINTS / "gepa")
    found = loaded.recipe
    assert found.kind == "gepa" and found.reflection_harness == "pi@0.85.1"
    assert found.reflection_model == "Qwen/Qwen3-8B"
    assert found.reflection_image == REFLECTION_IMAGE == "python:3.12-slim"
    assert loaded.agent.modules == "modules" and found.minibatch == 3 and found.pareto is None
    assert found.budget is None and found.patience is None and found.edits == "rewrite"
    for gone in ("tolerance", "max_metric_calls", "components", "rng_seed", "population"):
        assert not hasattr(found, gone)


def test_what_the_search_no_longer_takes_is_an_unknown_key(tmp_path: Path) -> None:
    text = GEPA + 'tolerance = 0.05\nmax_metric_calls = 10\nbudget = 0\nedits = "all"\n'
    with pytest.raises(ConfigError) as caught:
        load(write_blueprint(tmp_path, text))
    assert [p.split(":")[0] for p in caught.value.problems] == [
        "[recipe] budget",
        "[recipe] edits",
        "[recipe] tolerance",
        "[recipe] max_metric_calls",
    ]


def test_check_names_the_reflector_and_blocks_a_blank_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    home = _blueprint(tmp_path)
    found = check(home)
    assert all(f.level == "ok" for f in found)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert [f.text for f in check(home) if f.level == "warning"] == [
        "reflector claude-code: none of ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN is set, the "
        "keys Harbor hands it; its first call fails unless it signs in another way"
    ]
    texts = [f.text for f in found]
    assert texts[5].startswith("modules: 1 component(s) under")
    assert texts[6] == (
        "reflector claude-code, model anthropic/claude-sonnet-5, image python:3.12-slim"
    )
    assert texts[7] == "pareto: 1 of 2 tasks held out with seed 0 to select on, 1 to reflect on"
    blank = write_blueprint(
        tmp_path / "blank", GEPA.replace('"claude-code"', '" "').replace("anthropic/", "")
    )
    shutil.copytree(VALID, blank / "modules")
    assert Finding("blocked", NO_REFLECTOR) in check(blank)
    unnamed = GEPA.replace('reflection_model = "anthropic/claude-sonnet-5"\n', "")
    found = check(write_blueprint(tmp_path / "unnamed", unnamed))
    assert [f.text for f in found][6].startswith(
        "reflector claude-code, model the harness's own default, image"
    ), "after the blocked modules line: no modules/ beside this one"
    assert [f.text for f in found if f.level == "blocked"][0].startswith("[agent] modules: no")


def test_check_prints_the_resolved_config_and_the_modules_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(app, ["check", str(_blueprint(tmp_path)), "--verbose"])
    assert result.exit_code == 0, result.output
    assert (
        'kind = "gepa"' in result.output
        and 'reflection_image = "python:3.12-slim"' in result.output
    )
    assert "patience = 2" in result.output and "budget" not in result.output
    assert "ok  modules: 1 component(s) under" in result.output
    assert "ok  reflector claude-code" in result.output


def test_pareto_takes_a_dataset_name_or_a_count(tmp_path: Path) -> None:
    for value, loaded in (("3", 3), ("0", 0), ('"held"', "held")):
        text = GEPA + f"pareto = {value}\n"
        assert load(write_blueprint(tmp_path / value, text)).recipe.pareto == loaded
    for value in ("-1", "true", '" "', "2.5"):
        with pytest.raises(ConfigError) as refused:
            load(write_blueprint(tmp_path / f"bad{value}", GEPA + f"pareto = {value}\n"))
        assert refused.value.problems == [
            "[recipe] pareto: a dataset name, or how many of the run's tasks to hold out "
            "(0 or more)"
        ]


def test_the_sets_follow_pareto_and_check_says_how_they_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture_tasks(tmp_path)
    dataset(tmp_path / "tasks", "six", *"abcdef")

    def sets(text: str) -> tuple[list[str], list[str]]:
        cfg = load(write_blueprint(tmp_path / str(len(made)), text))
        found = search_sets(Generator.of(cfg).train, cfg.recipe.pareto, seed=cfg.data.seed)
        made.append(text)
        return [task.name for task in found[0]], [task.name for task in found[1]]

    made: list[str] = []
    six = GEPA.replace('"fixture"', '"six"')
    feedback, pareto = sets(six)
    assert (len(feedback), len(pareto)) == (2, 4), (
        "two thirds, as in three of GEPA's four benchmarks"
    )
    assert sets(six) == (feedback, pareto), "the same draw for the same [data] seed"
    assert sets(six.replace("batch_size = 1", "batch_size = 1\nseed = 5")) != (feedback, pareto)
    assert sets(six + "pareto = 1\n")[1] == [held_out(list("abcdef"), 1, seed=0)[1][0]]
    assert sets(six + "pareto = 0\n") == (list("abcdef"), list("abcdef"))
    assert sets(six + 'pareto = "fixture"\n') == (list("abcdef"), ["alpha", "beta"])

    def said(text: str) -> list[Finding]:
        home = write_blueprint(tmp_path / f"check{len(made)}", text)
        made.append(text)
        shutil.copytree(VALID, home / "modules")
        about = ("pareto", "[recipe] pareto", "dataset fixture", "no dataset at")
        return [f for f in check(home) if f.text.startswith(about)]

    assert said(six + 'pareto = "fixture"\n') == [
        Finding("ok", "dataset fixture: 2 tasks under tasks/fixture"),
        Finding(
            "ok", "pareto: dataset fixture's 2 tasks to select on, the run's 6 tasks to reflect on"
        ),
    ]
    assert said(six + "pareto = 0\n") == [
        Finding("ok", "pareto: none held out; 6 tasks both reflected and selected on")
    ]
    assert said(six + "pareto = 6\n") == [
        Finding("blocked", "[recipe] pareto: holding out 6 of 6 tasks leaves none to reflect on")
    ]
    [missing] = said(six + 'pareto = "nowhere"\n')
    assert missing.level == "blocked" and missing.text.startswith("no dataset at")
    dataset(tmp_path / "tasks", "one", "alpha")
    assert said(GEPA.replace('"fixture"', '"one"')) == [
        Finding(
            "blocked",
            "[recipe] pareto: one task cannot be split; 0 selects on the task the search "
            "reflects on, and a dataset name selects on that dataset",
        )
    ]
