"""Tests for the blueprint schema and the config that `check` resolves."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

from shipyard import config
from shipyard.cli import resolved
from shipyard.cli.preflight import Finding
from shipyard.config import Blueprint, ConfigError, load
from tests.trials import BLUEPRINTS, check, fixture_tasks, write_blueprint

#: Recipe kinds implemented as a `Gradient` recipe.
GRADIENT = ("dapo", "dr-grpo", "cispo", "opd", "fst")

MINIMAL = """
[agent]
model = "Qwen/Qwen3-8B"
harness = "pi@0.85.1"

[data]
train = "aime-train"
batch_size = 2
[rollout]
[recipe]
kind = "dapo"
learning_rate = 2e-5
steps = 1
"""


def _problems(tmp_path: Path, text: str) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        load(write_blueprint(tmp_path, text))
    return caught.value.problems


@pytest.mark.parametrize("kind", config.KINDS)
def test_each_fixture_loads(kind: str) -> None:
    found = load(BLUEPRINTS / kind)
    assert isinstance(found, Blueprint)
    assert found.recipe.kind == kind
    assert found.raw["recipe"]["kind"] == kind
    assert found.checkpoints.every == 1 and found.checkpoints.ttl_hours == 168.0
    assert isinstance(found.recipe, config.Gradient) == (kind in GRADIENT)


EXAMPLES = sorted((Path(__file__).parent.parent / "examples" / "blueprints").iterdir())


@pytest.mark.parametrize("home", EXAMPLES, ids=[one.name for one in EXAMPLES])
def test_each_example_loads_and_spells_its_sets_the_current_way(home: Path) -> None:
    loaded = load(home)
    assert "dataset" not in loaded.raw["data"]
    assert loaded.datasets or loaded.test_sets


def test_load_accepts_the_file_as_well_as_the_directory() -> None:
    assert load(BLUEPRINTS / "dapo" / "run.toml").recipe.kind == "dapo"


def test_dapo_fixture_defaults() -> None:
    found = load(BLUEPRINTS / "dapo")
    assert found.agent.served and found.agent.provider == "tinker"
    assert found.recipe.kind == "dapo" and found.recipe.learning_rate == 2e-5
    assert found.recipe.reference == "trainer" and found.recipe.substeps == 16
    assert found.recipe.clip_low == 0.2 and found.recipe.clip_high == 0.28
    assert found.recipe.kl_coef == 0.0 and found.agent.modules is None
    assert found.rollout.sandbox == "docker" and found.rollout.max_tokens == 8192
    assert found.agent.env == {} and found.recipe.lora_rank == 32


def test_dr_grpo_and_cispo_fixtures_carry_their_own_knobs() -> None:
    found = load(BLUEPRINTS / "dr-grpo").recipe
    assert found.kind == "dr-grpo" and found.clip == 0.2
    assert found.length_penalty == 0.2 and found.length_floor == 0
    assert found.substeps == 1, "Dr. GRPO: one substep"
    found = load(BLUEPRINTS / "cispo").recipe
    assert found.kind == "cispo" and found.reference == "sampler"
    assert found.clip_high == 3.0, "ScaleRL A.17.2 and FST appendix D: a ceiling of 4.0"
    assert found.substeps == 16, "MiniMax-M1 3.1"


def test_substeps_is_a_knob_and_refill_is_no_longer_one(tmp_path: Path) -> None:
    text = MINIMAL.replace("learning_rate = 2e-5", "learning_rate = 2e-5\nsubsteps = 2")
    assert load(write_blueprint(tmp_path / "set", text)).recipe.substeps == 2
    gone = MINIMAL.replace("learning_rate = 2e-5", "learning_rate = 2e-5\nrefill = 0")
    assert _problems(tmp_path / "gone", gone) == ["[recipe] refill: unknown key"]


def test_dapo_and_cispo_dock_overlong_rollouts_as_dapo_does_and_take_its_knobs(
    tmp_path: Path,
) -> None:
    for kind in ("dapo", "cispo"):
        found = load(BLUEPRINTS / kind).recipe
        assert (found.overlong_penalty, found.overlong_buffer) == (0.5, 0.2), "DAPO Eq. 13, 4.1"
    knobs = "learning_rate = 2e-5\noverlong_penalty = 0.0\noverlong_buffer = 1.0"
    found = load(write_blueprint(tmp_path / "set", MINIMAL.replace("learning_rate = 2e-5", knobs)))
    assert (found.recipe.overlong_penalty, found.recipe.overlong_buffer) == (0.0, 1.0)
    for at, bad in enumerate(("overlong_buffer = 0.0", "overlong_buffer = 1.5")):
        text = MINIMAL + f"{bad}\n"
        assert [p.split(":")[0] for p in _problems(tmp_path / str(at), text)] == [
            "[recipe] overlong_buffer"
        ], "a share of the budget, above 0 and at most all of it"
    text = MINIMAL + "overlong_penalty = -0.1\n"
    assert [p.split(":")[0] for p in _problems(tmp_path / "neg", text)] == [
        "[recipe] overlong_penalty"
    ]
    fst = (BLUEPRINTS / "fst" / "run.toml").read_text(encoding="utf-8")
    fst = fst.replace('kind = "fst"', 'kind = "fst"\noverlong_penalty = 0.5')
    assert _problems(tmp_path / "fst", fst) == ["[recipe] overlong_penalty: unknown key"]


@pytest.mark.parametrize("kind", ["dapo", "cispo"])
@pytest.mark.parametrize("value", ["inf", "nan"])
def test_the_overlong_penalty_is_a_finite_number(tmp_path: Path, kind: str, value: str) -> None:
    text = MINIMAL.replace('kind = "dapo"', f'kind = "{kind}"\noverlong_penalty = {value}')
    assert _problems(tmp_path, text) == [
        "[recipe] overlong_penalty: Input should be a finite number"
    ]


def _cap_warnings(home: Path) -> list[str]:
    return [
        found.text
        for found in check(home)
        if found.level == "warning" and found.text.startswith("[recipe] length_cap")
    ]


def test_the_length_cap_is_a_dr_grpo_knob_that_takes_infinity_and_warns_from_one(
    tmp_path: Path,
) -> None:
    """The warning applies only while the length penalty is on; with `length_penalty = 0`
    the cap has no effect."""
    assert load(BLUEPRINTS / "dr-grpo").recipe.length_cap == 0.5
    assert _cap_warnings(BLUEPRINTS / "dr-grpo") == []
    dr = (BLUEPRINTS / "dr-grpo" / "run.toml").read_text(encoding="utf-8")
    home = write_blueprint(tmp_path / "inf", dr + "length_cap = inf\n")
    assert math.isinf(load(home).recipe.length_cap)
    assert _cap_warnings(home) == [
        "[recipe] length_cap: inf lets the length rule take a solved answer's whole reward, "
        "so a long solved answer can score as low as a failure"
    ]
    assert _cap_warnings(write_blueprint(tmp_path / "high", dr + "length_cap = 0.9\n")) == []
    off = dr.replace("length_penalty = 0.2", "length_penalty = 0.0") + "length_cap = inf\n"
    assert _cap_warnings(write_blueprint(tmp_path / "off", off)) == [], "the rule is off"
    unset = dr.replace("length_penalty = 0.2\n", "") + "length_cap = 1.0\n"
    assert _cap_warnings(write_blueprint(tmp_path / "unset", unset)) == [], "off by default"
    assert [p.split(":")[0] for p in _problems(tmp_path / "zero", dr + "length_cap = 0.0\n")] == [
        "[recipe] length_cap"
    ]
    fst = (BLUEPRINTS / "fst" / "run.toml").read_text(encoding="utf-8")
    slow = fst.replace('kind = "fst"', 'kind = "fst"\nslow = "dr-grpo"\nlength_cap = 1.0')
    home = write_blueprint(tmp_path / "fst-off", slow)
    assert load(home).recipe.length_cap == 1.0 and _cap_warnings(home) == [], "no penalty set"
    home = write_blueprint(tmp_path / "fst", slow + "length_penalty = 0.1\n")
    assert load(home).recipe.length_cap == 1.0 and len(_cap_warnings(home)) == 1
    cispo = fst.replace('kind = "fst"', 'kind = "fst"\nlength_cap = 1.0')
    assert _problems(tmp_path / "cispo", cispo) == [
        "[recipe] length_cap: not a knob of slow = 'cispo', which takes clip_high"
    ]


def test_gepa_fixture_defaults() -> None:
    found = load(BLUEPRINTS / "gepa")
    assert found.recipe.kind == "gepa" and not isinstance(found.recipe, config.Gradient)
    assert found.agent.modules == "modules" and found.recipe.minibatch == 3
    assert found.recipe.budget is None and found.recipe.patience is None
    assert found.recipe.reflection_image == "python:3.12-slim"
    assert found.recipe.edits == "rewrite"


def test_evaluate_fixture_is_provider_served() -> None:
    found = load(BLUEPRINTS / "evaluate")
    assert found.recipe.kind == "evaluate"
    assert not found.agent.served and found.agent.provider == "openrouter"


def test_missing_required_keys_are_named(tmp_path: Path) -> None:
    problems = _problems(
        tmp_path,
        """
        [agent]
        [data]
        dataset = "aime-train"
        [recipe]
        kind = "dapo"
        """,
    )
    assert "[agent] model: field required" in problems
    assert "[data] batch_size: field required" in problems
    assert "[agent] harness: field required" in problems
    assert "[recipe] learning_rate: field required" in problems


def test_a_key_in_the_table_it_left_is_refused_naming_where_it_went(tmp_path: Path) -> None:
    text = """
[model]
name = "Qwen/Qwen3-8B"
lora_rank = 32
[data]
train = "aime-train"
batch_size = 2
[rollout]
harness = "pi@0.85.1"
[recipe]
kind = "dapo"
learning_rate = 2e-5
steps = 1
modules = "skills"
[wandb]
project = "p"
"""
    assert _problems(tmp_path, text) == [
        "[model] name: now [agent] model",
        "[model] lora_rank: now [recipe] lora_rank",
        "[rollout] harness: now [agent] harness",
        "[recipe] modules: now [agent] modules",
        "[wandb] project: now [logging.wandb] project",
    ]


def test_missing_table_is_named(tmp_path: Path) -> None:
    problems = _problems(tmp_path, MINIMAL.replace("[data]", "[other]"))
    assert "[data]: table required" in problems
    assert "[other]: unknown table" in problems


def test_rollout_is_optional_every_key_having_a_default(tmp_path: Path) -> None:
    found = load(write_blueprint(tmp_path, MINIMAL.replace("[rollout]\n", "")))
    assert found.rollout.sandbox == "docker" and found.rollout.concurrency == 4


def test_unknown_key_and_table_are_named(tmp_path: Path) -> None:
    text = MINIMAL.replace('harness = "pi@0.85.1"', 'harness = "pi@0.85.1"\nfoo = 1')
    problems = _problems(tmp_path, text + "\n[extra]\nbar = 2\n")
    assert problems == ["[agent] foo: unknown key", "[extra]: unknown table"]


def test_the_advantage_and_stabilize_tables_are_gone(tmp_path: Path) -> None:
    problems = _problems(tmp_path, MINIMAL + "\n[recipe.advantage]\nnormalize = true\n")
    assert problems == ["[recipe] advantage: unknown key"]
    problems = _problems(tmp_path / "s", MINIMAL + '\n[recipe.stabilize]\nloss_fn = "ppo"\n')
    assert problems == ["[recipe] stabilize: unknown key"]


@pytest.mark.parametrize(
    ("kind", "knob"),
    [
        ("dapo", "clip = 0.1"),
        ("dapo", "length_penalty = 0.1"),
        ("dr-grpo", "clip_high = 0.3"),
        ("cispo", "clip_low = 0.1"),
        ("cispo", "length_floor = 2"),
        ("dapo", "length_cap = 0.9"),
        ("dr-grpo", "overlong_penalty = 0.5"),
    ],
)
def test_a_knob_of_another_recipe_is_an_unknown_key(tmp_path: Path, kind: str, knob: str) -> None:
    text = MINIMAL.replace('kind = "dapo"', f'kind = "{kind}"\n{knob}')
    assert _problems(tmp_path, text) == [f"[recipe] {knob.split(' =')[0]}: unknown key"]


def test_each_gradient_recipe_takes_the_shared_knobs(tmp_path: Path) -> None:
    for at, kind in enumerate(GRADIENT):
        text = MINIMAL.replace(
            'kind = "dapo"',
            f'kind = "{kind}"\nsubsteps = 2\nreference = "sampler"\nkl_coef = 0.1\n'
            "lora_rank = 16\nrestore_optimizer = true\nteacher_coef = 0.5\n"
            'teacher = "Qwen/Qwen3-32B"'
            + ('\nreflection_harness = "claude-code"' if kind == "fst" else ""),
        )
        found = load(write_blueprint(tmp_path / str(at), text)).recipe
        assert found.kind == kind and found.substeps == 2 and found.reference == "sampler"
        assert found.kl_coef == 0.1 and found.lora_rank == 16 and found.restore_optimizer
        assert found.teacher_coef == 0.5 and found.teacher == "Qwen/Qwen3-32B"


def test_a_teacher_and_its_weight_come_together(tmp_path: Path) -> None:
    def problems(name: str, knobs: str, kind: str = "dapo") -> list[str]:
        text = MINIMAL.replace('kind = "dapo"', f'kind = "{kind}"\n{knobs}')
        return _problems(tmp_path / name, text)

    assert problems("weight", "teacher_coef = 0.5") == [
        "[recipe] teacher: teacher_coef = 0.5 weighs a teacher's logprobs; name the teacher"
    ]
    assert problems("teacher", 'teacher = "Qwen/Qwen3-32B"') == [
        "[recipe] teacher: set teacher_coef above 0 to weigh the teacher, or leave it unset"
    ]
    assert problems("state", 'teacher_coef = 1.0\nteacher = "tinker://r/weights/final"') == [
        "[recipe] teacher: a teacher is sampled, so a checkpoint is named by its sampler_path "
        "(tinker://.../sampler_weights/...)"
    ]
    assert problems("opd", "", kind="opd") == ["[recipe] teacher: field required"]
    found = load(BLUEPRINTS / "opd").recipe
    assert (found.teacher_coef, found.teacher) == (1.0, "Qwen/Qwen3-32B")
    path = 'teacher = "tinker://r/sampler_weights/final"'
    assert (
        load(
            write_blueprint(
                tmp_path / "path", MINIMAL.replace('kind = "dapo"', f'kind = "opd"\n{path}')
            )
        ).recipe.teacher
        == "tinker://r/sampler_weights/final"
    )


def test_train_takes_a_name_or_a_list_and_dataset_is_its_older_spelling(tmp_path: Path) -> None:
    old = MINIMAL.replace('train = "aime-train"', 'dataset = "aime-train"')
    found = load(write_blueprint(tmp_path, old))
    assert found.data.train == "aime-train" and found.datasets == ["aime-train"]
    assert found.data.test is None and found.test_sets == []
    listed = MINIMAL.replace('train = "aime-train"', 'train = ["a", "b"]\ntest = "c"')
    found = load(write_blueprint(tmp_path / "two", listed))
    assert found.data.train == ["a", "b"] and found.datasets == ["a", "b"]
    assert found.data.test == "c" and found.test_sets == ["c"]
    assert _problems(tmp_path / "three", MINIMAL.replace('"aime-train"', "3")) == [
        "[data] train: a dataset name or a list of them"
    ]
    both = MINIMAL.replace('train = "aime-train"', 'dataset = "a"\ntrain = "b"')
    assert _problems(tmp_path / "four", both) == [
        "[data]: `dataset` is the older spelling of `train`; name one of them"
    ]
    none = MINIMAL.replace('train = "aime-train"\n', "")
    assert _problems(tmp_path / "five", none) == ["[data] train: field required"]


def test_the_evaluate_kind_reads_dataset_as_test(tmp_path: Path) -> None:
    text = MINIMAL.replace('kind = "dapo"\nlearning_rate = 2e-5\nsteps = 1', 'kind = "evaluate"')
    text = text.replace('train = "aime-train"', 'dataset = "aime-train"')
    found = load(write_blueprint(tmp_path, text))
    assert found.data.train is None and found.data.test == "aime-train"
    assert found.datasets == [] and found.test_sets == ["aime-train"]
    none = text.replace('dataset = "aime-train"\n', "")
    assert _problems(tmp_path / "none", none) == ["[data] test: field required"]


def test_the_steps_the_test_group_size_and_the_evaluation_cadence_are_keys(
    tmp_path: Path,
) -> None:
    text = MINIMAL.replace(
        "batch_size = 2", 'batch_size = 2\ntest = "held"\ntest_group_size = 8'
    ).replace("steps = 1", "steps = 4\nevaluate_every = 2")
    found = load(write_blueprint(tmp_path, text))
    assert found.data.test_group_size == 8
    assert (found.recipe.steps, found.recipe.evaluate_every) == (4, 2)
    assert load(write_blueprint(tmp_path / "d", MINIMAL)).data.test_group_size is None
    bad = MINIMAL.replace("steps = 1", "steps = 0\nevaluate_every = -1")
    assert _problems(tmp_path / "bad", bad) == [
        "[recipe] steps: Input should be greater than or equal to 1",
        "[recipe] evaluate_every: Input should be greater than or equal to 0",
    ]
    unset = MINIMAL.replace("steps = 1\n", "")
    assert _problems(tmp_path / "unset", unset) == ["[recipe] steps: field required"]


def test_epochs_are_refused_with_where_the_count_went(tmp_path: Path) -> None:
    text = MINIMAL.replace("batch_size = 2", "batch_size = 2\nepochs = 2\nevaluate_every = 1")
    assert _problems(tmp_path, text) == [
        "[data]: epochs and evaluate_every: a run is counted in steps; set "
        "[recipe] steps and [recipe] evaluate_every"
    ]


def test_check_warns_on_the_older_spelling_an_overlap_and_a_cadence_past_the_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture_tasks(tmp_path)
    text = MINIMAL.replace(
        'train = "aime-train"\nbatch_size = 2',
        'dataset = "fixture"\ntest = "fixture"\nbatch_size = 2',
    ).replace("steps = 1", "steps = 1\nevaluate_every = 3")
    warned = [
        f.text
        for f in check(write_blueprint(tmp_path, text))
        if f.text.startswith(("[data]", "[recipe] evaluate_every"))
    ]
    assert warned == [
        "[data] dataset: the older spelling of train; still read",
        "[data] test: 2 tasks also in train, so the measurement is not held out",
        "[recipe] evaluate_every: 3 is past the 1 steps, so the run evaluates at its end only",
    ]


def test_wrong_kind_is_named(tmp_path: Path) -> None:
    problems = _problems(tmp_path, MINIMAL.replace('kind = "dapo"', 'kind = "train"'))
    assert problems == [
        "[recipe] kind: no recipe called 'train'; one of dapo, dr-grpo, cispo, opd, gepa, fst, "
        "evaluate"
    ]
    problems = _problems(tmp_path / "none", MINIMAL.replace('kind = "dapo"\n', ""))
    assert problems == ["[recipe] kind: field required"]


def test_bounds_and_choices_are_named(tmp_path: Path) -> None:
    problems = _problems(
        tmp_path,
        MINIMAL.replace("batch_size = 2", "batch_size = 0").replace("2e-5", "0.0")
        + 'reference = "x"\nkl_coef = -1.0\nclip_low = 1.5\n[checkpoints]\nttl_hours = 0.5\n',
    )
    assert [p.split(":")[0] for p in problems] == [
        "[data] batch_size",
        "[recipe] learning_rate",
        "[recipe] reference",
        "[recipe] kl_coef",
        "[recipe] clip_low",
        "[checkpoints] ttl_hours",
    ], "the SDK keeps a checkpoint for an hour at least; `check` says so before a run"
    problems = _problems(tmp_path / "c", MINIMAL.replace('"dapo"', '"cispo"') + "clip_high = 0.0\n")
    assert [p.split(":")[0] for p in problems] == ["[recipe] clip_high"], "an epsilon, above 0"


def test_gepa_requires_a_reflection_harness(tmp_path: Path) -> None:
    text = MINIMAL.replace('kind = "dapo"\nlearning_rate = 2e-5\nsteps = 1', 'kind = "gepa"')
    assert _problems(tmp_path, text) == ["[recipe] reflection_harness: field required"]


def test_evaluate_takes_no_extra_keys(tmp_path: Path) -> None:
    text = MINIMAL.replace('kind = "dapo"', 'kind = "evaluate"')
    assert _problems(tmp_path, text) == [
        "[recipe] learning_rate: unknown key",
        "[recipe] steps: unknown key",
    ]


def test_check_is_blocked_for_a_broken_file(tmp_path: Path) -> None:
    findings = check(tmp_path / "missing")
    assert [f.level for f in findings] == ["blocked"]
    assert "run.toml" in findings[0].text
    broken = write_blueprint(tmp_path, "[model\nname = 1")
    findings = check(broken)
    assert [f.level for f in findings] == ["blocked"]
    assert "not readable TOML" in findings[0].text
    bad = write_blueprint(tmp_path / "bad", MINIMAL.replace('model = "Qwen/Qwen3-8B"', ""))
    assert check(bad) == [Finding("blocked", "[agent] model: field required")]
    report = resolved.report(bad)
    assert report.config is None and report.resolution is None and report.text() == ""
    assert report.blocked and report.findings == check(bad)


def test_check_blocks_a_training_kind_on_a_provider_model(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        'model = "Qwen/Qwen3-8B"', 'model = "Qwen/Qwen3-8B"\nprovider = "openrouter"'
    )
    said = (
        "[recipe] kind = 'dapo' trains the weights this run serves, and "
        "[agent] provider = 'openrouter' serves them elsewhere"
    )
    assert Finding("blocked", said) in check(write_blueprint(tmp_path, text))
    text = text.replace('kind = "dapo"\nlearning_rate = 2e-5\nsteps = 1', 'kind = "evaluate"')
    found = check(write_blueprint(tmp_path / "evaluate", text))
    assert not any("serves them elsewhere" in f.text for f in found)


# ------------------------------------------------------------- the resolved config


def test_the_resolved_config_fills_every_table_and_leads_the_recipe_with_its_kind() -> None:
    found = resolved.resolved(load(BLUEPRINTS / "dapo"))
    assert list(found) == ["agent", "data", "rollout", "recipe", "checkpoints"]
    assert found["agent"] == {
        "model": "Qwen/Qwen3-8B",
        "provider": "tinker",
        "from_checkpoint": None,
        "harness": "pi@0.85.1",
        "kwargs": {},
        "env": {},
        "modules": None,
    }
    assert found["data"] == {
        "train": "aime-train",
        "test": None,
        "batch_size": 2,
        "group_size": 4,
        "test_group_size": None,
        "seed": 0,
    }
    assert found["rollout"]["concurrency"] == 2 and "env" not in found["rollout"]
    assert found["rollout"]["max_tokens"] == 8192 and found["rollout"]["timeout"] is None
    assert list(found["recipe"])[0] == "kind"
    assert found["recipe"] == {
        "kind": "dapo",
        "learning_rate": 2e-5,
        "steps": 2,
        "evaluate_every": 0,
        "substeps": 16,
        "warmup": 0,
        "reference": "trainer",
        "kl_coef": 0.0,
        "lora_rank": 32,
        "restore_optimizer": False,
        "teacher_coef": 0.0,
        "teacher": None,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "overlong_penalty": 0.5,
        "overlong_buffer": 0.2,
    }
    assert found["checkpoints"] == {"every": 1, "ttl_hours": 168.0}


def test_the_resolved_config_names_the_renderer_a_served_model_loads() -> None:
    loaded = load(BLUEPRINTS / "dapo")
    assert (
        loaded.rollout.renderer == ""
        and resolved.resolved(loaded)["rollout"]["renderer"] == "qwen3"
    )
    named = loaded.model_copy(
        update={"rollout": loaded.rollout.model_copy(update={"renderer": "x"})}
    )
    assert resolved.resolved(named)["rollout"]["renderer"] == "x"
    hosted = loaded.model_copy(
        update={"agent": loaded.agent.model_copy(update={"provider": "openrouter"})}
    )
    assert resolved.resolved(hosted)["rollout"]["renderer"] == ""
    assert resolved.recommended_renderer("nobody/knows-this") == ""


def test_the_resolved_toml_reads_back_as_the_resolved_config_without_its_nones() -> None:
    for kind in config.KINDS:
        found = resolved.resolved(load(BLUEPRINTS / kind))
        text = resolved.as_toml(found)
        expected = {
            table: {key: value for key, value in body.items() if value is not None}
            for table, body in found.items()
        }
        assert tomllib.loads(text) == expected, kind
        assert text.startswith("[agent]\n") and "\n[agent.env]\n" in text
        assert "\n[recipe]\n" + f'kind = "{kind}"\n' in text


def test_a_key_toml_cannot_leave_bare_is_quoted_so_the_output_reads_back() -> None:
    nested = {"rollout": {"kwargs": {"models": {"Qwen/Qwen3-8B": {"limit": {"context": 16384}}}}}}
    text = resolved.as_toml({**nested, "env": {"a.b": "x", "plain_key-1": "y"}})
    assert '[rollout.kwargs.models."Qwen/Qwen3-8B".limit]' in text
    assert tomllib.loads(text) == {**nested, "env": {"a.b": "x", "plain_key-1": "y"}}


def test_toml_values_are_spelled_as_toml() -> None:
    text = resolved.as_toml(
        {
            "t": {
                "yes": True,
                "n": -1,
                "f": 2e-5,
                "whole": 1.0,
                "far": float("inf"),
                "s": 'say "hi"',
                "names": ["a", "b"],
                "none": None,
                "sub": {"k": "v", "inner": {"deep": 1}},
            }
        }
    )
    assert tomllib.loads(text) == {
        "t": {
            "yes": True,
            "n": -1,
            "f": 2e-5,
            "whole": 1.0,
            "far": float("inf"),
            "s": 'say "hi"',
            "names": ["a", "b"],
            "sub": {"k": "v", "inner": {"deep": 1}},
        }
    }
    assert "none" not in text and "whole = 1.0" in text and "far = inf" in text


@pytest.mark.parametrize(
    ("kind", "line"),
    [
        (
            "dapo",
            "# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, "
            "averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token "
            "budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-08, weight decay "
            "0.1, gradient norm clipped at 1.0, no warm-up (the paper's is 20 steps); degenerate "
            "groups dropped",
        ),
        (
            "dr-grpo",
            "# dr-grpo: advantage = group mean, not divided by spread; loss = ppo, "
            "clip 0.2 / 0.2, summed over tokens; length penalty 0.2 over 0 tokens among "
            "solved answers, capped at 0.5; 1 substep; adamw betas 0.9 / 0.95, eps 1e-08, "
            "gradient norm clipped at 1.0; degenerate groups dropped",
        ),
        (
            "cispo",
            "# cispo: advantage = group mean, divided by spread; loss = cispo, "
            "weight truncated above 4.0, no lower bound, averaged per prompt; overlong penalty "
            "up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw "
            "betas 0.9 / 0.95, eps 1e-15; degenerate groups dropped",
        ),
        (
            "opd",
            "# opd: advantage = teacher - reference per token, no reward term; loss = "
            "importance_sampling, summed over tokens; 1 substep; adamw betas 0.9 / 0.95, eps "
            "1e-08; groups with equal rewards kept for the teacher; teacher 1.0 toward "
            "Qwen/Qwen3-32B",
        ),
        ("gepa", None),
        ("evaluate", None),
    ],
)
def test_the_resolution_line_states_what_the_recipe_name_resolves_to(
    kind: str, line: str | None
) -> None:
    report = resolved.report(BLUEPRINTS / kind)
    assert report.resolution == line
    text = report.text()
    assert text.startswith("[agent]\n")
    if line is None:
        assert text.endswith("ttl_hours = 168.0\n") and "#" not in text
    else:
        assert text.endswith(f"ttl_hours = 168.0\n{line}\n")


def test_the_resolution_line_names_a_kl_anchor_when_asked(tmp_path: Path) -> None:
    text = MINIMAL.replace('kind = "dapo"', 'kind = "dapo"\nkl_coef = 0.05')
    report = resolved.report(write_blueprint(tmp_path, text))
    assert report.resolution is not None
    assert report.resolution.endswith("degenerate groups dropped; kl 0.05 to the starting weights")


def test_the_resolution_line_names_a_teacher_and_the_groups_it_keeps(tmp_path: Path) -> None:
    taught = 'kind = "dapo"\nteacher_coef = 0.5\nteacher = "Qwen/Qwen3-32B"\nkl_coef = 0.05'
    report = resolved.report(write_blueprint(tmp_path, MINIMAL.replace('kind = "dapo"', taught)))
    assert report.resolution is not None
    assert report.resolution.endswith(
        "groups with equal rewards kept for the teacher; kl 0.05 to the starting weights; "
        "teacher 0.5 toward Qwen/Qwen3-32B"
    )


def test_the_resolution_line_says_how_far_the_length_rule_may_dock(tmp_path: Path) -> None:
    dr = (BLUEPRINTS / "dr-grpo" / "run.toml").read_text(encoding="utf-8")
    report = resolved.report(write_blueprint(tmp_path, dr + "length_cap = inf\n"))
    assert report.resolution is not None
    assert "length penalty 0.2 over 0 tokens among solved answers, uncapped;" in report.resolution
    assert "length_cap = inf" in report.text()


def test_the_resolution_line_says_what_one_substep_leaves(tmp_path: Path) -> None:
    text = MINIMAL.replace('kind = "dapo"', 'kind = "dapo"\nsubsteps = 1')
    report = resolved.report(write_blueprint(tmp_path / "one", text))
    assert report.resolution == (
        "# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, "
        "averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token "
        "budget; 1 substep; adamw betas 0.9 / 0.95, eps 1e-08, weight decay 0.1, gradient norm "
        "clipped at 1.0, no warm-up (the paper's is 20 steps); degenerate groups dropped"
    )
    text = MINIMAL.replace('kind = "dapo"', 'kind = "dapo"\nwarmup = 20')
    report = resolved.report(write_blueprint(tmp_path / "warm", text))
    assert (
        report.resolution is not None
        and "learning rate warmed up over 20 steps" in report.resolution
    )


def test_the_rollout_serving_keys_load_with_their_defaults(tmp_path: Path) -> None:
    found = load(write_blueprint(tmp_path, MINIMAL)).rollout
    assert (found.endpoint_url, found.host, found.bind, found.bind_port) == (
        "",
        "host.docker.internal",
        "0.0.0.0",
        0,
    )
    assert (found.temperature, found.top_p, found.top_k) == (1.0, 1.0, -1)
    assert (found.max_tokens, found.max_context, found.renderer) == (8192, 0, "")
    assert (found.fill_context, found.check_turns) == (False, False)
    text = MINIMAL.replace(
        "[rollout]",
        '[rollout]\nhost = "proxy.lan"\nbind = "127.0.0.1"\nbind_port = 8000\n'
        "fill_context = true\ncheck_turns = true\nmax_context = 32768\n"
        'renderer = "qwen3"\nendpoint_url = "https://p.example"',
    )
    found = load(write_blueprint(tmp_path / "set", text)).rollout
    assert (found.host, found.bind, found.bind_port) == ("proxy.lan", "127.0.0.1", 8000)
    assert (found.fill_context, found.check_turns) == (True, True)
    assert (found.max_context, found.renderer, found.endpoint_url) == (
        32768,
        "qwen3",
        "https://p.example",
    )
    problems = _problems(
        tmp_path / "bad", MINIMAL.replace("[rollout]", "[rollout]\nbind_port = 70000")
    )
    assert [p.split(":")[0] for p in problems] == ["[rollout] bind_port"]


def test_rollout_jobs_dir_defaults_to_jobs(tmp_path: Path) -> None:
    assert load(BLUEPRINTS / "evaluate").rollout.jobs_dir == "jobs"
    text = MINIMAL.replace("[rollout]", '[rollout]\njobs_dir = "scratch/jobs"')
    assert load(write_blueprint(tmp_path, text)).rollout.jobs_dir == "scratch/jobs"
