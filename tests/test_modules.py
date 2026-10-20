"""Tests for `shipyard.jobs.modules`: seeding, keeping and refusing modules, their delivery
to Harbor, and how a run carries them into its jobs and across a resume."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import AgentConfig
from harbor.utils.templating import render_prompt_template
from typer.testing import CliRunner

from proxies import MODEL
from shipyard import Group, Trainer
from shipyard.cli import app
from shipyard.config import Rollout
from shipyard.jobs.engine import DELIVERED, HarborEngine, Job, laid_over
from shipyard.jobs.harnesses import harness
from shipyard.jobs.modules import (
    KINDS,
    PROMPT,
    PROMPT_FILE,
    RESERVED,
    SKILL_FILE,
    SKILLS_DIR,
    Inadmissible,
    Module,
    Modules,
    Skill,
    Unsupported,
    deliver,
    keep,
    preamble,
    read,
    write,
)
from shipyard.models.endpoints import ProviderEndpoint
from shipyard.run import record
from shipyard.run.directory import open_run, reopened, running
from shipyard.trainer import CARRIED, carried
from tests.fakes import Alternating, FakeProxy, FakeTrials, Harbor, dataset
from tests.trainers import FakeService

FIXTURES = Path(__file__).parent / "modules"
VALID = FIXTURES / "valid"

#: Blueprint over the two-task `fixture` dataset whose `[agent] modules` names `modules/` beside
#: it. A provider serves the agent, so its rollouts need no proxy and no Tinker.
CARRYING = """
[agent]
model = "m"
provider = "openrouter"
harness = "pi@0.85.1"
modules = "modules"

[data]
train = "fixture"
batch_size = 2

[recipe]
steps = 1
"""


def _skill(body: str, *, name: str = "solving") -> str:
    return f"---\nname: {name}\ndescription: what to do.\n---\n\n{body}\n"


def _module(text: str = _skill("read it twice"), **beside: str) -> Module:
    return Module("solving", "skill", {SKILL_FILE: text, **beside})


def _prompt(text: str = "Use the project's own python.\n", *, name: str = PROMPT) -> Module:
    return Module(name, "prompt", {PROMPT_FILE: text})


def _files(home: Path) -> dict[str, str]:
    return {
        found.relative_to(home).as_posix(): found.read_text(encoding="utf-8")
        for found in sorted(home.rglob("*"))
        if found.is_file()
    }


# ------------------------------------------------------------------- read and keep


def test_a_valid_directory_reads_as_one_component_per_skill() -> None:
    candidate = Modules.read(VALID)
    assert list(candidate.components) == ["solving"]
    module = candidate.components["solving"]
    assert (module.name, module.kind) == ("solving", "skill")
    assert module.text == (VALID / SKILLS_DIR / "solving" / SKILL_FILE).read_text(encoding="utf-8")
    assert list(module.files) == [SKILL_FILE, "examples/worked.md"]
    assert len(candidate.digest) == 16 and len(module.digest) == 16


def test_the_layout_holds_a_prompt_at_the_top_and_skills_beneath(tmp_path: Path) -> None:
    write({PROMPT_FILE: "Run the tests.\n"}, into=tmp_path)
    shutil.copytree(VALID / SKILLS_DIR, tmp_path / SKILLS_DIR)
    candidate = Modules.read(tmp_path)
    assert list(candidate.components) == [PROMPT, "solving"]
    prompt = candidate.components[PROMPT]
    assert (prompt.name, prompt.kind, prompt.files) == (
        PROMPT,
        "prompt",
        {PROMPT_FILE: "Run the tests.\n"},
    )


@pytest.mark.parametrize("prompted", [False, True])
def test_keep_writes_the_layout_read_takes_so_a_winner_can_be_carried(
    tmp_path: Path, prompted: bool
) -> None:
    home = tmp_path / "modules"
    shutil.copytree(VALID, home)
    if prompted:
        write({PROMPT_FILE: "Run the tests.\n"}, into=home)
    candidate = Modules.read(home)
    kept = tmp_path / "kept"
    keep(candidate, kept)
    assert _files(kept) == _files(home)
    again = Modules.read(kept)
    assert again == candidate and again.digest == candidate.digest


def test_a_missing_or_empty_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Inadmissible, match="no modules at"):
        Modules.read(tmp_path / "none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(Inadmissible, match="no modules under"):
        Modules.read(tmp_path / "empty")
    # Dot-files and an empty skills directory hold no module.
    (tmp_path / "empty" / ".hidden" / "x").mkdir(parents=True)
    (tmp_path / "empty" / SKILLS_DIR).mkdir()
    with pytest.raises(Inadmissible, match="no modules under"):
        Modules.read(tmp_path / "empty")


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ("README.md", "'README.md' is not part of the layout"),
        ("solving/SKILL.md", "'solving' is not part of the layout"),
        ("SKILL.md", "'SKILL.md' is not part of the layout"),
        ("PROMPT.md/x.md", "'PROMPT.md' is not part of the layout"),
    ],
)
def test_an_entry_outside_the_layout_is_refused(tmp_path: Path, entry: str, why: str) -> None:
    """Covers the layout used before `skills/` existed: a skill directly under `modules/` is
    refused rather than guessed at."""
    shutil.copytree(VALID, tmp_path / "modules")
    write({entry: _skill("x")}, into=tmp_path / "modules")
    with pytest.raises(Inadmissible, match=re.escape(why) + ": a modules directory holds"):
        Modules.read(tmp_path / "modules")


# ------------------------------------------------------------------ what is refused


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "the text is empty"),
        ("# Solving\n\nread it twice", "does not open with a `---` frontmatter block"),
        ("---\nname: a\ndescription: d\n\nbody", "never closed by a second `---`"),
        ("---\nname: a\n---\n\nbody", "the frontmatter has no description"),
        ("---\ndescription: d\n---\n\nbody", "the frontmatter has no name"),
        ("---\nname:\ndescription:\n---\n\nbody", "the frontmatter has no name or description"),
    ],
)
def test_frontmatter_failures_are_named(text: str, reason: str) -> None:
    refused = Skill().check(_module(text))
    assert refused is not None and reason in refused
    assert Skill().check(_module(_skill("body"))) is None


def test_a_skill_without_frontmatter_is_refused_on_read_by_name() -> None:
    with pytest.raises(Inadmissible, match="'solving' cannot be delivered as a skill: it does"):
        Modules.read(FIXTURES / "bare")


def test_a_skill_directory_without_its_marker_is_refused_by_name() -> None:
    with pytest.raises(Inadmissible, match="'skills/solving' holds no SKILL.md"):
        Modules.read(FIXTURES / "unknown")


@pytest.mark.parametrize(("directory", "kind"), list(RESERVED.items()))
def test_a_reserved_directory_says_the_kind_is_not_supported(
    tmp_path: Path, directory: str, kind: str
) -> None:
    write({"lakehouse/server.py": "print('hello')\n"}, into=tmp_path / directory)
    with pytest.raises(Unsupported, match=f"{kind} modules are not supported in this version"):
        Modules.read(tmp_path)
    with pytest.raises(Inadmissible) as refused:  # an Unsupported is an Inadmissible
        Modules.read(tmp_path)
    assert isinstance(refused.value, Unsupported) and refused.value.kind == kind
    assert f"{directory!r}" in str(refused.value)


def test_the_kinds_table_holds_skills_and_prompts() -> None:
    assert list(KINDS) == ["skill", "prompt"]
    skill, prompt = KINDS["skill"], KINDS["prompt"]
    assert (skill.marker, skill.alone, skill.home("solving")) == (
        SKILL_FILE,
        False,
        "skills/solving",
    )
    assert (prompt.marker, prompt.alone, prompt.home(PROMPT)) == (PROMPT_FILE, True, "")
    assert RESERVED == {"tools": "tool", "harness": "harness"}


# ------------------------------------------------------------------------ prompts


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("  \n", "the text is empty"),
        ("a {% endraw %} b", "`{% endraw %}`"),
        ("a {%- endraw -%} b", "`{% endraw %}`"),
    ],
)
def test_a_prompt_is_refused_when_it_cannot_be_delivered_as_written(
    tmp_path: Path, text: str, why: str
) -> None:
    write({PROMPT_FILE: text}, into=tmp_path)
    with pytest.raises(
        Inadmissible, match="'prompt' cannot be delivered as a prompt: .*" + re.escape(why)
    ):
        Modules.read(tmp_path)


def test_a_prompt_is_its_marker_alone() -> None:
    refused = KINDS["prompt"].check(Module(PROMPT, "prompt", {PROMPT_FILE: "go", "notes.md": "x"}))
    assert refused is not None and "nothing beside it is delivered" in refused


def test_a_skill_may_not_take_the_prompts_name(tmp_path: Path) -> None:
    write({PROMPT_FILE: "go", f"{SKILLS_DIR}/{PROMPT}/{SKILL_FILE}": _skill("x")}, into=tmp_path)
    with pytest.raises(Inadmissible, match="'skills/prompt' takes the name of the PROMPT.md"):
        Modules.read(tmp_path)


def test_a_prompt_is_delivered_as_harbors_prompt_template(tmp_path: Path) -> None:
    """Harbor renders the instruction through the template, which yields the prompt text, a
    blank line, then the instruction. Jinja syntax in the prompt text is not evaluated."""
    text = "Use {{ python }} and {% if x %}this{% endif %}.\n\n"
    fields = deliver(Modules({PROMPT: _prompt(text)}), tmp_path / "into")
    template = Path(fields["kwargs"]["prompt_template_path"])
    assert template == (tmp_path / "into" / "prompt" / "template.j2").resolve()
    rendered = render_prompt_template(template, "Fix the bug.\n")
    assert rendered == "Use {{ python }} and {% if x %}this{% endif %}.\n\nFix the bug.\n"
    assert rendered == preamble(_prompt(text)) + "Fix the bug.\n"


def test_a_prompt_rides_with_skills_and_lays_over_the_blueprints_kwargs(tmp_path: Path) -> None:
    candidate = Modules({PROMPT: _prompt(), "solving": _module()})
    under: dict[str, Any] = {"kwargs": {"version": "1.0", "prompt_template_path": "theirs.j2"}}
    fields = laid_over(under, deliver(candidate, tmp_path / "into"))
    assert fields["skills"] == [str((tmp_path / "into" / "skills").resolve())]
    assert fields["kwargs"]["version"] == "1.0"
    assert fields["kwargs"]["prompt_template_path"].endswith("prompt/template.j2")


def test_a_candidate_carries_one_prompt(tmp_path: Path) -> None:
    two = Modules({"a": _prompt(name="a"), "b": _prompt("Other.\n", name="b")})
    with pytest.raises(Inadmissible, match="modules carry one prompt, and these have a, b"):
        deliver(two, tmp_path / "into")


def test_texts_written_in_code_are_refused_as_reading_refuses_them() -> None:
    """Modules made in code, by `with_texts` or by hand, are checked as read ones are:
    otherwise `deliver` writes a template Harbor cannot render, or a skill the harness never
    surfaces, and `keep` saves what a resume cannot read back."""
    carried = Modules({PROMPT: _prompt(), "solving": _module()})
    with pytest.raises(Inadmissible, match="'prompt' cannot be delivered as a prompt: .*endraw"):
        carried.with_texts({PROMPT: "Check it. {% endraw %} Then finish."})
    with pytest.raises(Inadmissible, match="'solving' cannot be delivered as a skill: .*---"):
        carried.with_texts({"solving": "Read it twice."})
    with pytest.raises(Inadmissible, match="'prompt' cannot be delivered as a prompt: .*empty"):
        Modules({PROMPT: _prompt("  \n")})
    assert carried.with_texts({PROMPT: "Check it.\n"}).texts[PROMPT] == "Check it.\n"


# ------------------------------------------------------------------------ digests


def test_the_digest_changes_with_any_file() -> None:
    base = _module()
    assert base.digest == _module().digest, "the same files, the same digest"
    assert base.digest != _module(_skill("read it three times")).digest
    assert base.digest != _module(**{"examples/one.md": "an example"}).digest
    with_a = _module(**{"a.md": "same"})
    with_b = _module(**{"b.md": "same"})
    assert with_a.digest != with_b.digest, "a file's path counts, not only its content"


def test_a_candidates_digest_covers_its_components_and_their_names() -> None:
    one = Modules({"solving": _module()})
    assert one.digest == Modules({"solving": _module()}).digest
    assert one.digest != Modules({"planning": _module()}).digest, "the name counts"
    two = Modules({"solving": _module(), "planning": _module(_skill("plan", name="planning"))})
    assert two.digest != one.digest
    changed = Modules({"solving": _module(_skill("read it three times"))})
    assert changed.digest != one.digest


# ----------------------------------------------------------------------- delivery


def test_deliver_names_the_skills_root_harbor_takes_and_writes_every_file(
    tmp_path: Path,
) -> None:
    candidate = Modules.read(VALID)
    into = tmp_path / "m"
    stale = into / "skills" / "stale" / SKILL_FILE
    stale.parent.mkdir(parents=True)
    stale.write_text(_skill("left over", name="stale"))
    fields = deliver(candidate, into)
    root = (into / "skills").resolve()
    assert fields == {"skills": [str(root)]} and root.is_absolute()
    assert _files(root / "solving") == _files(VALID / SKILLS_DIR / "solving")
    assert not stale.exists(), "emptied first: one candidate's text never beside another's"
    # The field is Harbor's: `AgentConfig.skills` takes the root as it is.
    assert AgentConfig(name="pi", **fields).skills == [str(root)]


async def test_a_kinds_fields_land_on_the_agent_config_beside_the_rollouts_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fields a kind's `deliver` returns merge with the ones the rollout sets. A future
    harness or tool kind can then be added as a class and a `KINDS` entry, with no change
    to the engine."""
    # These fields are made up, so Harbor's own check of the config would refuse them.
    monkeypatch.setattr("shipyard.jobs.engine.preflight", lambda *_: None)
    monkeypatch.setattr(
        "shipyard.jobs.engine.deliver",
        lambda candidate, into: {
            "name": "other",
            "kwargs": {"turns": 3},
            "env": {"TOOL": "on"},
            "skills": [str(into / "skills")],
        },
    )
    runner = FakeTrials()
    table = Rollout(sandbox="docker", concurrency=1, jobs_dir=str(tmp_path / "jobs"))
    engine = HarborEngine(table, env={"KEEP": "1"}, kwargs={"thinking": "high"}, run_trial=runner)
    job = Job("run-0000", modules=Modules.read(VALID), harness=harness("pi@0.85.1"))
    endpoint = ProviderEndpoint("m", "openrouter")
    await engine.run(job, [tmp_path / "tasks" / "alpha"], 1, endpoint=endpoint)
    agent = runner.configs[0].agent
    assert agent.name == "other" and agent.model_name == "openrouter/m"
    assert agent.kwargs == {"version": "0.85.1", "thinking": "high", "turns": 3}
    assert agent.env == {"KEEP": "1", "TOOL": "on"}
    assert agent.skills == [str(tmp_path / "jobs" / "run-0000" / DELIVERED / "skills")]


def test_a_relative_into_is_delivered_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fields = deliver(Modules.read(VALID), Path("m"))
    assert Path(fields["skills"][0]).is_absolute()
    assert Path(fields["skills"][0]) == (tmp_path / "m" / "skills").resolve()


def test_read_skips_the_incidental_and_write_refuses_an_escape(tmp_path: Path) -> None:
    source = tmp_path / "solving"
    (source / "examples" / "__pycache__").mkdir(parents=True)
    (source / SKILL_FILE).write_text("the document")
    (source / "examples" / SKILL_FILE).write_text("also named SKILL.md")
    (source / "examples" / "worked.md").write_text("and one that is not")
    (source / "examples" / "__pycache__" / "cached.pyc").write_text("x")
    (source / ".DS_Store").write_text("x")
    files = read(source)
    assert files == {
        SKILL_FILE: "the document",
        "examples/SKILL.md": "also named SKILL.md",
        "examples/worked.md": "and one that is not",
    }
    write(files, into=tmp_path / "round-trip")
    assert read(tmp_path / "round-trip") == files
    with pytest.raises(Inadmissible, match="inside the module"):
        write({"../escaped.md": "no"}, into=tmp_path / "solving")
    (source / "binary.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(Inadmissible, match="'binary.bin' in 'solving' is not UTF-8 text"):
        read(source)


# ------------------------------------------------------------------- the carrying


def _carrying(
    tmp_path: Path, text: str = CARRYING, modules: Path | None = VALID, *, at: str = "bp"
) -> Path:
    """Write a blueprint at `<tmp>/<at>` with a copy of `modules` beside it, over the fixture
    dataset in `<tmp>/tasks`."""
    if not (tmp_path / "tasks" / "fixture").is_dir():
        dataset(tmp_path / "tasks", "fixture", "alpha", "beta")
    home = tmp_path / at
    home.mkdir(parents=True)
    (home / "run.toml").write_text(text, encoding="utf-8")
    if modules is not None:
        shutil.copytree(modules, home / "modules")
    return home


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    return [tmp_path / "tasks" / "fixture" / name for name in names]


async def _roll(
    home: Path, root: Path, *jobs: tuple[list[Path], Mapping[str, Any]], update: bool = False
) -> tuple[Path, Trainer]:
    """Open a run for `home` and train it one step that rolls out each of `jobs`, a batch with
    the keyword arguments of its `rollout`; with `update`, it scores and trains on each."""

    class Rolling(Trainer):
        async def step(self) -> None:
            for tasks, named in jobs:
                groups = await self.rollout(tasks, **named)
                if update:
                    await self.score(groups)
                    await self.update(groups)

    directory, config = open_run(home, root=root)
    with running(directory), Rolling(config, directory) as trainer:
        await trainer.train()
    return directory, trainer


def _rows(directory: Path) -> list[dict[str, Any]]:
    return list(record.read(directory / record.JOBS))


async def test_a_run_keeps_the_carried_modules_once_and_each_job_delivers_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path)
    kept: list[Path] = []
    keeping = keep
    monkeypatch.setattr(
        "shipyard.trainer.keep", lambda c, into: (kept.append(into), keeping(c, into))
    )
    runner = harbor(FakeTrials())
    directory, trainer = await _roll(
        home,
        tmp_path / "runs",
        (_batch(tmp_path, "alpha", "beta"), {"attempts": 2}),
        (_batch(tmp_path, "alpha"), {"attempts": 1}),
    )
    assert trainer.state.modules == Modules.read(home / "modules")
    copy = directory / record.MODULES / CARRIED
    assert kept == [copy] and _files(copy) == _files(VALID)
    assert Modules.read(copy).digest == trainer.state.modules.digest
    with Trainer(trainer.config, directory) as again:
        assert again.state.modules == trainer.state.modules
    assert kept == [copy], "the carried copy is written once; later trainers read it back"
    first, second = f"{directory.name}-0000", f"{directory.name}-0001"
    assert len(runner.configs) == 5
    for config, name in zip(runner.configs, [first] * 4 + [second], strict=True):
        root = (tmp_path / "jobs" / name / DELIVERED / "skills").resolve()
        assert config.agent.skills == [str(root)]
        assert config.agent.name == "pi" and config.agent.kwargs == {"version": "0.85.1"}
        assert _files(root / "solving") == _files(VALID / SKILLS_DIR / "solving")
    rows = _rows(directory)
    assert [row["modules"] for row in rows] == [trainer.state.modules.digest] * 2
    head = ["at", "job", "dir", "purpose", "harness", "trials", "step", "tasks", "modules"]
    assert list(rows[0])[:9] == head and rows[0]["harness"] == "pi@0.85.1"
    assert sorted(p.name for p in (tmp_path / "jobs" / first).iterdir()) == sorted(
        [DELIVERED] + [config.trial_name for config in runner.configs[:4]]
    )


async def test_a_job_delivers_the_candidate_it_is_handed_and_nothing_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, CARRYING.replace('modules = "modules"\n', ""), modules=None)
    runner = harbor(FakeTrials())
    handed = Modules({"solving": _module(_skill("read it three times"))})
    alpha = _batch(tmp_path, "alpha")
    directory, trainer = await _roll(
        home,
        tmp_path / "runs",
        (alpha, {"attempts": 1}),
        (alpha, {"attempts": 1, "modules": handed}),
    )
    assert trainer.state.modules is None and not (directory / record.MODULES).exists()
    bare, carrying = f"{directory.name}-0000", f"{directory.name}-0001"
    assert runner.configs[0].agent.skills == []
    assert sorted(p.name for p in (tmp_path / "jobs" / bare).iterdir()) == [
        runner.configs[0].trial_name
    ], "nothing of ours in a job carrying nothing"
    root = (tmp_path / "jobs" / carrying / DELIVERED / "skills").resolve()
    assert runner.configs[1].agent.skills == [str(root)]
    assert (root / "solving" / SKILL_FILE).read_text() == _skill("read it three times")
    rows = _rows(directory)
    assert "modules" not in rows[0] and rows[1]["modules"] == handed.digest


async def test_a_rollout_runs_the_harness_it_names_and_its_row_says_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, CARRYING.replace('modules = "modules"\n', ""), modules=None)
    runner = harbor(FakeTrials())
    alpha = _batch(tmp_path, "alpha")
    named = {"attempts": 1, "harness": "claude-code@2.1.0"}
    directory, _ = await _roll(home, tmp_path / "runs", (alpha, {"attempts": 1}), (alpha, named))
    ran = [(one.agent.name, one.agent.kwargs.get("version")) for one in runner.configs]
    assert ran == [("pi", "0.85.1"), ("claude-code", "2.1.0")], "[agent] harness, then the named"
    assert [row["harness"] for row in _rows(directory)] == ["pi@0.85.1", "claude-code@2.1.0"]


#: CARRYING without modules, driven by two harnesses, in groups of two.
TWO = (
    CARRYING.replace('modules = "modules"\n', "")
    .replace('harness = "pi@0.85.1"', 'harness = ["pi@0.85.1", "claude-code@2.1.0"]')
    .replace("batch_size = 2", "batch_size = 2\ngroup_size = 2")
)


async def _two(tmp_path: Path, text: str) -> tuple[Path, list[Group]]:
    """Run one step that rolls out the fixture's two tasks under `text`, and return the run's
    directory and the groups the rollout returned."""
    found: list[Group] = []

    class Rolling(Trainer):
        async def step(self) -> None:
            found.extend(await self.rollout(_batch(tmp_path, "alpha", "beta")))

    directory, config = open_run(_carrying(tmp_path, text, modules=None), root=tmp_path / "runs")
    with running(directory), Rolling(config, directory) as trainer:
        await trainer.train()
    return directory, found


async def test_two_unmixed_harnesses_each_run_every_task_in_a_group_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    harbor(FakeTrials())
    directory, found = await _two(tmp_path, TWO)
    assert [(group.task.name, group.harness) for group in found] == [
        ("alpha", "pi@0.85.1"),
        ("alpha", "claude-code@2.1.0"),
        ("beta", "pi@0.85.1"),
        ("beta", "claude-code@2.1.0"),
    ], "in the order asked, a group of each task per harness"
    assert all(len(group.trials) == 2 for group in found), "each of group_size"
    assert [(row["harness"], row["tasks"], row["trials"]) for row in _rows(directory)] == [
        ("pi@0.85.1", 2, 4),
        ("claude-code@2.1.0", 2, 4),
    ]


async def test_mixed_harnesses_split_each_tasks_attempts_and_share_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    harbor(FakeTrials())
    mixed = TWO.replace("harness = [", "mix_harnesses = true\nharness = [")
    directory, found = await _two(tmp_path, mixed)
    assert [group.task.name for group in found] == ["alpha", "beta"]
    assert [[one.harness.name for one in group.trials] for group in found] == [
        ["pi", "claude-code"],
        ["pi", "claude-code"],
    ], "one attempt of each task per harness, in one group"
    assert [(row["harness"], row["trials"]) for row in _rows(directory)] == [
        ("pi@0.85.1", 2),
        ("claude-code@2.1.0", 2),
    ]


async def test_a_handed_candidate_wins_over_the_carried_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = harbor(FakeTrials())
    planning = Module("planning", "skill", {SKILL_FILE: _skill("plan first", name="planning")})
    other = Modules({"planning": planning})
    directory, _ = await _roll(
        _carrying(tmp_path), tmp_path / "runs", (_batch(tmp_path, "alpha"), {"modules": other})
    )
    root = Path(runner.configs[0].agent.skills[0])
    assert sorted(p.name for p in root.iterdir()) == ["planning"]
    [row] = _rows(directory)
    assert row["modules"] == other.digest != Modules.read(VALID).digest
    assert row["job"] == f"{directory.name}-0000"


async def test_a_job_hands_a_prompt_to_harbor_as_the_agents_prompt_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = harbor(FakeTrials())
    prompted = Modules({"guidance": _prompt("Run the tests.\n", name="guidance")})
    directory, _ = await _roll(
        _carrying(tmp_path), tmp_path / "runs", (_batch(tmp_path, "alpha"), {"modules": prompted})
    )
    kwargs = runner.configs[0].agent.kwargs
    job = f"{directory.name}-0000"
    template = (tmp_path / "jobs" / job / DELIVERED / "prompt" / "template.j2").resolve()
    assert kwargs == {"version": "0.85.1", "prompt_template_path": str(template)}
    assert render_prompt_template(template, "Fix it.") == "Run the tests.\n\nFix it."
    assert runner.configs[0].agent.skills == [], "a prompt is not a skill"
    assert _rows(directory)[0]["modules"] == prompted.digest


async def test_a_carried_prompt_replaces_the_blueprints_template_and_skills_leave_it_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harbor: Harbor
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = harbor(FakeTrials())
    text = CARRYING.replace("[recipe]", '[agent.kwargs]\nprompt_template_path = "t.j2"\n[recipe]')
    prompted = _carrying(tmp_path, text, modules=None, at="prompted")
    write({PROMPT_FILE: "Run the tests.\n"}, into=prompted / "modules")
    alpha = (_batch(tmp_path, "alpha"), {"attempts": 1})
    directory, _ = await _roll(prompted, tmp_path / "runs", alpha)
    template = tmp_path / "jobs" / f"{directory.name}-0000" / DELIVERED / "prompt" / "template.j2"
    assert runner.configs[0].agent.kwargs["prompt_template_path"] == str(template.resolve())
    await _roll(_carrying(tmp_path, text, at="skilled"), tmp_path / "runs", alpha)
    assert runner.configs[1].agent.kwargs["prompt_template_path"] == "t.j2"


def test_a_run_carries_its_modules_and_keeps_a_candidate_in_the_layout_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, config = open_run(_carrying(tmp_path), root=tmp_path / "runs")
    assert carried(config, directory) == Modules.read(VALID)
    assert (directory / record.MODULES / CARRIED / SKILLS_DIR / "solving" / SKILL_FILE).is_file()
    best = Modules({**Modules.read(VALID).components, PROMPT: _prompt("Run the tests.\n")})
    with Trainer(config, directory) as trainer:
        kept = trainer.keep(best, "best")
    assert kept == directory / record.MODULES / "best"
    assert Modules.read(kept) == best, "a kept candidate seeds a later run"
    assert Modules.read(directory / record.MODULES / CARRIED) == Modules.read(VALID), (
        "the carried copy stays"
    )


def test_a_resumed_run_carries_the_modules_its_checkpoint_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the seed: a search, such as FST's, moved on to other modules, which the state of
    its checkpoint holds whole."""
    monkeypatch.chdir(tmp_path)
    directory, config = open_run(_carrying(tmp_path), root=tmp_path / "runs")
    trainer = Trainer(config, directory)
    later = Modules({"solving": _module(_skill("read it three times"))})
    saved = Trainer.State(steps=4, updates=4, drawn=4, modules=later).model_dump(mode="json")
    record.write_json(directory / record.STATES / "step-4.json", saved)
    resumed = Trainer(config, directory)
    resumed.restore(
        directory, {"tag": "step-4", "step": 4, "state": f"{record.STATES}/step-4.json"}
    )
    assert resumed.state.modules == later != trainer.state.modules


async def test_a_resumed_run_carries_the_modules_it_kept_not_the_blueprints(
    tmp_path: Path, fakes: tuple[FakeProxy, FakeService], harbor: Harbor
) -> None:
    harbor(Alternating())
    trained = CARRYING.replace('model = "m"\nprovider = "openrouter"', f'model = "{MODEL}"')
    home = _carrying(tmp_path, trained + "learning_rate = 2e-5\n")
    twice = (_batch(tmp_path, "alpha"), {"attempts": 2})  # scored 1 and 0, so it updates
    directory, _ = await _roll(home, tmp_path / "runs", twice, update=True)
    shutil.rmtree(home / "modules")
    config, row = reopened(directory)
    saved = Trainer.State.model_validate(record.read_json(directory / row["state"]))
    assert row["tag"] == "final" and saved.modules == Modules.read(VALID), "in the state, whole"
    with Trainer(config, directory) as resumed:
        resumed.restore(directory, row)
        assert resumed.state.modules == Modules.read(VALID)


@pytest.mark.parametrize(
    ("modules", "reason"),
    [
        (None, "no modules at"),
        (FIXTURES / "bare", "'solving' cannot be delivered as a skill: it does not open with"),
        (FIXTURES / "unknown", "'skills/solving' holds no SKILL.md"),
    ],
)
def test_modules_the_run_cannot_read_fail_it_before_anything_is_kept_or_rolled_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
    modules: Path | None,
    reason: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = harbor(FakeTrials())
    home = _carrying(tmp_path, modules=modules)
    result = CliRunner().invoke(app, ["run", str(home), "--root", str(tmp_path / "runs")])
    assert result.exit_code == 1 and isinstance(result.exception, Inadmissible)
    (directory,) = (tmp_path / "runs").iterdir()
    note = record.read_json(directory / record.RUN)["process"]
    assert note["failed"] is True and note["error"].startswith("Inadmissible: ")
    assert reason in note["error"]
    if modules is None:
        assert str(home / "modules") in note["error"]
    assert not (directory / record.MODULES).exists() and runner.configs == []


def test_a_reserved_kind_stops_the_trainer_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, modules=None)
    write({"lakehouse/server.py": "print('x')\n"}, into=home / "modules" / "tools")
    directory, config = open_run(home, root=tmp_path / "runs")
    with pytest.raises(Unsupported) as refused:
        Trainer(config, directory)
    assert str(refused.value) == (
        "'tools' is for tool modules, and tool modules are not supported in this version"
    )
    assert not (directory / record.MODULES).exists()
