"""Modules: what a directory of them seeds, how a candidate is kept and delivered, what is
refused and why, and the carrying path: `Run.open` keeps the blueprint's modules once,
`Run.sample` delivers them into the job and names them on Harbor's agent config."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trial.config import AgentConfig
from harbor.utils.templating import render_prompt_template
from typer.testing import CliRunner

from shipyard import record
from shipyard.cli import app
from shipyard.config import Finding, check
from shipyard.modules import (
    KINDS,
    PROMPT,
    PROMPT_FILE,
    RESERVED,
    SKILL_FILE,
    SKILLS_DIR,
    Candidate,
    Inadmissible,
    Module,
    Skill,
    Unsupported,
    deliver,
    keep,
    merge_fields,
    preamble,
    read,
    seed,
    write,
)
from shipyard.rollout import DELIVERED, rollout
from shipyard.run import CARRIED, Run
from tests.trials import FakeTrials, fixture_tasks, write_blueprint

FIXTURES = Path(__file__).parent / "modules"
VALID = FIXTURES / "valid"
BLUEPRINTS = Path(__file__).parent / "blueprints"

#: An evaluate blueprint over the fixture dataset, carrying `modules/` beside it.
CARRYING = """
[model]
name = "m"
provider = "openrouter"
[data]
dataset = "fixture"
batch_size = 2
[rollout]
harness = "pi@0.85.1"
[recipe]
kind = "evaluate"
modules = "modules"
"""


def _skill(body: str, *, name: str = "solving") -> str:
    return f"---\nname: {name}\ndescription: what to do.\n---\n\n{body}\n"


def _module(text: str = _skill("read it twice"), **beside: str) -> Module:
    return Module("solving", "skill", {SKILL_FILE: text, **beside})


def _files(home: Path) -> dict[str, str]:
    return {
        found.relative_to(home).as_posix(): found.read_text(encoding="utf-8")
        for found in sorted(home.rglob("*"))
        if found.is_file()
    }


# ------------------------------------------------------------------- seed and keep


def test_a_valid_directory_seeds_one_component_per_skill() -> None:
    candidate = seed(VALID)
    assert list(candidate.components) == ["solving"] and len(candidate) == 1
    module = candidate.components["solving"]
    assert (module.name, module.kind) == ("solving", "skill")
    assert module.text == (VALID / SKILLS_DIR / "solving" / SKILL_FILE).read_text(encoding="utf-8")
    assert list(module.files) == [SKILL_FILE, "examples/worked.md"]
    assert len(candidate.digest) == 16 and len(module.digest) == 16


def test_the_layout_holds_a_prompt_at_the_top_and_skills_beneath(tmp_path: Path) -> None:
    write({PROMPT_FILE: "Run the tests.\n"}, into=tmp_path)
    shutil.copytree(VALID / SKILLS_DIR, tmp_path / SKILLS_DIR)
    candidate = seed(tmp_path)
    assert list(candidate.components) == [PROMPT, "solving"]
    prompt = candidate.components[PROMPT]
    assert (prompt.name, prompt.kind, prompt.files) == (
        PROMPT,
        "prompt",
        {PROMPT_FILE: "Run the tests.\n"},
    )


@pytest.mark.parametrize("prompted", [False, True])
def test_keep_writes_the_layout_seed_reads_so_a_winner_is_a_seed(
    tmp_path: Path, prompted: bool
) -> None:
    home = tmp_path / "modules"
    shutil.copytree(VALID, home)
    if prompted:
        write({PROMPT_FILE: "Run the tests.\n"}, into=home)
    candidate = seed(home)
    kept = tmp_path / "kept"
    keep(candidate, kept)
    assert _files(kept) == _files(home)
    again = seed(kept)
    assert again == candidate and again.digest == candidate.digest


def test_a_missing_or_empty_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Inadmissible, match="no modules at"):
        seed(tmp_path / "none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(Inadmissible, match="no modules under"):
        seed(tmp_path / "empty")
    # Dot-files and an empty skills directory hold no module.
    (tmp_path / "empty" / ".hidden" / "x").mkdir(parents=True)
    (tmp_path / "empty" / SKILLS_DIR).mkdir()
    with pytest.raises(Inadmissible, match="no modules under"):
        seed(tmp_path / "empty")


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
    """A skill directly under modules/, the layout before skills/, is refused, not guessed."""
    shutil.copytree(VALID, tmp_path / "modules")
    write({entry: _skill("x")}, into=tmp_path / "modules")
    with pytest.raises(Inadmissible, match=re.escape(why) + ": a modules directory holds"):
        seed(tmp_path / "modules")


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


def test_a_skill_without_frontmatter_is_refused_at_the_seed_by_name() -> None:
    with pytest.raises(Inadmissible, match="'solving' cannot be delivered as a skill: it does"):
        seed(FIXTURES / "bare")


def test_a_skill_directory_without_its_marker_is_refused_by_name() -> None:
    with pytest.raises(Inadmissible, match="'skills/solving' holds no SKILL.md"):
        seed(FIXTURES / "unknown")


@pytest.mark.parametrize(("directory", "kind"), list(RESERVED.items()))
def test_a_reserved_directory_says_the_kind_is_not_supported(
    tmp_path: Path, directory: str, kind: str
) -> None:
    write({"lakehouse/server.py": "print('hello')\n"}, into=tmp_path / directory)
    with pytest.raises(Unsupported, match=f"{kind} modules are not supported in this version"):
        seed(tmp_path)
    try:
        seed(tmp_path)
    except Inadmissible as refused:  # an Unsupported is an Inadmissible, with its kind
        assert isinstance(refused, Unsupported) and refused.kind == kind
        assert f"{directory!r}" in str(refused)


def test_the_kinds_table_holds_skills_and_prompts() -> None:
    assert list(KINDS) == ["skill", "prompt"]
    skill, prompt = KINDS["skill"], KINDS["prompt"]
    assert (skill.marker, skill.alone, skill.home("solving")) == (
        SKILL_FILE,
        False,
        "skills/solving",
    )
    assert (prompt.marker, prompt.alone, prompt.home(PROMPT)) == (PROMPT_FILE, True, "")
    assert skill.reference() is None
    assert "before the task's own instruction" in (prompt.reference() or "")
    assert RESERVED == {"tools": "tool", "harness": "harness"}


# ------------------------------------------------------------------------ prompts


def _prompt(text: str = "Use the project's own python.\n", *, name: str = PROMPT) -> Module:
    return Module(name, "prompt", {PROMPT_FILE: text})


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
        seed(tmp_path)


def test_a_prompt_is_its_marker_alone() -> None:
    refused = KINDS["prompt"].check(Module(PROMPT, "prompt", {PROMPT_FILE: "go", "notes.md": "x"}))
    assert refused is not None and "nothing beside it is delivered" in refused


def test_a_skill_may_not_take_the_prompts_name(tmp_path: Path) -> None:
    write({PROMPT_FILE: "go", f"{SKILLS_DIR}/{PROMPT}/{SKILL_FILE}": _skill("x")}, into=tmp_path)
    with pytest.raises(Inadmissible, match="'skills/prompt' takes the name of the PROMPT.md"):
        seed(tmp_path)


def test_a_prompt_is_delivered_as_harbors_prompt_template(tmp_path: Path) -> None:
    """Harbor renders the instruction through the template: the text exactly, a blank line,
    then the instruction. Jinja syntax in the text stays text."""
    text = "Use {{ python }} and {% if x %}this{% endif %}.\n\n"
    fields = deliver(Candidate({PROMPT: _prompt(text)}), tmp_path / "into")
    template = Path(fields["kwargs"]["prompt_template_path"])
    assert template == (tmp_path / "into" / "prompt" / "template.j2").resolve()
    rendered = render_prompt_template(template, "Fix the bug.\n")
    assert rendered == "Use {{ python }} and {% if x %}this{% endif %}.\n\nFix the bug.\n"
    assert rendered == preamble(_prompt(text)) + "Fix the bug.\n"


def test_a_prompt_rides_with_skills_and_lays_over_the_blueprints_kwargs(tmp_path: Path) -> None:
    candidate = Candidate({PROMPT: _prompt(), "solving": _module()})
    fields: dict[str, Any] = {"kwargs": {"version": "1.0", "prompt_template_path": "theirs.j2"}}
    merge_fields(fields, deliver(candidate, tmp_path / "into"))
    assert fields["skills"] == [str((tmp_path / "into" / "skills").resolve())]
    assert fields["kwargs"]["version"] == "1.0"
    assert fields["kwargs"]["prompt_template_path"].endswith("prompt/template.j2")


def test_a_candidate_carries_one_prompt(tmp_path: Path) -> None:
    two = Candidate({"a": _prompt(name="a"), "b": _prompt("Other.\n", name="b")})
    with pytest.raises(Inadmissible, match="a candidate carries one prompt, and this one has a, b"):
        deliver(two, tmp_path / "into")


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
    one = Candidate({"solving": _module()})
    assert one.digest == Candidate({"solving": _module()}).digest
    assert one.digest != Candidate({"planning": _module()}).digest, "the name counts"
    two = one.with_component("planning", _module(_skill("plan", name="planning")))
    assert list(two.components) == ["solving", "planning"] and two.digest != one.digest
    assert list(one.components) == ["solving"], "with_component makes a new candidate"
    changed = one.with_component("solving", _module(_skill("read it three times")))
    assert changed.digest != one.digest and len(changed) == 1


# ----------------------------------------------------------------------- delivery


def test_deliver_names_the_skills_root_harbor_takes_and_writes_every_file(
    tmp_path: Path,
) -> None:
    candidate = seed(VALID)
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


def test_merge_fields_joins_lists_merges_dicts_and_replaces_the_rest() -> None:
    fields: dict[str, Any] = {
        "skills": ["/a"],
        "kwargs": {"version": "1", "turns": 2},
        "name": "pi",
    }
    merge_fields(
        fields, {"skills": ["/b"], "kwargs": {"turns": 3}, "env": {"X": "1"}, "name": "acp:x"}
    )
    assert fields == {
        "skills": ["/a", "/b"],
        "kwargs": {"version": "1", "turns": 3},
        "env": {"X": "1"},
        "name": "acp:x",
    }


async def test_a_kinds_fields_land_on_the_agent_config_beside_the_rollouts_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam a harness or tool kind would use: fields the rollout sets itself merge,
    so such a kind is a class and a table row and needs no rollout edit."""
    monkeypatch.setattr(
        "shipyard.rollout.deliver",
        lambda candidate, into: {
            "name": "other",
            "kwargs": {"turns": 3},
            "env": {"TOOL": "on"},
            "skills": [str(into / "skills")],
        },
    )
    runner = FakeTrials()
    await rollout(
        [tmp_path / "tasks" / "alpha"],
        "run-0000",
        harness="pi@0.85.1",
        model="m",
        sandbox="docker",
        concurrency=1,
        rollouts=1,
        jobs_dir=tmp_path / "jobs",
        env={"KEEP": "1"},
        kwargs={"thinking": "high"},
        run_trial=runner,
        modules=seed(VALID),
    )
    agent = runner.configs[0].agent
    assert agent.name == "other" and agent.model_name == "m"
    assert agent.kwargs == {"version": "0.85.1", "thinking": "high", "turns": 3}
    assert agent.env == {"KEEP": "1", "TOOL": "on"}
    assert agent.skills == [str(tmp_path / "jobs" / "run-0000" / DELIVERED / "skills")]


def test_a_relative_into_is_delivered_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fields = deliver(seed(VALID), Path("m"))
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


def _carrying(tmp_path: Path, text: str = CARRYING, modules: Path | None = VALID) -> Path:
    """A blueprint carrying a copy of `modules` beside it, in a workspace of fixture tasks."""
    fixture_tasks(tmp_path)
    home = write_blueprint(tmp_path, text)
    if modules is not None:
        shutil.copytree(modules, home / "modules")
    return home


def _batch(tmp_path: Path, *names: str) -> list[Path]:
    return [tmp_path / "tasks" / "fixture" / name for name in names]


def _rows(opened: Run) -> list[dict[str, Any]]:
    return list(record.read(opened.directory / record.JOBS))


async def test_open_keeps_the_carried_modules_once_and_sample_delivers_them_per_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path)
    kept: list[Path] = []
    keeping = keep
    monkeypatch.setattr("shipyard.run.keep", lambda c, into: (kept.append(into), keeping(c, into)))
    opened = Run.open(home, root=tmp_path / "runs")
    assert opened.carried is not None and opened.carried == seed(home / "modules")
    carried = opened.directory / record.MODULES / CARRIED
    assert kept == [carried] and _files(carried) == _files(VALID)
    assert seed(carried).digest == opened.carried.digest
    opened.run_trial = runner = FakeTrials()
    first = await opened.sample(_batch(tmp_path, "alpha", "beta"), rollouts=2, index=0)
    second = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=1)
    assert kept == [carried], "the carried copy is written once, at open"
    assert len(runner.configs) == 5
    for config, job in zip(runner.configs, [first.job] * 4 + [second.job], strict=True):
        root = (tmp_path / "jobs" / job / DELIVERED / "skills").resolve()
        assert config.agent.skills == [str(root)]
        assert config.agent.name == "pi" and config.agent.kwargs == {"version": "0.85.1"}
        assert _files(root / "solving") == _files(VALID / SKILLS_DIR / "solving")
    rows = _rows(opened)
    assert [row["modules"] for row in rows] == [opened.carried.digest] * 2
    assert list(rows[0])[:7] == ["at", "job", "purpose", "trials", "batch", "tasks", "modules"]
    assert sorted(p.name for p in (tmp_path / "jobs" / first.job).iterdir()) == sorted(
        [DELIVERED] + [trial.name for trial in first.trials]
    )


async def test_sample_delivers_the_candidate_it_is_handed_and_nothing_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(BLUEPRINTS / "evaluate", root=tmp_path / "runs")
    assert opened.carried is None and not (opened.directory / record.MODULES).exists()
    opened.run_trial = runner = FakeTrials()
    bare = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=0)
    assert runner.configs[0].agent.skills == []
    assert sorted(p.name for p in (tmp_path / "jobs" / bare.job).iterdir()) == [
        bare.trials[0].name
    ], "nothing of ours in a job carrying nothing"
    handed = seed(VALID).with_component("solving", _module(_skill("read it three times")))
    carried = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=1, modules=handed)
    root = (tmp_path / "jobs" / carried.job / DELIVERED / "skills").resolve()
    assert runner.configs[1].agent.skills == [str(root)]
    assert (root / "solving" / SKILL_FILE).read_text() == _skill("read it three times")
    rows = _rows(opened)
    assert "modules" not in rows[0] and rows[1]["modules"] == handed.digest


async def test_a_handed_candidate_wins_over_the_carried_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_carrying(tmp_path), root=tmp_path / "runs")
    opened.run_trial = runner = FakeTrials()
    planning = _module(_skill("plan first", name="planning"))
    other = Candidate({}).with_component("planning", planning)
    assert other.components["planning"].name == "planning", "named by its component"
    rolled = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=0, modules=other)
    root = Path(runner.configs[0].agent.skills[0])
    assert sorted(p.name for p in root.iterdir()) == ["planning"]
    assert _rows(opened)[0]["modules"] == other.digest != opened.carried.digest
    assert rolled.job == f"{opened.id}-0000"


async def test_sample_hands_a_prompt_to_harbor_as_the_agents_prompt_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened = Run.open(_carrying(tmp_path), root=tmp_path / "runs")
    opened.run_trial = runner = FakeTrials()
    prompted = Candidate({}).with_component("guidance", _prompt("Run the tests.\n"))
    rolled = await opened.sample(_batch(tmp_path, "alpha"), rollouts=1, index=0, modules=prompted)
    kwargs = runner.configs[0].agent.kwargs
    template = (tmp_path / "jobs" / rolled.job / DELIVERED / "prompt" / "template.j2").resolve()
    assert kwargs == {"version": "0.85.1", "prompt_template_path": str(template)}
    assert render_prompt_template(template, "Fix it.") == "Run the tests.\n\nFix it."
    assert runner.configs[0].agent.skills == [], "a prompt is not a skill"
    assert _rows(opened)[0]["modules"] == prompted.digest


def test_check_warns_when_a_prompt_replaces_the_blueprints_template(tmp_path: Path) -> None:
    text = CARRYING.replace("[recipe]", '[rollout.kwargs]\nprompt_template_path = "t.j2"\n[recipe]')
    home = write_blueprint(tmp_path, text)
    write({PROMPT_FILE: "Run the tests.\n"}, into=home / "modules")
    found = [one.text for one in check(home) if one.level == "warning"]
    assert any(one.startswith("[rollout.kwargs] prompt_template_path: the prompt") for one in found)
    (home / "modules" / PROMPT_FILE).unlink()
    shutil.copytree(VALID, home / "modules", dirs_exist_ok=True)
    found = [one.text for one in check(home) if one.level == "warning"]
    assert not any("prompt_template_path" in one for one in found), "skills leave it alone"


def test_a_gradient_recipe_carries_and_a_gepa_run_seeds_its_own(tmp_path: Path) -> None:
    text = (BLUEPRINTS / "dapo" / "run.toml").read_text(encoding="utf-8")
    home = _carrying(tmp_path, text.replace("[recipe]", '[recipe]\nmodules = "modules"'))
    opened = Run.open(home, root=tmp_path / "runs")
    assert opened.carried == seed(VALID)
    assert (
        opened.directory / record.MODULES / CARRIED / SKILLS_DIR / "solving" / SKILL_FILE
    ).is_file()
    gepa = Run.open(BLUEPRINTS / "gepa", root=tmp_path / "runs")
    assert gepa.config.recipe.modules == "modules" and gepa.carried is None
    assert not (gepa.directory / record.MODULES).exists()


def test_open_refuses_a_broken_seed_before_writing_anything(tmp_path: Path) -> None:
    home = _carrying(tmp_path, modules=FIXTURES / "bare")
    with pytest.raises(Inadmissible, match="'solving' cannot be delivered as a skill"):
        Run.open(home, root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------- check's findings


def test_check_reports_the_modules_it_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path)
    digest = seed(VALID).digest
    found = check(home)
    assert Finding("ok", f"modules: 1 component(s) under {home / 'modules'} ({digest})") in found
    assert all(f.level == "ok" for f in found)
    assert [f.text for f in found][4:6] == [
        "dataset fixture: 2 tasks under tasks/fixture",
        f"modules: 1 component(s) under {home / 'modules'} ({digest})",
    ], "after the datasets, before the proxy's findings"
    without = write_blueprint(tmp_path / "plain", CARRYING.replace('modules = "modules"\n', ""))
    assert not any("modules" in f.text for f in check(without))


@pytest.mark.parametrize(
    ("modules", "reason"),
    [
        (None, "no modules at"),
        (FIXTURES / "bare", "'solving' cannot be delivered as a skill: it does not open with"),
        (FIXTURES / "unknown", "'skills/solving' holds no SKILL.md"),
    ],
)
def test_check_blocks_a_seed_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modules: Path | None, reason: str
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, modules=modules)
    blocked = [f for f in check(home) if f.level == "blocked"]
    assert len(blocked) == 1 and blocked[0].text.startswith("[recipe] modules: ")
    assert reason in blocked[0].text
    if modules is None:
        assert str(home / "modules") in blocked[0].text


def test_check_blocks_a_reserved_kind_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    home = _carrying(tmp_path, modules=None)
    write({"lakehouse/server.py": "print('x')\n"}, into=home / "modules" / "tools")
    [blocked] = [f.text for f in check(home) if f.level == "blocked"]
    assert blocked == (
        "[recipe] modules: 'tools' is for tool modules, and tool modules are not "
        "supported in this version"
    )


def test_check_prints_the_modules_finding_under_verbose_and_blocks_a_bad_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    home = _carrying(tmp_path)
    result = runner.invoke(app, ["check", str(home)])
    assert result.exit_code == 0, result.output
    assert 'modules = "modules"' in result.output and "ok  " not in result.output
    verbose = runner.invoke(app, ["check", str(home), "--verbose"])
    assert verbose.exit_code == 0 and "ok  modules: 1 component(s) under" in verbose.output
    # The gepa fixture carries a skill under `modules/`; here only its datasets are missing.
    gepa = runner.invoke(app, ["check", str(BLUEPRINTS / "gepa"), "--verbose"])
    assert "ok  modules: 1 component(s) under" in gepa.output
    blocked = [line for line in gepa.output.splitlines() if line.startswith("blocked")]
    assert len(blocked) == 2 and all("no dataset at" in line for line in blocked)
    shutil.rmtree(home / "modules")
    result = runner.invoke(app, ["check", str(home)])
    assert result.exit_code == 1 and "blocked  [recipe] modules: no modules at" in result.output
