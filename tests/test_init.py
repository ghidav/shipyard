"""Tests for `shipyard init`: the recipes and the env template it copies into the directory it
runs in, and what it keeps."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from shipyard.cli import app, initialized, shipped

RECIPES = Path(__file__).parents[1] / "recipes"


def files(root: Path) -> set[str]:
    """Every file under `root`, as relative paths, leaving out bytecode and hidden files."""
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    }


def test_init_copies_every_recipe_and_the_env_template(tmp_path: Path) -> None:
    recipes, env = shipped()
    copied, kept, wrote = initialized(tmp_path)
    folders = sorted(path.name for path in recipes.iterdir() if path.is_dir())
    assert copied == [name for name in folders if not name.startswith((".", "_"))]
    assert kept == [] and wrote
    assert files(tmp_path / "recipes") == files(recipes), "every file, and no bytecode"
    assert (tmp_path / ".env.example").read_text() == env.read_text()


def test_init_again_keeps_what_is_already_there(tmp_path: Path) -> None:
    """A recipe already there is left as it is, edits and all, and so are both env files."""
    initialized(tmp_path)
    mine = tmp_path / "recipes" / "dapo" / "run.toml"
    mine.write_text("# mine\n")
    (tmp_path / ".env.example").write_text("# mine\n")
    (tmp_path / ".env").write_text("TINKER_API_KEY=mine\n")
    copied, kept, wrote = initialized(tmp_path)
    assert copied == [] and "dapo" in kept and not wrote
    assert mine.read_text() == "# mine\n"
    assert (tmp_path / ".env.example").read_text() == "# mine\n"
    assert (tmp_path / ".env").read_text() == "TINKER_API_KEY=mine\n"


def test_the_cli_init_says_what_a_first_run_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert "copied recipes/" in result.output and "wrote .env.example" in result.output
    assert "shipyard run recipes/dapo" in result.output
    assert (tmp_path / "recipes" / "dapo" / "trainer.py").is_file()


@pytest.mark.parametrize(
    "recipe", sorted(one.name for one in RECIPES.iterdir() if (one / "run.toml").is_file())
)
def test_each_recipe_says_in_its_readme_what_it_needs(recipe: str) -> None:
    readme = (RECIPES / recipe / "README.md").read_text(encoding="utf-8")
    assert f"shipyard run recipes/{recipe}" in readme and "**Needs**" in readme
