"""`show`'s rows at the console's width: a table when it fits, and each row as an indented
`key  value` block when the table would be wider, since rich would otherwise fold every
cell of a fifteen-key job row to an ellipsis at eighty columns."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from shipyard import record, show
from shipyard.cli import app
from shipyard.run import Run

DAPO = Path(__file__).parent / "blueprints" / "dapo"
runner = CliRunner()
#: A job row as a served run writes it now: fifteen keys with `at`.
ROW = {
    "at": "2026-10-07T20:57:50.871956+00:00",
    "job": "02-eval-tinker-docker__eTNXY25-0000",
    "purpose": "rollout",
    "trials": 2,
    "batch": 0,
    "tasks": 1,
    "graded": 2,
    "served": "tinker",
    "bridged": 2,
    "party": "tinker",
    "input_tokens": 1992,
    "cache_tokens": 4224,
    "output_tokens": 727,
    "sandbox": "docker",
    "sandbox_seconds": 131.392899,
}


def printed(monkeypatch: pytest.MonkeyPatch, width: int, rows: list[dict]) -> str:
    out = io.StringIO()
    monkeypatch.setattr(show, "console", Console(file=out, width=width, color_system=None))
    show.print_rows("jobs", {"count": len(rows), "rows": rows}, None)
    return out.getvalue()


def test_a_fifteen_key_row_is_a_block_at_eighty_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(ROW) == 15
    text = printed(monkeypatch, 80, [ROW, {**ROW, "job": "next-0001", "masked": {"timeout": 1}}])
    lines = text.splitlines()
    assert lines[0] == "jobs  2 rows"
    assert "…" not in text and "┃" not in text and "│" not in text
    assert lines[1] == "  at               2026-10-07T20:57:50"
    assert "  job              02-eval-tinker-docker__eTNXY25-0000" in lines
    assert "  sandbox_seconds  131.392899" in lines and "  bridged          2" in lines
    assert lines.index("") > lines.index("  sandbox_seconds  131.392899"), "a gap between rows"
    assert '  masked           {"timeout": 1}' in lines and "  job              next-0001" in lines
    assert all(len(line) <= 80 for line in lines)


def test_the_table_is_kept_when_it_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    wide = printed(monkeypatch, 400, [ROW])
    assert "┃" in wide and "sandbox_seconds" in wide.splitlines()[2]
    small = printed(monkeypatch, 80, [{"name": "ckpt-1", "step": 1}])
    assert "┃" in small and "ckpt-1" in small


def test_show_at_eighty_columns_prints_the_jobs_as_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Run.open(DAPO, root=tmp_path) as run:
        record.append(
            run.directory / record.JOBS, {key: v for key, v in ROW.items() if key != "at"}
        )
    monkeypatch.setenv("COLUMNS", "80")
    shown = runner.invoke(app, ["show", run.id, "--root", str(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert "  purpose          rollout" in shown.output and "…" not in shown.output
    monkeypatch.setenv("COLUMNS", "400")
    shown = runner.invoke(app, ["show", run.id, "--root", str(tmp_path)])
    header = next(line for line in shown.output.splitlines() if "purpose" in line)
    assert "job" in header and "sandbox_seconds" in header


def test_the_process_section_names_the_proxy_the_backend_and_a_smoke_only_when_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    with Run.open(DAPO, root=tmp_path, smoke=True) as run:
        run.note(proxy={"placement": "tunnel", "origin": "https://w.trycloudflare.com"})
    shown = runner.invoke(app, ["show", run.id, "--root", str(tmp_path)]).output
    assert '  proxy      {"placement": "tunnel", "origin": "https://w.trycloudflare.com"}' in shown
    assert "  smoke      true" in shown
    assert "  backend    https://" in shown, "a served run notes its tinker_base_url"
    with Run.open(Path(__file__).parent / "blueprints" / "evaluate", root=tmp_path) as plain:
        pass
    shown = runner.invoke(app, ["show", plain.id, "--root", str(tmp_path)]).output
    assert "proxy" not in shown and "smoke" not in shown and "backend" not in shown
