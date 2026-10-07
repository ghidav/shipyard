"""The record's writers: stamped, flushed, tolerant of a cut last line, atomic."""

from __future__ import annotations

import json
from pathlib import Path

from shipyard import record


def test_append_stamps_at_first_and_persists_the_row(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    row = record.append(path, {"seq": 1, "reward": 0.5})
    assert list(row) == ["at", "seq", "reward"]
    assert json.loads(path.read_text(encoding="utf-8")) == row


def test_append_repairs_a_missing_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    path.write_bytes(b'{"job": "first"}')
    record.append(path, {"job": "second"})
    rows = list(record.read(path))
    assert [row["job"] for row in rows] == ["first", "second"]
    assert path.read_bytes().count(b"\n") == 2


def test_read_skips_a_truncated_last_line_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"seq": 1}\n\n{"seq": 2}\n{"seq": 3, "rew', encoding="utf-8")
    assert [row["seq"] for row in record.read(path)] == [1, 2]


def test_read_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert list(record.read(tmp_path / "nothing.jsonl")) == []


def test_write_json_leaves_no_writing_file_and_reads_whole(tmp_path: Path) -> None:
    path = tmp_path / "process.json"
    for step in range(5):
        record.write_json(path, {"step": step, "padding": "x" * 1000})
        assert not path.with_name("process.json.writing").exists()
        assert record.read_json(path) == {"step": step, "padding": "x" * 1000}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["process.json"]


def test_read_json_tolerates_missing_and_broken_files(tmp_path: Path) -> None:
    assert record.read_json(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert record.read_json(broken) == {}
    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2]", encoding="utf-8")
    assert record.read_json(not_object) == {}
