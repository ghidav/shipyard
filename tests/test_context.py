"""Tests for `shipyard.models.context`: placing a text in a copy of a sequence, before its
task's instruction."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_proxy.wire import IMAGE_TOKEN, Picture
from proxies import ids
from shipyard.groups import Sequence, Trial
from shipyard.jobs.harnesses import harness
from shipyard.models.context import boundary, insert, placed
from tests.trainers import CharTokenizer


def _sequence(*parts: str | int, completions: tuple[str, ...], task: Path = Path(".")) -> Sequence:
    """Build a sequence of a rollout on `task` from text parts and image runs (an int is a run
    of that many image tokens), whose completions are the parts named in `completions`."""
    tokens: list[int] = []
    spans: list[tuple[int, int]] = []
    images: list[Picture] = []
    for part in parts:
        if isinstance(part, int):
            images.append(Picture(offset=len(tokens), length=part, format="png", data=""))
            tokens += [IMAGE_TOKEN] * part
            continue
        if part in completions:
            spans.append((len(tokens), len(tokens) + len(part)))
        tokens += ids(part)
    trial = Trial("job", task, harness("pi"), task / "trial")
    return Sequence(tuple(tokens), tuple(spans), None, tuple(images), trial)


def _text(sequence: Sequence) -> str:
    return CharTokenizer().decode([one for one in sequence.tokens if one != IMAGE_TOKEN])


def test_the_text_goes_at_the_last_token_boundary_before_the_instruction() -> None:
    table = {1: "<user>", 2: "\n\nWri", 3: "te it", 4: "!"}

    def decode(tokens: list[int]) -> str:
        return "".join(table[one] for one in tokens)

    assert boundary([1, 2, 3, 4], "Write it", decode) == 1, "the token merging the newlines in"
    assert boundary([1, 2, 3, 4], "<user>", decode) == 0
    assert boundary([1, 2, 3], "nowhere", decode) is None


def test_a_placed_copy_shifts_its_completions_and_the_images_after_the_text() -> None:
    sequence = _sequence(
        "<u>", 2, "Do it.", "<a>", "ok", "<t>", 3, "<a>", "fine", completions=("ok", "fine")
    )
    copy = placed(sequence, "Hint. ", "Do it.\n", CharTokenizer())
    assert copy is not None and copy.trial is sequence.trial
    assert _text(copy) == "<u>Hint. Do it.<a>ok<t><a>fine"
    assert [one.offset for one in copy.images] == [3, sequence.images[1].offset + 6]
    assert [copy.tokens[at] for at in copy.targets] == [
        sequence.tokens[at] for at in sequence.targets
    ]


def test_no_copy_without_the_instruction_before_the_first_completion() -> None:
    later = _sequence("<u>go<a>", "ok", "<t>Do it.<a>", "fine", completions=("ok", "fine"))
    assert placed(later, "Hint. ", "Do it.", CharTokenizer()) is None
    missing = _sequence("<u>go<a>", "ok", completions=("ok",))
    assert placed(missing, "Hint. ", "Do it.", CharTokenizer()) is None
    assert placed(missing, "Hint. ", "  \n", CharTokenizer()) is None, "a blank instruction"


def test_insert_reads_each_tasks_instruction_and_says_what_it_could_not_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "instruction.md").write_text("Do it.\n")
    monkeypatch.setattr("shipyard.models.context.tokenizer", lambda model: CharTokenizer())
    asked = _sequence("<u>Do it.<a>", "ok", completions=("ok",), task=tmp_path)
    elsewhere = _sequence("<u>go<a>", "ok", completions=("ok",), task=tmp_path)
    found = insert(
        [asked, asked, asked, elsewhere, asked], ["Hint. ", "", None, "Hint. ", "x" * 99], "m", 20
    )
    first, plain, unread, unmatched, overflowed = found
    assert first is not None and _text(first) == "<u>Hint. Do it.<a>ok"
    assert plain is asked and unread is None and unmatched is None and overflowed is None
    (said,) = [r.getMessage() for r in caplog.records if "read no text" in r.getMessage()]
    assert "1 with no instruction" in said and "1 past the context of 20 tokens" in said


def test_insert_loads_no_tokenizer_without_a_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(model: str) -> None:
        raise AssertionError("no text, no tokenizer")

    monkeypatch.setattr("shipyard.models.context.tokenizer", refuse)
    sequence = _sequence("<u>go<a>", "ok", completions=("ok",))
    assert insert([sequence, sequence], ["", None], "m", None) == [sequence, None]
