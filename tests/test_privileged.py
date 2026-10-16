"""Tests for placing privileged text in a copy of a sequence, before the task's instruction."""

from __future__ import annotations

from shipyard.proxy.wire import IMAGE_TOKEN, Picture
from shipyard.training.privileged import boundary, placed
from shipyard.types import Sequence
from tests.proxies import ids
from tests.trainers import CharTokenizer


def _sequence(*parts: str | int, completions: tuple[str, ...]) -> Sequence:
    """Build a sequence from text parts and image runs (an int is a run of that many image
    tokens), whose completions are the parts named in `completions`, in order."""
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
    return Sequence(tuple(tokens), tuple(spans), None, tuple(images))


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
    assert copy is not None
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
