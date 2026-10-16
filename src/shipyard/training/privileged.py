"""Privileged sequences: a copy of the policy's sequence with text placed before the task's
instruction, for a teacher to read before it scores the policy's tokens.

The records keep token ids only, so the text is placed by finding the instruction in the
decoded tokens and inserting the text's own tokens at the token boundary before it. The
copy's completions and images shift by the inserted length, so the teacher's logprobs at
the copy's targets line up with the original's."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence as Values
from dataclasses import replace
from typing import Any

from shipyard.proxy.wire import IMAGE_TOKEN
from shipyard.types import Sequence

#: Characters of the task's instruction, from its start, looked for in the decoded tokens.
PROBE = 200


def placed(sequence: Sequence, text: str, instruction: str, tokenizer: Any) -> Sequence | None:
    """Return a copy of `sequence` with `text` placed before the first `instruction` that
    comes ahead of its first completion, or None when there is no such instruction."""
    probe = instruction.strip()[:PROBE]
    if not probe or not sequence.spans:
        return None
    head = sequence.tokens[: sequence.spans[0][0]]
    at = boundary(head, probe, tokenizer.decode)
    if at is None:
        return None
    return spliced(sequence, at, tuple(tokenizer.encode(text, add_special_tokens=False)))


def boundary(tokens: Values[int], probe: str, decode: Callable[[list[int]], str]) -> int | None:
    """Return the last token boundary at or before where `probe` first starts in the decoded
    `tokens`, or None when it does not appear. Image tokens decode to nothing."""

    def text(end: int) -> str:
        return decode([one for one in tokens[:end] if one != IMAGE_TOKEN])

    start = text(len(tokens)).find(probe)
    if start < 0:
        return None
    low, high = 0, len(tokens)
    while low < high:
        middle = (low + high + 1) // 2
        if len(text(middle)) <= start:
            low = middle
        else:
            high = middle - 1
    return low


def spliced(sequence: Sequence, at: int, inserted: tuple[int, ...]) -> Sequence:
    """Return a copy of `sequence` with `inserted` before position `at`, its completions and
    the images after `at` shifted to match."""
    shift = len(inserted)
    return replace(
        sequence,
        tokens=(*sequence.tokens[:at], *inserted, *sequence.tokens[at:]),
        spans=tuple((start + shift, end + shift) for start, end in sequence.spans),
        images=tuple(
            replace(one, offset=one.offset + shift) if one.offset >= at else one
            for one in sequence.images
        ),
    )
