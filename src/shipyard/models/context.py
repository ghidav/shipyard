from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Sequence as Values
from dataclasses import replace
from typing import Any

from harness_proxy.wire import IMAGE_TOKEN
from shipyard.groups import Sequence

logger = logging.getLogger(__name__)

#: Characters of the instruction, from its start, looked for in the decoded tokens.
PROBE = 200


def insert(
    sequences: Values[Sequence], texts: Values[str | None], model: str, limit: int | None
) -> list[Sequence | None]:
    """Return a copy of each sequence with its text placed before its task's instruction."""
    reader = tokenizer(model) if any(texts) else None
    found: list[Sequence | None] = []
    unmatched = overflowed = 0
    for sequence, text in zip(sequences, texts, strict=True):
        if not text:
            found.append(None if text is None else sequence)
            continue
        copy = placed(sequence, text, sequence.trial.instruction, reader)
        if copy is None:
            unmatched += 1
        elif limit and len(copy.tokens) > limit:
            copy, overflowed = None, overflowed + 1
        found.append(copy)
    if unmatched or overflowed:
        logger.warning(
            "%d sequence(s) read no text: %d with no instruction before the first completion, "
            "%d past the context of %s tokens with it",
            unmatched + overflowed,
            unmatched,
            overflowed,
            limit,
        )
    return found


def placed(sequence: Sequence, text: str, instruction: str, reader: Any) -> Sequence | None:
    """Return a copy of `sequence` with `text` before its task's instruction, or None."""
    probe = instruction.strip()[:PROBE]
    if not probe or not sequence.spans:
        return None
    head = sequence.tokens[: sequence.spans[0][0]]
    at = boundary(head, probe, reader.decode)
    if at is None:
        return None
    return spliced(sequence, at, tuple(reader.encode(text, add_special_tokens=False)))


def boundary(tokens: Values[int], probe: str, decode: Callable[[list[int]], str]) -> int | None:
    """Return the last token boundary at or before where `probe` starts in the decoded `tokens`."""

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
    """Return a copy of `sequence` with `inserted` before position `at`, the rest shifted."""
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


def shares_tokenizer(model: str, other: str) -> bool:
    """Return whether two models read the same token ids: the same vocabulary, id for id."""
    return model == other or tokenizer(model).get_vocab() == tokenizer(other).get_vocab()


def tokenizer(model: str) -> Any:
    """Return the model's tokenizer, as the Tinker cookbook loads and caches it."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)
