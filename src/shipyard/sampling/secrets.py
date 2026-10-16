"""What a trial's environment holds that no other trial in flight may share: a credential
that is an identity, held by one trial for exactly its duration and scrubbed from what the
trial leaves behind."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shipyard.types import Trial

#: How a pool's variable separates its values: commas, whitespace, or both.
SEPARATORS = re.compile(r"[\s,]+")
#: Bytes of a file read to tell text from binary before it is scrubbed.
SAMPLE = 8192


class Secrets:
    """What each trial acquires before Harbor starts it and releases after its teardown. This
    base gives nothing; `Pool` leases what `[rollout.secrets]` names, and a trainer's
    `secrets` may return any subclass."""

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        """The variables `trial` holds, by name, for exactly its duration. Waits while there
        are none to give."""
        return {}

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        """`trial` is over, however it ended: what it held is free again."""


class Pool(Secrets):
    """Identities by position: the i-th value of every name the pool holds. A trial waits for
    a free one, so no more trials run at once than there are identities."""

    def __init__(self, identities: Sequence[Mapping[str, str]]) -> None:
        if not identities:
            raise ValueError("a pool needs at least one identity")
        self.identities = [dict(one) for one in identities]
        #: By trial directory, the identity it holds.
        self._held: dict[Path, int] = {}
        self._free: asyncio.Queue[int] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def of(cls, table: Mapping[str, str], environ: Mapping[str, str] | None = None) -> Pool:
        """The pool `[rollout.secrets]` names: for each name, the values the variable it
        names holds. Raises if one holds none, or two hold different counts."""
        environ = os.environ if environ is None else environ
        values = {name: pooled(environ, source) for name, source in table.items()}
        if empty := [table[name] for name, found in values.items() if not found]:
            raise ValueError(f"[rollout.secrets]: {', '.join(empty)} unset or empty")
        if len({len(found) for found in values.values()}) > 1:
            counts = ", ".join(f"{table[name]} {len(found)}" for name, found in values.items())
            raise ValueError(f"[rollout.secrets]: the pools differ in size ({counts})")
        size = len(next(iter(values.values())))
        return cls([{name: found[at] for name, found in values.items()} for at in range(size)])

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        index = await self._queue().get()
        self._held[trial.dir] = index
        return self.identities[index]

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        self._queue().put_nowait(self._held.pop(trial.dir))

    def _queue(self) -> asyncio.Queue[int]:
        """The free identities, on the running loop: a queue is bound to the loop it first
        waits on, and a trainer may run on more than one, one after another."""
        loop = asyncio.get_running_loop()
        if self._free is None or self._loop is not loop:
            self._loop, self._free = loop, asyncio.Queue()
            held = set(self._held.values())
            for index in range(len(self.identities)):
                if index not in held:
                    self._free.put_nowait(index)
        return self._free


def pooled(environ: Mapping[str, str], source: str) -> list[str]:
    """The values the variable `source` holds, separated by commas or whitespace."""
    return [one for one in SEPARATORS.split(environ.get(source, "").strip()) if one]


def redacted(value: str) -> str:
    """`value` as Harbor redacts a sensitive one when it saves a trial's config
    (`redact_sensitive_value`, harbor/utils/env.py)."""
    return "****" if len(value) <= 8 else value[:4] + "****" + value[-3:]


def scrub(directory: Path, held: Mapping[str, str]) -> None:
    """Replace every value `held`, and the form Harbor redacts it to, with `${NAME}` in each
    text file under `directory`. A form as short as Harbor's bare `****` is left alone."""
    swaps = [(value, f"${{{name}}}") for name, value in held.items() if value]
    swaps += [(redacted(value), f"${{{name}}}") for name, value in held.items() if len(value) > 8]
    if not swaps or not directory.is_dir():
        return
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with path.open("rb") as handle:
                if b"\0" in handle.read(SAMPLE):
                    continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        clean = text
        for value, name in swaps:
            clean = clean.replace(value, name)
        if clean != text:
            path.write_text(clean, encoding="utf-8")
