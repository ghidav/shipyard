from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.utils.env import redact_sensitive_value

if TYPE_CHECKING:
    from shipyard.groups import Trial

logger = logging.getLogger(__name__)

#: Separator between the values in a pool's environment variable: commas, whitespace, or both.
SEPARATORS = re.compile(r"[\s,]+")
#: Bytes read from a file to tell text from binary before scrubbing it.
SAMPLE = 8192


class Secrets:
    """Secrets each trial acquires before Harbor starts it and releases after its teardown."""

    async def acquire(self, trial: Trial) -> Mapping[str, str]:
        """Return the environment variables, by name, that `trial` holds for its duration."""
        return {}

    async def release(self, trial: Trial, held: Mapping[str, str]) -> None:
        """Release `held`, the secrets `trial` acquired, once the trial ends for any reason."""


class Pool(Secrets):
    """A fixed set of identities, each leased to one trial at a time."""

    def __init__(self, identities: Sequence[Mapping[str, str]]) -> None:
        self.identities = [dict(one) for one in identities]
        #: Index of the identity each trial holds, keyed by trial directory.
        self._held: dict[Path, int] = {}
        self._free: asyncio.Queue[int] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def of(cls, table: Mapping[str, str], environ: Mapping[str, str] | None = None) -> Pool:
        """Build the pool that `[rollout.secrets]` names."""
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
        """Return the queue of free identities for the running event loop."""
        loop = asyncio.get_running_loop()
        if self._free is None or self._loop is not loop:
            self._loop, self._free = loop, asyncio.Queue()
            held = set(self._held.values())
            for index in range(len(self.identities)):
                if index not in held:
                    self._free.put_nowait(index)
        return self._free


def pooled(environ: Mapping[str, str], source: str) -> list[str]:
    """Return the values in the environment variable `source`, split on commas or whitespace."""
    return [one for one in SEPARATORS.split(environ.get(source, "").strip()) if one]


def scrub(directory: Path, held: Mapping[str, str]) -> None:
    """Replace each value in `held` with `${NAME}` in every text file under `directory`."""
    swaps = [(value, f"${{{name}}}") for name, value in held.items() if value]
    swaps += [
        (redact_sensitive_value(value), f"${{{name}}}")
        for name, value in held.items()
        if len(value) > 8
    ]
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
            try:
                path.write_text(clean, encoding="utf-8")
            except OSError:
                try:
                    path.unlink()
                except OSError:
                    logger.error("%s holds a secret it could not be scrubbed of or removed", path)
