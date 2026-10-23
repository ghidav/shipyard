"""A task's hooks: the functions its `hooks.py` defines, each named after an event of a trial,
which Harbor calls at that event of every trial of the task, such as setting up an outside
system the task uses before the agent starts, and tearing it down once the trial is done.

They run on this machine, in the run's process. Harbor's own rules apply: a hook that raises at
`start` skips `end`; one that raises at `environment_start` fails its trial, which is masked,
and `end` still runs; one that raises at `end` leaves the trial's grade as it was."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType

from harbor.trial.hooks import TrialEvent, TrialHookEvent

logger = logging.getLogger(__name__)

#: The file a task's hooks live in, beside its `task.toml`.
HOOKS_FILE = "hooks.py"
#: The events a hook is named after, by name: Harbor's `environment-start` is `environment_start`.
EVENTS = {event.value.replace("-", "_"): event for event in TrialEvent}

Hook = Callable[[TrialHookEvent], Awaitable[None]]


def hooks_of(task: Path) -> Mapping[TrialEvent, Hook]:
    """Return the hooks of the task at `task`, by event, each logging its run and how it ended;
    none for a task without a `hooks.py`. A task's file is loaded once. Raise `ValueError` for a
    file that cannot be loaded, a public function no event is named after, or a hook that is not
    `async`."""
    return _loaded(Path(task).resolve())


@cache
def _loaded(task: Path) -> Mapping[TrialEvent, Hook]:
    file = task / HOOKS_FILE
    if not file.is_file():
        return MappingProxyType({})
    name = "shipyard_hooks_" + hashlib.sha256(str(file).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ValueError(f"{file} cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ValueError(f"{file} could not be loaded: {error!r}") from error
    found: dict[TrialEvent, Hook] = {}
    for attribute, value in vars(module).items():
        # Helpers are private, and what the file imports is not its own.
        if attribute.startswith("_") or not inspect.isfunction(value):
            continue
        if value.__module__ != name:
            continue
        event = EVENTS.get(attribute)
        if event is None:
            raise ValueError(
                f"{file}: {attribute} is named after no event of a trial: a hook is one of "
                f"{', '.join(EVENTS)}, and a helper's name starts with '_'"
            )
        if not inspect.iscoroutinefunction(value):
            raise ValueError(f"{file}: {attribute} is a hook, so it is written `async def`")
        found[event] = _logged(value, attribute)
    return MappingProxyType(found)


def _logged(hook: Hook, name: str) -> Hook:
    """Return `hook`, logging each of its runs and how it ended; a failure is raised as it was."""

    async def run(event: TrialHookEvent) -> None:
        began = time.monotonic()
        try:
            await hook(event)
        except Exception as error:
            took = time.monotonic() - began
            logger.warning(
                "hook %s of %s failed after %.1f s: %r", name, event.trial_name, took, error
            )
            raise
        took = time.monotonic() - began
        logger.info("hook %s of %s done in %.1f s", name, event.trial_name, took)

    return run
