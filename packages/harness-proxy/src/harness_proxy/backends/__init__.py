from __future__ import annotations

from types import ModuleType

from harness_proxy.backends import tinker, together

#: Every backend, by the name a run gives it: a module with `PREFIX`, `connect`, `sampler`,
#: `context_of` and `close`.
BACKENDS: dict[str, ModuleType] = {"tinker": tinker, "together": together}


def named(name: str) -> ModuleType:
    """Return the backend called `name`. Raise ValueError for an unknown one."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"no backend {name!r}: the proxy samples from {', '.join(sorted(BACKENDS))}"
        ) from None
