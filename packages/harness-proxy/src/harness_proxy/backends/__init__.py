"""The services the proxy samples from, one module each.

A backend has `PREFIX`, which begins the name of any weights it serves; `connect`, which opens
the service; `sampler`, which returns a client on the named weights, or on the base model for
None, whose `sample_async` answers with Tinker's `SampleResponse`; `context_of`, the base
model's context length; and `close`, which lets the service go."""

from __future__ import annotations

from types import ModuleType

from harness_proxy.backends import tinker, together

#: Every backend, by the name a run gives it.
BACKENDS: dict[str, ModuleType] = {"tinker": tinker, "together": together}


def named(name: str) -> ModuleType:
    """Return the backend called `name`. Raise ValueError for an unknown one."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"no backend {name!r}: the proxy samples from {', '.join(sorted(BACKENDS))}"
        ) from None
