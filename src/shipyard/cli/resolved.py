"""The report `check` prints: the blueprint as TOML with defaults filled in, and its findings.

For a gradient recipe, a comment line after the TOML says what the recipe name resolves to.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipyard.cli.preflight import Finding, findings
from shipyard.config import Blueprint, ConfigError, load

#: Keys TOML accepts unquoted; any other key is written as a quoted string.
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Report:
    """Result of `check` on a blueprint.

    `config` is None when the blueprint did not load. `resolution` is the comment line saying
    what a gradient recipe resolves to, and None for any other recipe.
    """

    config: dict[str, Any] | None
    resolution: str | None
    findings: list[Finding]

    @property
    def blocked(self) -> bool:
        return any(found.level == "blocked" for found in self.findings)

    def text(self) -> str:
        """Return the resolved TOML followed by the resolution line, or "" if nothing loaded."""
        if self.config is None:
            return ""
        toml = as_toml(self.config)
        return f"{toml}{self.resolution}\n" if self.resolution else toml


def report(blueprint: Path, trainer: Any = None) -> Report:
    """Load a blueprint and report its resolved config, resolution line and findings.

    When `trainer` is given, the resolution line is computed for it, and an `ok` finding notes
    when it defines its own `secrets`. A blueprint that does not load yields only its load
    errors, as blocking findings.
    """
    from shipyard.training.recipes import resolution
    from shipyard.training.trainer import Trainer

    try:
        loaded = load(blueprint)
    except ConfigError as error:
        return Report(None, None, [Finding("blocked", problem) for problem in error.problems])
    found = findings(loaded)
    if trainer is not None and trainer.secrets is not Trainer.secrets:
        found.append(Finding("ok", f"secrets from {trainer.__name__}.secrets"))
    return Report(resolved(loaded), resolution(loaded.recipe, trainer), found)


def resolved(loaded: Blueprint) -> dict[str, Any]:
    """Return every blueprint table as plain data with defaults filled in, in schema order.

    Optional tables that are unset are left out. `kind` comes first in the recipe table, as in
    the file. For a served model, `renderer` is the renderer the proxy will load.
    """
    dumped = {
        key: value
        for key, value in loaded.model_dump(mode="json").items()
        if value and any(one is not None for one in value.values())
    }
    recipe = dumped["recipe"]
    dumped["recipe"] = {"kind": recipe["kind"], **{k: v for k, v in recipe.items() if k != "kind"}}
    if loaded.agent.served:
        dumped["rollout"]["renderer"] = served_renderer(loaded)
    return dumped


def served_renderer(loaded: Blueprint) -> str:
    """Return the renderer the proxy will load: `[rollout] renderer`, else the model's own."""
    return loaded.rollout.renderer or recommended_renderer(loaded.agent.model)


def recommended_renderer(model: str) -> str:
    """Return the Tinker cookbook's renderer for `model`, read from its model table offline.

    Returns "" for a model the table does not know; the proxy refuses such a model at start.
    """
    from tinker_cookbook.model_info import get_recommended_renderer_name

    try:
        return get_recommended_renderer_name(model)
    except Exception:  # noqa: BLE001 - the proxy's start is where an unknown model is refused
        return ""


def as_toml(config: Mapping[str, Any]) -> str:
    """Render a resolved config as TOML.

    Each top-level key becomes a table, and each dict value a nested table after its parent's
    keys. None values are left out, since TOML has no null.
    """
    lines: list[str] = []
    for name, body in config.items():
        _table(lines, name, body)
    return "\n".join(lines) + "\n"


def _table(lines: list[str], name: str, body: Mapping[str, Any]) -> None:
    if lines:
        lines.append("")
    lines.append(f"[{name}]")
    nested: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in body.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested.append((f"{name}.{_key(key)}", value))
            continue
        lines.append(f"{_key(key)} = {_value(value)}")
    for sub, inner in nested:
        _table(lines, sub, inner)


def _key(key: str) -> str:
    """Return `key` bare if TOML allows it, else quoted, as in `"Qwen/Qwen3-8B"`."""
    return key if BARE_KEY.fullmatch(key) else json.dumps(key)


def _value(value: Any) -> str:
    """Render one value as TOML: lowercase booleans, floats that keep their point or exponent,
    `inf` and `nan`, basic strings, and inline lists."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return json.dumps(str(value))
