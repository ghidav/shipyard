"""The blueprint as `check` prints it: every table with its defaults filled in, as TOML,
and what a gradient recipe's name resolves to, on one comment line after it."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipyard.config import Blueprint, ConfigError, Finding, findings, load


@dataclass(frozen=True)
class Report:
    """What `check` has to say: the resolved config (None when it did not load), the
    recipe's resolution line (None but for a gradient recipe), and the findings."""

    config: dict[str, Any] | None
    resolution: str | None
    findings: list[Finding]

    @property
    def blocked(self) -> bool:
        return any(found.level == "blocked" for found in self.findings)

    def text(self) -> str:
        """The resolved TOML with the resolution line trailing it; "" when nothing loaded."""
        if self.config is None:
            return ""
        toml = as_toml(self.config)
        return f"{toml}{self.resolution}\n" if self.resolution else toml


def report(blueprint: Path) -> Report:
    """`check` over a blueprint path: the problems alone when it does not load."""
    from shipyard.recipes import resolution

    try:
        loaded = load(blueprint)
    except ConfigError as error:
        return Report(None, None, [Finding("blocked", problem) for problem in error.problems])
    return Report(resolved(loaded), resolution(loaded.recipe), findings(loaded))


def resolved(loaded: Blueprint) -> dict[str, Any]:
    """Every table as plain data with the defaults in, in the schema's order; `kind`
    leads the recipe table as it does in the file."""
    dumped = loaded.model_dump(mode="json")
    recipe = dumped["recipe"]
    dumped["recipe"] = {"kind": recipe["kind"], **{k: v for k, v in recipe.items() if k != "kind"}}
    return dumped


def as_toml(config: Mapping[str, Any]) -> str:
    """The resolved config as TOML: one table per top-level key, a nested table per dict
    value after its table's keys, and no line for a None (TOML has no null)."""
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
            nested.append((f"{name}.{key}", value))
            continue
        lines.append(f"{key} = {_value(value)}")
    for sub, inner in nested:
        _table(lines, sub, inner)


def _value(value: Any) -> str:
    """One TOML value: booleans lower-cased, floats with their point or exponent kept,
    inf and nan as TOML spells them, strings as basic strings, lists inline."""
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
