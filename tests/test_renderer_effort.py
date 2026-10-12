"""Inkling's effort levels as renderer names: `tml_v0_effort_<e>` builds every prompt at `e`."""

from __future__ import annotations

from typing import Any

import pytest

from shipyard.proxy import cookbook


class Inner:
    """A renderer that records what each prompt was built with."""

    strip_thinking_from_history = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.built: list[dict[str, Any]] = []

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> str:
        self.built.append(kwargs)
        return "prompt"


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names: list[str] = []

    def get_renderer(name: str, tokenizer: Any, **_: Any) -> Inner:
        names.append(name)
        return Inner(name)

    monkeypatch.setattr(cookbook, "get_renderer", get_renderer)
    monkeypatch.setattr(cookbook, "get_tokenizer", lambda model: object())
    monkeypatch.setattr(cookbook, "image_processor_for", lambda model: None)
    monkeypatch.setattr(cookbook, "get_recommended_renderer_name", lambda model: "tml_v0")
    return names


def test_an_effort_name_builds_tml_v0_and_every_prompt_at_that_effort(made: list[str]) -> None:
    renderer = cookbook.renderer_for("thinkingmachines/Inkling-Small", "tml_v0_effort_0.7")
    assert made == ["tml_v0"]
    assert isinstance(renderer, cookbook.AtEffort) and renderer.effort == 0.7
    renderer.build_generation_prompt([])
    renderer.build_generation_prompt([], effort=0.2)
    assert renderer._inner.built == [{"effort": 0.7}, {"effort": 0.2}]


def test_the_wrapper_passes_through_what_the_bridge_and_vision_read(made: list[str]) -> None:
    renderer = cookbook.renderer_for("thinkingmachines/Inkling-Small", "tml_v0_effort_0.5")
    assert renderer.strip_thinking_from_history is False
    assert renderer._inner.name == "tml_v0"


def test_plain_and_default_names_are_built_as_they_are(made: list[str]) -> None:
    assert not isinstance(cookbook.renderer_for("m", "tml_v0"), cookbook.AtEffort)
    assert not isinstance(cookbook.renderer_for("m"), cookbook.AtEffort)
    assert made == ["tml_v0", "tml_v0"]


@pytest.mark.parametrize("name", ["tml_v0_effort_1.0", "tml_v0_effort_1.5"])
def test_an_effort_outside_the_renderers_range_is_refused(made: list[str], name: str) -> None:
    with pytest.raises(ValueError, match="effort must be in"):
        cookbook.renderer_for("m", name)
    assert made == []


class Stripping(Inner):
    """A renderer that drops the history's thinking by default, as the cookbook's Qwen3
    and Qwen3.5 renderers do."""

    strip_thinking_from_history = True


class Plain:
    """A renderer with no such setting."""

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> str:
        return "prompt"


def test_the_history_keeps_its_thinking_whatever_the_renderers_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, Any] = {"qwen3_5": Stripping("qwen3_5"), "tml_v0": Plain()}
    monkeypatch.setattr(cookbook, "get_renderer", lambda name, tokenizer, **_: built[name])
    monkeypatch.setattr(cookbook, "get_tokenizer", lambda model: object())
    monkeypatch.setattr(cookbook, "image_processor_for", lambda model: None)
    assert cookbook.renderer_for("Qwen/Qwen3.5-4B", "qwen3_5").strip_thinking_from_history is False
    assert not hasattr(cookbook.renderer_for("m", "tml_v0"), "strip_thinking_from_history")
