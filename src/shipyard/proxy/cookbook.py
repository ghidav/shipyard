"""Imports `tinker_cookbook` for the proxy. Every symbol the proxy uses is named here, so a
cookbook rename fails at import time."""

from __future__ import annotations

import re
from typing import Any

import httpx
import tinker
from tinker_cookbook.capture.proxy import app as _app
from tinker_cookbook.capture.proxy.app import ProxyDeps as ProxyDeps
from tinker_cookbook.capture.proxy.app import make_app as make_app
from tinker_cookbook.capture.scope import current_scope
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import ParseTermination as ParseTermination
from tinker_cookbook.renderers import ToolCall as ToolCall
from tinker_cookbook.renderers import get_renderer as get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

#: The cookbook app's private symbols that thinking, vision and keepalive wrap: the two
#: parsers, the SSE writer and the 400 type. A patch of any other name is refused.
PATCHABLE = ("_parse_anthropic", "_parse_openai", "_serve_sse", "_BadRequest")


def private(name: str) -> Any:
    """The cookbook app's private symbol as it currently stands, patched or not. Raises
    RuntimeError for a name outside `PATCHABLE` or one the installed cookbook lacks."""
    if name not in PATCHABLE or not hasattr(_app, name):
        raise RuntimeError(
            f"tinker_cookbook's capture proxy has no patchable {name!r}. shipyard patches "
            f"{', '.join(PATCHABLE)}, and the installed version must define each."
        )
    return getattr(_app, name)


def patch(**replacements: Any) -> None:
    """Replace private symbols of the cookbook app in place. Thinking, vision and keepalive
    install their wrappers through this function."""
    for name in replacements:
        private(name)
    for name, value in replacements.items():
        setattr(_app, name, value)


def tagged(wrapper: Any, inner: Any, tag: str) -> Any:
    """Mark `wrapper` with `tag` and with the symbol it wraps (`__wrapped__`), and return
    it. A later install finds the mark even when other wraps sit over it."""
    wrapper.shipyard_wrap, wrapper.__wrapped__ = tag, inner
    return wrapper


def wrapped_by(name: str, tag: str) -> bool:
    """Whether a wrap tagged `tag` is already on the private symbol, possibly under later
    wraps. Each module installs once per process, in any order."""
    found = private(name)
    while found is not None:
        if getattr(found, "shipyard_wrap", None) == tag:
            return True
        found = getattr(found, "__wrapped__", None)
    return False


def refused(message: str) -> Exception:
    """The cookbook's 400 with `message`. Unlike `too_long`, it does not use Tinker's error
    type, so a budget refusal is not mistaken for a context overflow, which a harness
    compacts on."""
    return private("_BadRequest")(message)


def too_long(message: str) -> Exception:
    """A refusal in Tinker's error type for a prompt that does not fit. The cookbook
    classifies it by message and answers with the "prompt is too long" 400 that a harness
    compacts on."""
    return tinker.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://shipyard.invalid")),
        body=None,
    )


#: Inkling's renderer takes its thinking effort when it builds each prompt, so its levels are
#: named here, as the cookbook names Qwen3.8's: `tml_v0_effort_0.7` is `tml_v0` at 0.7.
#: Plain `tml_v0` builds at the renderer's own default.
EFFORT_RENDERER = re.compile(r"^(tml_v0)_effort_(\d*\.?\d+)$")


class AtEffort:
    """A renderer that builds every prompt at one thinking effort. Everything else is the
    inner renderer's."""

    def __init__(self, inner: Any, effort: float) -> None:
        self._inner = inner
        self.effort = effort

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("effort", self.effort)
        return self._inner.build_generation_prompt(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def renderer_for(base_model: str, name: str | None = None) -> Any:
    """The cookbook renderer for the model, or the renderer named `name`, built on the
    model's tokenizer and image processor. `tml_v0_effort_<e>` is `tml_v0` at effort `e`.
    Fetches the tokenizer over the network."""
    chosen = name or get_recommended_renderer_name(base_model)
    effort = None
    if matched := EFFORT_RENDERER.match(chosen):
        chosen, effort = matched.group(1), float(matched.group(2))
        if not 0.0 <= effort < 1.0:
            raise ValueError(f"renderer {name}: the effort must be in [0, 1), got {effort:g}")
    renderer = get_renderer(
        chosen,
        get_tokenizer(base_model),
        image_processor=image_processor_for(base_model),
        model_name=base_model,
    )
    # Most cookbook renderers drop the history's thinking by default, so the generation
    # prompt is not a prefix of the next render and the bridge cannot extend it. The
    # history keeps what the model sampled, as a bridged prompt does and as the cookbook
    # advises for multi-turn RL.
    if hasattr(renderer, "strip_thinking_from_history"):
        renderer.strip_thinking_from_history = False
    return renderer if effort is None else AtEffort(renderer, effort)


def image_processor_for(base_model: str) -> Any:
    """The model's image processor, or None for a text-only model or one whose processor
    cannot be built here. A renderer without a processor refuses images."""
    try:
        attributes = get_model_attributes(base_model)
    except Exception:  # noqa: BLE001 - a model the cookbook does not know may still be VL
        attributes = None
    if attributes is not None and not attributes.is_vl:
        return None
    try:
        return get_image_processor(base_model)
    except Exception:  # noqa: BLE001 - a failed build means no processor
        return None


def scope_value(key: str) -> str | None:
    """A value from the capture scope that the cookbook's handler samples under, or None.
    The scope holds the trial (from the `/r/...` address) and the model the harness asked
    for."""
    value = dict(current_scope()).get(key)
    return str(value) if value else None
