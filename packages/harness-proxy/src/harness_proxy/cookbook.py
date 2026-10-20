"""Single point of import for the parts of `tinker_cookbook` the proxy uses: the renderers, their
message types and the tokenizers.

Every cookbook symbol the proxy uses is named here, so a rename in the cookbook fails at
import time."""

from __future__ import annotations

import re
from typing import Any

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.renderers import Message as Message
from tinker_cookbook.renderers import ParseTermination as ParseTermination
from tinker_cookbook.renderers import ToolCall as ToolCall
from tinker_cookbook.renderers import ToolSpec as ToolSpec
from tinker_cookbook.renderers import get_renderer as get_renderer
from tinker_cookbook.renderers import get_text_content as get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

#: Renderer names that fix a thinking effort. Inkling's renderer takes the effort when it
#: builds each prompt, so the levels are named here, as the cookbook names Qwen3.8's:
#: `tml_v0_effort_0.7` is `tml_v0` at 0.7. Plain `tml_v0` uses the renderer's default.
EFFORT_RENDERER = re.compile(r"^(tml_v0)_effort_(\d*\.?\d+)$")


class AtEffort:
    """Renderer wrapper that builds every prompt at a fixed thinking effort.

    All other attributes are delegated to the inner renderer."""

    def __init__(self, inner: Any, effort: float) -> None:
        self._inner = inner
        self.effort = effort

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("effort", self.effort)
        return self._inner.build_generation_prompt(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def renderer_for(base_model: str, name: str | None = None) -> Any:
    """Build the cookbook renderer `name`, or the model's recommended one, for `base_model`.

    The renderer uses the model's tokenizer, fetched over the network, and its image
    processor. `tml_v0_effort_<e>` builds `tml_v0` at effort `e`, which must be in [0, 1)."""
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
    """Return the model's image processor, or None for a text-only model or a failed build.

    A renderer without an image processor refuses images."""
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
