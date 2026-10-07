"""The one module that imports `tinker_cookbook`: every symbol the proxy uses, named once,
so a cookbook rename fails here at import and not inside a request."""

from __future__ import annotations

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

#: The app's private symbols thinking, vision and keepalive wrap: the two parsers, the
#: SSE writer and the 400 type. Named once so a patch of anything else is refused.
PATCHABLE = ("_parse_anthropic", "_parse_openai", "_serve_sse", "_BadRequest")


def private(name: str) -> Any:
    """The cookbook app's private symbol as it stands now, patched or not. Refuses a name
    outside `PATCHABLE` or one the installed cookbook lacks, before a request meets it."""
    if name not in PATCHABLE or not hasattr(_app, name):
        raise RuntimeError(
            f"tinker_cookbook's capture proxy has no patchable {name!r}; shipyard wraps "
            f"only {', '.join(PATCHABLE)}, and the installed version must carry each."
        )
    return getattr(_app, name)


def patch(**replacements: Any) -> None:
    """Replace private symbols of the cookbook app in place: thinking, vision and keepalive
    install their wrappers through this, so nothing else writes into the cookbook."""
    for name in replacements:
        private(name)
    for name, value in replacements.items():
        setattr(_app, name, value)


def tagged(wrapper: Any, inner: Any, tag: str) -> Any:
    """A wrap marked with the module that made it and the symbol it wraps, so a second
    install finds the first whatever was wrapped over it since."""
    wrapper.shipyard_wrap, wrapper.__wrapped__ = tag, inner
    return wrapper


def wrapped_by(name: str, tag: str) -> bool:
    """Whether a wrap tagged `tag` already sits on the private symbol, under any wraps
    made since: an install is once per process, in whichever order the modules asked."""
    found = private(name)
    while found is not None:
        if getattr(found, "shipyard_wrap", None) == tag:
            return True
        found = getattr(found, "__wrapped__", None)
    return False


def refused(message: str) -> Exception:
    """The cookbook's own 400 with the message as given. Not Tinker's type: a budget refusal
    must not read as an overflow, which is the message a harness compacts on."""
    return private("_BadRequest")(message)


def too_long(message: str) -> Exception:
    """A refusal in Tinker's type, for a prompt that does not fit: the cookbook classifies
    it by message and answers the "prompt is too long" 400 a harness compacts on."""
    return tinker.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://shipyard.invalid")),
        body=None,
    )


def renderer_for(base_model: str, name: str | None = None) -> Any:
    """The cookbook's renderer for the model, or the one `name` chooses, on the model's
    tokenizer and image processor. Reaches the network for the tokenizer."""
    chosen = name or get_recommended_renderer_name(base_model)
    return get_renderer(
        chosen,
        get_tokenizer(base_model),
        image_processor=image_processor_for(base_model),
        model_name=base_model,
    )


def image_processor_for(base_model: str) -> Any:
    """The model's image processor, or None for a text-only model or one whose processor
    cannot be built here: a renderer without one refuses images rather than drop them."""
    try:
        attributes = get_model_attributes(base_model)
    except Exception:  # noqa: BLE001 - a model the cookbook does not know may still be VL
        attributes = None
    if attributes is not None and not attributes.is_vl:
        return None
    try:
        return get_image_processor(base_model)
    except Exception:  # noqa: BLE001 - no processor is the answer, whatever stood in the way
        return None


def scope_value(key: str) -> str | None:
    """A value of the capture scope the cookbook's handler samples under, or None: the
    trial from the `/r/...` address, the model the harness asked for."""
    value = dict(current_scope()).get(key)
    return str(value) if value else None
