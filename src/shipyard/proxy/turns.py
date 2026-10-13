"""System-role messages inside the message list become user turns where they stand.

Claude Code's wire puts system-role messages inside `messages`: one block of agent text
after the first user message, and on every call a new block at the end with the tokens
left and any note about a file that changed on disk. The cookbook's parser folds them all
into the system prompt, so a block appended at the end of the conversation lands at the
front of the prompt, the previous prompt is no longer its prefix, and the bridge declines.
Read as user turns in place, as prime-rl's Anthropic dialect reads them, each call extends
the previous one. The request's own `system` stays the one system message."""

from __future__ import annotations

from typing import Any

from shipyard.proxy import cookbook

TAG = "turns"


def install() -> None:
    """Wrap the cookbook's Anthropic parser, once per process."""
    if cookbook.wrapped_by("_parse_anthropic", TAG):
        return
    anthropic = cookbook.private("_parse_anthropic")

    def parse_anthropic(body: dict[str, Any]) -> Any:
        return anthropic(as_user_turns(body))

    cookbook.patch(_parse_anthropic=cookbook.tagged(parse_anthropic, anthropic, TAG))


def as_user_turns(body: dict[str, Any]) -> dict[str, Any]:
    """The body with every system-role message of its list given the user role, or the
    body itself when it has none. The top-level `system` is untouched."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(_is_system(m) for m in messages):
        return body
    return {**body, "messages": [{**m, "role": "user"} if _is_system(m) else m for m in messages]}


def _is_system(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"
