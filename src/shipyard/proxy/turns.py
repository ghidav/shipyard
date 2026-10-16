"""System-role messages inside the message list are read as user turns where they stand.

Claude Code's wire puts system-role messages inside `messages`: one block of agent text
after the first user message and, on every call, a new block at the end with the tokens
left and any note about a file that changed on disk. The cookbook's Anthropic parser folds
them all into the system prompt, so a block appended at the end of the conversation lands
at the front of the prompt, the previous prompt is no longer a prefix, and the bridge
declines. The cookbook's OpenAI parser refuses such a block outright when the request
carries tools. Read as user turns in place, as prime-rl's Anthropic dialect reads them,
each call extends the previous one. One system prompt remains: the request's own `system`
on the Anthropic wire, and the system messages that open the list on the OpenAI wire."""

from __future__ import annotations

from typing import Any

from shipyard.proxy import cookbook

TAG = "turns"


def install() -> None:
    """Wrap the cookbook's two parsers, once per process."""
    if cookbook.wrapped_by("_parse_anthropic", TAG):
        return
    anthropic = cookbook.private("_parse_anthropic")
    openai = cookbook.private("_parse_openai")

    def parse_anthropic(body: dict[str, Any]) -> Any:
        return anthropic(as_user_turns(body))

    def parse_openai(body: dict[str, Any]) -> Any:
        return openai(as_user_turns(body, keep_opening=True))

    cookbook.patch(
        _parse_anthropic=cookbook.tagged(parse_anthropic, anthropic, TAG),
        _parse_openai=cookbook.tagged(parse_openai, openai, TAG),
    )


def as_user_turns(body: dict[str, Any], *, keep_opening: bool = False) -> dict[str, Any]:
    """Return the body with every system-role message in its list given the user role.

    The body is returned unchanged when it has no such message. With `keep_opening`, the
    system messages at the start of the list are kept, since on the OpenAI wire they are
    the system prompt. The top-level `system` is untouched."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    start = 0
    if keep_opening:
        while start < len(messages) and _is_system(messages[start]):
            start += 1
    if not any(_is_system(m) for m in messages[start:]):
        return body
    turned = [{**m, "role": "user"} if _is_system(m) else m for m in messages[start:]]
    return {**body, "messages": [*messages[:start], *turned]}


def _is_system(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"
