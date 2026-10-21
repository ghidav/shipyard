"""Tests for rewriting in-list system messages, which Claude Code sends, as user turns."""

from __future__ import annotations

from typing import Any

import pytest

from harness_proxy.chat import Refused
from harness_proxy.dialects import anthropic, openai
from proxies import BridgeRenderer, FakeSampler, ask, ask_anthropic, chars, ids, only, server


@pytest.fixture
def parsers():
    """The two wires' readers."""
    return anthropic.read, openai.read


def _claude_code_body() -> dict[str, Any]:
    """Anthropic body shaped like Claude Code's: a system prompt, then agent text and a
    per-call block as system-role messages inside the list."""
    return {
        "model": "m",
        "max_tokens": 8,
        "system": [{"type": "text", "text": "Be brief."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is here?"}]},
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "# Environment\nA sandbox.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Listing."}]},
            {"role": "user", "content": [{"type": "text", "text": "go on"}]},
            {
                "role": "system",
                "content": [{"type": "text", "text": "<total_tokens>9 left</total_tokens>"}],
            },
        ],
    }


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": "ls", "description": "List", "parameters": {"type": "object"}},
    }
]


def _openai_body() -> dict[str, Any]:
    """The same conversation as an OpenAI body, where two system messages open the list."""
    return {
        "model": "m",
        "tools": OPENAI_TOOLS,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": [{"type": "text", "text": "# Environment\nA sandbox."}]},
            {"role": "user", "content": "What is here?"},
            {"role": "assistant", "content": "Listing."},
            {"role": "user", "content": "go on"},
            {"role": "system", "content": "<total_tokens>9 left</total_tokens>"},
        ],
    }


def test_in_list_system_messages_are_user_turns_in_place(parsers) -> None:
    anthropic, _ = parsers
    parsed = anthropic(_claude_code_body())
    assert parsed.system == "Be brief."
    assert [(m["role"], m["content"]) for m in parsed.messages] == [
        ("user", "What is here?"),
        ("user", "# Environment\nA sandbox."),
        ("assistant", "Listing."),
        ("user", "go on"),
        ("user", "<total_tokens>9 left</total_tokens>"),
    ]


def test_on_the_openai_wire_the_opening_system_messages_stay(parsers) -> None:
    """With tools, the system messages that open the list form the system prompt the renderer
    writes the tools into. A later one is a user turn, not an error."""
    _, openai = parsers
    parsed = openai(_openai_body())
    assert parsed.system == "Be brief.\n\n# Environment\nA sandbox."
    assert [(m["role"], m["content"]) for m in parsed.messages] == [
        ("user", "What is here?"),
        ("assistant", "Listing."),
        ("user", "go on"),
        ("user", "<total_tokens>9 left</total_tokens>"),
    ]


def test_without_them_the_roles_are_as_sent(parsers) -> None:
    read_anthropic, read_openai = parsers
    plain = read_anthropic({"messages": [{"role": "user", "content": "go"}]})
    assert [m["role"] for m in plain.messages] == ["user"] and plain.system is None
    opened = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]}
    assert [m["role"] for m in read_openai(opened).messages] == ["system", "user"]
    assert [m["role"] for m in read_anthropic(opened).messages] == ["user", "user"], (
        "the Anthropic wire keeps none"
    )
    with pytest.raises(Refused, match="non-empty list"):
        read_anthropic({"messages": "x"})


def test_the_top_level_system_stays_and_the_input_is_not_modified(parsers) -> None:
    read_anthropic, _ = parsers
    body = {"system": "Be brief.", "messages": [{"role": "system", "content": "late"}]}
    call = read_anthropic(body)
    assert call.system == "Be brief."
    assert [(m["role"], m["content"]) for m in call.messages] == [("user", "late")]
    assert body["messages"] == [{"role": "system", "content": "late"}]


async def test_a_call_that_ends_with_a_new_block_extends_the_previous_prompt() -> None:
    """The second call appends a tool result and a new system-role block, as Claude Code
    does after a file changes on disk. The block becomes the closing user turn of the tail,
    so the new prompt is the previous prompt and sampled tokens plus that tail."""
    sampler = FakeSampler(["CALL ls {}", "two files"])
    tools = [{"name": "ls", "description": "List", "input_schema": {"type": "object"}}]
    body = {"system": "Be brief.", "tools": tools}
    agent = {"role": "system", "content": [{"type": "text", "text": "# Environment\nA sandbox."}]}
    note = "Note: a.txt changed on disk.\n\n<total_tokens>9 left</total_tokens>"
    async with server(sampler, renderer=BridgeRenderer()) as started:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "look"}, agent]
        first = await ask_anthropic(started, "t", messages=messages, **body)
        assert first.status_code == 200, first.text
        said = first.json()["content"]
        call = next(block for block in said if block["type"] == "tool_use")
        messages += [
            {"role": "assistant", "content": said},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "a b"}],
            },
            {"role": "system", "content": [{"type": "text", "text": note}]},
        ]
        second = await ask_anthropic(started, "t", messages=messages, **body)
        assert second.status_code == 200, second.text
        one, two = only(started).records_for("t")
    assert one.prompt_token_ids == ids(
        "system:Be brief.<tools:ls>\nuser:look\nuser:# Environment\nA sandbox.\n>"
    )
    assert two.bridged
    prefix = (*one.prompt_token_ids, *one.completion_token_ids)
    assert two.prompt_token_ids[: len(prefix)] == prefix
    tail = f"\x00\ntool[{call['id']}]:a b\nuser:{note}\n>"
    assert chars(two.prompt_token_ids[len(prefix) :]) == tail
    assert sampler.prompts[1].to_ints() == list(two.prompt_token_ids)


async def test_the_same_on_the_openai_wire() -> None:
    """A harness on the OpenAI wire that appends a system message after a tool result gets
    the same treatment: the message becomes the closing user turn of the tail and the call
    bridges."""
    sampler = FakeSampler(["CALL ls {}", "two files"])
    note = "Note: a.txt changed on disk."
    async with server(sampler, renderer=BridgeRenderer()) as started:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "look"},
        ]
        first = await ask(started, "t", messages=messages, tools=OPENAI_TOOLS)
        assert first.status_code == 200, first.text
        said = first.json()["choices"][0]["message"]
        (call,) = said["tool_calls"]
        messages += [
            said,
            {"role": "tool", "tool_call_id": call["id"], "content": "a b"},
            {"role": "system", "content": note},
        ]
        second = await ask(started, "t", messages=messages, tools=OPENAI_TOOLS)
        assert second.status_code == 200, second.text
        one, two = only(started).records_for("t")
    assert one.prompt_token_ids == ids("system:Be brief.<tools:ls>\nuser:look\n>")
    assert two.bridged
    prefix = (*one.prompt_token_ids, *one.completion_token_ids)
    assert two.prompt_token_ids[: len(prefix)] == prefix
    tail = f"\x00\ntool[{call['id']}]:a b\nuser:{note}\n>"
    assert chars(two.prompt_token_ids[len(prefix) :]) == tail
    assert sampler.prompts[1].to_ints() == list(two.prompt_token_ids)
