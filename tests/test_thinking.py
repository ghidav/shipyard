"""The model's thinking on the wire: handed to the harness in its wire's form, read back
from whatever the harness kept, and bridged only when kept as written. Through the real
app on both wires, renderer and sampler faked; one test runs the real Qwen3 renderer."""

from __future__ import annotations

import json
from typing import Any

import pytest

from shipyard.proxy import cookbook, thinking
from tests.proxies import (
    TOOLS_ANTHROPIC,
    TOOLS_OPENAI,
    BridgeRenderer,
    FakeResponse,
    FakeSequence,
    ScriptedSampler,
    ask,
    ask_anthropic,
    chars,
    endpoint,
)


def _events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        head, data = block.split("\n", 1)
        events.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


# ------------------------------------------------------------------ the Anthropic wire


async def _anthropic_turns(sent_back: Any) -> list[Any]:
    """Two turns of a tool loop: the first reply's blocks, rewritten by `sent_back`, then
    a tool result. Returns the trial's records."""
    sampler = ScriptedSampler("", answers=["{look}CALL ls {}", "{done}two files"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        body = {"system": "be brief", "tools": TOOLS_ANTHROPIC}
        messages: list[dict[str, Any]] = [{"role": "user", "content": "look around"}]
        first = await ask_anthropic(started, "t", messages=messages, **body)
        assert first.status_code == 200, first.text
        blocks = first.json()["content"]
        tool = next(b for b in blocks if b["type"] == "tool_use")
        messages.append({"role": "assistant", "content": sent_back(blocks)})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool["id"], "content": "a b"}],
            }
        )
        second = await ask_anthropic(started, "t", messages=messages, **body)
        assert second.status_code == 200, second.text
        return started.records_for("t")


async def test_the_reply_carries_its_thinking_as_a_signed_block_first() -> None:
    sampler = ScriptedSampler("", answers=["{look}CALL ls {}"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        answered = await ask_anthropic(started, "t", tools=TOOLS_ANTHROPIC)
        blocks = answered.json()["content"]
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["thinking"] == "look"
    assert blocks[0]["signature"] == thinking.signature("look") != ""


async def test_a_reply_without_thinking_carries_no_block() -> None:
    async with endpoint(
        ScriptedSampler("", answers=["plain"]), renderer=BridgeRenderer()
    ) as started:
        blocks = (await ask_anthropic(started, "t")).json()["content"]
    assert blocks == [{"type": "text", "text": "plain"}]


async def test_a_streamed_reply_carries_the_block_at_index_zero() -> None:
    sampler = ScriptedSampler("", answers=["{look}CALL ls {}"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        events = _events(
            (await ask_anthropic(started, "t", tools=TOOLS_ANTHROPIC, stream=True)).text
        )
    names = [name for name, _ in events]
    assert names[:5] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
    ]
    assert events[1][1]["content_block"]["type"] == "thinking"
    assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "look"}
    assert events[3][1]["delta"] == {
        "type": "signature_delta",
        "signature": thinking.signature("look"),
    }
    starts = [data for name, data in events if name == "content_block_start"]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [
        (0, "thinking"),
        (1, "tool_use"),
    ]
    assert {data["index"] for name, data in events if name.startswith("content_block_")} == {0, 1}


async def test_thinking_sent_back_as_written_is_bridged() -> None:
    one, two = await _anthropic_turns(lambda blocks: blocks)
    assert two.bridged
    assert two.prompt_token_ids[: len(one.prompt_token_ids)] == one.prompt_token_ids
    assert chars(two.prompt_token_ids).count("{look}") == 1


async def test_thinking_the_harness_dropped_is_not_read_back() -> None:
    one, two = await _anthropic_turns(lambda blocks: [b for b in blocks if b["type"] != "thinking"])
    assert not two.bridged
    assert "{look}" not in chars(two.prompt_token_ids) and "{LOOK}" not in chars(
        two.prompt_token_ids
    )


async def test_thinking_the_harness_changed_is_read_as_it_sent_it() -> None:
    def changed(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**b, "thinking": "looked"} if b["type"] == "thinking" else b for b in blocks]

    one, two = await _anthropic_turns(changed)
    assert not two.bridged
    prompt = chars(two.prompt_token_ids)
    assert "{LOOKED}" in prompt and "{look}" not in prompt


async def test_a_redacted_block_is_accepted_and_carries_nothing() -> None:
    def redacted(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"type": "redacted_thinking", "data": "opaque"} if b["type"] == "thinking" else b
            for b in blocks
        ]

    one, two = await _anthropic_turns(redacted)
    assert not two.bridged
    assert "{look}" not in chars(two.prompt_token_ids) and "{LOOK}" not in chars(
        two.prompt_token_ids
    )


# ------------------------------------------------------------------- the OpenAI wire


async def test_the_openai_reply_carries_reasoning_and_its_details() -> None:
    sampler = ScriptedSampler("", answers=["{look}CALL ls {}"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        message = (await ask(started, "t", tools=TOOLS_OPENAI)).json()["choices"][0]["message"]
    assert message["reasoning"] == "look" and "reasoning_content" not in message
    assert message[thinking.DETAILS] == [
        {"type": "reasoning.text", "text": "look", "format": "unknown", "index": 0}
    ]
    assert message["tool_calls"][0]["function"]["name"] == "ls"


async def test_the_openai_stream_carries_the_reasoning_before_any_text() -> None:
    async with endpoint(
        ScriptedSampler("", answers=["{look}hello"]), renderer=BridgeRenderer()
    ) as started:
        text = (await ask(started, "t", stream=True)).text
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in text.strip().split("\n\n")
        if line != "data: [DONE]"
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
    assert deltas[0]["reasoning"] == "look" and "content" not in deltas[0]
    assert deltas[0]["reasoning_details"][0]["text"] == "look"
    assert "".join(d.get("content") or "" for d in deltas[1:]) == "hello"
    assert len({chunk["id"] for chunk in chunks}) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"reasoning_content": "look"},
        {"reasoning": "look"},
        {"reasoning": "look", "reasoning_details": [{"type": "reasoning.text", "text": "look"}]},
        {"reasoning_details": [{"type": "reasoning.text", "text": "look"}]},
    ],
)
async def test_reasoning_sent_back_in_any_field_is_read_once_and_bridged(
    fields: dict[str, Any],
) -> None:
    sampler = ScriptedSampler("", answers=["{look}CALL ls {}", "done"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        first = await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
        said = first.json()["choices"][0]["message"]
        history.append(
            {"role": "assistant", "content": None, "tool_calls": said["tool_calls"], **fields}
        )
        history.append(
            {"role": "tool", "tool_call_id": said["tool_calls"][0]["id"], "content": "a"}
        )
        await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
        one, two = started.records_for("t")
    assert two.bridged and chars(two.prompt_token_ids).count("{look}") == 1


def test_the_parsers_turn_thinking_into_parts_once() -> None:
    thinking.install()
    parsed = cookbook.private("_parse_openai")(
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning": "look",
                    "reasoning_details": [{"type": "reasoning.text", "text": "look"}],
                },
                {"role": "user", "content": "more"},
            ]
        }
    )
    assert parsed.messages[1]["content"] == [
        {"type": "thinking", "thinking": "look"},
        {"type": "text", "text": "ok"},
    ]
    parsed = cookbook.private("_parse_anthropic")(
        {
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "only"}]},
                {"role": "user", "content": "more"},
            ],
        }
    )
    # A reply that was thinking alone is still a message.
    assert parsed.messages[1]["content"] == [{"type": "thinking", "thinking": "only"}]


def test_install_is_once_whatever_the_order() -> None:
    thinking.install()
    before = cookbook.private("_parse_openai")
    thinking.install()
    assert cookbook.private("_parse_openai") is before
    assert cookbook.wrapped_by("_parse_openai", thinking.TAG)
    assert cookbook.wrapped_by("_serve_sse", thinking.TAG)
    assert not cookbook.wrapped_by("_parse_openai", "nothing")


# ------------------------------------------------------------------ the real renderer


@pytest.fixture(scope="module")
def qwen():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", local_files_only=True
        )
    except Exception:  # noqa: BLE001 - not cached here
        pytest.skip("Qwen3's tokenizer is not cached locally")
    return tokenizer, cookbook.get_renderer("qwen3", tokenizer)


class TokenSampler(ScriptedSampler):
    """Answers with token ids, as Tinker does."""

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        self.asked.append(sampling_params)
        self.prompts.append(prompt)
        tokens = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return FakeResponse([FakeSequence(tokens=list(tokens), logprobs=[-0.5] * len(tokens))])


@pytest.mark.parametrize("kept", [True, False])
async def test_qwen_reads_its_thinking_exactly_when_claude_code_sends_it_back(
    qwen: Any, kept: bool
) -> None:
    tokenizer, renderer = qwen
    call = (
        "<think>\nI should list the files.\n</think>\n\nLooking.\n\n<tool_call>\n"
        '{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call><|im_end|>'
    )
    first = tokenizer.encode(call, add_special_tokens=False)
    last = tokenizer.encode(
        "<think>\nDone.\n</think>\n\nTwo files.<|im_end|>", add_special_tokens=False
    )
    sampler = TokenSampler("", answers=[first, last])
    tools = [
        {
            "name": "Bash",
            "description": "Run a command",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }
    ]
    async with endpoint(sampler, renderer=renderer) as started:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "What is here?"}]
        body = {"system": "Be brief.", "tools": tools, "stream": True}
        reply = await ask_anthropic(started, "t", messages=messages, **body)
        assert reply.status_code == 200, reply.text
        events = _events(reply.text)
        said = events[2][1]["delta"]["thinking"]
        assert said == "I should list the files."
        tool = next(
            data["content_block"]
            for name, data in events
            if name == "content_block_start" and data["content_block"]["type"] == "tool_use"
        )
        # What Claude Code keeps of a streamed reply: each block as its deltas built it.
        text = next(
            data["delta"]["text"]
            for name, data in events
            if name == "content_block_delta" and data["delta"]["type"] == "text_delta"
        )
        blocks = [
            {"type": "tool_use", "id": tool["id"], "name": "Bash", "input": {"command": "ls"}}
        ]
        if kept:
            blocks.insert(
                0, {"type": "thinking", "thinking": said, "signature": thinking.signature(said)}
            )
        messages += [
            {"role": "assistant", "content": [{"type": "text", "text": text}, *blocks]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool["id"], "content": "a b"}],
            },
        ]
        second = await ask_anthropic(started, "t", messages=messages, **body)
        assert second.status_code == 200, second.text
        one, two = started.records_for("t")
    prefix = (*one.prompt_token_ids, *one.completion_token_ids)
    shown = tokenizer.decode(list(two.prompt_token_ids))
    if kept:
        assert two.bridged and two.prompt_token_ids[: len(prefix)] == prefix
        assert shown.count("I should list the files.") == 1
    else:
        assert not two.bridged and "I should list the files." not in shown
    assert sampler.prompts[1].to_ints() == list(two.prompt_token_ids)
