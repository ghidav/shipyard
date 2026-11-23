"""GPT-OSS writes a tool call as a Harmony message addressed to a function, and the cookbook's
parser sets aside one whose arguments are not JSON. A server of the model hands such a call to the
harness with its arguments as sampled, and the proxy does the same: the harness is the one to
answer it. A reply of calls alone has no text."""

from __future__ import annotations

from typing import Any

from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

from harness_proxy import cookbook
from proxies import FakeRenderer, FakeSampler, ask, server

STOPS = {"<|return|>": 1, "<|call|>": 2}
#: The call of GPT-OSS that came back as an empty final answer: an edit through `apply_patch`,
#: its arguments closed as Codex's argv form is, with `"]}`.
ARGUMENTS = '{"command":"apply_patch <<\'PATCH\'\\n*** Begin Patch\\n*** End Patch\\nPATCH"]}'
CALL = "<|channel|>commentary to=functions.bash <|constrain|>json<|message|>"
THOUGHT = "<|channel|>analysis<|message|>Edit the file.<|end|><|start|>assistant"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]


class Chars:
    """A tokenizer of code points, with Harmony's two stop tokens as ids of their own."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, at = [], 0
        while at < len(text):
            stop = next((one for one in STOPS if text.startswith(one, at)), None)
            ids.append(STOPS[stop] if stop else ord(text[at]) + 10)
            at += len(stop) if stop else 1
        return ids

    def decode(self, ids: Any) -> str:
        named = {token: stop for stop, token in STOPS.items()}
        return "".join(named.get(int(one)) or chr(int(one) - 10) for one in ids)


def gpt_oss() -> GptOssRenderer:
    return GptOssRenderer(
        Chars(),  # type: ignore[arg-type]
        use_system_prompt=True,
        reasoning_effort="medium",
        current_date="2026-01-01",
    )


def read(sampled: str) -> tuple[Any, Any]:
    """Return a sampled reply as the cookbook parses it, and as the proxy hands it on."""
    renderer = gpt_oss()
    parsed, _ = renderer.parse_response(Chars().encode(sampled))
    return parsed, cookbook.as_written(renderer, parsed)


def test_a_call_whose_arguments_are_not_json_is_handed_over_as_sampled() -> None:
    parsed, handed = read(f"{THOUGHT}{CALL}{ARGUMENTS}<|call|>")
    assert not parsed.get("tool_calls") and parsed["unparsed_tool_calls"], "the parser's own"
    (call,) = handed["tool_calls"]
    assert (call.function.name, call.function.arguments) == ("bash", ARGUMENTS)
    assert "unparsed_tool_calls" not in handed
    assert cookbook.get_text_content(handed) == ""
    assert handed["content"] == parsed["content"], "its thinking is kept"


def test_a_reply_of_a_call_alone_has_no_text() -> None:
    for arguments in ('{"command":"ls"}', ARGUMENTS):
        parsed, handed = read(f"{CALL}{arguments}<|call|>")
        assert CALL in parsed["content"], "the parser falls back on the raw markup"
        (call,) = handed["tool_calls"]
        assert call.function.arguments == arguments
        assert handed["content"] == ""


def test_a_reply_the_parser_read_whole_is_returned_as_it_is() -> None:
    for sampled in (
        f'{THOUGHT}{CALL}{{"command":"ls"}}<|call|>',
        "<|channel|>analysis<|message|>Done.<|end|><|start|>assistant<|channel|>final"
        "<|message|>Fixed.<|return|>",
    ):
        parsed, handed = read(sampled)
        assert handed == parsed


def test_a_header_named_in_the_thinking_adds_no_call() -> None:
    """The parser counts `to=functions.` anywhere in the reply, so a thought that names one
    makes it report the last message, the real call, as a block it could not read."""
    named = "<|channel|>analysis<|message|>I call to=functions.bash now.<|end|><|start|>assistant"
    parsed, handed = read(f'{named}{CALL}{{"command":"ls"}}<|call|>')
    assert parsed["unparsed_tool_calls"], "the parser's guess"
    assert handed == parsed
    assert len(handed["tool_calls"]) == 1


def test_another_renderers_reply_is_returned_as_it_is() -> None:
    parsed, _ = read(f"{CALL}{ARGUMENTS}<|call|>")
    assert cookbook.as_written(FakeRenderer(), parsed) is parsed


async def test_the_harness_gets_the_call_and_the_model_reads_it_back_as_it_wrote_it() -> None:
    chars = Chars()
    replies = [
        chars.encode(f"{THOUGHT}{CALL}{ARGUMENTS}<|call|>"),
        chars.encode("<|channel|>final<|message|>Fixed.<|return|>"),
    ]
    sampler = FakeSampler(replies)
    async with server(sampler, renderer=gpt_oss()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "fix it"}]
        first = await ask(started, "t", messages=history, tools=TOOLS)
        assert first.status_code == 200, first.text
        choice = first.json()["choices"][0]
        said = choice["message"]
        (call,) = said["tool_calls"]
        assert choice["finish_reason"] == "tool_calls"
        assert said["content"] is None
        assert call["function"] == {"name": "bash", "arguments": ARGUMENTS}
        # The harness answers the call it could not read, as it would any server's.
        result = {"role": "tool", "tool_call_id": call["id"], "content": "not a JSON object"}
        second = await ask(started, "t", messages=[*history, said, result], tools=TOOLS)
        assert second.status_code == 200, second.text
        assert second.json()["choices"][0]["message"]["content"] == "Fixed."
    read_back = chars.decode(sampler.prompts[1].to_ints())
    assert f"{ARGUMENTS}<|call|><|start|>functions.bash to=assistant" in read_back
    assert "Edit the file." in read_back, "with the thinking it sampled before the call"
