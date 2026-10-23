"""A trial's calls encode each text once: each re-renders the whole conversation, and the trial's
renderer remembers what its tokenizer encoded. Runs on Qwen3's real renderer, skipped when its
tokenizer is not cached here."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from harness_proxy import cookbook
from proxies import FakeSampler, ask, recorded, server

MODEL = "Qwen/Qwen3-8B"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]
REPLY = (
    "<think>\nI will look.\n</think>\n\n<tool_call>\n"
    '{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call><|im_end|>'
)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached here
        pytest.skip("Qwen3's tokenizer is not cached locally")


async def converse(started: Any, trial: str, turns: int) -> None:
    """Run `turns` calls of a tool loop for `trial`, quoting each reply back as a harness does."""
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
    ]
    for turn in range(turns):
        answered = await ask(started, trial, messages=history, tools=TOOLS)
        assert answered.status_code == 200, answered.text
        said = answered.json()["choices"][0]["message"]
        history += [
            said,
            {
                "role": "tool",
                "tool_call_id": said["tool_calls"][0]["id"],
                "content": f"src/\ntests/\nturn {turn}: " + "line of output\n" * 40,
            },
        ]


async def test_a_trial_encodes_each_text_once_and_another_trial_has_its_own(
    tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cookbook, "get_tokenizer", lambda model: tokenizer)
    renderer = cookbook.renderer_for(MODEL)
    sampler = FakeSampler([tokenizer.encode(REPLY, add_special_tokens=False)])
    encoded: Counter[str] = Counter()
    encode = tokenizer.encode

    def counted(text: Any, *args: Any, **kwargs: Any) -> Any:
        encoded[text] += 1
        return encode(text, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", counted)
    async with server(sampler, renderer=renderer, max_context=40_000) as started:
        await converse(started, "t", turns=5)
        records = recorded(started, "t")
        assert [one.bridged for one in records] == [False, True, True, True, True], (
            "every call after the first is bridged: about three renders of the conversation"
        )
        assert encoded and max(encoded.values()) == 1, "each text encoded once in the trial"
        await converse(started, "u", turns=5)
    assert set(encoded.values()) == {2}, "the other trial's renderer encodes for itself"
    first = renderer.build_generation_prompt(
        [
            *renderer.create_conversation_prefix_with_tools(
                [one["function"] for one in TOOLS], system_prompt="You are a coding agent."
            ),
            {"role": "user", "content": "Fix the failing test."},
        ]
    )
    assert list(records[0].prompt_token_ids) == first.to_ints(), "rendered as without the memo"
