"""The digest and the narrow bridge: what identifies a message, which replies the index
finds, and the five reasons a prompt is rendered afresh instead, through the real app on
a renderer whose fresh render never equals the sampled tokens."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from shipyard.proxy.bridge import (
    Index,
    Reply,
    bridge,
    chain,
    digest,
    digests_of,
    ids_in,
    tail_extends,
)
from shipyard.proxy.cookbook import ToolCall
from shipyard.proxy.exchange import Exchange, exchange
from shipyard.proxy.recorder import IMAGE_TOKEN, Recorder
from tests.proxies import (
    TOOLS_ANTHROPIC,
    TOOLS_OPENAI,
    BridgeRenderer,
    ScriptedSampler,
    ask,
    ask_anthropic,
    chars,
    endpoint,
    fetch,
    ids,
)

STOP = BridgeRenderer.STOP


def call(name: str, arguments: str, id: str | None = None) -> dict[str, Any]:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ----------------------------------------------------------------------- the digest


def test_a_tool_call_is_its_name_and_id_so_a_rewritten_argument_is_the_same_call() -> None:
    sampled = {"role": "assistant", "content": "", "tool_calls": [call("edit", '{"p": "a"}', "c1")]}
    echoed = {**sampled, "tool_calls": [call("edit", '{"replace_all": false, "p": "a"}', "c1")]}
    assert digest(sampled) == digest(echoed)
    parsed = ToolCall(id="c1", function=ToolCall.FunctionBody(name="edit", arguments="{}"))
    assert digest({"role": "assistant", "content": "", "tool_calls": [parsed]}) == digest(sampled)
    assert digest(sampled) != digest({**sampled, "tool_calls": [call("edit", '{"p": "a"}', "c2")]})
    assert digest(sampled) != digest({**sampled, "tool_calls": [call("write", '{"p": "a"}', "c1")]})
    assert digest(sampled) != digest({**sampled, "tool_calls": []})


def test_whitespace_in_a_messages_text_is_not_part_of_it() -> None:
    reads = [call("read", "{}", "c1"), call("read", "{}", "c2")]
    between = [{"type": "text", "text": "\n"}, {"type": "text", "text": "\n\n"}]
    sampled = {"role": "assistant", "content": between, "tool_calls": reads}
    assert digest(sampled) == digest({"role": "assistant", "content": [], "tool_calls": reads})
    said = {"role": "assistant", "content": "two files\n"}
    assert digest(said) == digest({"role": "assistant", "content": "two files"})
    assert digest(said) == digest({"role": "assistant", "content": " two  files"})
    assert digest(said) != digest({"role": "assistant", "content": "two filez"})


def test_a_tool_result_is_its_name_and_text_and_never_its_call_id() -> None:
    one = {"role": "tool", "content": "a b", "tool_call_id": "call_1", "name": "ls"}
    assert digest(one) == digest({**one, "tool_call_id": "call_2"})
    assert digest(one) != digest({**one, "name": "cat"})
    assert digest(one) != digest({**one, "content": "a c"})


def test_thinking_is_part_of_a_message_so_a_harness_that_strips_it_has_changed_it() -> None:
    kept = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "ok"}],
    }
    stripped = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    changed = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"}],
    }
    assert len({digest(kept), digest(stripped), digest(changed)}) == 3
    assert digest(stripped) == digest({"role": "assistant", "content": "ok"})


def test_images_count_by_their_pixels() -> None:
    def seen(color: str) -> dict[str, Any]:
        image = Image.new("RGB", (4, 4), color)
        return {
            "role": "user",
            "content": [{"type": "text", "text": "x"}, {"type": "image", "image": image}],
        }

    assert digest(seen("red")) == digest(seen("red")) != digest(seen("blue"))
    assert digest(seen("red")) != digest({"role": "user", "content": "x"})


def test_the_chain_of_a_prefix_is_its_own_and_the_index_finds_the_longest_reply() -> None:
    digests = digests_of([{"role": "user", "content": c} for c in ("a", "b", "c", "d")])
    assert chain(digests[:2]) != chain(digests[:3]) and chain(()) == ""
    index = Index()
    first = Reply((1,), (2,), True, (1,))
    second = Reply((1, 2, 3), (4,), False, (1, 2, 3))
    index.add(digests[:1], digests[1], first)
    index.add(digests[:3], digests[3], second)
    assert index.find(digests) == (3, second)
    assert index.find(digests[:3]) == (1, first)
    assert index.find(digests[:1]) is None and index.find(()) is None
    assert index.find((digests[1], digests[0])) is None, "order is the chain"
    assert len(index) == 2


def test_two_different_replies_under_one_history_leave_it_ambiguous() -> None:
    digests = digests_of([{"role": "user", "content": c} for c in ("a", "b")])
    index = Index()
    index.add(digests[:1], digests[1], Reply((1,), (2,), True, (1,)))
    index.add(digests[:1], digests[1], Reply((1,), (2,), True, (1,)))
    assert index.find(digests) == (1, Reply((1,), (2,), True, (1,))), "the same reply twice"
    index.add(digests[:1], digests[1], Reply((1,), (3,), True, (1,)))
    assert index.find(digests) is None


def test_two_different_replies_to_one_history_sharing_a_call_id_are_both_passed_over() -> None:
    """A model that writes its own ids gives two samples of one history the same id. If
    the harness trims the one it kept (drops its thinking, or a call), the echo can digest
    as the other: both are passed over."""
    user = digests_of([{"role": "user", "content": "look"}])
    kept, other = Reply((1,), (2,), True, (1,)), Reply((1,), (3,), True, (1,))
    index = Index()
    index.add(user, "with-thinking", kept, ["functions.ls:0"])
    index.add(user, "without", other, ["functions.ls:0"])
    assert index.find([*user, "with-thinking"]) is None
    assert index.find([*user, "without"]) is None
    elsewhere = digests_of([{"role": "user", "content": "else"}])
    index.add(elsewhere, "without", other, ["functions.ls:0"])
    assert index.find([*elsewhere, "without"]) == (1, other), "another history is its own"


# ------------------------------------------------------------------ the rule, by hand


def test_only_tool_results_and_one_closing_user_message_extend() -> None:
    tool, user, assistant = {"role": "tool"}, {"role": "user"}, {"role": "assistant"}
    assert tail_extends([tool]) and tail_extends([tool, tool]) and tail_extends([user])
    assert tail_extends([tool, user])
    assert not tail_extends([]), "a resample, not an extension"
    assert not tail_extends([tool, assistant, tool]), "an assistant message would be re-tokenized"
    assert not tail_extends([user, user]) and not tail_extends([user, tool])


def _indexed(
    renderer: Any, history: list[dict[str, Any]], completion: str, *, ended: bool
) -> Index:
    """An index holding one reply: `history[:-1]` was its prompt, rendered afresh."""
    prompt = tuple(renderer.build_generation_prompt(history[:-1]).to_ints())
    index = Index()
    index.add(
        digests_of(history[:-1]), digest(history[-1]), Reply(prompt, ids(completion), ended, prompt)
    )
    return index


def test_the_bridge_is_the_stored_prompt_the_sampled_reply_its_stop_and_the_tail() -> None:
    renderer = BridgeRenderer()
    history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "hi"}]
    index = _indexed(renderer, history, "hi", ended=False)
    messages = [*history, {"role": "user", "content": "more"}]
    assert bridge(messages, renderer, index).ids == (
        *ids("user:go\n>"),
        *ids("hi"),
        STOP,
        *ids("\nuser:more\n>"),
    )
    closed = _indexed(renderer, history, "hi\x00", ended=True)
    assert bridge(messages, renderer, closed).ids == (
        *ids("user:go\n>"),
        *ids("hi\x00"),
        *ids("\nuser:more\n>"),
    )


def test_nothing_after_the_reply_or_no_reply_at_all_renders_afresh() -> None:
    renderer = BridgeRenderer()
    history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "hi"}]
    index = _indexed(renderer, history, "hi", ended=False)
    assert bridge(history, renderer, index) is None, "a resample"
    assert bridge([history[0], {"role": "user", "content": "more"}], renderer, index) is None
    assert bridge([*history, {"role": "user", "content": "more"}], renderer, Index()) is None


def test_a_renderer_without_token_stops_or_a_prompt_with_an_image_chunk_cannot_be_bridged() -> None:
    renderer = BridgeRenderer()
    history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "hi"}]
    index = _indexed(renderer, history, "hi", ended=False)
    messages = [*history, {"role": "user", "content": "more"}]
    assert bridge(messages, renderer, index) is not None
    renderer.get_stop_sequences = lambda: ["</s>"]  # type: ignore[method-assign]
    assert bridge(messages, renderer, index) is None
    assert (
        ids_in(SimpleNamespace(chunks=[SimpleNamespace(tokens=[1]), SimpleNamespace(length=2)]))
        is None
    )
    assert ids_in(SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2])])) == (1, 2)


def test_a_record_whose_prompt_held_an_image_is_never_indexed() -> None:
    recorder = Recorder(object())
    found = Exchange(trial="t", prompt_digests=("a",))
    token = exchange.set(found)
    try:
        recorder._add("t", (1, IMAGE_TOKEN), {}, None, completion=(3,))
        recorder._add("t", (1, 2), {}, None, completion=(3,))
    finally:
        exchange.reset(token)
    recorder.reply("t", 1, "r1", ended_with_stop=True)
    assert len(recorder.index_for("t")) == 0
    recorder.reply("t", 2, "r2", ended_with_stop=True)
    assert len(recorder.index_for("t")) == 1
    assert [r.reply_digest for r in recorder.records_for("t")] == ["r1", "r2"]
    recorder.reply("t", 9, "nobody", ended_with_stop=True)
    assert len(recorder.index_for("t")) == 1


# ------------------------------------------------------------- the rule, over the wire


async def _tool_loop(answers: list[str], *, retitle: Any = None) -> tuple[Any, Any]:
    """Three turns on the OpenAI wire: a tool call, its result, a reply with thinking, a
    closing user message. `retitle` may edit the history before the third request."""
    sampler = ScriptedSampler("", answers=answers)
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        first = await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
        assert first.status_code == 200, first.text
        said = first.json()["choices"][0]["message"]
        history.append({"role": "assistant", "content": None, "tool_calls": said["tool_calls"]})
        history.append(
            {"role": "tool", "tool_call_id": said["tool_calls"][0]["id"], "content": "a b"}
        )
        second = await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
        assert second.status_code == 200, second.text
        history.append(second.json()["choices"][0]["message"])
        history.append({"role": "user", "content": "thanks"})
        if retitle is not None:
            retitle(history)
        third = await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
        assert third.status_code == 200, third.text
        return started.records_for("t"), sampler


async def test_a_tool_loop_bridges_and_every_prompt_extends_the_last() -> None:
    records, sampler = await _tool_loop(["CALL ls {}", "{done}two files", "bye"])
    one, two, three = records
    assert [r.bridged for r in records] == [False, True, True]
    assert one.prompt_token_ids == ids("system:<tools:ls>\nuser:look\n>")
    call_id = chars(two.prompt_token_ids).split("tool[")[1].split("]")[0]
    assert call_id.startswith("call_"), "the proxy names the call, in the wire's form"
    assert two.prompt_token_ids == (
        *one.prompt_token_ids,
        *one.completion_token_ids,
        STOP,
        *ids(f"\ntool[{call_id}]:a b\n>"),
    )
    assert three.prompt_token_ids == (
        *two.prompt_token_ids,
        *two.completion_token_ids,
        STOP,
        *ids("\nuser:thanks\n>"),
    )
    assert "CALL LS" not in chars(three.prompt_token_ids), "the sampled tokens, not a re-render"
    assert [p.to_ints() for p in sampler.prompts] == [list(r.prompt_token_ids) for r in records]
    assert [r.reply_digest is not None for r in records] == [True, True, True]
    assert len(one.prompt_digests) == 2 and len(three.prompt_digests) == 6


async def test_a_stripped_thinking_replay_does_not_bridge() -> None:
    def strip(history: list[dict[str, Any]]) -> None:
        history[3] = {"role": "assistant", "content": history[3]["content"]}

    records, _ = await _tool_loop(["CALL ls {}", "{done}two files", "bye"], retitle=strip)
    assert [r.bridged for r in records] == [False, True, False]
    shown = chars(records[2].prompt_token_ids)
    assert ">TWO FILES\x00" in shown and "done" not in shown.lower()


async def test_a_user_message_after_a_non_tool_tail_does_not_bridge() -> None:
    def interject(history: list[dict[str, Any]]) -> None:
        history.insert(4, {"role": "assistant", "content": "note to self"})

    records, _ = await _tool_loop(["CALL ls {}", "{done}two files", "bye"], retitle=interject)
    assert [r.bridged for r in records] == [False, True, False]
    assert ">NOTE TO SELF\x00" in chars(records[2].prompt_token_ids)


async def test_a_head_that_re_renders_differently_does_not_bridge() -> None:
    """The digest does not see a tool result's call id; this renderer renders it. A history
    whose result's id changed matches by digest and renders another head, so it forks."""

    def renumber(history: list[dict[str, Any]]) -> None:
        history[2] = {**history[2], "tool_call_id": "call_other"}

    records, _ = await _tool_loop(["CALL ls {}", "{done}two files", "bye"], retitle=renumber)
    assert [r.bridged for r in records] == [False, True, False]
    assert "tool[call_other]" in chars(records[2].prompt_token_ids)


async def test_the_anthropic_wire_bridges_a_tool_loop_too() -> None:
    sampler = ScriptedSampler("", answers=["CALL ls {}", "two entries"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        body = {"system": "be brief", "tools": TOOLS_ANTHROPIC}
        first = await ask_anthropic(started, "t", messages=messages, **body)
        assert first.status_code == 200, first.text
        block = first.json()["content"][0]
        assert block["type"] == "tool_use" and block["name"] == "ls"
        assert block["id"].startswith("toolu_"), "the proxy names the call, in the wire's form"
        messages.append({"role": "assistant", "content": [block]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": "a b"}],
            }
        )
        second = await ask_anthropic(started, "t", messages=messages, stream=True, **body)
        assert second.status_code == 200, second.text
        one, two = started.records_for("t")
    assert two.bridged and two.prompt_token_ids[
        : len(one.prompt_token_ids) + len(one.completion_token_ids)
    ] == (
        *one.prompt_token_ids,
        *one.completion_token_ids,
    )


async def test_a_retry_replays_one_record_with_its_digest_and_the_index_drains_with_the_trial() -> (
    None
):
    sampler = ScriptedSampler("", answers=["hi"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        await ask(started, "t")
        await ask(started, "t", headers={"x-stainless-retry-count": "1"})
        (record,) = started.records_for("t")
        assert record.reply_digest == digest(
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
        )
        assert len(started.recorder.index_for("t")) == 1 and len(sampler.asked) == 1
        fetched = await fetch(started, "t")
        assert "t" not in started.recorder.indexes
    assert fetched.json()["records"][0]["reply_digest"] == record.reply_digest
    assert fetched.json()["records"][0]["bridged"] is False


async def test_a_refused_request_leaves_a_record_with_its_digests_and_no_reply() -> None:
    async with endpoint(
        ScriptedSampler("", answers=["hi"]), renderer=BridgeRenderer(), max_context=4
    ) as started:
        refused = await ask(started, "t")
        assert refused.status_code == 400
        (record,) = started.records_for("t")
    assert record.error == "context" and record.reply_digest is None
    assert record.prompt_digests == digests_of([{"role": "user", "content": "go"}])


@pytest.mark.parametrize("stream", [False, True])
async def test_a_bridged_prompt_is_what_the_sampler_is_given(stream: bool) -> None:
    sampler = ScriptedSampler("", answers=["hi", "bye"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        history = [{"role": "user", "content": "go"}]
        await ask(started, "t", messages=history, stream=stream)
        history += [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
        await ask(started, "t", messages=history, stream=stream)
        one, two = started.records_for("t")
    assert two.bridged and sampler.prompts[1].to_ints() == list(two.prompt_token_ids)
    assert two.prompt_token_ids == (*one.prompt_token_ids, *ids("hi"), STOP, *ids("\nuser:more\n>"))


def test_a_renderer_that_strips_history_thinking_renders_a_user_turn_afresh() -> None:
    """Qwen3's template drops earlier thinking at a new user turn: bridging there would
    keep thinking the template's own render leaves out, so the turn is rendered afresh."""
    renderer = BridgeRenderer()
    history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "hi"}]
    index = _indexed(renderer, history, "hi", ended=False)
    tool = {"role": "tool", "content": "ok", "tool_call_id": "c"}
    renderer.strip_thinking_from_history = True
    assert bridge([*history, {"role": "user", "content": "more"}], renderer, index) is None
    assert bridge([*history, tool], renderer, index) is not None, (
        "a tool result alone still extends"
    )
    renderer.strip_thinking_from_history = False
    assert bridge([*history, {"role": "user", "content": "more"}], renderer, index) is not None


# --------------------------------------------------------- a call known by its id


async def _reply(started: Any, history: list[dict[str, Any]], **headers: str) -> dict[str, Any]:
    answered = await ask(started, "t", messages=history, tools=TOOLS_OPENAI, headers=headers)
    assert answered.status_code == 200, answered.text
    return answered.json()["choices"][0]["message"]


def _echo(said: dict[str, Any], arguments: str, result: str) -> list[dict[str, Any]]:
    """The reply's call as a harness sends it back with its arguments rewritten, and its
    result: Claude Code echoes the input it normalized, not the one the model wrote."""
    (sent,) = said["tool_calls"]
    rewritten = {**sent, "function": {**sent["function"], "arguments": arguments}}
    return [
        {"role": "assistant", "content": None, "tool_calls": [rewritten]},
        {"role": "tool", "tool_call_id": sent["id"], "content": result},
    ]


def _extends(later: Any, earlier: Any) -> bool:
    head = (*earlier.prompt_token_ids, *earlier.completion_token_ids)
    return later.prompt_token_ids[: len(head)] == head


async def test_an_echo_whose_arguments_the_harness_rewrote_still_bridges() -> None:
    sampler = ScriptedSampler("", answers=['CALL ls {"p": "a"}', "done"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        history += _echo(await _reply(started, history), '{"all": false, "p": "a"}', "alpha")
        await _reply(started, history)
        one, two = started.records_for("t")
    assert two.bridged and _extends(two, one)
    shown = chars(two.prompt_token_ids)
    assert 'CALL ls {"p": "a"}' in shown and '"ALL"' not in shown, "the call the model wrote"


async def test_of_two_replies_to_one_history_the_one_the_harness_kept_is_bridged() -> None:
    """Two samples of one history call the same tool on different files. The harness kept
    the first and rewrote its arguments; its ids say which it was."""
    answers = ['CALL ls {"p": "a"}', 'CALL ls {"p": "b"}', "done"]
    async with endpoint(ScriptedSampler("", answers=answers), renderer=BridgeRenderer()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        a = await _reply(started, history)
        b = await _reply(started, history)
        assert a["tool_calls"][0]["id"] != b["tool_calls"][0]["id"]
        history += _echo(a, '{"p": "a", "n": 1}', "alpha")
        await _reply(started, history)
        first, _, third = started.records_for("t")
    assert third.bridged and _extends(third, first)


class ModelNamedCalls(BridgeRenderer):
    """A renderer whose model writes its own call ids, numbered as Kimi numbers them: two
    samples of one history give their calls the same id."""

    def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
        message, termination = super().parse_response(response)
        calls = message.get("tool_calls") or []
        message["tool_calls"] = [
            c.model_copy(update={"id": f"functions.{c.function.name}:0"}) for c in calls
        ]
        return message, termination


async def test_a_model_that_names_its_own_calls_keeps_its_ids_and_two_samples_fork() -> None:
    answers = ['CALL ls {"p": "a"}', 'CALL ls {"p": "b"}', "done"]
    async with endpoint(
        ScriptedSampler("", answers=answers), renderer=ModelNamedCalls()
    ) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        a = await _reply(started, history)
        b = await _reply(started, history)
        assert a["tool_calls"][0]["id"] == b["tool_calls"][0]["id"] == "functions.ls:0"
        history += _echo(a, '{"p": "a", "n": 1}', "alpha")
        await _reply(started, history)
        assert not started.records_for("t")[-1].bridged, "two replies fit: neither is spliced"
    answers = ['CALL ls {"p": "a"}', "done"]
    async with endpoint(
        ScriptedSampler("", answers=answers), renderer=ModelNamedCalls()
    ) as started:
        history = [{"role": "user", "content": "look"}]
        history += _echo(await _reply(started, history), '{"p": "a", "n": 1}', "alpha")
        await _reply(started, history)
        one, two = started.records_for("t")
    assert two.bridged and _extends(two, one), "one reply fits: its rewritten echo bridges"


async def test_a_replayed_retry_carries_the_ids_its_first_answer_did() -> None:
    sampler = ScriptedSampler("", answers=["CALL ls {}"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        history = [{"role": "user", "content": "look"}]
        first = await _reply(started, history)
        replayed = await _reply(started, history, **{"x-stainless-retry-count": "1"})
    assert len(sampler.asked) == 1
    assert replayed["tool_calls"][0]["id"] == first["tool_calls"][0]["id"]


async def test_a_sibling_the_trimmed_echo_of_a_kept_reply_digests_as_is_not_spliced() -> None:
    """Kimi-like ids: the kept reply had thinking, its sibling had none. The harness echoes
    the kept one without its thinking, which digests as the sibling; the shared id forks."""
    answers = ['{plan}CALL ls {"p": "a"}', 'CALL ls {"p": "b"}', "done"]
    async with endpoint(
        ScriptedSampler("", answers=answers), renderer=ModelNamedCalls()
    ) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        a = await _reply(started, history)
        await _reply(started, history)
        history += _echo(a, '{"p": "a"}', "alpha")
        await _reply(started, history)
        assert not started.records_for("t")[-1].bridged


async def test_two_turns_that_sample_the_same_call_get_their_own_ids() -> None:
    answers = ["CALL ls {}", "CALL ls {}", "done"]
    async with endpoint(ScriptedSampler("", answers=answers), renderer=BridgeRenderer()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        first = await _reply(started, history)
        history += _echo(first, "{}", "alpha")
        second = await _reply(started, history)
    assert first["tool_calls"][0]["id"] != second["tool_calls"][0]["id"]


async def test_an_anthropic_echo_that_drops_a_whitespace_only_block_bridges() -> None:
    """Claude Code drops a text block holding only whitespace when it echoes a reply."""
    sampler = ScriptedSampler("", answers=["\n\nCALL ls {}", "done"])
    async with endpoint(sampler, renderer=BridgeRenderer()) as started:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        body = {"system": "be brief", "tools": TOOLS_ANTHROPIC}
        first = await ask_anthropic(started, "t", messages=messages, **body)
        blocks = first.json()["content"]
        assert [b["type"] for b in blocks] == ["text", "tool_use"]
        (use,) = [b for b in blocks if b["type"] == "tool_use"]
        messages.append({"role": "assistant", "content": [use]})
        result = {"type": "tool_result", "tool_use_id": use["id"], "content": "a b"}
        messages.append({"role": "user", "content": [result]})
        second = await ask_anthropic(started, "t", messages=messages, **body)
        assert second.status_code == 200, second.text
        one, two = started.records_for("t")
    assert two.bridged and _extends(two, one)
