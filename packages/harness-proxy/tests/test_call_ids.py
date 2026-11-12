"""A renderer may name calls by their place in a reply, as tml-renderers 0.4 does for Inkling, so
its ids repeat from turn to turn. Claude Code drops a call whose id it has seen, and its result,
and sends "(no content)" in their place."""

from __future__ import annotations

from typing import Any

from proxies import TOOLS_OPENAI, BridgeRenderer, FakeSampler, ask, server


class PlacedCalls(BridgeRenderer):
    """Bridge renderer whose parser names each call by its place in the reply."""

    def parse_response(self, response: list[int]) -> tuple[dict[str, Any], Any]:
        message, termination = super().parse_response(response)
        calls = message.get("tool_calls") or []
        message["tool_calls"] = [
            call.model_copy(update={"id": f"functions.{call.function.name}:{at}"})
            for at, call in enumerate(calls)
        ]
        return message, termination


async def _reply(started: Any, history: list[dict[str, Any]]) -> dict[str, Any]:
    answered = await ask(started, "t", messages=history, tools=TOOLS_OPENAI)
    assert answered.status_code == 200, answered.text
    return answered.json()["choices"][0]["message"]


async def test_a_call_id_an_earlier_call_holds_is_replaced() -> None:
    answers = ["CALL ls {}", "CALL ls {}", "CALL ls {}", "done"]
    async with server(FakeSampler(answers), renderer=PlacedCalls()) as started:
        history: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
        named = []
        for result in ("alpha", "beta", "gamma"):
            said = await _reply(started, history)
            (call,) = said["tool_calls"]
            named.append(call["id"])
            history += [said, {"role": "tool", "tool_call_id": call["id"], "content": result}]
    assert named[0] == "functions.ls:0", "an id no earlier call holds is the renderer's"
    assert len(set(named)) == len(named), f"every turn's call has its own id: {named}"
