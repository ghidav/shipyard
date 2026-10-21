from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker

from harness_proxy.bridge import Index, bridge, chain, digest, digests_of, int_stops
from harness_proxy.chat import Call, Caller, Refused, Reply, thinking_of
from harness_proxy.cookbook import Message, get_text_content
from harness_proxy.vision import has_images, takes_images
from harness_proxy.wire import model_input

if TYPE_CHECKING:
    from harness_proxy.recorder import Recorder

#: What Tinker's 400 for a prompt that does not fit says. A harness compacts its history on the
#: refusal this becomes, so no other refusal may say it.
_OVERFLOW = (
    re.compile(r"prompt length plus max_tokens exceeds", re.IGNORECASE),
    re.compile(r"\d+\s*prompt tokens\s*\+\s*\d+\s*max_tokens\s*>\s*\d+", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
)


@dataclass(frozen=True)
class Prompt:
    """A call's prompt as rendered: what is sampled, the digests of its messages, and for a
    prompt the bridge built, the fresh render that a later prompt's head must equal to extend
    it."""

    model_input: Any
    digests: tuple[str, ...] = ()
    rendered: tuple[int, ...] | None = None

    @property
    def bridged(self) -> bool:
        return self.rendered is not None


async def complete(
    call: Call, caller: Caller, renderer: Any, recorder: Recorder, default_max_tokens: int
) -> Reply:
    """Render the call, sample one reply through `recorder`, parse it, and index it for the
    bridge; raise `Refused` if it cannot be sampled."""
    if call.tools:
        try:
            prefix = renderer.create_conversation_prefix_with_tools(
                call.tools, system_prompt=call.system or ""
            )
        except (NotImplementedError, AttributeError):
            raise Refused(
                "the configured renderer does not support tool calling; omit 'tools'"
            ) from None
        messages = [*prefix, *call.messages]
    elif call.system is not None:
        messages = [Message(role="system", content=call.system), *call.messages]
    else:
        messages = list(call.messages)
    prompt = render(messages, renderer, recorder.index_for(caller.trial))
    stop: list[str] | list[int] = renderer.get_stop_sequences()
    string_stops = bool(stop) and isinstance(stop[0], str)
    if call.stop:
        if stop and not string_stops:
            # SamplingParams.stop holds strings or ids, not both.
            raise Refused(
                "client stop sequences are not supported with this model's renderer, which "
                "uses token-id stop conditions; omit stop/stop_sequences"
            )
        stop = [*[s for s in stop if isinstance(s, str)], *call.stop]
    max_tokens = call.max_tokens or default_max_tokens
    asked = {
        "temperature": call.temperature,
        "top_p": call.top_p,
        "top_k": call.top_k,
        "seed": call.seed,
    }
    try:
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            stop=stop,
            **{key: value for key, value in asked.items() if value is not None},
        )
    except (ValueError, TypeError) as bad:
        raise Refused(f"invalid sampling parameters: {bad}") from None
    length = prompt.model_input.length
    try:
        sampled = await recorder.sample(caller, prompt, params, requested_model=call.model)
    except tinker.BadRequestError as failed:
        if any(pattern.search(str(failed)) for pattern in _OVERFLOW):
            raise Refused(
                f"prompt is too long: {length} prompt tokens plus {max_tokens} "
                "max_tokens exceed the model's context window"
            ) from failed
        raise
    tokens = list(sampled.tokens)
    message, termination = renderer.parse_response(tokens)
    message = with_call_ids(message, tokens, caller, prompt.digests)
    parsed = get_text_content(message)
    made = list(message.get("tool_calls") or [])
    text, calls = parsed, [] if call.no_tools else made
    stop_reason = sampled.stop_reason or "stop"
    client_stop, matched = False, None
    if call.stop and not calls and stop_reason == "stop" and not termination.is_clean:
        # A stop the sampler excluded parses as unclean too, so only a stop seen in the text,
        # or the only kind of stop sampled with, says the harness's own stop fired.
        hits = [(text.find(s), s) for s in call.stop if s in text]
        if hits:
            at, matched = min(hits)
            text, client_stop = text[:at], True
        elif not string_stops:
            client_stop = True
            matched = call.stop[0] if len(call.stop) == 1 else None
    if sampled.seq is not None:
        recorder.reply(
            caller.trial,
            sampled.seq,
            digest(message),
            ended_with_stop=bool(tokens) and int(tokens[-1]) in int_stops(renderer),
            rendered=prompt.rendered,
            call_ids=tuple(str(_call_id(one)) for one in made),
            # The bridge splices a reply's tokens into a history that quotes it as sampled, and
            # the harness never gets a reply cut at its stop or stripped of its calls as such.
            bridgeable=text == parsed and calls == made,
        )
    return Reply(
        text=text,
        thinking=thinking_of(message),
        tool_calls=calls,
        prompt_tokens=length,
        completion_tokens=len(tokens),
        stop_reason=stop_reason,
        client_stop=client_stop,
        matched_stop=matched,
    )


def render(messages: Sequence[Message], renderer: Any, index: Index) -> Prompt:
    """Render `messages`, from the tokens of a sampled reply in `index` they quote when the bridge
    can splice them; `Refused` for an image the model takes none of."""
    messages = list(messages)
    if has_images(messages) and not takes_images(renderer):
        # Answer with a 400 so the harness sees the request was the problem.
        raise Refused(
            "this request carries an image and the model this endpoint serves takes "
            "none: its renderer was built without an image processor"
        )
    digests = digests_of(messages)
    built = bridge(messages, renderer, index, digests)
    if built is not None:
        return Prompt(model_input(built.ids, built.pictures), digests, built.rendered)
    return Prompt(renderer.build_generation_prompt(messages), digests)


def with_call_ids(message: Any, tokens: Any, caller: Caller, digests: Sequence[str]) -> Any:
    """Give every tool call without an id one, derived so a replayed retry gets the same ids."""
    calls = list(message.get("tool_calls") or [])
    if all(_call_id(call) for call in calls):
        return message
    prefix = "toolu_" if caller.wire == "anthropic" else "call_"
    seed = "|".join([caller.trial, chain(digests), ",".join(str(int(token)) for token in tokens)])
    named = []
    for at, call in enumerate(calls):
        if _call_id(call):
            named.append(call)
            continue
        minted = prefix + hashlib.sha256(f"{seed}|{at}".encode()).hexdigest()[:32]
        named.append(
            {**call, "id": minted}
            if isinstance(call, dict)
            else call.model_copy(update={"id": minted})
        )
    return {**message, "tool_calls": named}


def _call_id(call: Any) -> str | None:
    return call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
