"""The half of a harness given as a file that runs in the trial's sandbox, beside the file;
`file.py` is the half on this machine. It gives the harness its way to the model, records every
call the harness makes, and says how the harness ended. What follows is all a harness file is
written against.

The file defines `run(task, model)`, and all else in it is the harness's own. `run` is called
once, in the directory the task starts in, with the task's instruction as text.

`model(messages, tools=None, **params)` asks the run's proxy for one chat completion, in OpenAI's
shape, and returns the assistant message: a dict to append to `messages` as it is, with the call's
token counts as `.usage` and why it stopped as `.finish`. It raises `model.Refused`, with the
proxy's reason, for a call the proxy will not answer: a prompt too long for the model's context,
or a trial that has sampled all it may.

The harness reads the rest from its environment: `HARNESS_SECONDS`, the seconds the trial gives
it, after which the runner ends it; `HARNESS_SKILLS`, the directory of the skills the trial
carries, a `<name>/SKILL.md` each, set when it carries any; and the `env` its run.toml gives it.

The record holds a line per call, and the trial's trajectory is made from it. This file runs on
the sandbox's own Python, 3.8 or later, with the standard library alone, as a harness file does."""

import importlib.util
import json
import os
import signal
import sys
import time
import traceback
import urllib.error
import urllib.request

#: Exit status when the model could not be reached: the trial says nothing about the harness.
UNREACHABLE = 75
#: Seconds to wait before each new try of a call that did not come back.
WAITS = (2, 10, 30)
#: Seconds one call may take.
CALL_SECONDS = 900
#: Statuses the proxy answers a call with that it will never answer otherwise.
REFUSALS = (400, 413, 422)


class Refused(Exception):
    """The proxy will not answer this call, and says why."""


class Unreachable(Exception):
    """The model could not be reached, or failed, on every try."""


class OutOfTime(BaseException):
    """The harness ran past the seconds a trial gives it."""


class Reply(dict):
    """The assistant message of one call, as the next call sends it back."""

    #: `prompt_tokens`, `completion_tokens` and `total_tokens` of the call.
    usage = {}
    #: Why the model stopped: `stop`, `tool_calls` or `length`.
    finish = ""


class Model:
    """The harness's way to the model: called with the messages, it returns the reply."""

    Refused = Refused

    def __init__(self, base, key, name, record):
        self.url = base.rstrip("/") + "/chat/completions"
        self.key, self.name, self.record = key, name, record
        # The last call's messages and tools, as JSON: a line holds only what changed.
        self._sent, self._tools = [], None

    def __call__(self, messages, tools=None, **params):
        body = dict(params, model=self.name, messages=messages, stream=True)
        body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
        line = self._line(messages, tools, params)
        started = time.time()
        try:
            reply = self._ask(json.dumps(body).encode("utf-8"))
        except (Refused, Unreachable) as failed:
            self.note(line, seconds=time.time() - started, error=str(failed))
            raise
        self.note(
            line, seconds=time.time() - started, reply=reply, usage=reply.usage, finish=reply.finish
        )
        return reply

    def note(self, line=None, **said):
        """Add a line to the record."""
        self.record.write(json.dumps(dict(line or {}, **said)) + "\n")
        self.record.flush()

    def _line(self, messages, tools, params):
        """Return the call's line: the messages past the ones the last call sent unchanged, and
        the tools when they are not the last call's."""
        sent = [json.dumps(one, sort_keys=True) for one in messages]
        kept = 0
        while kept < min(len(sent), len(self._sent)) and sent[kept] == self._sent[kept]:
            kept += 1
        line = {
            "at": time.time(),
            "kept": kept,
            "messages": list(messages[kept:]),
            "params": params,
        }
        given = json.dumps(tools, sort_keys=True)
        if given != self._tools:
            line["tools"] = tools
        self._sent, self._tools = sent, given
        return line

    def _ask(self, body):
        """Send the call until it comes back: the reply, or `Refused`, or `Unreachable` once
        every try has failed."""
        failed = ""
        for wait in (0,) + WAITS:
            time.sleep(wait)
            request = urllib.request.Request(
                self.url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
            )
            try:
                with urllib.request.urlopen(request, timeout=CALL_SECONDS) as response:
                    return _read(response)
            except urllib.error.HTTPError as answered:
                said = _reason(answered.read().decode("utf-8", "replace")) or str(answered)
                if answered.code in REFUSALS:
                    raise Refused(said) from None
                failed = f"{answered.code}: {said}"
            except (OSError, ValueError, Unreachable) as lost:
                failed = f"{type(lost).__name__}: {lost}"
        raise Unreachable(failed)


def _read(response):
    """Read a streamed completion into a `Reply`. A stream that ends in an error, or without the
    model stopping, is a call that failed."""
    text, reasoning, calls, usage, finish = [], "", [], {}, ""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("error"):
            raise Unreachable(str(event["error"].get("message") or event["error"]))
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            reasoning += delta.get("reasoning") or ""
            for call in delta.get("tool_calls") or []:
                calls.append({key: value for key, value in call.items() if key != "index"})
            finish = choice.get("finish_reason") or finish
    if not finish:
        raise Unreachable("the reply ended before the model stopped")
    reply = Reply(role="assistant", content="".join(text))
    if calls:
        reply["tool_calls"] = calls
    if reasoning:
        reply["reasoning"] = reasoning
    reply.usage, reply.finish = usage, finish
    return reply


def _reason(text):
    """Return the message of an error body, or the body when it is not the proxy's JSON."""
    try:
        return str(json.loads(text)["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return text.strip()


def _harness(path):
    spec = importlib.util.spec_from_file_location("harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _limit(seconds):
    """Raise `OutOfTime` in the harness once `seconds` have passed, and again every few seconds
    after, for a harness that catches it."""

    def over(signum, frame):
        raise OutOfTime()

    signal.signal(signal.SIGALRM, over)
    signal.setitimer(signal.ITIMER_REAL, seconds, 5)


def main():
    """Run the harness on the task, and return the exit status: 0 when it returned or ran out of
    the trial's seconds, `HARNESS_SECONDS`, 1 when it raised, `UNREACHABLE` when the model could
    not be reached."""
    with open(os.environ["HARNESS_TASK"], encoding="utf-8") as held:
        task = held.read()
    with open(os.environ["HARNESS_RECORD"], "a", encoding="utf-8") as record:
        # Taken out of the environment, so the commands the harness runs do not hold them.
        base, key = os.environ.pop("OPENAI_BASE_URL"), os.environ.pop("OPENAI_API_KEY")
        model = Model(base, key, os.environ["HARNESS_MODEL"], record)
        seconds = float(os.environ.get("HARNESS_SECONDS") or 0)
        try:
            if seconds:
                _limit(seconds)
            _harness(os.environ["HARNESS_FILE"]).run(task, model)
        except OutOfTime:
            signal.setitimer(signal.ITIMER_REAL, 0)
            model.note(end="out of time")
            return 0
        except Unreachable as lost:
            model.note(end="unreachable", error=str(lost))
            return UNREACHABLE
        except BaseException:  # noqa: BLE001 - whatever the harness raised is what the trial shows
            model.note(end="raised", error=traceback.format_exc())
            return 1
        signal.setitimer(signal.ITIMER_REAL, 0)
        model.note(end="returned")
        return 0


if __name__ == "__main__":
    sys.exit(main())
