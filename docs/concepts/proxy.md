# The proxy

The **proxy** is one process, `shipyard serve`, that stands in for a model
provider. It serves the run's weights to every harness in every sandbox, and
keeps a **record** of every model call. A run with a served model (see
[Rollouts](rollouts.md)) starts it before its first job and stops it when the
run ends. Its output goes to `runs/<id>/proxy.log`, which holds the line the
run waits for:

```text
serving Qwen/Qwen3-8B at http://host.docker.internal:56479 (control: on)
```

## Where it runs

For a sandbox on this machine (`docker`, `podman`, `apple-container`), the
proxy runs here too. It listens on `[rollout] bind` and `bind_port` (every
interface and any free port, by default), and the sandboxes reach it as
`[rollout] host`, `host.docker.internal` by default. A remote sandbox reaches
the proxy through a tunnel the run starts.

Two tokens guard the proxy, both environment variables and never config
keys, since `run.toml` is copied into the run directory. The harness token,
`SHIPYARD_PROXY_TOKEN`, opens every route but `/healthz` and
`/control/weights`, and every sandbox holds it. One of those routes is
`GET /records/<trial>`, which hands over a trial's records and drains them, so
a sandbox could read another trial's. The control token,
`SHIPYARD_CONTROL_TOKEN`, alone opens `POST /control/weights`, which points the
proxy at new weights: the model under training cannot re-point its own proxy.
Before each batch, a training
run publishes its weights and points the proxy at them; the addresses already
handed out keep answering.

A run that starts the proxy takes both tokens from its environment, or makes
them. `[rollout] endpoint_url` names a proxy the run does not start, one
started elsewhere with `shipyard serve`. The run then needs the harness token
in its environment (`check` blocks without it), and the control token to point
that proxy at new weights. It asks the proxy's `/healthz` before it opens any
sandbox.

## Two dialects

Each trial has its own address, `http://<host>:<port>/r/trial/<trial>/v1`.
Under it the proxy speaks both dialects: `POST .../chat/completions` (OpenAI)
and `POST .../messages` (Anthropic). The harness's profile picks one. The
proxy renders the messages to tokens with the renderer the Tinker cookbook
recommends for the model (`[rollout] renderer` names another), and samples on
Tinker. The `model` a request names is recorded and ignored: the proxy serves
one model.

## What a record holds

One record per model call:

| field | what it holds |
|---|---|
| `seq`, `at` | the call's number in its trial, from 1, and its time |
| `prompt_token_ids`, `completion_token_ids` | what the model read, and what it wrote |
| `inference_logprobs` | the sampler's logprob of each completion token |
| `stop_reason`, `sampling_params`, `sample_ms` | why it stopped, the parameters it was sampled with, how long it took |
| `requested_model` | the model name the harness sent |
| `served` | the `tinker://` path that answered; `null` for the base model |
| `cached_tokens` | prompt tokens the backend's cache served |
| `request_id` | the request's `Idempotency-Key` or `x-request-id` |
| `bridged` | whether the bridge built the prompt (below) |
| `prompt_digests`, `reply_digest` | a digest per prompt message, and one of the parsed reply |
| `error` | set on a call that was refused or failed, which wrote nothing |

After each job the run drains every trial's records and writes one row per
record to `runs/<id>/requests.jsonl`. One trial's two rows from the live run,
with `at`, `job`, `sample_ms` and `served` left out:

```json
{"trial": "hello-world__csFTaNW", "seq": 1, "prompt_tokens": 1424, "cached_tokens": 0, "completion_tokens": 239, "stop_reason": "stop", "request_id": null, "bridged": false, "error": null}
{"trial": "hello-world__csFTaNW", "seq": 2, "prompt_tokens": 1681, "cached_tokens": 1408, "completion_tokens": 128, "stop_reason": "stop", "request_id": null, "bridged": true, "error": null}
```

The run also checks that every record of a job was served by the weights it
pointed the proxy at, and stops if one was not.

## Pinned sampling

`[rollout] temperature`, `top_p` and `top_k` (defaults `1.0`, `1.0`, `-1`: the
model's whole distribution) replace whatever the harness sends, so every
rollout samples the distribution the run chose. `[rollout] max_tokens`
(default `8192`) only caps: a call that asks for more, or names no limit, gets
`max_tokens`; one that asks for less keeps its own.

## Volatile lines

Claude Code's system prompt carries a per-request `<total_tokens>` line. A
line that changes per request makes each prompt differ from the
last at the system prompt: nothing is cached, and no call extends the one
before, so every call forks (below).
The proxy cuts the profile's volatile lines from system messages before
rendering, and the job row counts the cuts as `cut`.
`[rollout] cut_volatile = false` keeps them.

## The token budget and the context

`[rollout] max_context` is the model's context length; `0`, the default, asks
the backend. When no length is known, nothing is refused. Two refusals follow
from it, each a 400 to the harness and a record with `error` set:

- **Context fit.** A call whose prompt plus `max_tokens` exceeds `max_context`
  is refused with `error = "context"`, as a "prompt is too long" 400: the
  signal a harness compacts its history on. With `[rollout] fill_context =
  true`, a call whose prompt still fits is served instead, its `max_tokens`
  cut to the room left.
- **Token budget.** A trial may sample `max_context` tokens in all. A call
  that would overrun is cut to what is left; a call after the budget is spent
  is refused with `error = "budget"` and a 400 that does not read as an
  overflow, so the harness does not compact and retry:
  `This rollout has spent its budget: ... sampled tokens of ..., the model's context length. Nothing more is served to it.`

[Admission](admission.md) reads both: a budget cut scores 0, and so does a
filled context the harness then gave up on; a first call that never fit is
masked.

## Retries are replayed

The OpenAI and Anthropic SDKs mark a retry with `x-stainless-retry-count`. A
retry gets the stored answer when its trial already sent the same prompt with
the same parameters, or the same `Idempotency-Key`: no second sample, no
second record. A first attempt with an `Idempotency-Key` is matched by that
key alone, so the same prompt under a new key is a new sample. A call that
failed stored nothing, and its retry samples anew.

## Thinking goes both ways

A reply's thinking goes back to the harness in its dialect's own form: a
`thinking` block ahead of the text on the Anthropic wire, with a signature
that is a digest of the text and that the proxy never checks; `reasoning` and
one `reasoning_details` entry on the OpenAI wire. Thinking the harness sends
back (Anthropic `thinking` blocks; OpenAI `reasoning_content`, `reasoning`,
`reasoning_text` or `reasoning_details`) becomes the message's thinking again.
`redacted_thinking` blocks are dropped. A harness that drops a reply's thinking
has changed the message, and its next call forks (below).

## Images and PDFs

The proxy takes images and documents on both wires: Anthropic `image` and
`document` blocks, also inside tool results; OpenAI `image_url` parts and
`file` parts. Images and PDFs come as base64 bytes; an image given by URL is
refused. An image is scaled to at most 1536 pixels on its longest side. A PDF
becomes one image per page, at most 16 pages, with a note when it is cut. An
image in a system prompt is replaced by a note saying so. A model whose
renderer has no image processor refuses any image with a 400.

Each image token stands as `-1` in the record's prompt ids, so the record
keeps its length; [admission](admission.md) masks the rollout `multimodal`.

## Forks and the bridge

Training reads the records as **sequences**, by the token-prefix rule: a call
joins the sequence whose last prompt plus completion begins its prompt, or
else starts a new one, a **fork**. Each completion is a target exactly once,
and the model is trained on exactly what it read, so a fork is always correct.
In a tool loop it is wasteful: a fresh render of the re-sent history need not
give back the tokens the model sampled, so each call would fork, and ten calls
would train as ten sequences, each repeating the one before.

The **bridge** keeps a tool loop as one sequence. When a call quotes a reply
this proxy sampled in the same trial, the proxy builds the prompt from that
reply's tokens. Every rule must hold, or the call is rendered afresh:

1. **Exact message match.** The call's messages, up to and including the
   reply, digest the same as an earlier call's messages plus its reply. A
   digest covers role, text, thinking, images by their pixels, tool calls by
   name and parsed arguments, and a tool result's tool name; never a tool-call
   id. The longest such match wins.
2. The reply is not the call's first message.
3. After the reply come only tool results, optionally closed by one user
   message. An empty tail is a resample, not an extension. The closing user
   message is refused when the renderer drops thinking from earlier turns, as
   Qwen3's does: the fresh render of that turn drops the reply's thinking,
   and the bridge would keep it.
4. The renderer has stop tokens that are token ids.
5. **Exact token equality.** The fresh render of the messages before the reply
   equals, token for token, the fresh render of the earlier call's messages.
6. The fresh render of the whole call begins with that head, and has a stop
   token after it, where the reply ends.

The prompt is then the earlier call's prompt, the reply's completion (and its
stop token, if the sampler left it off), and the fresh render after that stop.

The live run shows it. In trial `hello-world__csFTaNW`, pi's first call had a
prompt of 1,424 tokens; the model sampled 239: its thinking and a call to
`write`. pi ran the tool and called again with the history and the tool
result. That history matched the first call and its reply, and its tail was
one tool result, so the bridge built the prompt: the 1,424 prompt tokens, the
239 sampled ones, then 18 from the fresh render (the tool result and the
opening of the next turn). The second prompt was 1,681 tokens, beginning with
the first call's 1,663, and its record says `"bridged": true`. A gradient
recipe would read the trial as one sequence of 1,809 tokens with two targets:
the 239 tokens and the 128 that followed. The other trial bridged the same
way, and the job row counts both calls as `"bridged": 2`.
