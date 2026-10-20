# harness-proxy

A model server acting as a drop-in provider for agent harnesses. It supports both the OpenAI Chat Completions and Anthropic Messages APIs (streaming and non-streaming), backed by a Tinker sampling client or by the generator of a Together post-training session. For every trial, each call is logged as exact token IDs and sampler logprobs, providing the precise data needed for downstream training.

On each call the proxy:

- renders the request with the model's cookbook renderer, including images, PDFs and system messages in the middle of the conversation;
- builds the prompt from the tokens it already sampled when the history quotes one of its own replies;
- samples with the run's sampling settings, a per-trial token budget and the model's context length, and answers an SDK retry with the original reply;
- returns the reply's thinking in the API's own format, and sends keepalive frames on slow streams.

## Run it

```bash
harness-proxy --model Qwen/Qwen3-8B --advertise host.docker.internal
# or: python -m harness_proxy.serve ...
```

It prints one line, `serving <model> at http://<host>:<port> (control: on)`, and serves
until SIGTERM. Tokens come from `HARNESS_PROXY_TOKEN` (harness routes) and
`HARNESS_PROXY_CONTROL_TOKEN` (the run's routes: records and weights), or are generated and
printed.

## Routes

| Route | Who | What |
|---|---|---|
| `/r/trial/<trial>/v1/chat/completions`, `/r/trial/<trial>/v1/messages` | harness, with the harness token | the model, recorded under `<trial>` |
| `/r/trial/<trial>/renderer/<renderer>/v1/...` | harness, with the harness token | the same, rendered with `<renderer>`, one of the proxy's; any other is refused |
| `GET /records/<trial>` | trainer, with the control token | drains the trial's records (`?delta=1` sends prompts as extensions of earlier records; `?again=1` resends the last answer) |
| `POST /control/weights` | trainer, with the control token | `{"model_path": "tinker://..." \| null}` points the proxy at new weights (on Together, `together://...` names the weights the run published); trial URLs keep working |
| `GET /healthz` | anyone | the served model, the per-trial budget, and the renderers a trial's address may name |

## From Python

```python
from harness_proxy.client import Proxy

# Or Proxy.tunnelled, or Proxy.remote for one already running elsewhere.
proxy = await Proxy.local(settings, host="host.docker.internal")
await proxy.ready()
base_url = proxy.address_for("trial-0")  # give this to the harness, with proxy.token
...
records = await proxy.records("trial-0")  # harness_proxy.wire.Record, in request order
await proxy.swap("tinker://.../sampler_weights/000010")
proxy.close()
```

`--renderer` (the `renderer` setting) overrides the model's own cookbook renderer. Given more than
once, such as `--renderer qwen3_8_low_reasoning --renderer qwen3_8_xhigh_reasoning` for two
reasoning efforts, it builds each, and `proxy.address_for("trial-0", "qwen3_8_low_reasoning")`
is an address whose calls are rendered with that one; an address that names none takes the
first. Every renderer samples the same weights.

`Proxy.tunnelled` puts a Cloudflare quick tunnel in front of a local proxy, for sandboxes
with no route back to this machine (Modal, Daytona). The proxy opens the tunnel and closes it when
it stops. A proxy the run starts stops once the run is gone, even a run killed outright, so it
leaves no process, port, Tinker session or tunnel behind.

## Backends

The `backend` setting picks what the proxy samples: `tinker` (the default), or `together`,
which attaches to the run's Together post-training session, named by the `session` setting,
and samples its generator. That one serves the weights the run published last, takes text
only, and stops on strings or at the end of sequence. It needs the `together` extra
(`harness-proxy[together]`) and `TOGETHER_API_KEY`.

## Inside

A call runs through the following pipeline.

| Module | What it does |
|---|---|
| `dialects/openai.py`, `dialects/anthropic.py` | read a request into a `Call`; write the reply as JSON or SSE, and refusals as the wire's 400 |
| `chat.py` | completes a `Call`: render, sample, parse; the checks both wires share |
| `rendering.py`, `bridge.py` | the renderer a call goes through: the bridge first, then tool call ids and the reply index |
| `recorder.py` | samples from the backend with the run's settings, budget and context; replays retries; keeps the records |
| `backends/tinker.py`, `backends/together.py` | what a call samples from: a Tinker sampling client on any saved weights, or the generator of the run's Together session |
| `vision.py` | images and PDF pages as image parts |
| `keepalive.py` | SSE frames while a stream waits on its sample |
| `server.py`, `serve.py`, `tunnel.py` | the HTTP server; the process, with the tunnel it opens |
| `client.py`, `wire.py` | the run's side, and the `Record` both sides share |
| `cookbook.py` | the one import of `tinker_cookbook`: its renderers and tokenizers |

## Develop

```bash
cd packages/harness-proxy && pytest
```

The tests run on fake samplers and renderers (`tests/proxies.py`): no Tinker, no tokenizer. A test that needs a real tokenizer skips when it is not cached.
