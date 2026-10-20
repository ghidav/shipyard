# harness-proxy

A model server that stands in for a provider to agent harnesses: OpenAI Chat Completions and Anthropic Messages, streaming or not, sampled from Tinker or from a Together post-training session. It records every call of every trial as token ids and sampler logprobs, so a trainer can train on exactly what was sampled.

```mermaid
flowchart LR
    harness["harness"] -->|"OpenAI or Anthropic call"| dialect["dialects"]
    dialect --> chat["chat: complete the call"]
    chat -->|render| rendering["rendering, bridge"]
    chat -->|sample| recorder["recorder: budget, records"]
    recorder --> backend["backends: tinker, together"]
    run["the run, through client"] -->|"records, weights"| recorder
```

On each call it renders with the model's cookbook renderer, images, PDFs and mid-conversation system messages included; builds the prompt from its own tokens when the history quotes one of its replies; answers an SDK retry with the first reply; returns thinking in the API's own format; and keeps slow streams alive.

## Run it

```bash
harness-proxy --model Qwen/Qwen3-8B --advertise host.docker.internal
```

It prints `serving <model> at http://<host>:<port> (control: on)` and serves until SIGTERM. Its tokens come from `HARNESS_PROXY_TOKEN` (the harnesses') and `HARNESS_PROXY_CONTROL_TOKEN` (the run's), or are generated and printed.

| Route | Token | What |
|---|---|---|
| `/r/trial/<trial>/v1/chat/completions`, `/r/trial/<trial>/v1/messages` | harness | the model, recorded under `<trial>` |
| `/r/trial/<trial>/renderer/<renderer>/v1/...` | harness | the same, rendered with one of the proxy's renderers |
| `GET /records/<trial>` | control | drains the trial's records (`?delta=1` sends prompts as extensions; `?again=1` resends the last answer) |
| `POST /control/weights` | control | `{"model_path": "tinker://..." \| null}` switches the weights; trial URLs keep working |
| `GET /healthz` | none | the model, the per-trial budget, and the renderers |

## From Python

```python
from harness_proxy.client import Proxy

proxy = await Proxy.local(settings, host="host.docker.internal")  # or .tunnelled, or .remote
await proxy.ready()
base_url = proxy.address_for("trial-0")  # for the harness, with proxy.token
records = await proxy.records("trial-0")  # harness_proxy.wire.Record, in request order
await proxy.swap("tinker://.../sampler_weights/000010")
proxy.close()
```

- **Renderers:** `--renderer`, given more than once, builds each, such as two reasoning efforts; `address_for("trial-0", name)` renders with one. All of them sample the same weights.
- **Tunnels:** `Proxy.tunnelled` puts a Cloudflare quick tunnel in front, for sandboxes with no route back (Modal, Daytona). A proxy the run starts stops once the run is gone, leaving no process, port, session or tunnel.
- **Together:** `backend = "together"` samples the generator of the run's session (`session`): the weights published last, text only, stopping on strings or at the end of sequence. It needs `harness-proxy[together]` and `TOGETHER_API_KEY`.

## Develop

```bash
cd packages/harness-proxy && pytest
```

The tests run on fake samplers and renderers (`tests/proxies.py`); one that needs a real tokenizer skips when it is not cached.
