# Models Architecture (`models/`)

This package manages model execution backends and bridges training with rollout environments. It exposes an interface for policies, handles Tinker training/sampling sessions, and manages a proxy to record exact generation trajectories from agent harnesses.

Recipes interact only with high-level abstractions (`Policy`, `rollout`, and `update`) and do not import directly from this package.


## 1. Module Overview

| Module | Primary Role |
| --- | --- |
| **`agent.py`** | Defines runtime backends (`Session`, `Frozen`, `Provider`), initial reference policies (`INITIAL`), and scoring routines. |
| **`client.py`** | Wraps the Tinker training client (`TinkerClient`) and formats sequences/token shifts (`shifted`, `datum`) for training passes. |
| **`endpoints.py`** | Configures connection endpoints (`ProxyEndpoint`, `ProviderEndpoint`) that route harness calls to models. |
| **`context.py`** | Utility to insert text (e.g., critic advice) prior to the task instruction via token splicing. |


## 2. Policy Backends

When `Trainer.train` begins, each `Policy` the trainer holds (as an attribute, or in a dict, list or tuple attribute) is bound to a backend:

| Backend | Purpose | Sampling (`sample`) | Logprob Scoring (`logprobs`) | Billing / Cost Attribution |
| --- | --- | --- | --- | --- |
| **`Session`** | The active agent trained on Tinker. | Refused *(sampled externally by harness during `rollout`)*. | 1 batched forward pass via training client, at the current weights, whenever `logprobs` is called (`update` calls it only for sequences recorded without logprobs). | Billed to Tinker (`[agent] model`). |
| **`Frozen`** | Reference model (e.g. `INITIAL`, `tinker://` checkpoint, or baseline). | Refused. | Prefill/cache/sample via Tinker sampling client. | Billed to Tinker as base model. |
| **`Provider`** | External commercial LLMs (OpenAI, Anthropic). | Routed via LiteLLM (concurrency capped at 8, 4 retries). | Refused. | Billed to provider directly. |

> **Note:** Calling `sample` or `logprobs` on a policy before it is bound raises `RuntimeError` (`"not connected yet"`).


## 3. Training & Session Lifecycle

`Trainer.train` orchestrates the Tinker session and harness proxy through four discrete stages:

```
[1. Training Client Init] ──> [2. Frozen Models Init] ──> [3. Restore State] ──> [4. Proxy Startup]
     (Base model + LoRA)        (Open frozen samplers)        (Weights + Optim)       (Serve 'start' weights)

```

1. **Initialize Training Client:** Allocates base model on Tinker with a LoRA adapter (rank defined by `[recipe] lora_rank`, default `32`) or restores from checkpoint.
2. **Open Frozen Models:** Opens a sampling client for each frozen policy before training starts. `INITIAL` opens on the weights as the run began: the base model, or, for a run from `[agent] from_checkpoint`, sampler weights saved as `anchor` and retained for `[checkpoints] ttl_hours`.
3. **Restore Checkpoint:** Re-applies optimizer states and model weights if resuming an existing run.
4. **Launch Proxy:** Starts `harness-proxy` (or connects to the remote one) and points it at the session's current weights, saved as `start` (on a resume, the restored weights).

### State & Checkpoint Retention

* **Serving Weights:** Weights published for sampling (`start`, and subsequent `sample-<n>` after updates) expire automatically after **12 hours** (`SERVE_TTL`) to prevent orphaned resources.
* **Teardown:** On session completion, the proxy closes and the Tinker session transitions to `success`, `errored`, or `interrupted`.


## 4. The Harness Proxy (`harness-proxy`)

The proxy intercepts and logs all LLM requests made by agent harnesses during trials, capturing token IDs and sampling logprobs for exact on-policy data reproduction.

### Deployment Modes

`ProxyEndpoint` determines network topology based on `[rollout]` configuration:

| Mode | Trigger | Routing |
| --- | --- | --- |
| **`remote`** | Explicit `endpoint_url` set | Uses existing instance. Needs `HARNESS_PROXY_TOKEN` (the harnesses' token) and `HARNESS_PROXY_CONTROL_TOKEN` (records and weight swaps). |
| **`local`** | Local sandboxes (`docker`, `podman`, `apple-container`) | Runs locally; exposed to containers via `host.docker.internal`. |
| **`tunnel`** | Other isolated remote environments | Runs locally behind an automated Cloudflare quick tunnel. |

### Post-Job Health Checks

* **`record`:** Drains each trial's records and raises `ValueError` if any names weights other than those the run last served (`answered_by_ours`). A failed fetch raises `ProxyGone` if the proxy has exited.
* **`check`:** Raises `ProxyGone` if the proxy process the run started has exited, or `NothingServed` if no trial reached the proxy or every request failed at the sampler.


## 5. Training Data Pipeline (`update`)

During an `update`, trajectories captured by the proxy are formatted for Tinker:

* **Token Shifting (`shifted`):** Standard autoregressive alignment—inputs are `tokens[:-1]` predicting targets at `tokens[1:]` (target index $p$ mapped to $p - 1$). Image tokens are never targets.
* **Datum Assembly (`datum`):** Constructs training records with per-target advantage values. Sampling logprobs (or, for a sequence recorded without them, logprobs scored at the current weights) and target masks are aligned at $p - 1$.
* **Execution:**
* `forward_backward`: Ships inputs, target IDs, sampling logprobs and advantages to Tinker (masks remain local) to compute the recipe's loss (`importance_sampling` by default) and its gradients, and return the trained weights' logprobs.
* `optim_step`: Runs Adam optimizer step at the scheduled learning rate.



## 6. Token Manipulation (`context.py`)

`insert` allows splicing arbitrary text (e.g., critic feedback) immediately before a task instruction within raw token sequences:

1. Decodes the tokens before the first completion and searches them for the first 200 characters of the task's instruction (`PROBE`).
2. Splices the text's tokens at the last token boundary at or before where the instruction starts.
3. Shifts subsequent completions and multimodal offsets by the inserted length to preserve target alignment.
4. Returns `None` if context length is exceeded or no instruction is detected.


## 7. Testing

| Test Suite | Coverage |
| --- | --- |
| `tests/test_client.py` | Tinker client lifecycle (open, anchor, samplers, saves, close), the pipelined training step and `compute_logprobs`. |
| `tests/test_pack.py` | Verifies `shifted` and `datum` correctness against standard reference implementations. |
| `tests/test_endpoints.py` | Proxy placement and tokens, connection strings, records and post-job checks, and `ProviderEndpoint`'s LiteLLM calls. |
| `tests/test_context.py` | Token splicing and offset alignment mechanics. |
| `tests/test_trainer.py` | End-to-end backend integration tests using the `fakes` fixture (`FakeService`, `FakeProxy`, `FlatAnchor`). |