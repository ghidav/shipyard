# Security

## Reporting a vulnerability

Report it privately, through the repository's **Security** tab: **Report a vulnerability**. Do not open a public issue or pull request for it. Say what an attacker can do, and how to reproduce it.

## What the proxy exposes

A run that serves Tinker weights starts the **proxy**, `shipyard serve`: an HTTP server that samples from the backend on the run's `TINKER_API_KEY`. Whoever can call it spends that key's credit.

It listens on `[rollout] bind` (default `0.0.0.0`, every interface) and `[rollout] bind_port` (default `0`, any free port). Behind a tunnel, for a remote sandbox, it has a public URL. Its routes:

| Route | What it does | Token |
|---|---|---|
| `GET /healthz` | says the proxy is up, and which model it serves | none |
| `POST /v1/chat/completions`, `POST /v1/messages`, and both under `/r/trial/<trial>/v1/` | samples a reply, OpenAI or Anthropic dialect, and records the call | harness token |
| `GET /records/<trial>` | returns a trial's records (prompt and completion token ids, logprobs) and drains them | harness token |
| `POST /control/weights` | points the proxy at other weights | control token |

### The two tokens

- The **harness token**, `SHIPYARD_PROXY_TOKEN`, opens every route but `/healthz` and `/control/weights`. A request presents it as `Authorization: Bearer <token>` or `x-api-key: <token>`. Every trial's harness gets it as its provider key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), so anything running in the sandbox, the model under training included, can read it.
- The **control token**, `SHIPYARD_CONTROL_TOKEN`, is the only key to `/control/weights`, as `Authorization: Bearer <token>`. It never enters a sandbox: the model under training must not re-point its own proxy.

Both are environment variables and never config keys. A run copies its `run.toml` into its record, and a secret written there would be in every copy. For the same reason, keep secrets out of `[rollout] env`.

When one is unset, the run makes a fresh one for its own proxy and hands it over in the proxy's environment. A `shipyard serve` started by hand without them makes its own and prints them after its first line; treat that output as a secret.

### What the harness token does not protect

The harness token is shared by every trial of a run. A process in one sandbox that holds it can sample on the run's key until the run stops the proxy, and can fetch another trial's records from `/records/<trial>`.
