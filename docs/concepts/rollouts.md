# Rollouts

A **trial** is one task attempted once, in one container, by one harness. A
**harness** is the agent program in that container: `pi`, `claude-code`,
`opencode`, `terminus-2`, or any other Harbor agent. A **rollout** is a trial
seen from training: the model calls it made, and its reward.

## A batch is one Harbor job

A recipe hands the run one batch of tasks at a time (see
[Datasets](datasets.md)). The run turns each batch into one Harbor **job**,
named `<run id>-NNNN` and kept under `jobs/`. The job holds one trial per task
and rollout: each task of the batch, `[data] group_size` times. This order is
the **plan order**.

Up to `[rollout] concurrency` trials run at once. They finish in any order and
come back in plan order, so the rollouts of one task always sit together.

A batch of one task with `group_size = 2` makes this job:

```text
jobs/02-eval-tinker-docker__eTNXY25-0000/
  hello-world__csFTaNW/
  hello-world__XdGKSa2/
```

Harbor names each trial `<task>__<7 chars>` and writes its own files there:
`config.json`, `result.json`, `trial.log`, `agent/`, `verifier/`. The
verifier grades every trial, and the container is deleted when the trial ends.

A trial that crashes keeps its slot. If Harbor raises, the run logs
`trial <name> could not run` and goes on. The slot has no `result.json`, and
[admission](admission.md) masks it `env_error`. The task's group keeps its
size, so no rollout slides into another task's group.

## Served or provider-served

`[model] provider` decides who answers the harness.

- `provider = "tinker"`, the default: the model is **served**. The run starts
  [the proxy](proxy.md), and every trial calls the model through it. The proxy
  records every call. Only a served model can be trained, since the gradient
  recipes train on those records.
- Any other provider: the model is **provider-served**. The harness calls the
  provider itself, under `[model] name` as written, with the provider's key
  from your environment. The token counts are the ones the harness reported
  to Harbor.

Two real blueprints, one of each:

```toml
# blueprints/01-eval-provider-docker/run.toml
[model]
name = "claude-sonnet-5-5"
provider = "anthropic"

# blueprints/02-eval-tinker-docker/run.toml
[model]
name = "Qwen/Qwen3-8B"
```

`shipyard check --verbose` says which is which:

```text
ok  model claude-sonnet-5-5, provider anthropic
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
```

Each job row says it again, as `"served": "provider"` or `"served": "tinker"`.
A gradient recipe on a provider-served model stops with:

```text
[recipe] kind = 'dapo' trains the weights this run serves, and [model] provider = 'anthropic' serves them elsewhere
```

## How a harness reaches the proxy

Harbor's agents already know how to reach a model provider. Each reads a base
URL and an API key from its environment, and takes a model named
`<provider>/<model>`. This is Harbor's model connection. For a served model,
each trial gets:

- `OPENAI_BASE_URL`: the proxy's address, with the trial's name in it;
- `OPENAI_API_KEY`: the proxy's harness token;
- the model name `openai/<[model] name>`.

A harness whose [profile](#harness-profiles) speaks the Anthropic dialect gets `ANTHROPIC_BASE_URL`,
`ANTHROPIC_API_KEY` and `anthropic/<[model] name>` instead. opencode's profile
names a provider of its own, so opencode gets `SHIPYARD_BASE_URL`,
`SHIPYARD_API_KEY` and `shipyard/<[model] name>`. Here is a trial's
`config.json` from the served run above, the token elided:

```json
"agent": {
    "name": "pi",
    "model_name": "openai/Qwen/Qwen3-8B",
    "kwargs": {"version": "0.85.1", "model_api": "openai-completions"},
    "env": {
        "OPENAI_BASE_URL": "http://host.docker.internal:56479/r/trial/hello-world__csFTaNW/v1",
        "OPENAI_API_KEY": "****"
    }
}
```

The trial's name in the address is how the proxy files each call under its
trial. These names win over `[rollout] env`. When a task's
agent phase runs in Harbor's allowlist network mode, the proxy's host is added
to the agent's allowed hosts.

## Harness profiles

A **profile** is shipyard's small entry for one harness: the dialect it
speaks, and what it needs beyond the model connection. Four harnesses have one:

| harness | dialect | what the profile adds |
|---|---|---|
| `pi` | OpenAI | the agent kwarg `model_api = "openai-completions"`; its log's last stop reason, which says whether its last call failed (see [Admission](admission.md)) |
| `claude-code` | Anthropic | the base URL without `/v1`, which Claude Code appends itself; its per-request `<total_tokens>` system line, cut unless `[rollout] cut_volatile = false`; `DISABLE_COMPACT=1` and `DISABLE_AUTO_COMPACT=1` when `[rollout] fill_context = true`; a turn counter: distinct request ids in its log |
| `opencode` | OpenAI | the provider `shipyard` in the agent kwarg `opencode_config`: opencode's bundled `@ai-sdk/openai-compatible` package, which posts to `/chat/completions`, its base URL and key read from `SHIPYARD_BASE_URL` and `SHIPYARD_API_KEY`; `agent.title.disable = true` in the same config, so it makes no title call; a turn counter: lines carrying `step-start` in its log |
| `terminus-2` | OpenAI | nothing |

The turn counters serve `[rollout] check_turns` (see
[Admission](admission.md)). The profile's kwargs are laid over
`[rollout] kwargs` table by table: an `opencode_config` in the blueprint keeps
its own keys, such as a model's limits under `provider.shipyard.models`, and
the profile's keys win where both set one.

!!! note "terminus-2"
    terminus-2 cannot use a served model in v1. It makes its model calls
    through litellm inside the Harbor process, from its `api_base` option and
    the host's environment, so the per-trial environment above never reaches
    them. Since it has a profile, `check` does not warn.

A version rides on the name: `harness = "pi@0.85.1"` runs the agent `pi` and
hands Harbor `0.85.1` as its `version` kwarg, as in the `config.json` above. A
name with a colon, an import path, is passed whole.

Any other Harbor agent runs with the generic profile: the OpenAI dialect,
nothing added. For a served model, `check` and `run` both say so:

```text
warning  no profile for harness <harness>; generic OpenAI wiring
```

## The [rollout] keys that shape a trial

The served run above used:

```toml
[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 2
max_tokens = 4096
```

| key | default | what it does |
|---|---|---|
| `harness` | required | the Harbor agent, optionally `name@version`; `check` blocks an empty one, since Harbor reads an unnamed agent as its oracle, which runs the task's reference solution |
| `sandbox` | `"docker"` | Harbor's environment type: `docker`, `podman`, `apple-container`, `modal`, and the others Harbor lists; `check` blocks a name Harbor does not know |
| `concurrency` | `4` | how many trials of a job run at once |
| `timeout` | unset | overrides the agent's timeout, in seconds |
| `setup_timeout` | unset | overrides the agent's setup timeout, in seconds |
| `env` | `{}` | environment variables for every trial's agent; `check` blocks a key starting with `TINKER_`, and `SHIPYARD_CONTROL_TOKEN`, and a `${...}` value naming one |
| `kwargs` | `{}` | agent kwargs for every trial |
| `jobs_dir` | `"jobs"` | where the job directories go |
| `fill_context` | `false` | lays the profile's environment on each served trial, and lets a turn fill the context (see [The proxy](proxy.md)); `check` warns when `env` sets one of the profile's keys to the profile's own value without it, since the proxy then still refuses an overflowing call |
| `check_turns` | `false` | compares the harness's own turn count with the proxy's records (see [Admission](admission.md)) |

The other `[rollout]` keys shape the proxy, not the trial: see
[The proxy](proxy.md). Every key with its default is in
[Configuration](../reference/configuration.md).
