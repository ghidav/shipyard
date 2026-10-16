# Rollouts

A **trial** is one task attempted once, in one container, by one harness. A
**harness** is the agent program in that container: `pi`, `claude-code`,
`opencode`, `terminus-2`, or any other Harbor agent. A **rollout** is a trial
seen from training: the model calls it made, and its reward.

## A batch is one Harbor job

A recipe gives the run one batch of tasks at a time (see
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
[admission](admission.md) masks it `env_error`. The task's group keeps its size.

## Served or provider-served

`[agent] provider` decides who answers the harness.

- `provider = "tinker"`, the default: the model is **served**. The run starts
  [the proxy](proxy.md), and every trial calls the model through it. The proxy
  records every call. Only a served model can be trained, since the gradient
  recipes train on those records.
- Any other provider: the model is **provider-served**. The harness calls the
  provider itself, under `[agent] model` as written, with the provider's key
  from your environment. The token counts are the ones the harness reported
  to Harbor.

Two real blueprints, one of each:

```toml
# blueprints/01-eval-provider-docker/run.toml
[agent]
model = "claude-sonnet-5-5"
provider = "anthropic"

# blueprints/02-eval-tinker-docker/run.toml
[agent]
model = "Qwen/Qwen3-8B"
```

`shipyard check --verbose` says which is which:

```text
ok  model claude-sonnet-5-5, provider anthropic
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
```

Each job row records it as `"served": "provider"` or `"served": "tinker"`.
A gradient recipe on a provider-served model stops with:

```text
[recipe] kind = 'dapo' trains the weights this run serves, and [agent] provider = 'anthropic' serves them elsewhere
```

## How a harness reaches the proxy

Harbor's agents reach a model provider through a base URL and an API key in
their environment, and take a model named `<provider>/<model>`. This is
Harbor's model connection. For a served model, each trial gets:

- `OPENAI_BASE_URL`: the proxy's address, with the trial's name in it;
- `OPENAI_API_KEY`: the proxy's harness token;
- the model name `openai/<[agent] model>`.

A harness whose [profile](#harness-profiles) speaks the Anthropic dialect gets `ANTHROPIC_BASE_URL`,
`ANTHROPIC_API_KEY` and `anthropic/<[agent] model>` instead. opencode's profile
names a provider of its own, so opencode gets `SHIPYARD_BASE_URL`,
`SHIPYARD_API_KEY` and `shipyard/<[agent] model>`. Here is a trial's
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

The trial's name in the address tells the proxy which trial a call belongs to.
These variables win over `[agent] env`. When a task's agent phase runs in
Harbor's allowlist network mode, the proxy's host is added to the agent's
allowed hosts.

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
`[agent] kwargs` table by table. An `opencode_config` in the blueprint keeps
its own keys, such as a model's limits under `provider.shipyard.models`. The
profile's keys win where both set one.

!!! note "terminus-2"
    terminus-2 cannot use a served model in v1. It makes its model calls
    through litellm inside the Harbor process, from its `api_base` option and
    the host's environment, so the per-trial environment above does not reach
    them. Since it has a profile, `check` does not warn.

A version follows the name: `harness = "pi@0.85.1"` runs the agent `pi` and
passes Harbor `0.85.1` as its `version` kwarg, as in the `config.json` above. A
name with a colon (an import path) is passed whole.

Any other Harbor agent runs with the generic profile: the OpenAI dialect,
nothing added. For a served model, `check` and `run` both say so:

```text
warning  no profile for harness <harness>; generic OpenAI wiring
```

## The keys that shape a trial

The served run above used:

```toml
[agent]
model = "Qwen/Qwen3-8B"
harness = "pi@0.85.1"

[rollout]
sandbox = "docker"
concurrency = 2
max_tokens = 4096
```

What acts in the trial is `[agent]`'s:

| key | default | what it does |
|---|---|---|
| `harness` | required | the Harbor agent, optionally `name@version`; `check` blocks an empty one, since Harbor reads an unnamed agent as its oracle, which runs the task's reference solution |
| `env` | `{}` | environment variables for every trial's agent; `check` blocks a key starting with `TINKER_`, and `SHIPYARD_CONTROL_TOKEN`, and a `${...}` value naming one |
| `kwargs` | `{}` | agent kwargs for every trial |

How the trial runs is `[rollout]`'s:

| key | default | what it does |
|---|---|---|
| `sandbox` | `"docker"` | Harbor's environment type: `docker`, `podman`, `apple-container`, `modal`, and the others Harbor lists; `check` blocks a name Harbor does not know |
| `concurrency` | `4` | how many trials of a job run at once |
| `timeout` | unset | overrides the agent's timeout, in seconds |
| `setup_timeout` | unset | overrides the agent's setup timeout, in seconds |
| `jobs_dir` | `"jobs"` | where the job directories go |
| `fill_context` | `false` | lays the profile's environment on each served trial, and lets a turn fill the context (see [The proxy](proxy.md)); `check` warns when `[agent] env` sets one of the profile's keys to the profile's own value without it, since the proxy then still refuses an overflowing call |
| `check_turns` | `false` | compares the harness's own turn count with the proxy's records (see [Admission](admission.md)) |
| `secrets` | `{}` | a variable each trial holds alone, one value per trial in flight (see [Secrets that are identities](#secrets-that-are-identities)) |

The other `[rollout]` keys configure the proxy (see [The proxy](proxy.md)).
Every key with its default is in
[Configuration](../reference/configuration.md).

## Secrets that are identities

A task reads a secret from its environment: a `${BAUPLAN_API_KEY}` in its `[environment.env]`
is filled by Harbor from the process that runs it. A secret every trial may share needs nothing
more. A secret that is an identity does: a service keeps state per identity, so trials that
run at once under one key wipe and read each other's state. Such a secret needs one value per
trial in flight.

`[rollout.secrets]` names each such variable, and the variable of shipyard's own environment
holding its values, separated by commas. The values stay in `.env`, never in `run.toml`:

```toml
[rollout.secrets]
BAUPLAN_API_KEY = "BAUPLAN_API_KEYS"
```

```sh title=".env"
BAUPLAN_API_KEYS=key-one,key-two,key-three
```

Each trial acquires one value before Harbor starts it and gives it back after its teardown, so
the task's prepare step, the agent, the healthcheck and the tests all see the same identity,
and no other trial holds it meanwhile. A trial waits for a free one, so no more trials run at
once than there are values, whatever `concurrency` says. Several variables pair by position: the
second trial in flight might hold the second key and the second user.

The value reaches the trial's environment and its verifier's, over what the task's
`${BAUPLAN_API_KEY}` resolves to. shipyard sets the variable to `""` in its own environment, so
the template always resolves and a path that missed the trial's value authenticates as nobody.
After the trial, the value is replaced with `${BAUPLAN_API_KEY}` in every text file of the
trial's directory, Harbor's saved config and the agent's logs included. It is never in
`run.toml`, the `check` output, or a job row. An agent that holds the key can still read it and
send it to the model, and then it is in that call's tokens: give each identity no more than the
task needs.

`check` blocks a variable that is unset or empty, pools of different sizes, a name that
`[agent] env` also sets, and a run secret such as `TINKER_API_KEY`. It warns when `concurrency` is
above the number of values.

### Secrets of your own

A trainer decides what each trial holds through its `secrets` method, which returns the pools
by default. For other logic, such as a key made for each trial and revoked after it, a trainer
file returns its own `Secrets`. A trainer file is a Python file defining one `Trainer`
subclass, passed after the blueprint: `shipyard run blueprints/bauplan mine.py`.

```python title="mine.py"
from shipyard.sampling import Secrets
from shipyard.training import DAPOTrainer


class FreshUser(Secrets):
    async def acquire(self, trial):
        user = await admin.create_user(f"shipyard-{trial.name}")
        return {"BAUPLAN_API_KEY": user.key}

    async def release(self, trial, held):
        await admin.delete_user(held["BAUPLAN_API_KEY"])


class Mine(DAPOTrainer):
    def secrets(self):
        return FreshUser()
```

`acquire` gets the trial, so it can also keep one identity per task or draw from a vault, and
it may wait, which bounds how many trials run at once. The rest stays shipyard's: when a trial
acquires and releases, where the values go, and the scrub. `check` prints
`secrets from Mine.secrets` and cannot look further; `acquire` raises at the first trial if
something is wrong.
