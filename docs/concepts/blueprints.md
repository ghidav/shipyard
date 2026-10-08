# Blueprints

A blueprint is a directory holding one file, `run.toml`, and optionally the modules directory
that `[recipe] modules` names. It says everything a run needs: the model, the tasks, how the
tasks are rolled out, and what the run does with the results.

`shipyard check` reads a blueprint and says whether it could run. `shipyard run` executes
it and writes the record of what happened, a [run](runs.md).

## The file

```toml
# blueprints/02-eval-tinker-docker/run.toml
[model]
name = "Qwen/Qwen3-8B"

[data]
dataset = "hello-world"
batch_size = 1
group_size = 2

[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 2
max_tokens = 4096

[recipe]
kind = "evaluate"
```

Five tables, each with one job:

| Table | Says |
|---|---|
| `[model]` | which model, and who serves it |
| `[data]` | which [datasets](datasets.md), and how they are cut into batches |
| `[rollout]` | which harness runs each task, in which sandbox, sampled how |
| `[recipe]` | what the run does with the rollouts; `kind` names it |
| `[checkpoints]` | when a training run saves its weights; optional |

The first four tables are required. Every key, with its type and default, is in the
[configuration reference](../reference/configuration.md).

A model is *served* when `[model] provider` is `"tinker"`, the default: the run starts its
own [proxy](proxy.md) and samples the weights from a Tinker-API backend. Any other provider,
such as `"anthropic"`, means the harness calls that provider itself.

## Recipe names

`[recipe] kind` is one of five names:

| Kind | Does |
|---|---|
| `dapo`, `dr-grpo`, `cispo` | train the served weights by gradient |
| `gepa` | search over the text in `modules/`; the weights never move |
| `evaluate` | measure the policy; nothing changes |

A name is a preset: it fixes how the advantage is formed and which loss and clipping are
used. Each name takes its own knobs; a knob from another recipe is an unknown key. See
[Recipes](recipes.md).

A gradient recipe trains the weights the run serves. With any provider but `"tinker"`, the
run fails as it starts, saying `[recipe] kind = 'dapo' trains the weights this run serves`.

## Every problem at once

A key the schema does not know is an error. `check` lists every problem in one pass, one per
line, as `[table] key: problem`:

```toml
[model]
name = "Qwen/Qwen3-8B"
base = "Qwen/Qwen3-8B"

[data]
dataset = "hello-world"
batch_size = 0

[rollout]
harness = "pi"
model = "Qwen/Qwen3-8B"

[recipe]
kind = "dapo"
learning_rate = 2e-5
clip = 0.2

[logging]
wandb = true
```

```text
blocked  [model] base: unknown key
blocked  [data] batch_size: Input should be greater than or equal to 1
blocked  [rollout] model: unknown key
blocked  [recipe] clip: unknown key
blocked  [logging]: unknown table
```

`clip` is a `dr-grpo` knob, so `dapo` refuses it. A misspelt kind names the five:

```text
blocked  [recipe] kind: no recipe called 'grpo'; one of dapo, dr-grpo, cispo, gepa, evaluate
```

## What check prints

`check` prints three things, in order.

**The config as resolved.** Every table, with the defaults filled in, as TOML. A key that is
unset and has no default, such as `[rollout] timeout`, gets no line.

**One comment line, for a gradient recipe only.** It says what the name and its knobs resolve to:

```text
[checkpoints]
every = 1
ttl_hours = 168.0
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-08, no warm-up (the paper's is 20 steps); degenerate groups dropped and refilled from the plan, up to 9 more rounds
```

**The findings.** Each is `ok`, `warning` or `blocked`. The `ok` lines are hidden unless you
pass `--verbose`:

```text
$ shipyard check blueprints/02-eval-tinker-docker --verbose
...
ok  recipe evaluate
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
ok  dataset hello-world
ok  harness pi@0.85.1, sandbox docker
ok  dataset hello-world: 1 task under tasks/hello-world
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

Without `--verbose`, a blueprint with nothing to report prints the config alone. A warning
always shows:

```text
warning  no profile for harness codex; generic OpenAI wiring
```

`check` exits 1 when any finding is `blocked`, and 0 otherwise; a warning does not block.
`shipyard run` makes the same checks first, and on a `blocked` it prints
`blocked; nothing was started`.

What `check` looks at includes:

- each dataset under `tasks/` in the working directory, with its task count;
- the `modules` directory, when the recipe names one;
- for `gepa`, the reflector's harness, model and image;
- that the harness is named and the sandbox is one Harbor knows;
- for a `docker` sandbox, that Docker is running, and whether earlier runs left 20 or more
  trial networks behind, as a warning;
- for a `modal` sandbox with a token, whether Modal takes it: a network call, a warning
  when Modal does not answer within 15 seconds;
- that `[rollout] env` holds no key starting with `TINKER_` and no `SHIPYARD_CONTROL_TOKEN`,
  which must never enter a sandbox;
- a key that `.env` assigns twice, as a warning, since the last one wins;
- for a served model: where the proxy will stand, whether the harness has a profile,
  whether `[rollout] env` switches its compaction off without `fill_context`, and whether
  the backend serves the model. That last
  one is a network call, made only when `TINKER_API_KEY` is set; without it, `check` warns.

`shipyard` reads `.env` in the working directory before any command; a variable already set
in the shell wins.

## As JSON

`--json` prints one object, `{"config": ..., "findings": [{"level": ..., "text": ...}]}`:
the resolved config, with unset keys as `null`, and every finding, `ok` included. It carries
no comment line. `config` is `null` when the file did not load. The exit code is the same.
