# shipyard

shipyard post-trains models on Harbor jobs.

You write a **blueprint**: a directory holding a `run.toml`. `shipyard run` reads it and writes a **run**: a directory that records what happened.

## Blueprints in, runs out

This blueprint measures Qwen3-8B, served from Tinker. The `pi` harness, the agent program that attempts each task, runs in a Docker sandbox:

```toml title="blueprints/02-eval-tinker-docker/run.toml"
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

`shipyard run blueprints/02-eval-tinker-docker` left this run:

```
runs/02-eval-tinker-docker__eTNXY25/
├── run.toml         the blueprint, copied byte for byte
├── process.json     when it ran and how it ended
├── metrics.jsonl    a row per step or batch, then one for the run
├── jobs.jsonl       a row per Harbor job
├── requests.jsonl   a row per model call
├── costs.json       tokens per party, sandbox seconds
└── proxy.log        what the proxy, the process serving the weights, printed
```

A training run adds `checkpoints.jsonl`. A run that carries or evolves skills adds `modules/`. [Runs](concepts/runs.md) describes every file.

## Where Harbor and Tinker sit

A **dataset** is a directory of Harbor tasks under `tasks/<name>/`. One task attempted once, in one container, is a **trial**. The program that attempts it is the **harness**: `pi`, `claude-code`, `opencode`, `terminus-2`, or any other Harbor agent. With weights the run serves itself (below), `terminus-2` does not work in this version. A harness shipyard has no profile for gets a generic OpenAI wiring ([Rollouts](concepts/rollouts.md)).

Four parts do the work:

| Part | What it does | Chosen by |
|---|---|---|
| Harbor | runs each trial in a sandbox, runs the harness there, grades the result: the **reward** | `[rollout] harness`, `[rollout] sandbox` |
| Tinker-API backend | holds the weights; samples from them, trains them, saves checkpoints | `[model] name`, `TINKER_BASE_URL` |
| The proxy | serves the backend's weights to the harness and records every call as token ids | started by the run for a Tinker-served model |
| shipyard | cuts the dataset into batches, starts one Harbor **job** per batch, turns rewards into a training step, writes the run | `[recipe] kind` |

The backend is Thinking Machines' Tinker by default. `TINKER_BASE_URL` points shipyard at another backend that implements the same API, such as Fireworks or Baseten.

A model from another provider, such as `[model] provider = "anthropic"`, needs neither the proxy nor the backend. The harness calls the provider through Harbor's own model connection. Only `evaluate` and `gepa` take such a model.

## The six recipes

The **recipe** is the `[recipe] kind`: what the run does with its batches.

| `kind` | What it does | Model |
|---|---|---|
| `evaluate` | measures: the mean reward per batch and over the run | any |
| `dapo` | trains: reward minus the group mean, divided by the spread; PPO loss, clip 0.2 / 0.28; a penalty for running into the token budget; degenerate groups refilled from the next batches | Tinker-served |
| `dr-grpo` | trains: reward minus the group mean, not divided; PPO loss, clip 0.2; an optional length rule | Tinker-served |
| `cispo` | trains: dapo's advantage, overlong penalty and refill; CISPO loss, weight truncated above 4.0 | Tinker-served |
| `gepa` | evolves the skills the harness reads; no gradient | any |
| `fst` | both, in cycles: gepa on the next batches, then steps with every group split across the top skills | Tinker-served |

`shipyard check` prints what a gradient recipe resolves to, on one comment line after the config:

```
# dapo: advantage = group mean, divided by spread; loss = ppo, clip 0.2 / 0.28, averaged per prompt; overlong penalty up to 0.5 over the last 20% of the token budget; 16 substeps by prompt; adamw betas 0.9 / 0.95, eps 1e-08, weight decay 0.1, gradient norm clipped at 1.0, no warm-up (the paper's is 20 steps); degenerate groups dropped and refilled from the plan, up to 9 more rounds
```

[Recipes](concepts/recipes.md) has every knob.

## Where to go next

- New here: [Installation](getting-started/installation.md), then [Measure a policy](tutorials/measure-a-policy.md).
- To train: [Train with dapo](tutorials/train-with-dapo.md).
- To improve a harness's instructions without a gradient: [Evolve a skill](tutorials/evolve-a-skill.md).
- How each part works: [Blueprints](concepts/blueprints.md), [Datasets](concepts/datasets.md), [Rollouts](concepts/rollouts.md), [The proxy](concepts/proxy.md), [Admission](concepts/admission.md), [Modules](concepts/modules.md), [gepa](concepts/gepa.md), [Costs](concepts/costs.md).
- Every key and every command: [Configuration](reference/configuration.md), [Commands](reference/cli.md).
- Why it is built this way: [Decisions](decisions.md).
