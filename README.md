# shipyard

Post-training on Harbor jobs.

- A **blueprint** is a directory holding a `run.toml`. `shipyard run` reads it and writes a **run**: a directory under `runs/` that records what happened.
- Harbor runs each task in a container, runs an agent program (the **harness**) there, and grades the result.
- A Tinker-API backend holds the weights, samples from them and trains them: Thinking Machines' Tinker by default, or another backend that implements the Tinker API, such as Fireworks or Baseten, named by `TINKER_BASE_URL`.
- Between the two, shipyard's proxy serves the weights to the harness and records every model call as tokens.
- Six recipes: `evaluate` measures a policy, `dapo`, `dr-grpo` and `cispo` train it, `gepa` evolves the skills its harness reads, and `fst` does both in cycles.

## Install

```sh
uv add shipyard
```

shipyard needs Python 3.12 or newer. A Docker sandbox needs Docker running. A cloud sandbox needs the matching Harbor extra, such as `uv add "harbor[modal]"`. Your workspace is the directory holding your blueprints, `tasks/` and `.env`, and you run every command there. Keys go in its `.env`. [Installation](docs/getting-started/installation.md) has the details.

## A blueprint

`blueprints/02-eval-tinker-docker/run.toml` measures Qwen3-8B, served from Tinker, with the `pi` harness in Docker:

```toml
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

## The commands

From the workspace, with its venv active, or each prefixed with `uv run`:

```sh
shipyard check blueprints/02-eval-tinker-docker   # the resolved config and every problem, before anything is spent
shipyard run blueprints/02-eval-tinker-docker     # runs the recipe; the record goes to runs/<blueprint>__<7 chars>/
shipyard runs                                     # every run, newest first
shipyard show 02-eval-tinker-docker__eTNXY25      # one run: process, metrics, jobs, checkpoints, costs
shipyard serve --help                             # the proxy as a process of its own; a run starts its own
```

The last metrics row of that run, as `shipyard show` prints it:

```
metrics  2 rows
  at          2026-10-07T20:57:50.873722+00:00
  seq         2
  evaluation  true
  batches     1
  rollouts    2
  graded      2
  masked      0
  mean        1.0
```

Two rollouts, both graded, both solved.

## Documentation

The pages live under `docs/`. `uvx --with mkdocs-material mkdocs serve` builds the site.

- Getting started: [Installation](docs/getting-started/installation.md)
- Concepts: [Blueprints](docs/concepts/blueprints.md), [Runs](docs/concepts/runs.md), [Datasets](docs/concepts/datasets.md), [Rollouts](docs/concepts/rollouts.md), [The proxy](docs/concepts/proxy.md), [Admission](docs/concepts/admission.md), [Recipes](docs/concepts/recipes.md), [Modules](docs/concepts/modules.md), [gepa](docs/concepts/gepa.md), [Costs](docs/concepts/costs.md)
- Tutorials: [Measure a policy](docs/tutorials/measure-a-policy.md), [Train with dapo](docs/tutorials/train-with-dapo.md), [Evolve a skill](docs/tutorials/evolve-a-skill.md)
- Reference: [Configuration](docs/reference/configuration.md), [Commands](docs/reference/cli.md)
- [Decisions](docs/decisions.md)

## Project

- [CHANGELOG.md](CHANGELOG.md): what each release holds.
- [CONTRIBUTING.md](CONTRIBUTING.md): the checks, and how a recipe or a module kind is added.
- [SECURITY.md](SECURITY.md): how to report a vulnerability, and what the proxy exposes.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
