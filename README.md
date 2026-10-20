# Shipyard

Post-training framework for AI agents on Harbor benchmark tasks, built as composable reinforcement learning recipes.

A recipe subclasses `Trainer` and defines its optimization loop using four core primitives:

```python
async def step(self):
    # 1. Sample tasks and gather rollout trajectories
    groups = await self.rollout(self.draw())

    # 2. Score rollouts (e.g., group-relative advantage: R - μ_group)
    await self.score(groups)

    # 3. Update parameters on Tinker
    await self.update(groups)

```

* **`draw`**: Returns the next deterministic batch of training tasks from a dataset.
* **`rollout`**: Executes agent harnesses inside isolated Harbor sandboxes, evaluates trials with ground-truth verifiers, and groups outcomes by task.
* **`score`**: The hook a recipe overrides to turn rewards into per-token advantages. It can call the fourth primitive, **`policy.logprobs`**, which evaluates per-token log probabilities across target sequences under any bound policy (active agent, frozen baseline, or teacher).
* **`update`**: Runs distributed optimization passes (PPO, CISPO, importance sampling) on Tinker and serves the new weights.

Under the hood, when Tinker serves the agent, **`harness-proxy`** intercepts all LLM requests from agent harnesses, recording exact token IDs and sampler log probabilities so the trainer updates on the true generation distribution.



## Architecture Overview

```
shipyard/
├── src/shipyard/        # Public Trainer API and root primitives (README.md)
│   ├── models/          # Policy backends, Tinker clients, and proxy endpoints
│   ├── jobs/            # Harbor execution engine, harness profiles, and verdicts
│   └── run/             # Run management, telemetry streams, and dataset generators
├── packages/
│   └── harness-proxy/   # Serves the policy to harnesses and records every call
└── recipes/             # Algorithmic implementations (Dr. GRPO, DAPO, CISPO, OPD, GEPA)

```


## Installation

Requires **Python 3.12+** and [uv](https://github.com/astral-sh/uv):

```sh
# Base installation
uv sync

# With prompt/skill evolution search extras (GEPA)
uv sync --extra gepa

```

### Sandbox Providers

Local Docker sandboxes (the default) need no extra, only a running Docker daemon. To run agent trials in cloud sandboxes, add the corresponding Harbor provider package:

```sh
uv add "harbor[modal]"     # For Modal-backed sandboxes

```

Configure API keys and endpoints in a `.env` file at the root of your workspace:

```sh
cp .env.example .env

```



## Quickstart

### 1. Download Benchmark Tasks

Fetch Harbor-compatible datasets into your working directory:

```sh
harbor datasets download <dataset_name> -o tasks

```

Tasks reside at `tasks/<dataset>/<task>/`.

### 2. Launch a Recipe

Each recipe lives in a self-contained folder containing a configuration blueprint (`run.toml`) and an optional implementation file (`trainer.py`):

```sh
# Run a recipe
uv run shipyard run recipes/dapo

# Artifacts, metrics, and checkpoints are stored under:
# runs/<recipe>__<id>/

```

### 3. Resume an Interrupted Run

Runs can be resumed seamlessly from their latest (or a targeted) checkpoint:

```sh
# Resume from latest checkpoint
uv run shipyard resume runs/dapo__a1b2c3d

# Resume from a specific tag
uv run shipyard resume runs/dapo__a1b2c3d --tag step-50

```



## Customizing Runs & Algorithms

* **Modify Hyperparameters & Tasks:** Edit the recipe's `run.toml` (controls model selection, LoRA rank, learning rate, harness configurations, and dataset batching).
* **Modify the Algorithm:** Subclass `Trainer` in `trainer.py` and override `step()` or `score()`, declaring any extra `Policy` in `__init__`.
* **Available Recipes:** Check out the [`recipes/` directory](recipes/README.md) for the built-in algorithms: **Dr. GRPO**, **DAPO**, **CISPO**, **On-Policy Distillation (OPD)**, and **GEPA**.



## License

Apache-2.0. See [LICENSE](LICENSE) for details.