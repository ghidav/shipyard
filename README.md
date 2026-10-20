# Shipyard

Post-training framework for AI agents on Harbor benchmark tasks, built as composable reinforcement learning recipes.



## Installation

Requires **Python 3.12+** and [uv](https://github.com/astral-sh/uv):

```sh
# Base installation
uv tool install shipyard

```

### Extras

Additional modules install with the same command.

```sh
uv tool install "shipyard[gepa]"                 # GEPA prompt/skill evolution search (the gepa recipe)
uv tool install "shipyard[wandb]"                # Mirror metrics to Weights & Biases ([logging.wandb])
uv tool install "shipyard[together]"             # Train on Together post-training ([agent] provider = "together")
uv tool install shipyard --with "harbor[modal]"  # Run agent trials in Modal sandboxes
uv tool install "shipyard[gepa,wandb]" --with "harbor[modal]"  # Several at once

```

Local Docker sandboxes (the default) need no extra, only a running Docker daemon.



## Quickstart

### 1. Get the Recipes

In the directory you will run from, copy the built-in recipes into `recipes/`, each with a `README.md` saying what it needs, and configure API keys and endpoints in a `.env` file:

```sh
shipyard init
cp .env.example .env

```

### 2. Download Tasks

Fetch Harbor-compatible datasets into your working directory:

```sh
uvx harbor datasets download <dataset_name> -o tasks

```

Tasks reside at `tasks/<dataset>/<task>/`.

### 3. Launch a Recipe

Each recipe lives in a self-contained folder containing a configuration blueprint (`run.toml`) and an optional implementation file (`trainer.py`):

```sh
# Run a recipe
shipyard run recipes/dapo

# Artifacts, metrics, and checkpoints are stored under:
# runs/<recipe>__<id>/

```



## Customizing Runs & Algorithms

* **Modify Hyperparameters & Tasks:** Edit the recipe's `run.toml` (controls model selection, LoRA rank, learning rate, harness configurations, and dataset batching).
* **Modify the Algorithm:** Subclass `Trainer` in `trainer.py` and override `step()` or `score()`, declaring any extra `Policy` in `__init__` (see [Writing a Recipe](#writing-a-recipe)).
* **Available Recipes:** Check out the [`recipes/` directory](recipes/README.md) for the built-in algorithms: **Dr. GRPO**, **DAPO**, **CISPO**, **On-Policy Distillation (OPD)**, and **GEPA**.



## Writing a Recipe

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



## License

Apache-2.0. See [LICENSE](LICENSE) for details.
