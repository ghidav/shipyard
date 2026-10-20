# Recipes (`recipes/`)

A **Recipe** defines an RL algorithm or exploration strategy by subclassing `Trainer` and authoring an asynchronous `step()` method, or only its `score` hook. Recipes compose four foundational primitives to drive data sampling, trajectory collection, logprob scoring, and distributed parameter updates.



## 1. The Four Primitives

| Primitive | Signature / Returns | Semantic Role |
| --- | --- | --- |
| **`self.draw()`** | `-> list[Path]` | Returns the next deterministic batch of `[data] batch_size` task paths from the task dataset generator. |
| **`self.rollout(tasks, ...)`** | `(tasks, ...) -> list[Group]` | Dispatches trials across Harbor sandboxes using `[data] group_size` attempts per task, evaluates verifier outcomes, and returns structured `Group`s with attached proxy records. |
| **`policy.logprobs(seqs, prefix=...)`** | `(seqs, ...) -> list[Tensor \| None]` | Evaluates target-token log probabilities under any bound policy, optionally conditioning on text spliced immediately prior to the task instruction. |
| **`self.update(groups, ...)`** | `(groups, loss=..., ...) -> Update \| None` | Runs distributed forward/backward optimization passes on all sequences assigned advantages, performs Adam steps, and publishes updated weights as `sample-<n>`. |

### The Base Policy Gradient Step

The baseline `Trainer.step` implements a plain Group-Relative Policy Optimization (GRPO-style) flow, without reward normalization or ratio clipping:

```python
async def step(self) -> None:
    # 1. Sample tasks and gather rollout trajectories
    groups = await self.rollout(self.draw())

    # 2. Score trials (trial reward minus group mean reward)
    await self.score(groups)

    # 3. Update parameters using importance sampling loss
    await self.update(groups)
```



## 2. Policy Instantiation & Declarative Binding

Every model in a recipe is an instance of `Policy`. The agent policy is `self.policy`, configured under the `[agent]` table in `run.toml`; it is trained when Tinker serves it.

Auxiliary policies (reference models, teacher models, reward critics) are declared directly in `__init__`. The trainer automatically detects, opens, and connects them at session startup:

```python
def __init__(self, config: Blueprint, directory: Path) -> None:
    super().__init__(config, directory)
    self.anchor = Policy(INITIAL)  # Frozen origin weights
    self.teacher = Policy("Qwen/Qwen3-32B")  # Frozen Tinker model or tinker:// checkpoint
    self.critic = Policy("claude-sonnet-5-5", provider="anthropic")  # External provider via LiteLLM
```

* **`Session` (Agent):** Trained via Tinker; sampling is delegated externally to the proxy during `rollout`.
* **`Frozen` (Auxiliary Tinker Models):** Hosted inside the same Tinker session; provides logprob evaluation via Tinker's sampling client.
* **`Provider` (External LLMs):** Invocations route through LiteLLM. Sequence logprobs are unavailable (`logprobs` calls are refused).



## 3. Recipe Configuration Contracts (`Recipe`)

Standard hyperparameters (`steps`, `learning_rate`, `substeps`, `warmup`, `evaluate_every`, `lora_rank`, `restore_optimizer`, `grade_timeouts`) live in `[recipe]`.

Recipes declaring custom hyperparameters define a nested `Recipe` schema subclassing `Trainer.Recipe`. Fields without defaults are strictly enforced at launch time:

```python
class DAPOTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        clip_low: float
        clip_high: float
        overlong_budget: int | None = None  # None indicates a default derived at runtime
```

`shipyard run` validates `run.toml` against this schema **before** provisioning directories or reserving compute:

* Missing mandatory fields raise validation errors immediately.
* Undeclared or misspelled keys in `[recipe]` trigger pre-execution failures.



## 4. Recipe Catalog

```
recipes/
├── dr-grpo/            # Dr. GRPO (unscaled advantages + clipped PPO token loss)
├── dapo/               # Dynamic sampling, overlong penalties, prompt-averaged PPO
├── cispo/              # Clipped Importance Sampling Policy Optimization
├── opd/                # On-Policy Distillation from a frozen teacher policy
└── gepa/               # Discrete prompt/skill candidate search using GEPA

```

### Detailed Recipe Profiles

| Recipe | Algorithmic Formulation & Characteristics | Primary Objective / Loss |
| --- | --- | --- |
| **`dr-grpo`** | **Dr. GRPO**: Centered advantages without variance normalization. Token-level clipped surrogate objective summed across target spans. | PPO (`clip`) |
| **`dapo`** | **DAPO**: Group-normalized rewards with length-penalty dampening on overlong generations. Token losses are averaged per prompt; includes an iterative re-sampling loop in `step`. | Clipped PPO (prompt-normalized) |
| **`cispo`** | **CISPO**: Employs Tinker’s `cispo` objective, which clamps token-level importance weights rather than truncating surrogate objectives directly. | `cispo` |
| **`opd`** | **On-Policy Distillation**: Matches token generation distributions against a stronger frozen teacher (`self.teacher.logprobs`). | Per-token reverse KL as advantages (`importance_sampling`) |
| **`gepa`** | Genetic Pareto search over prompt templates and skill files (`Candidate`). Evaluates candidates via `rollout(tasks, modules=...)`. Requires `shipyard[gepa]`. | Black-box discrete search (no weight updates) |



## 5. Execution & Dispatching

Launch runs by specifying either the recipe directory containing a `run.toml` and `trainer.py`, or by passing a custom configuration blueprint explicitly:

```bash
# Launch a preconfigured recipe
shipyard run recipes/opd

# Launch using a standalone blueprint and explicit trainer implementation
shipyard run configs/agent_dapo.toml recipes/dapo/trainer.py --root ./runs

# Resume execution from a persisted checkpoint
shipyard resume runs/opd__a1b2c3d --tag step-100

```

> **Fallback Rule:** If a directory lacks a custom `trainer.py`, `shipyard run` defaults to executing the root `Trainer` baseline (Group-Relative Policy Optimization using Tinker's `importance_sampling` loss).