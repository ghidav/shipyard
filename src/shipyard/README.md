# Shipyard Architecture (`shipyard/`)

The top level of `shipyard` defines the public programming model for RL recipes. A recipe is a `Trainer` subclass whose `step()` method is composed from four primitives: `draw`, `rollout`, `Policy.logprobs`, and `update`. 

The framework encapsulates session lifecycle, policy-to-backend binding, token expenditure tracking, step telemetry, checkpoint persistence, evaluation loops, and state restoration.

Recipes interact exclusively with the root exports: `Trainer`, `Policy`, `INITIAL`, `Group`, `Trial`, `Blueprint`, `Modules`, `Inadmissible`, and `load`. Subsystem implementations are partitioned into three dedicated packages:

* **[`models/`](models/README.md):** Model backends, Tinker training/sampling sessions, and the trial proxy.
* **[`jobs/`](jobs/README.md):** Harbor sandbox orchestration, harness profiles, secret pooling, and admittance judging.
* **[`run/`](run/README.md):** Directory layout, telemetry streams (`metrics.jsonl`, `jobs.jsonl`), cost accounting, and datasets.



## 1. Module Overview

| Module | Primary Role |
| :--- | :--- |
| **`trainer.py`** | Houses `Trainer`: implements `draw`, `rollout` and `update`, baseline `step` and `score` hooks, policy discovery, session loops, checkpointing, and evaluation. Manages harness distribution (`split`) and carried modules (`carried`). |
| **`policy.py`** | Defines `Policy`: encapsulates inference and sequence scoring (`logprobs`) across heterogeneous backends (Tinker or external providers); re-exports the `INITIAL` starting-weights sentinel. |
| **`groups.py`** | Data containers returned by rollouts: `Group` $\to$ `Trial` $\to$ `Sequence`. Manages advantage assignment, trajectory packing, and token-level target masking. |
| **`config.py`** | Validates blueprints against recipe schemas (`Blueprint`, `Recipe`, `load`, `ConfigError`). |
| **`cli.py`** | Command-line entry points (`shipyard run`, `shipyard resume`) and graceful signal propagation (SIGINT, SIGTERM, SIGHUP). |



## 2. The Four Primitives

| Primitive | Signature / Returns | Semantic Behavior |
| :--- | :--- | :--- |
| **`self.draw()`** | `-> list[Path]` | Returns the next `[data] batch_size` task paths from a deterministic, resumable generator stream. |
| **`self.rollout(tasks, ...)`** | `(list[Path], ...) -> list[Group]` | Dispatches agent trials across tasks, evaluates verifier outcomes, and returns structured `Group`s aligned to `tasks`. |
| **`policy.logprobs(seqs, ...)`** | `(list[Sequence], prefix=...) -> list[Tensor \| None]` | Evaluates target-token log probabilities under the policy, optionally conditioning on spliced prefix text. |
| **`self.update(groups, ...)`** | `(list[Group], loss=..., ...) -> Update \| None` | Runs distributed optimization passes on all sequences assigned advantages, updating weights and publishing new sampling checkpoints. |

The baseline `Trainer.step` implements standard group-relative policy optimization (GRPO-style). Custom algorithms typically override the `score` hook:

```python
async def step(self) -> None:
    groups = await self.rollout(self.draw())
    await self.score(groups)  # Assigns advantages: (reward - group_mean)
    await self.update(groups)  # Tinker importance-sampling loss by default
```

> **Note:** The default `score` implementation skips groups where all trial rewards are identical (zero variance yields zero gradient). `update` trains only sequences carrying explicit advantages.

### Rollout Configuration Options

| Parameter | Default | Description |
| --- | --- | --- |
| `attempts` | `[data] group_size` | Number of independent trials dispatched per task. |
| `modules` | `self.state.modules` | The prompt template and skills (`Modules`) delivered to sandboxes. Overridden by search recipes. |
| `harness` | `[agent] harness` | Target Harbor agent string (e.g., `claude-code@2.1.0`). When omitted, each harness listed in `[agent] harness` runs its own job. |
| `renderer` | `[rollout] renderer` | One of the renderers in `[rollout] renderer`; any other is refused. When omitted, the tasks take the run's renderers in turn. |
| `purpose` | `"training"` | Labels the job's row in `jobs.jsonl`. Only `"training"` rollouts feed the step's `train` row; `evaluate` passes `"evaluation"`, and search recipes pass `"search"`. |

#### Multi-Harness Dispatching

When `[agent] harness` defines multiple agents:

* **`mix_harnesses = true`:** Each task's `attempts` are distributed evenly across harnesses; every `Group` contains mixed-agent trials.
* **`mix_harnesses = false` (default):** Tasks are interleaved round-robin across harnesses (offsetting by one harness each step); each `Group` belongs to a single harness.

#### Several Renderers (Reasoning Efforts)

When `[rollout] renderer` lists several cookbook renderers, such as the reasoning efforts `qwen3_8_low_reasoning` and `qwen3_8_xhigh_reasoning`, one proxy renders with all of them and each trial's address names its own. Tasks are interleaved round-robin across the renderers (offsetting by one each step), so each `Group` belongs to a single renderer and its baseline compares an effort only with itself. With harnesses unmixed, one round-robin runs over every (harness, renderer) pair; with `mix_harnesses = true`, each task's attempts are split across the harnesses at its renderer. `evaluate` runs every test task at each renderer.

### Trainer Runtime State

| Attribute | Type | Description |
| --- | --- | --- |
| `self.config` / `self.recipe` | `Blueprint` / `Recipe` | Validated configuration and recipe-specific schema bindings. |
| `self.policy` | `Policy` | The active agent policy instantiated from `[agent]`. |
| `self.state` | `State` | What the run has done so far, which a checkpoint saves and a resume restores, plus a recipe's own fields (§3). |
| `self.state.modules` | `Modules \| None` | Global baseline modules deployed with trials. |
| `self.metrics` | `dict[str, Any]` | Ephemeral telemetry dictionary logged to `metrics.jsonl` and reset each step. |
| `self.state.steps` / `.updates` / `.drawn` | `int` | Monotonic counters tracking elapsed steps, `update` calls, and drawn task batches. |
| `self.state.evaluated` | `int \| None` | The step after which the test set was last evaluated. |
| `self.directory` | `Path` | Active run directory root. |



## 3. Run Lifecycle

```
[Bind Policies] ──> [Open Session] ──> [Step Loop] ──> [Final Checkpoint]
 (Introspect &       (Tinker training   (Runs [recipe]   (Saves the state,
  assign backends)    client & proxy)    steps)           and any weights)

```

1. **Policy Discovery & Binding:** Introspects trainer attributes, binding each `Policy` instance to its target backend (§4).
2. **Session Initialization:** For an agent on Tinker, initializes the training client, registers frozen baseline models, loads the weights the restored state names (if any), and boots the harness proxy.
3. **Step Loop:** Iterates for `[recipe] steps`:
* Resets `self.metrics`.
* Executes `await self.step()`.
* Evaluates on the test set every `[recipe] evaluate_every` steps, and after the last step, if the run has one.
* Writes the step's row to `metrics.jsonl`, its evaluation's numbers among it as `eval_` fields.
* Persists periodic checkpoints (`step-<n>`) every `[checkpoints] every` steps.


4. **Final Checkpoint:** Saves the state as `final`, and, if updates moved the weights, the training state and sampling weights, kept permanently. A run that moves no weights, such as a module search, saves its state alone.

### Checkpointing & Resumption

* **`checkpoint(tag, keep=False)`:** Saves the agent's training state via Tinker once an update has moved it (it expires after `[checkpoints] ttl_hours` unless `keep=True`, which keeps it for good and exports the weights for sampling), then `self.state` as `states/step-<n>.json`, then the row naming both. `final` and the checkpoint of the last step share that step's state and differ in their weights. A provider's agent, or one no update has moved, saves its state alone.
* **`restore(origin, row)`:** Reads the state the row of the run in `origin` names into `self.State`, refusing a state whose fields no longer fit it and dropping, with a warning, those the recipe no longer declares. The weights it names load when `train` opens the session.
* **`keep(modules, name)`:** Persists modules (prompt and skills) to `modules/<name>/`, for a later run to carry. A resume needs none: the state holds the modules whole.

### A Recipe's Own State

A recipe whose step builds something a resume must find again, such as a memory bank or a population of prompts, declares it as its `State`, as it declares its `Recipe`:

```python
class FSTTrainer(Trainer):
    class State(Trainer.State):
        population: list[Modules] = []  # each field has a default
        pending: list[list[Path]] = []

    async def step(self) -> None:
        batch = self.state.pending.pop(0)  # saved at every checkpoint, restored on resume
```

Fields are saved as JSON: numbers, text, paths, lists and dicts of them, pydantic models, and `Modules`, whole. What can be recomputed, such as embeddings, is better left out and rebuilt.



## 4. Policy Abstraction (`policy.py`)

Every model is represented as a `Policy`. The agent is `self.policy`. Auxiliary models are declared during `Trainer.__init__` as direct attributes, lists, tuples, or dicts:

```python
def __init__(self, config: Blueprint, directory: Path) -> None:
    super().__init__(config, directory)
    self.anchor = Policy(INITIAL)  # Agent's frozen origin weights
    self.reference = Policy("Qwen/Qwen3-32B")  # Frozen Tinker baseline or checkpoint
    self.critic = Policy("claude-sonnet-5-5", provider="anthropic")  # External provider model
```

### Backend Resolution Matrix

| Policy Declaration | Backend | Sampling (`sample`) | Logprob Scoring (`logprobs`) |
| --- | --- | --- | --- |
| **`self.policy`** (Tinker) | `Session` | Refused *(sampled externally via proxy in `rollout`)* | Single forward pass via Tinker training client |
| **`self.policy`** (`provider` set) | `Provider` | Routed via LiteLLM | Refused |
| **`Policy(INITIAL)`** | `Frozen` | Refused | Evaluated via sampling client at the agent's starting weights (base model or `[agent] from_checkpoint`) |
| **`Policy("<model \| tinker://>")`** | `Frozen` | Refused | Evaluated via target sampling client |
| **`Policy("<model>", provider=...)`** | `Provider` | Routed via LiteLLM | Refused *(providers omit exact sequence logprobs)* |

> **Constraint:** Auxiliary Tinker policies share the agent's active Tinker session. If the agent runs against a commercial provider, Tinker-based auxiliary policies raise a binding error (`ValueError`) when `train` binds them.

### Prefix Injection Scoring (`logprobs(sequences, prefix=...)`)

Evaluates sequence logprobs as if the policy had ingested text prior to the task instruction (e.g., privileged critic advice or chain-of-thought hints):

* The prefix is spliced immediately before the task instruction, found by its first 200 characters in the decoded prompt, preserving the leading system prompt.
* Tokenized using the agent's tokenizer; the spliced sequence must fit the agent's context as the proxy reports it (`[rollout] max_context`, else the model's own).
* If a sequence contains no identifiable prompt boundary or exceeds context limits post-splice, `logprobs` returns `None` for that entry, and a single warning reports how many were dropped.



## 5. Rollout Hierarchy & Data Structures (`groups.py`)

```
Group                                Task-level collection of trials
 └── Trial                           Single harness execution inside sandbox
      ├── verdict                    Outcome status (numerical reward or mask reason)
      ├── records                    Raw proxy request/response logs (token IDs, logprobs)
      └── Sequence                   Chained turn trajectory flattened to token arrays
           ├── tokens, spans, images 
           ├── sampled_logprobs      Exact per-token logprobs captured by sampler
           └── advantages            Token-level advantage targets consumed by update()

```

### Record Chaining (`chains`, `grouped`)

Upon rollout completion, raw proxy logs for measured trials are assembled into training sequences:

1. **Prefix Matching:** Requests whose prompts begin with the prompt + completion prefix of a prior call are chained sequentially. Branches resolve to the longest matching sequence.
2. **Filtering:** Calls refused by the token budget or context limit, calls that failed upstream, and calls with an image the proxy did not keep are dropped.
3. **Packaging:** Each chain consolidates into a `Sequence` representing the terminal conversation state. Target positions correspond to completion tokens across spans.
4. **Logprob Validation:** `sampled_logprobs` is populated only if the proxy captured a logprob for every completion token of every call in the chain.

### Advantage Assignment

* **Scalar Assignment:** `trial.set_advantage(val)` broadcasts a scalar advantage to all target tokens across all sequences in the trial.
* **Dense Token Assignment:** Assigning a 1D tensor to `sequence.advantages` (one entry per target position) provides fine-grained, token-level credit assignment.



## 6. Optimization Passes (`self.update`)

```
[Gather Sequences] ──> [Resolve Reference] ──> [Substep Partitions] ──> [Publish Weights] ──> [Metrics]
 (Filter sequences       (Proxy logprobs,        (Mini-batch forward/     (Save sample-<n>,     (KL divergence,
  with advantages)        or current weights)     backward + step)         point proxy)          policy entropy)

```

1. **Filtering:** Gathers all sequences containing assigned advantages from valid groups.
2. **Importance Weights:** Ratios are computed using the proxy's recorded `sampled_logprobs`. Sequences missing sampler logprobs fall back to forward-pass scoring against current weights.
3. **Partitioned Substeps:** Groups are partitioned into `[recipe] substeps` chunks (at most one per group with advantages). Groups are never split across mini-batches, preserving prompt-set integrity. Forward-backward evaluations and optimizer steps are pipelined asynchronously.
4. **Serving Update:** Sampling weights are published to Tinker as `sample-<n>`, and proxy routing pointers shift to ensure subsequent rollouts remain strictly on-policy.
5. **Telemetry:** Returns an `Update` object logging policy entropy, `kl_v1`, `kl_v2` (drift between sampler and trainer), and learning rate.

> **Loss Normalization Warning:** Tinker accumulates raw token-level losses. Algorithms that average the loss per sequence (GRPO) or per prompt (DAPO) must scale each sequence's advantages accordingly prior to calling `update`.



## 7. Blueprints & Recipes (`config.py`)

Runs are declared in a `run.toml` blueprint validated against the recipe's configuration schema:

| Table | Target Scope | Key Parameters |
| --- | --- | --- |
| **`[agent]`** | Model definition & harnesses | `model`, `provider` (`"tinker"`, or `"together"`), `from_checkpoint`, `harness`, `mix_harnesses` (`false`), `kwargs`, `env`, `modules` |
| **`[data]`** | Tasks & sampling batch sizes | `train`, `test`, `batch_size`, `group_size` (`1`), `test_group_size`, `seed` (`0`) |
| **`[rollout]`** | Sandbox & proxy topology | `sandbox` (`"docker"`), `concurrency` (`4`), `timeout`, `max_tokens` (`8192`), `max_context`, `fill_context`, `renderer` (the model's own), `endpoint_url`, `host`, `secrets` |
| **`[recipe]`** | Optimization hyperparameters | `steps`, `learning_rate`, `substeps` (`1`), `warmup` (`0`), `evaluate_every` (`0`), `lora_rank` (`32`), `restore_optimizer` (`false`), `grade_timeouts` (`false`) |
| **`[checkpoints]`** | Retention rules | `every` (`1`), `ttl_hours` (`168`) |
| **`[logging.wandb]`** | Cloud metrics synchronization | `project`, `entity` |

### Custom Recipe Parameters

Custom recipes define hyperparameter contracts by subclassing `Trainer.Recipe`:

```python
class DAPOTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        clip_low: float
        clip_high: float
```

Declared keys carry no defaults, so the blueprint sets every one; a `None` default marks a key whose value is derived from the data when unset. Missing or undeclared keys trigger explicit, pre-execution `ConfigError` validation summaries.



## 8. CLI & Signal Handling (`cli.py`)

```bash
# Copy the built-in recipes into ./recipes, and write ./.env.example, keeping what is there
shipyard init

# Launch a new run from a recipe folder (its run.toml, and the trainer.py beside it)
shipyard run recipes/dapo --root ./runs

# Or name the blueprint and the trainer file explicitly
shipyard run recipes/dapo/run.toml recipes/dapo/trainer.py

# Go on from a checkpoint (the latest by default): a new run beside it, from the state it saved
shipyard resume ./runs/dapo__a1b2c3d --tag step-50

```

### Signal Handling Lifecycle

In-flight runs intercept `SIGINT`, `SIGTERM`, and `SIGHUP` to prevent orphaned sandbox containers and dangling Tinker sessions:

```
[Signal Received] ──> [Cancel asyncio Tasks] ──> [Drain Cleanup Handlers] ──> [Persist run.json] ──> [Exit 130]
 (SIGINT, SIGTERM,      (Halt trial runners)     (Close proxy, tunnel, Tinker) (Record the stop)
  SIGHUP)

```

The active Tinker session closes as `interrupted`, `run.json` records the stop (e.g. `stopped: SIGTERM`), and `shipyard resume` goes on from the last checkpoint, as a new run.



## 9. Testing & Fakes

| Test Suite | Verification Surface |
| --- | --- |
| **`tests/test_trainer.py`** | Full-loop step orchestration, policy binding, checkpoint restoration, evaluation loops, and signal traps. |
| **`tests/test_pack.py`** | Token-prefix trajectory reconstruction (`chains`), span extraction, and Tinker datum alignment. |
| **`tests/test_modules.py`** | Multi-harness round-robin and mixed trial scheduling; prompt and skill packaging. |

All integration tests execute hermetically using mocks from `tests/fakes.py` (Harbor trial runners, proxy stand-ins) and `tests/trainers.py` (Tinker mock clients, tokenizers), requiring no GPUs or active sandbox daemon.
