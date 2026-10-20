# Shipyard Architecture (`shipyard/`)

The top level of `shipyard` defines the public programming model for RL recipes. A recipe is a `Trainer` subclass whose `step()` method is composed from four primitives: `draw`, `rollout`, `Policy.logprobs`, and `update`. 

The framework encapsulates session lifecycle, policy-to-backend binding, token expenditure tracking, step telemetry, checkpoint persistence, evaluation loops, and state restoration.

Recipes interact exclusively with the root exports: `Trainer`, `Policy`, `INITIAL`, `Group`, `Trial`, `Blueprint`, `Candidate`, and `load`. Subsystem implementations are partitioned into three dedicated packages:

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
| `modules` | `self.modules` | The prompt template and skill package (`Candidate`) delivered to sandboxes. Overridden by search recipes. |
| `harness` | `[agent] harness` | Target Harbor agent string (e.g., `claude-code@2.1.0`). When omitted, each harness listed in `[agent] harness` runs its own job. |
| `purpose` | `"training"` | Labels the job's row in `jobs.jsonl`. Only `"training"` rollouts feed the step's `train` row; `evaluate` passes `"evaluation"`, and search recipes pass `"search"`. |

#### Multi-Harness Dispatching

When `[agent] harness` defines multiple agents:

* **`mix_harnesses = true`:** Each task's `attempts` are distributed evenly across harnesses; every `Group` contains mixed-agent trials.
* **`mix_harnesses = false` (default):** Tasks are interleaved round-robin across harnesses (offsetting by one harness each step); each `Group` belongs to a single harness.

### Trainer Runtime State

| Attribute | Type | Description |
| --- | --- | --- |
| `self.config` / `self.recipe` | `Blueprint` / `Recipe` | Validated configuration and recipe-specific schema bindings. |
| `self.policy` | `Policy` | The active agent policy instantiated from `[agent]`. |
| `self.modules` | `Candidate \| None` | Global baseline modules deployed with trials. |
| `self.metrics` | `dict[str, Any]` | Ephemeral telemetry dictionary logged to `metrics.jsonl` and reset each step. |
| `self.steps` / `self.updates` / `self.drawn` | `int` | Monotonic counters tracking elapsed steps, `update` calls, and drawn task batches. Fully restored on resume. |
| `self.directory` | `Path` | Active run directory root. |



## 3. Run Lifecycle

```
[Resume Check] ──> [Bind Policies] ──> [Open Session] ──> [Step Loop] ──> [Final Eval] ──> [Final Checkpoint]
 (Exits if resumed  (Introspect &       (Tinker training   (Runs [recipe]   (Runs if test    (Saves weights if
  from final)        assign backends)    client & proxy)    steps)           set exists)      updates occurred)

```

1. **Resume Verification:** If resumed from a `final` checkpoint, the run terminates immediately.
2. **Policy Discovery & Binding:** Introspects trainer attributes, binding each `Policy` instance to its target backend (§4).
3. **Session Initialization:** For an agent on Tinker, initializes the training client, registers frozen baseline models, loads checkpoint weights (if resuming), and boots the harness proxy.
4. **Step Loop:** Iterates for `[recipe] steps`:
* Resets `self.metrics`.
* Executes `await self.step()`.
* Appends summary telemetry to `metrics.jsonl`.
* Persists periodic checkpoints (`step-<n>`) every `[checkpoints] every` updates.
* Runs validation rollouts on test tasks every `[recipe] evaluate_every` steps.


5. **Final Evaluation:** Runs test-set evaluation if configured and not yet executed at the final step.
6. **Final Checkpoint:** If model parameters were updated during the run, saves the training state and sampling weights as `final`, kept permanently. Pure exploration runs (e.g., discrete module searches) bypass checkpoint generation.

### Checkpointing & Resumption

* **`checkpoint(tag, keep=False)`:** Serializes training states via Tinker. Checkpoints expire after `[checkpoints] ttl_hours` unless `keep=True`, which marks weights permanent and exports sampling artifacts.
* **`restore(row)`:** Restores the step, update and draw counters, and checks out the module configuration matching the checkpoint digest. Weights and optimizer state load when `train` opens the session.
* **`keep(modules, name)`:** Persists candidate prompts/skills to `modules/<name>/`. Required whenever a recipe replaces the carried modules (`self.modules`), so resumed runs can find them by digest.



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
| **`[agent]`** | Model definition & harnesses | `model`, `provider` (`"tinker"`), `from_checkpoint`, `harness`, `mix_harnesses` (`false`), `kwargs`, `env`, `modules` |
| **`[data]`** | Tasks & sampling batch sizes | `train`, `test`, `batch_size`, `group_size` (`1`), `test_group_size`, `seed` (`0`) |
| **`[rollout]`** | Sandbox & proxy topology | `sandbox` (`"docker"`), `concurrency` (`4`), `timeout`, `max_tokens` (`8192`), `max_context`, `fill_context`, `endpoint_url`, `host`, `secrets` |
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
# Launch a new run from a recipe folder (its run.toml, and the trainer.py beside it)
shipyard run recipes/dapo --root ./runs

# Or name the blueprint and the trainer file explicitly
shipyard run recipes/dapo/run.toml recipes/dapo/trainer.py

# Resume execution from a specific checkpoint tag (the latest by default)
shipyard resume ./runs/dapo__a1b2c3d --tag step-50

```

### Signal Handling Lifecycle

In-flight runs intercept `SIGINT`, `SIGTERM`, and `SIGHUP` to prevent orphaned sandbox containers and dangling Tinker sessions:

```
[Signal Received] ──> [Cancel asyncio Tasks] ──> [Drain Cleanup Handlers] ──> [Persist process.json] ──> [Exit 130]
 (SIGINT, SIGTERM,      (Halt trial runners)     (Close proxy, tunnel, Tinker) (Record the stop)
  SIGHUP)

```

The active Tinker session closes as `interrupted`, `process.json` records the stop (e.g. `stopped: SIGTERM`), and `shipyard resume` continues from the last checkpoint.



## 9. Testing & Fakes

| Test Suite | Verification Surface |
| --- | --- |
| **`tests/test_trainer.py`** | Full-loop step orchestration, policy binding, checkpoint restoration, evaluation loops, and signal traps. |
| **`tests/test_pack.py`** | Token-prefix trajectory reconstruction (`chains`), span extraction, and Tinker datum alignment. |
| **`tests/test_modules.py`** | Multi-harness round-robin and mixed trial scheduling; candidate prompt/skill packaging. |

All integration tests execute hermetically using mocks from `tests/fakes.py` (Harbor trial runners, proxy stand-ins) and `tests/trainers.py` (Tinker mock clients, tokenizers), requiring no GPUs or active sandbox daemon.
