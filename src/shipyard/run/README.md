# Run Management (`run/`)

This package manages the lifecycle, file structure, and logging artifacts of a Shipyard training run. It orchestrates directory initialization (`shipyard run`), state restoration (`shipyard resume`), telemetry and metric serialization via `Logs`, execution cost tracking, and task dataset batching.

Recipes do not import from this package directly; they publish to `self.metrics` and persist artifacts using `keep`.



## 1. Module Overview

| Module | Primary Role |
| --- | --- |
| **`directory.py`** | Creates run directories (`open_run`, and `continued` for one that goes on from another run's checkpoint), reads a run's checkpoint rows (`reopened`), records the process that runs a run and how it ended in its `run.json` (`running`), and dynamically imports isolated recipe trainers (`trainer_in`). |
| **`record.py`** | Atomic JSON document writers, resilient JSONL row appenders, and directory path constants. |
| **`logs.py`** | Central logging manager (`Logs`) aggregating metrics across steps, jobs, requests, evaluations, and checkpoints, with optional Weights & Biases mirroring (`Wandb`). |
| **`costs.py`** | Counts token volume (by provider, model, and kind) and container runtime (`Costs`), and totals the usage ledger (`Costs.of`). |
| **`data.py`** | Discovers Harbor task suites and manages deterministic, resumable infinite batch streams (`tasks`, `Generator`). |



## 2. Directory Layout & Storage

Run directories follow the path format `<root>/<blueprint>__<7-chars>/`, where `<blueprint>` is the name of the folder holding `run.toml` (first 32 characters; a run that goes on from another takes that one's), and `<root>` resolves in order of precedence: `--root` flag $\to$ `SHIPYARD_RUNS` environment variable $\to$ `./runs`.

```
<run_directory>/
├── run.toml              # Frozen blueprint specification copied at initialization
├── run.json              # What the run is: its recipe's place and code fingerprint, the checkpoint it went on from, the process that ran it
├── metrics.jsonl         # Interleaved sequence-indexed training steps and eval metrics
├── jobs.jsonl            # Summary row per rollout job (task counts, mask reasons, verdicts)
├── requests.jsonl        # Fine-grained log of every model call proxied during rollouts
├── checkpoints.jsonl     # One row per checkpoint: its tag and step, its state file and its weights
├── states/              # The state after each step a checkpoint was taken at, as step-<n>.json: counters, modules, a recipe's own fields
├── usage.jsonl           # The usage ledger: one row per spend (step, role, tokens or sandbox time), as it happens
├── costs.json            # Totals of usage.jsonl: tokens by provider, model and kind, and sandbox seconds
├── proxy.log             # stdout/stderr stream from the local harness proxy and tunnels
└── modules/
    ├── carried/          # Baseline module configurations active for the run
    └── <kept>/           # Module checkpoints explicitly saved by the recipe via keep

```

> **Atomicity & Resilience:** JSON documents (`costs.json`, `run.json`, `states/step-<n>.json`) are updated via atomic rename. JSONL streams are append-only; truncated rows caused by process crashes are automatically skipped by consumers. Harbor trial directories are kept outside the run, under `<[rollout] jobs_dir>/<run>-NNNN/` (default `jobs/`, relative to the working directory).



## 3. Initialization & Resumption

### Opening a Run (`open_run`)

```
[Import Trainer] ──> [Validate Blueprint] ──> [Create Directory] ──> [Fingerprint Recipe]
 (trainer_in)         (its Recipe schema)      (copy run.toml)        (path + source hash)

```

1. **Module Isolation:** `trainer_in` loads the recipe's folder as an isolated top-level package and the trainer file as a module of it, so relative imports (e.g., `from .helpers import shaped`) do not collide across different recipes.
2. **Blueprint Validation:** Verifies configuration keys against the trainer's `Recipe` schema before touching disk.
3. **Provenance Tracking:** `run.json` records, under `recipe`, the trainer file's absolute path (`trainer`) and a hash of the Python files in its folder (`fingerprint`). The run does not vendor recipe code; it executes in place. `shipyard run` then instantiates the trainer.

### Going On From a Checkpoint (`continued`)

A run's directory is written by one process, from start to end, and never reopened. `shipyard resume <run>` goes on from the run's checkpoint row matching `--tag` (`reopened`; the last row by default) as a new run beside it:

* `continued` opens the new run with the old one's `run.toml`, recipe and carried modules, and notes in its `run.json` its `origin`: the run, the tag, and the first run of the chain. The new run restores the state the checkpoint saved, and its weights.
* Its rows and costs are its own, its steps go on from the checkpoint's, and the run it went on from is left as it was: a curve across a resume is that run's rows up to the checkpoint, then the new run's.
* A run whose checkpoint is at `[recipe] steps` has finished: `resume` says so and opens nothing. Raising `steps` in its `run.toml` lets it go on.
* **Code Drift Handling:** If the recipe’s source code changed since the run started, a warning is emitted and the new run uses the updated code. If the recipe file was deleted, the resume aborts.



## 4. Telemetry & Log Streams

Each stream is the run's own, written by its one process; `requests.jsonl` rows join to their job by `job`. In every stream, `step` is the step a row belongs to, counted from 1: the jobs, spends and row of the third step, its checkpoint `step-3`, and an evaluation right after it all say 3.

### Primary Stream Schemas

* **`metrics.jsonl`:** One row per step: the step (`step`), whether it trained (`trained`), the jobs its groups came from (`jobs`), its counts (`groups`, `trials`, `graded`, `masked`, and `uniform`: groups whose graded trials all scored the same, so nothing to learn from), cohort statistics (`reward_mean`, `reward_spread`, `sampled_mean`: tokens the agent sampled per graded trial, and with several renderers each one's `reward_mean_<renderer>` and `sampled_mean_<renderer>`), the update (`sequences`, `train_tokens`, `kl_v1`, `kl_v2`, `entropy`, `learning_rate`, `substeps`, `loss`, `update_seconds`), the evaluation that belongs to the step, if any, as `eval_` fields (`eval_tasks`, `eval_trials`, `eval_graded`, `eval_masked`, `eval_reward_mean`, and each renderer's), and the recipe's own `self.metrics` (a name the row already holds is written as `recipe_<key>`). The run evaluates every `[recipe] evaluate_every` steps and after its last step.


* **`jobs.jsonl`:** Per-job breakdown including the folder its trials are in (`dir`, absolute: Harbor's `result.json`, `agent/` and `verifier/` per trial), its `harness` and `renderer`, target `tasks`, modules digests, counts of outcomes (`graded`, `masked`, with the reasons in `masked_by`, and Harbor's exceptions in `exceptions`), what served the agent (`backend`: `tinker`, `together` or `provider`), and the proxy's refusals. What the job spent is in `usage.jsonl`, under its name.
* **`requests.jsonl`:** One row per model call the proxy served: its trial and the call's number in it (`call`), token accounting (prompt, cached, completion), response latency, the weights that served it (`weights`), stop condition, and upstream errors.
* **`checkpoints.jsonl`:** One row per checkpoint: its `tag` and `step`, the state file it saved (`state`, `states/step-<n>.json`, which `final` shares with the last step's checkpoint), the agent's training state on Tinker (`weights`, null for a provider's agent or one no update moved), the weights exported for sampling (`sampler`, `final` only) and how long the saved weights are kept (`ttl_hours`, null for good).

### External Sync (Weights & Biases)

When `[logging.wandb]` is configured, each step's row is mirrored to W&B at its step, its evaluation's numbers under `eval/`. The W&B run's id and name are the run directory's name; the runs of a chain, each going on from a checkpoint of the one before, share a W&B group named after the first.



## 5. Cost Tracking (`costs.py`)

Resource expenditure tracks raw consumption rather than monetary costs:

### Token Accounting Categories

| Category | Definition |
| --- | --- |
| **`prefill`** | Uncached prompt tokens processed by the inference engine. |
| **`cache`** | Prompt tokens served from the provider's context cache. |
| **`sample`** | Generation tokens emitted by the sampler. |
| **`train`** | Tokens through the training client: forward-backward passes during optimization, and forward passes scoring the agent's logprobs. |

### The Usage Ledger

Every spend is a row of `usage.jsonl` the moment it happens, and `costs.json` holds the totals, rewritten with each row (`Costs.of` rebuilds them from the ledger). A row names the `step` it happened in and the `role` that spent it:

* **`agent`:** a job's rollouts (one row per provider and model, and one for sandbox time, each with its `job`), and the agent's training and scoring passes on Tinker.
* **Any other policy:** the name the trainer holds it by, such as `teacher`, `critics[main]` or `teachers[0]`. Each call is filed as it ends: a provider batch with a failed call still files the calls that answered, and a step that then fails loses nothing it spent.
* **Spend outside a policy:** a recipe files it with `self.logs.spend(role, provider, model, **tokens)`, such as an embedding model's prefill.

### Attribution Hierarchy

* **Proxied Agent Calls:** Derived directly from proxy sampling logs (`sampled`). Multi-sequence generations bill shared prompt prefixes only once; rejected or out-of-context requests incur no charge.
* **Direct Provider Calls:** Extracted from Harbor telemetry (`reported`), attributed to the specific provider and model identified by the harness.
* **Trainer In-Process Calls:** Logged by the active `models/agent.py` backend (training passes, logprob scoring, and a provider's model sampled from recipe code).
* **Sandbox Computation:** For each trial with a result, the span from the start of its first timed Harbor phase (environment setup) to the end of its last (usually the verifier); environment teardown is not counted.



## 6. Datasets & Task Generation (`data.py`)

Datasets reside under `tasks/<dataset>/<task>/` using standard Harbor layout conventions.

* **Task Discovery:** `tasks(dataset)` validates and returns a sorted index of runnable task directories, raising an error if no tasks are discovered.
* **Resumable Generation:** `Generator.of(config).batches(start)` yields deterministic task batches indefinitely. Pass $n$ over the dataset shuffles entries using `[data] seed + n`. Setting `start` offsets the generator state directly to the batch index required upon resuming, without repeating or dropping tasks.



## 7. Testing

| Test Suite | Scope |
| --- | --- |
| `tests/test_record.py` | JSONL append operations, atomic JSON write safety, and stream parsing. |
| `tests/test_costs.py` | Token category bucketing, prompt caching deduplication, and sandbox duration aggregation. |
| `tests/test_trainer.py` | Integration tests covering directory initialization, recipe module isolation, resuming logic, the recipes and the CLI. |
| `tests/test_logs.py` | `jobs.jsonl`, `requests.jsonl`, `usage.jsonl` and `metrics.jsonl` rows. |
