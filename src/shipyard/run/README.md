# Run Management (`run/`)

This package manages the lifecycle, file structure, and logging artifacts of a Shipyard training run. It orchestrates directory initialization (`shipyard run`), state restoration (`shipyard resume`), telemetry and metric serialization via `Logs`, execution cost tracking, and task dataset batching.

Recipes do not import from this package directly; they publish to `self.metrics` and persist artifacts using `keep`.



## 1. Module Overview

| Module | Primary Role |
| --- | --- |
| **`directory.py`** | Creates or reopens run directories (`open_run`, `reopened`), records each invocation and how it ended in `process.json` (`running`), and dynamically imports isolated recipe trainers (`trainer_in`). |
| **`record.py`** | Atomic JSON document writers, resilient JSONL row appenders, and directory path constants. |
| **`logs.py`** | Central logging manager (`Logs`) aggregating metrics across steps, jobs, requests, evaluations, and checkpoints, with optional Weights & Biases mirroring (`Wandb`). |
| **`costs.py`** | Aggregates token volume (by provider, model, and kind) and container runtime across rollouts and training steps (`Costs`). |
| **`data.py`** | Discovers Harbor task suites and manages deterministic, resumable infinite batch streams (`tasks`, `Generator`). |



## 2. Directory Layout & Storage

Run directories follow the path format `<root>/<blueprint>__<7-chars>/`, where `<blueprint>` is the name of the folder holding `run.toml` (first 32 characters), and `<root>` resolves in order of precedence: `--root` flag $\to$ `SHIPYARD_RUNS` environment variable $\to$ `./runs`.

```
<run_directory>/
├── run.toml              # Frozen blueprint specification copied at initialization
├── recipe.json           # Recipe pointer and SHA-256 fingerprint of sibling Python source
├── process.json          # Process runtime metadata (PIDs, start/end, exit errors, Tinker host)
├── metrics.jsonl         # Interleaved sequence-indexed training steps and eval metrics
├── jobs.jsonl            # Summary row per rollout job (task counts, mask reasons, verdicts)
├── requests.jsonl        # Fine-grained log of every model call proxied during rollouts
├── checkpoints.jsonl     # Registered checkpoints (tags, Tinker weight paths, iteration offsets)
├── costs.json            # Continuously aggregated token usage and sandbox container seconds
├── proxy.log             # stdout/stderr stream from the local harness proxy and tunnels
└── modules/
    ├── carried/          # Baseline module configurations active for the run
    └── <kept>/           # Module checkpoints explicitly saved by the recipe via keep

```

> **Atomicity & Resilience:** JSON documents (`costs.json`, `process.json`) are updated via atomic rename. JSONL streams are append-only; truncated rows caused by process crashes are automatically skipped by consumers. Harbor trial directories are kept outside the run, under `<[rollout] jobs_dir>/<run>-NNNN/` (default `jobs/`, relative to the working directory).



## 3. Initialization & Resumption

### Opening a Run (`open_run`)

```
[Import Trainer] ──> [Validate Blueprint] ──> [Create Directory] ──> [Fingerprint Recipe]
 (trainer_in)         (its Recipe schema)      (copy run.toml)        (path + source hash)

```

1. **Module Isolation:** `trainer_in` loads the recipe's folder as an isolated top-level package and the trainer file as a module of it, so relative imports (e.g., `from .helpers import shaped`) do not collide across different recipes.
2. **Blueprint Validation:** Verifies configuration keys against the trainer's `Recipe` schema before touching disk.
3. **Provenance Tracking:** `recipe.json` records the trainer file's absolute path (`trainer`) and a hash of the Python files in its folder (`fingerprint`). The run does not vendor recipe code; it executes in place. `shipyard run` then instantiates the trainer.

### Resuming an Existing Run (`reopened`)

* Reloads the configuration from the run's `run.toml`, and restores counters, carried modules and Tinker state from the checkpoint row matching `--tag` (defaults to the last row of `checkpoints.jsonl`).
* **Code Drift Handling:** If the recipe’s source code changed since initialization, a warning is emitted and execution proceeds using the updated code. If the recipe file was deleted, the resume operation aborts.
* **Attempts:** Each resume files the previous process note under `earlier` in `process.json`; a row's `attempt` is the length of that list. Steps redone after a resume are appended again, not replaced: nothing drops the earlier rows, so a reader wanting one row per step keeps the one with the highest `attempt`.



## 4. Telemetry & Log Streams

Every row of `metrics.jsonl`, `jobs.jsonl` and `checkpoints.jsonl` records an `attempt` integer (starting at `0`, incremented per resume); `requests.jsonl` rows join to their job by `job`.

### Primary Stream Schemas

* **`metrics.jsonl`:**
* **`train` rows:** Sequence index (`seq`), the step (`step`), whether it trained (`trained`), its counts (`groups`, `rollouts`, `graded`, `masked`, `degenerate`), cohort statistics (`reward_mean`, `reward_spread`, `length_mean`), loss and optimizer details (`kl_v1`, `kl_v2`, `entropy`, `learning_rate`, `substeps`, `loss_fn`), the update's duration (`seconds`), and arbitrary user metrics defined in `self.metrics` (a name the row already holds, or `attempt`/`seq`, is written as `recipe_<key>`).
* **`eval` rows:** Evaluation payload produced by `Trainer.evaluate`.


* **`jobs.jsonl`:** Per-job breakdown including target `tasks`, module candidate digests, categorical counts of outcomes (`graded`, `masked`, and Harbor exception breakdown in `ended`), routing type (`tinker` vs. `provider`), proxy refusals, and sandbox runtime.
* **`requests.jsonl`:** Detailed record of proxied completions: token accounting (prompt, cached, completion), response latency, serving weights, stop condition, and upstream errors.
* **`checkpoints.jsonl`:** Pointers to saved Tinker training states (`state_path`), sampling weights (`sampler_path`), retention TTL, module digests, and training counters.

### External Sync (Weights & Biases)

When `[logging.wandb]` is configured, `metrics.jsonl` entries are automatically forwarded to W&B, plotting numerical metrics against `step`. The W&B run's id and name are the run directory's name, and it is opened with `resume="allow"`, so resumed attempts log to the same W&B run.



## 5. Cost Tracking (`costs.py`)

Resource expenditure tracks raw consumption rather than monetary costs:

### Token Accounting Categories

| Category | Definition |
| --- | --- |
| **`prefill`** | Uncached prompt tokens processed by the inference engine. |
| **`cache`** | Prompt tokens served from the provider's context cache. |
| **`sample`** | Generation tokens emitted by the sampler. |
| **`train`** | Tokens through the training client: forward-backward passes during optimization, and forward passes scoring the agent's logprobs. |

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
| `tests/test_logs.py` | `jobs.jsonl` and `requests.jsonl` rows, and `seq` numbering across reopenings. |
