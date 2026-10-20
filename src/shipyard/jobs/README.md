# Jobs Architecture (`jobs/`)

A **Job** manages a batch of Harbor trials evaluating an agent against a set of tasks. `Trainer.rollout` generates a `Job`, passes it to `HarborEngine.run`, and receives back the trial records, verifier outputs, and parsed verdicts.

Recipes reach this package only through `Trainer` (`rollout`, `keep`) and the `Modules` that `shipyard` re-exports; none imports from `shipyard.jobs`.

## 1. Module Overview

| Module | Primary Role |
| --- | --- |
| **`engine.py`** | Coordinates execution: builds Harbor configurations, runs trials concurrently, pairs proxy records, and evaluates outcomes (`Job`, `HarborEngine`). |
| **`admit.py`** | Decides whether a trial is awarded a reward or masked from training/metrics (`Verdict`, `judge`). |
| **`harnesses.py`** | Manages agent integration configurations, wire protocols, and environment bindings (`Harness`, `Profile`, `PROFILES`). |
| **`secrets.py`** | Leases isolated credential sets to concurrent trials and scrubs leaked values from trial artifacts (`Secrets`, `Pool`, `scrub`). |
| **`modules.py`** | Manages the prompt template and skill files injected into trial sandboxes (`Module`, `Modules`, `deliver`). |
| **`progress.py`** | Renders terminal progress bars showing completed vs. masked trials per job (`Bars`). |

## 2. Job Execution Lifecycle

A call to `HarborEngine.run(job, tasks, attempts, *, endpoint)`, which runs `job.harness`, executes through four phases:

```
[1. Config Assembly] ──> [2. Preflight Checks] ──> [3. Concurrent Trials] ──> [4. Record Binding & Admittance]
  Modules + Harness       Network & Validation        Bounded concurrency         Attach records & assign verdicts

```

1. **Config Assembly:** Resolves the harness agent (e.g., `pi@0.85.1`), injects environment variables/kwargs, and renders the modules under `jobs/<job>/modules/`.
2. **Preflight Checks:** Validates Harbor compatibility and aborts early if a task disables networking while the harness relies on container-local model calls.
3. **Trial Execution:** Runs `attempts` trials per task, throttled by `[rollout] concurrency`. Each trial leases a secret identity (when `[rollout.secrets]` sets a pool) and connects to the active endpoint. Failures during runner startup are automatically masked.
4. **Admittance:** Drains proxy interaction records onto each trial and assigns a `Verdict`. Each trial's files (`result.json`, `agent/`, `verifier/`) sit in `<jobs_dir>/<run>-NNNN/<trial>/`, beside the job's `modules/`. The trainer then appends the job's row to the run's `jobs.jsonl`.

## 3. Verdicts & Masking Rules

A `Verdict` resolves to either a numerical reward or a specific mask reason. **Masked trials are excluded from mean metrics and gradient updates.**

### Evaluation Precedence for Served Trials (Proxy-Backed)

For a trial served through the proxy, its records (even none) are read before anything in Harbor's result:

| Order | Condition on Proxy Records | Verdict |
| --- | --- | --- |
| **1** | No calls reached the proxy | `env_error` *(masked)* |
| **2** | First request refused for context length; no requests served | `context_overflow` *(masked)* |
| **3** | Request refused by token budget | `reward = 0` |
| **4** | Context filled after $\ge 1$ answer, then harness errored (non-timeout) | `reward = 0` |
| **5** | Final request failed at the sampler | `api_error` *(masked)* |
| **6** | Harness logged a failed request unseen by the proxy (network drop) | `api_error` *(masked)* |
| **7** | A served call's prompt holds an image its record did not keep (or its completion holds an image token) | `multimodal` *(masked)* |

### Fallback Evaluation (Provider Trials or Normal Completions)

When proxy records pass or an external commercial provider was used, Harbor's outcome (`Trial.result`) decides:

| Condition on `Trial.result` | Verdict |
| --- | --- |
| Missing or unparseable `result.json` | `env_error` *(masked)* |
| `AgentTimeoutError` | `timeout` *(masked)*, unless `[recipe] grade_timeouts = true` |
| Exception type in `ENDPOINT_FAILED` (or named `*ApiError`) | `api_error` *(masked)* |
| `reward` in the verifier's `rewards` | `reward = rewards["reward"]` |
| No verifier output, or no `reward` in it | `grading_error` *(masked)* |

> `ENDPOINT_FAILED` (`admit.py`) is Harbor's `ApiError` subtree plus the login, model-not-found and network failures Harbor raises beside it (`AgentAuthenticationError`, `ModelNotFoundError`, `NetworkConnectionError`, `ApiConnectionError`), minus the policy-caused `OutputTokenExceededError`, `ContextWindowExceededError` and `AgentSafetyRefusalError`. Any other name ending in `ApiError` also counts.

## 4. Harness Profiles

A harness identifier (`[agent] harness`) points to a Harbor agent specification (e.g., `pi@0.85.1`, `acp:x.y:Z@1`). Its `Profile` configures connection mechanics:

* **`dialect`:** Wire protocol (`openai` or `anthropic`).
* **`provider`:** Custom provider slug used to route model paths (`<provider>/<model>`) and credentials (`<PROVIDER>_BASE_URL`).
* **`kwargs` / `env`:** Extra Harbor agent kwargs and environment flags (e.g., disabling internal context compaction when `[rollout] fill_context` is enabled).
* **`strip_v1`:** Strips `/v1` from the proxy address for clients that append it automatically.
* **`in_process`:** For a harness whose model calls Harbor makes in its own process: builds its Harbor agent kwargs from the proxy's loopback address and token, since the trial env does not reach those calls.

### Preconfigured Harness Profiles

| Harness | Protocol / Dialect | Key Profile Attributes |
| --- | --- | --- |
| `pi` | OpenAI (`openai-completions`) | Configured for custom base URLs; inspects local logs for dropped calls. |
| `claude-code` | Anthropic | `strip_v1 = true`; disables auto-compaction during context filling. |
| `opencode` | OpenAI | Targets `shipyard` provider via `@ai-sdk/openai-compatible`; disables background title generation. |
| `terminus-2` | OpenAI, via in-process LiteLLM | Connects via the proxy's loopback address and its harness token. |
| *Default / Other* | OpenAI | Standard fallback using `OPENAI_BASE_URL` and `OPENAI_API_KEY`. |

## 5. Secrets Isolation & Scrubbing

To prevent rate-limit contention and credential leakage, `[rollout.secrets]` pools credentials for concurrent sandboxes:

```
Table:         [rollout.secrets] API_KEY = "API_KEYS"
                     │
Process Env:   API_KEYS="key_A, key_B, key_C"
                     │
Pool Lease:    Trial 1 ──> Leases [key_A] (Index 0)
               Trial 2 ──> Leases [key_B] (Index 1)
                     │
Post-Trial:    scrub(Trial Artifacts, key_A -> "${API_KEY}") ──> Release key_A

```

* **Lease Queue:** If all identities are checked out, pending trials block until an active trial finishes.
* **Overrides:** Leased keys are injected into the trial sandbox and verifier scope, overriding task `${NAME}` defaults.
* **Post-Trial Scrubbing (`scrub`):** After the trial ends, before its identity is released, raw values and Harbor-redacted variations (of values over 8 characters) are replaced in place with `${NAME}` placeholders across all text files in the trial directory.

## 6. Prompt & Skill Modules (`modules.py`)

Modules live in a standard directory layout read by `Modules.read` and written by `keep`:

```
├── PROMPT.md                  # Text prepended to each task's instruction
└── skills/
    └── <skill_name>/
        ├── SKILL.md           # Skill definition (requires name & description frontmatter)
        └── ...                # Auxiliary UTF-8 text files (scripts, references)

```

`deliver` deploys these components to Harbor:

* **Skills:** Written to `modules/skills/<name>/` and passed as `AgentConfig.skills`; Harbor uploads each to the container's skills directory (`/harbor/skills/<name>/` by default) before agent setup.
* **Prompt Template:** Wrapped into a Jinja template (`prompt/template.j2`) as `{% raw %}<PROMPT.md>{% endraw %}{{ instruction }}` and passed via `prompt_template_path`.

Search recipes (e.g., `gepa`) inspect and mutate component texts using `Modules.texts` and `Modules.with_texts`; each job's row in `jobs.jsonl` records its modules' `digest`.

## 7. Progress & Testing

### Terminal Progress (`progress.py`)

`Bars` renders live terminal progress indicators tracking completed trials, masked runs, and elapsed duration. Output is suppressed when running in non-interactive environments or when `SHIPYARD_QUIET=1` is set.

### Test Suites

| Test File | Focus Area |
| --- | --- |
| `tests/test_engine.py` | Orchestration, concurrency limits, and record binding. |
| `tests/test_admit.py` | Verdict evaluation rules and priority ordering. |
| `tests/test_endpoints.py` | Wire dialect resolution and connection parameters per harness. |
| `tests/test_secrets.py` | Identity leasing, concurrency blocking, and artifact scrubbing. |
| `tests/test_modules.py` | Skill frontmatter validation, Jinja prompt assembly, and mutation digests. |
| `tests/test_progress.py` | Terminal progress display rendering. |

*Mocks:* `tests/fakes.py` provides standalone Harbor runners (`FakeTrials`, `Alternating`) for use with `HarborEngine(run_trial=...)` or the `harbor` pytest fixture.