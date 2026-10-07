# Changelog

## 0.1.0 (unreleased)

The first release.

- **Blueprints.** A blueprint is a directory holding a `run.toml` with `[model]`, `[data]`, `[rollout]`, `[recipe]` and `[checkpoints]`. `shipyard check` prints the config as resolved and every problem at once, before anything is spent.
- **Five recipes.** `evaluate` measures; `dapo`, `dr-grpo` and `cispo` train, each a fixed preset of advantage, loss and clipping; `gepa` evolves the skills a harness reads; `fst` alternates the two in cycles (fast-slow training). Any recipe can carry skills, directories holding a `SKILL.md`, into every rollout.
- **Runs.** Every run writes `runs/<blueprint>__<7 chars>/`: `run.toml` copied byte for byte, `process.json`, `metrics.jsonl`, `jobs.jsonl`, `requests.jsonl`, `checkpoints.jsonl`, `costs.json`, `modules/`, and `proxy.log` when it serves. `shipyard runs` lists them; `shipyard show` prints one.
- **Datasets as directories.** A dataset is `tasks/<name>/`, one Harbor task per subdirectory. A blueprint may name a list. Batches are a seeded reshuffle per epoch.
- **Rollouts on Harbor.** One Harbor job per batch, under `jobs/<run id>-NNNN/`, on any Harbor sandbox. Profiles wire `pi`, `claude-code`, `opencode` and `terminus-2`; any other Harbor agent runs with the generic OpenAI wiring.
- **The proxy.** `shipyard serve` serves Tinker weights in the OpenAI and Anthropic dialects, behind a harness token, and records every model call as token ids and logprobs. A separate control token re-points it at new weights.
- **Training targets by token prefix.** A call whose prompt extends the previous prompt and completion continues its sequence; any other call starts a new one. Each completion is trained on exactly once.
- **Admission.** A rollout with no result, no reward, no model calls on record, a cut clock, a failed endpoint, an image, or a first prompt that did not fit is masked, with its reason on the job row. A masked rollout enters no mean; a zero does.
- **The owned step.** Publish the weights, point the proxy at them, sample, credit, `forward_backward`, `optim_step`, over the Tinker SDK and any backend that implements it (`TINKER_BASE_URL`). Checkpoints every `[checkpoints] every` steps, and a final one always. A continuation is a new run from `[model] from_checkpoint`.
- **Costs as counts.** Tokens in, cached and out per party, training tokens, and sandbox seconds per sandbox, in `costs.json`.
