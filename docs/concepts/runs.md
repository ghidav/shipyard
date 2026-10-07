# Runs

A run is one execution of a [blueprint](blueprints.md). Its record is a directory named
`<blueprint>__<7 chars>`: the first 32 characters of the blueprint directory's name, and seven
random characters. Every `shipyard run` opens a new one.

Runs live under `--root` when a command is given one, else under `$SHIPYARD_RUNS`, else under
`./runs`. `run`, `runs` and `show` all look there.

## The run directory

```text
runs/02-eval-tinker-docker__eTNXY25/
├── costs.json
├── jobs.jsonl
├── metrics.jsonl
├── process.json
├── proxy.log
├── requests.jsonl
└── run.toml
```

| File | What it holds | Written |
|---|---|---|
| `run.toml` | the blueprint's file, copied byte for byte | when the run opens |
| `process.json` | who ran it, and how it ended | when the run opens, and when it ends |
| `jobs.jsonl` | one row per Harbor job | as each job's trials come back |
| `requests.jsonl` | one row per model call the proxy served | as each job's trials come back |
| `metrics.jsonl` | the recipe's rows | per batch, step or round |
| `checkpoints.jsonl` | one row per saved checkpoint | at each checkpoint |
| `costs.json` | counts per party and per sandbox | after each job and each step |
| `modules/` | the text the run carried or searched | see below |
| `proxy.log` | the proxy's output | while the proxy runs |

A file the run had no reason to write is absent. An `evaluate` run has no `checkpoints.jsonl`.
A run whose model is not served (any `[model] provider` but `"tinker"`) has no
`requests.jsonl` and no `proxy.log`. A run whose `[rollout] endpoint_url` names a proxy started
elsewhere has no `proxy.log` either. Without that key, a remote sandbox reaches the proxy through
a tunnel the run starts, and the tunnel's command and output go to `tunnel.log`.

`modules/` holds `carried/` when a gradient or `evaluate` recipe names `[recipe] modules`,
copied when the run opens. A `gepa` run writes `seed/` when it starts and `best/` when its
search ends. See [Modules](modules.md).

The trials live outside the run directory, in one Harbor job directory per batch under
`[rollout] jobs_dir`: `jobs/<run id>-0000/`, `-0001/` and so on. Each trial in a job is a
Harbor trial directory, such as `jobs/02-eval-tinker-docker__eTNXY25-0000/hello-world__csFTaNW/`,
with its `result.json`, its agent's logs and its verifier's output.

## The files, one row each

`process.json` is written when the run opens and completed when it ends. `tinker_base_url`
appears only for a served model:

```json
{
  "pid": 18425,
  "started_at": "2026-10-07T20:56:22.261688+00:00",
  "version": "0.1.0",
  "blueprint": "/.../blueprints/02-eval-tinker-docker",
  "kind": "evaluate",
  "tinker_base_url": "https://tinker.thinkingmachines.dev/services/tinker-prod",
  "finished_at": "2026-10-07T20:57:52.361554+00:00",
  "failed": false,
  "error": null
}
```

A served run also notes `proxy` at its first job: the proxy's `placement` (`local`, `tunnel`
or `remote`) and the `origin` its sandboxes dialled. `shipyard show` prints it, and prints
`tinker_base_url` as `backend`.

A `jobs.jsonl` row says what one Harbor job was and what it cost. `purpose` is `rollout`, or
`reflection` for a `gepa` reflector. `served` is `tinker` when the proxy answered and
`provider` otherwise. `masked` counts masked trials by reason ([Admission](admission.md)),
and `ended` counts the exceptions Harbor recorded on trials, by type; each appears only when it
has something to count. The token and sandbox counts are explained in [Costs](costs.md).

```json
{"at": "2026-10-07T20:57:50.871956+00:00", "job": "02-eval-tinker-docker__eTNXY25-0000", "purpose": "rollout", "trials": 2, "batch": 0, "tasks": 1, "graded": 2, "served": "tinker", "bridged": 2, "party": "tinker", "input_tokens": 1992, "cache_tokens": 4224, "output_tokens": 727, "sandbox": "docker", "sandbox_seconds": 131.392899}
```

A `requests.jsonl` row is one model call, in the order the trial made them:

```json
{"at": "2026-10-07T20:57:50.840828+00:00", "job": "02-eval-tinker-docker__eTNXY25-0000", "trial": "hello-world__csFTaNW", "seq": 2, "prompt_tokens": 1681, "cached_tokens": 1408, "completion_tokens": 128, "stop_reason": "stop", "sample_ms": 2630.4, "served": null, "request_id": null, "bridged": true, "error": null}
```

`metrics.jsonl` is the recipe's own. Every row has `at` and `seq`, a count from 1, since two
rows can share a timestamp. `evaluate` writes a row per batch and a closing row:

```json
{"at": "2026-10-07T20:57:50.873382+00:00", "seq": 1, "batch": 0, "tasks": 1, "rollouts": 2, "graded": 2, "masked": 0, "mean": 1.0}
{"at": "2026-10-07T20:57:50.873722+00:00", "seq": 2, "evaluation": true, "batches": 1, "rollouts": 2, "graded": 2, "masked": 0, "mean": 1.0}
```

A gradient recipe writes a row per step ([Recipes](recipes.md)); `gepa` a row per round and a
closing row ([gepa](gepa.md)).

`checkpoints.jsonl` has one row per save, each with `tag`, `state_path`, `sampler_path` and
`ttl_hours`. The tag is `step-<n>` every `[checkpoints] every` steps, and `final` once the
last step is done. The state path rebuilds a trainer; the sampler path only serves.

## Four states

`process.json`, and whether its `pid` is still alive, decide a run's state:

| State | Means |
|---|---|
| `finished` | `finished_at` is set and `failed` is false |
| `failed` | `failed` is true; `error` says why |
| `running` | no `finished_at`, and the `pid` is alive on this machine |
| `died` | no `finished_at`, and the `pid` is gone |

A run that raises, or is stopped by Ctrl-C, `SIGTERM` or `SIGHUP`, still closes its record:
`failed` is true and `error` holds `<Exception>: <message>` or `stopped: SIGTERM`. A run
killed in a way it cannot catch never writes `finished_at`, and reads `died` once its `pid` is
gone.

## Reading runs

`shipyard runs` lists every run under the root, newest first, with its kind, state and times.
`shipyard show <run id>` prints one run in five sections: `process`, `metrics`, `jobs`,
`checkpoints` and `costs`, leaving out a section whose file is absent. `process` adds an `error`
line for a failed run. `metrics` shows the row count and the last row; `jobs` the last 10 rows,
or every row with `--full`. `--json` prints the five sections as one object.
[Commands](../reference/cli.md#shipyard-runs) shows the output of both.

## Append and flush, never amended

A `.jsonl` file only grows. Each row is one line, stamped `at`, and flushed before the write
returns, so a row on disk is a row that happened. A reader skips a cut last line.
`process.json` and `costs.json` are whole documents, each written beside the old one and
renamed over it, so a reader sees the old document or the new one, never half.

Once a run ends, nothing writes to its directory again.

## Going on from a checkpoint

To go on training after a run ends, start a new run from one of its checkpoints. Copy the
`state_path` of a `checkpoints.jsonl` row into `[model] from_checkpoint`, and set
`restore_optimizer = true` to load the optimizer state with the weights:

```toml
[model]
name = "Qwen/Qwen3-8B"
from_checkpoint = "tinker://<id>/weights/step-2"
restore_optimizer = true
```

The rest of the blueprint is as before. The new run has its own id and an empty record. Its
batches start again from the first epoch, so change `[data] seed` for another order.

The same path also goes to the proxy, which loads it as it starts, before the first step points
it at newly published weights. With `kl_coef > 0` it is the anchor of the KL term too. So it
must be a path the backend can both train from and sample from.

To measure a checkpoint without training, name its `sampler_path` in an `evaluate`
blueprint's `from_checkpoint`.
