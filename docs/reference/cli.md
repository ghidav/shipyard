# Commands

`shipyard` has five commands: `check`, `run`, `runs`, `show` and `serve`. Each takes `-h` or `--help`.

Run them from your workspace. At start, shipyard reads `.env` from the working directory; a variable already set in the shell wins over the file. Datasets are read from `tasks/` under the working directory. Runs live under `--root`, else `$SHIPYARD_RUNS`, else `./runs`.

## shipyard check

```
shipyard check BLUEPRINT [--json] [--verbose]
```

Says whether a blueprint could run, before anything is spent.

| | |
|---|---|
| `BLUEPRINT` | a directory holding `run.toml`, or the file itself |
| `--verbose` | print the `ok` lines too |
| `--json` | print one JSON object instead |

It prints the blueprint with every default filled in, as TOML. For a Tinker-served model with no `[rollout] renderer`, the `renderer` line names the one the proxy will load. For `dapo`, `dr-grpo`, `cispo` and `fst` a comment line follows, saying what the recipe's name resolves to. Then come the findings, one per line: `blocked`, `warning`, and, under `--verbose`, `ok`. Every problem is listed at once.

For a Tinker-served model, `check` asks the backend whether it serves the model. That network call is made only when `TINKER_API_KEY` is set; without it, the line is a warning.

For a sandbox elsewhere, `check` blocks when any package of Harbor's extra for it is missing. For a `gepa` or `fst` reflector, it warns when none of the API keys Harbor hands that harness is set.

For `sandbox = "modal"`, with Harbor's modal extra installed and a token set, `check` makes one more network call: it looks an app up on Modal, which creates nothing. When Modal refuses the token, `check` blocks and says where the token came from: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment, and whether the working directory's `.env` assigns them, or a profile in `~/.modal.toml`, or in the file `MODAL_CONFIG_PATH` names. Modal reads each variable before the profile, so when `MODAL_TOKEN_ID` differs from the `token_id` of the profile it shadows, `check` says so; it compares ids and reads no secret. Any other failure, or no answer within 15 seconds, is a warning.

For `sandbox = "docker"`, `check` warns when 20 or more of Harbor's trial networks have no container on them. A run killed hard leaves its trials' networks behind, and Docker's default address pools hold about 30, past which no trial starts. The warning names a command that removes those networks and no others:

```sh
docker network ls -q --filter dangling=true --filter name=__env_default --filter name=__verifier__ | xargs docker network rm
```

`check` exits 1 when any finding is `blocked`, else 0.

```console
$ shipyard check blueprints/typo
blocked  [data] batch_size: field required
blocked  [data] batchsize: unknown key
blocked  [recipe] kind: no recipe called 'evalute'; one of dapo, dr-grpo, cispo, gepa, evaluate
```

A blueprint that does not load prints no config. Under `--json` the output is `{"config": ..., "findings": [{"level": ..., "text": ...}]}`, with every finding, `ok` included, and `config` null when the blueprint did not load. The JSON has no comment line.

## shipyard run

```
shipyard run BLUEPRINT [--root PATH]
```

Checks the blueprint, opens a run, hands it to its recipe, and records how it ended.

| | |
|---|---|
| `BLUEPRINT` | a directory holding `run.toml`, or the file itself |
| `--root PATH` | where to write the run; `$SHIPYARD_RUNS` or `./runs` by default |

`run` makes the same checks as `check` and prints every finding that is not `ok`. If one is `blocked`, it prints `blocked; nothing was started` and exits 1. Otherwise it creates `<root>/<blueprint>__<7 chars>/`, copies `run.toml` into it, and prints the run's id and directory. It refuses a run directory that already exists.

On a terminal it draws a progress bar per job; `SHIPYARD_QUIET=1` turns them off. At the end it prints the error, if there was one, then the id and the state.

| exit | when |
|---|---|
| 0 | the run finished |
| 1 | blocked, refused, or the recipe raised; the traceback is printed |
| 130 | stopped by Ctrl-C, SIGTERM or SIGHUP |

A stopped run is recorded as failed, with the error `stopped: SIGINT` (or the signal's name).

```console
$ shipyard run blueprints/02-eval-tinker-docker
02-eval-tinker-docker__eTNXY25  /path/to/workspace/runs/02-eval-tinker-docker__eTNXY25
[10/07/26 22:56:29] INFO     proxy serving Qwen/Qwen3-8B on port 56479 as host.docker.internal
...
02-eval-tinker-docker__eTNXY25  finished
```

See [Runs](../concepts/runs.md) for what the run directory holds.

## shipyard runs

```
shipyard runs [--root PATH]
```

Lists every run under the root, newest first: its id, recipe kind, state, and when it started and finished. It prints `(no runs)` when there are none. When the table is wider than the terminal, each run is printed as a block of `key  value` lines instead, so nothing is cut.

The state is one of:

| state | meaning |
|---|---|
| `running` | no finish recorded, and its process is alive |
| `finished` | it ended without an error |
| `failed` | it ended with an error, or was stopped |
| `died` | no finish recorded, and its process is gone |

```console
$ shipyard runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ id                               ┃ kind     ┃ state    ┃ started             ┃ finished            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ 01-eval-provider-docker__g9SJNBf │ evaluate │ finished │ 2026-10-07T20:58:46 │ 2026-10-07T21:00:15 │
│ 02-eval-tinker-docker__eTNXY25   │ evaluate │ finished │ 2026-10-07T20:56:22 │ 2026-10-07T20:57:52 │
└──────────────────────────────────┴──────────┴──────────┴─────────────────────┴─────────────────────┘
```

## shipyard show

```
shipyard show RUN_ID [--root PATH] [--full] [--json]
```

Prints one run, section by section.

| | |
|---|---|
| `RUN_ID` | a run id, `<blueprint>__<7 chars>` |
| `--root PATH` | where runs live; `$SHIPYARD_RUNS` or `./runs` by default |
| `--full` | every job row, not the last 10 |
| `--json` | one JSON object with the five sections |

The sections, in order:

| section | from | printed as |
|---|---|---|
| `process` | `process.json` | the state, the error if any, pid, started, finished, version, kind, blueprint, directory; then `proxy` and `backend` (its `tinker_base_url`) when the run noted them |
| `metrics` | `metrics.jsonl` | the row count and the last row, one field per line |
| `jobs` | `jobs.jsonl` | the row count and a table of the last 10 rows; a block of `key  value` lines per row when the table is wider than the terminal |
| `checkpoints` | `checkpoints.jsonl` | the row count and a table of every row, as blocks like `jobs` when it is wider than the terminal |
| `costs` | `costs.json` | the document as JSON |

A section whose file is absent is not printed. Under `--json` it is null; `metrics` is `{"count", "last"}` and `jobs` and `checkpoints` are `{"count", "rows"}` with every row.

A run id that is not under the root prints `no run named <id> under <root>` and exits 1.

```console
$ shipyard show 01-eval-provider-docker__g9SJNBf
01-eval-provider-docker__g9SJNBf
process
  state      finished
  pid        19984
  started    2026-10-07T20:58:46.566055+00:00
  finished   2026-10-07T21:00:15.724357+00:00
  version    0.1.0
  kind       evaluate
  blueprint  /path/to/workspace/blueprints/01-eval-provider-docker
  directory  /path/to/workspace/runs/01-eval-provider-docker__g9SJNBf
metrics  2 rows
  at          2026-10-07T21:00:15.723661+00:00
  seq         2
  evaluation  true
  batches     1
  rollouts    1
  graded      1
  masked      0
  mean        1.0
...
```

## shipyard serve

```
shipyard serve --model MODEL [--weights PATH] [--bind HOST] [--port N]
               [--advertise NAME] [--renderer NAME] [--settings JSON]
```

Serves a Tinker model to a harness in a sandbox and records every model call: the proxy. A run that serves its own weights starts this process itself, so you rarely type it. See [The proxy](../concepts/proxy.md).

| | |
|---|---|
| `--model MODEL` | required: the base model the weights are of |
| `--weights PATH` | a `tinker://` path to serve; without one, the base model |
| `--bind HOST` | the interface to listen on; `0.0.0.0` by default |
| `--port N` | the port to listen on; 0, the default, for any free port |
| `--advertise NAME` | the name a sandbox reaches the proxy by, when it is not the bind |
| `--renderer NAME` | a cookbook renderer to use instead of the model's own |
| `--settings JSON` | the run's other endpoint keys, as a JSON object: `temperature`, `top_p`, `top_k`, `max_tokens`, `max_context`, `fill_context`, `volatile` |

The harness presents `SHIPYARD_PROXY_TOKEN` as its API key. A run presents `SHIPYARD_CONTROL_TOKEN` to point the proxy at other weights. Each is read from the environment; one that is unset is generated, and printed after the first line as `<NAME> <value>`.

A bind of `0.0.0.0`, the default, is every interface, which no sandbox can dial, so `--advertise` must name the host it can. Without it, or with a `--settings` key it does not take, `serve` prints why and exits 2. Otherwise it serves until SIGINT or SIGTERM.

```console
$ shipyard serve --model Qwen/Qwen3-8B --advertise host.docker.internal --settings '{"max_tokens": 4096}'
```

Its first line is the one a run reads. The run's own proxy printed this, in `proxy.log`:

```
serving Qwen/Qwen3-8B at http://host.docker.internal:56479 (control: on)
```

To use a proxy you started yourself, name its address in `[rollout] endpoint_url` and export the same `SHIPYARD_PROXY_TOKEN` in the shell that runs `shipyard run`; `check` blocks without it. A run that changes the served weights also needs the same `SHIPYARD_CONTROL_TOKEN`. Such a run starts neither a proxy nor a tunnel. Without `endpoint_url`, a remote sandbox reaches the proxy through a tunnel the run starts.

## Environment

| variable | read by | what it does |
|---|---|---|
| `SHIPYARD_RUNS` | `run`, `runs`, `show` | where runs live when `--root` is not given |
| `SHIPYARD_QUIET` | `run` | `1` draws no progress bars |
| `SHIPYARD_PROXY_TOKEN` | `check`, `run`, `serve` | the harness's key to the proxy; generated per run when unset; `check` blocks without it when `[rollout] endpoint_url` is set |
| `SHIPYARD_CONTROL_TOKEN` | `run`, `serve` | the run's key to point the proxy at weights; generated per run when unset |
| `TINKER_API_KEY` | `check`, `run`, `serve` | the Tinker-API backend's key |
| `TINKER_BASE_URL` | `check`, `run`, `serve` | another Tinker-API backend; costs are then filed under `tinker@<host>` |

See [Configuration](configuration.md) for every `run.toml` key.
