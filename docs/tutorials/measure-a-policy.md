# Measure a policy

A policy is the model that answers a harness's calls. To measure it is to run it on tasks and read its mean reward, with no training. The `evaluate` recipe does this.
This tutorial starts from an empty directory. It measures one task twice: once with a model a provider serves, and once with weights that Tinker serves through the run's proxy.

You need shipyard installed (see [Installation](../getting-started/installation.md)) and Docker running. You also need `ANTHROPIC_API_KEY` for the first blueprint and `TINKER_API_KEY` for the second.

## The workspace

Shipyard works relative to the directory you run it from, the workspace ([Installation](../getting-started/installation.md#a-workspace) shows its layout). Run every command from there. Shipyard reads `.env` from the workspace when it starts. A variable already set in the shell wins over the file.

```bash title=".env"
ANTHROPIC_API_KEY=...
TINKER_API_KEY=...
```

## The dataset

A dataset is a directory of Harbor tasks under `tasks/<name>/`. Harbor downloads one:

```console
$ harbor datasets download hello-world -o tasks
Downloading dataset: hello-world@latest
Successfully downloaded 1 task(s)
```

Harbor adds the dataset's name under the directory you give it, so the task lands where shipyard looks:

```
tasks/hello-world/hello-world/
├── environment/Dockerfile
├── instruction.md     Create a file called hello.txt with "Hello, world!" as the content.
├── solution/solve.sh
├── task.toml
└── tests/
```

Give `-o` the `tasks` directory itself: `-o tasks/hello-world` puts the task one level too deep ([Datasets](../concepts/datasets.md#getting-one) shows how to flatten it).

## A provider-served blueprint

A provider-served model is one whose `[model] provider` is not `tinker`. The harness calls the provider itself.

```toml title="blueprints/01-eval-provider-docker/run.toml"
[model]
name = "claude-sonnet-5-5"
provider = "anthropic"

[data]
dataset = "hello-world"
batch_size = 1
group_size = 1

[rollout]
harness = "claude-code"
sandbox = "docker"
concurrency = 1

[recipe]
kind = "evaluate"
```

`group_size` is how many times each task is attempted; each attempt is a trial, run in its own container. `batch_size` is how many tasks go into one Harbor job, the unit the trials run in. The harness is the agent program in the container, here Claude Code.

## A Tinker-served blueprint

When `provider` is left out it is `tinker`, and the run serves the weights itself. It starts the proxy, which samples from Tinker and records every model call the harness makes. See [The proxy](../concepts/proxy.md).

```toml title="blueprints/02-eval-tinker-docker/run.toml"
[model]
name = "Qwen/Qwen3-8B"

[data]
dataset = "hello-world"
batch_size = 1
group_size = 2

[rollout]
harness = "pi@0.85.1"
sandbox = "docker"
concurrency = 2
max_tokens = 4096

[recipe]
kind = "evaluate"
```

`pi@0.85.1` pins the harness's version. `group_size = 2` attempts the task twice, and `concurrency = 2` runs both trials at once. `max_tokens` caps each reply.

## Check

`shipyard check` prints the blueprint with every default filled in, then what it found. With `--verbose` it also prints the `ok` lines:

```console
$ shipyard check blueprints/02-eval-tinker-docker --verbose
...
ok  recipe evaluate
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
ok  dataset hello-world
ok  harness pi@0.85.1, sandbox docker
ok  dataset hello-world: 1 task under tasks/hello-world
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

A problem prints as `blocked`, and `check` exits 1. A misspelled dataset reads:

```
blocked  no dataset at tasks/hello-wrld: no valid Harbor task directory under it; `harbor datasets download` fills it
```

## Run

```console
$ shipyard run blueprints/01-eval-provider-docker
01-eval-provider-docker__8Y8Byb4  /path/to/workspace/runs/01-eval-provider-docker__8Y8Byb4
...
01-eval-provider-docker__8Y8Byb4  finished
```

The first line names the run: the blueprint's directory, two underscores, seven random characters. Harbor's own logs, and a progress bar per job on a terminal, come in between. Run `blueprints/02-eval-tinker-docker` the same way.

## Read the record

`shipyard runs` lists the runs with their state (see [Commands](../reference/cli.md)). `shipyard show` prints one run, section by section:

```console
$ shipyard show 02-eval-tinker-docker__9YEWVfo
02-eval-tinker-docker__9YEWVfo
process
  state      finished
  pid        78770
  started    2026-10-07T22:33:37.769228+00:00
  finished   2026-10-07T22:34:35.996552+00:00
  version    0.1.0
  kind       evaluate
  blueprint  /path/to/workspace/blueprints/02-eval-tinker-docker
  directory  /path/to/workspace/runs/02-eval-tinker-docker__9YEWVfo
  proxy      {"placement": "local", "origin": "http://host.docker.internal:58735"}
  backend    https://tinker.thinkingmachines.dev/services/tinker-prod
metrics  2 rows
  at          2026-10-07T22:34:34.999038+00:00
  seq         2
  evaluation  true
  batches     1
  rollouts    2
  graded      2
  masked      0
  mean        1.0
jobs  1 row
  at               2026-10-07T22:34:34
  job              02-eval-tinker-docker__9YEWVfo-0000
  purpose          rollout
  trials           2
  batch            0
  tasks            1
  graded           2
  served           tinker
  bridged          2
  party            tinker
  input_tokens     1882
  cache_tokens     4224
  output_tokens    525
  sandbox          docker
  sandbox_seconds  69.63496599999999
costs
  {
    "parties": {
      "tinker": {
        "trials": 2,
        "input_tokens": 1882,
        "cache_tokens": 4224,
        "output_tokens": 525
      }
    },
    "sandbox": {
      "docker": {
        "trials": 2,
        "seconds": 69.63496599999999
      }
    }
  }
```

`process` says where the proxy ran (`local`, on this machine, reached from the containers as `host.docker.internal`) and which Tinker backend served the weights. `metrics.jsonl` holds a row per batch, then one row for the whole run, marked `evaluation`. Both trials were graded and both scored 1, so the mean is 1.0. A masked trial, one the run could not measure fairly, would be left out of the mean, never counted as 0 (see [Admission](../concepts/admission.md)). The run took 58 seconds; the two containers ran for 70 seconds between them.

The jobs section holds `jobs.jsonl`, one row per Harbor job, printed as a block when it is wider than the terminal.

The provider-served run has the same files with its own counts: one trial, mean 1.0, party `anthropic`, 30,901 input tokens (25,837 cached) and 116 output tokens, 68 sandbox seconds. Those token counts are what Claude Code reported to Harbor.

## Read requests.jsonl

A Tinker-served run also writes `requests.jsonl`: one row per model call the proxy received, including any it refused for the budget or the context. This run made four:

| trial | seq | prompt_tokens | cached_tokens | completion_tokens | bridged |
|---|---|---|---|---|---|
| hello-world__3a7jB84 | 1 | 1424 | 0 | 183 | false |
| hello-world__3a7jB84 | 2 | 1625 | 1408 | 126 | true |
| hello-world__3vQgSv9 | 1 | 1424 | 1408 | 191 | false |
| hello-world__3vQgSv9 | 2 | 1633 | 1408 | 25 | true |

Each row also carries `at`, `job`, `stop_reason`, `sample_ms`, `served`, `request_id` and `error`. Here `served` is null because the base model answered, and `error` is null because no call was refused or failed.

- The prompts add up to 6,106 tokens: the 1,882 uncached plus the 4,224 cached in `costs.json`. The completions add up to 525.
- Pi's own counts in each trial's `result.json` add up to the same 6,106 and 525. The proxy saw every call.
- Each trial's second call is `bridged`. Its prompt was built from the first reply's tokens, so the trial would train as one sequence.

`proxy.log` holds what the proxy printed, including where it served:

```
serving Qwen/Qwen3-8B at http://host.docker.internal:58735 (control: on)
```

A provider-served run has neither file.

## Next

- [Train with dapo](train-with-dapo.md) trains the Tinker-served policy on the same task.
- [Evolve a skill](evolve-a-skill.md) improves the text a provider-served policy reads.
- [Runs](../concepts/runs.md) describes every file of the record.
