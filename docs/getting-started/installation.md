# Installation

shipyard is a Python package with one command, `shipyard`. You run it from a **workspace**: a directory that holds your blueprints, datasets, runs and keys.

## shipyard

shipyard needs Python 3.12 or newer. In a uv project (`uv init` makes one):

```sh
uv add shipyard
```

Commands then run as `uv run shipyard ...`, or as `shipyard ...` with the project's venv activated. Harbor comes with it, so `uv run harbor ...` works too.

## A sandbox

`[rollout] sandbox` names a Harbor environment. The default, `docker`, needs Docker running. `podman` and `apple-container` use their own runtimes. These three are **local**: on this machine. Every other sandbox is **remote**.

A remote sandbox needs its provider's SDK, which Harbor ships as an extra:

```sh
uv add "harbor[modal]"     # or harbor[daytona], harbor[e2b], harbor[runloop], ...
```

A run whose model is served from Tinker, the default `[model] provider`, starts a **proxy**: the process that serves the weights to the harness. A remote sandbox reaches it through a tunnel the run starts. The tunnel uses the `cloudflared` binary when it is on your `PATH` (`brew install cloudflared`), else the `cloudflare/cloudflared` image through Docker. With neither, `check` blocks: install one, or name a proxy you started yourself in `[rollout] endpoint_url` ([CLI](../reference/cli.md#shipyard-serve)).

## Keys

shipyard reads `.env` from the working directory at every command. A variable already set in the shell wins over the file. Harbor reads its keys from the same environment.

Start from this file, which the repository also ships as `docs/.env.example`, and uncomment what you use:

```sh title=".env"
# A model served from Tinker
# TINKER_API_KEY=
# Another backend that implements the Tinker API; unset: Thinking Machines' Tinker
# TINKER_BASE_URL=

# Optional: the proxy fetches the model's tokenizer from Hugging Face
# HF_TOKEN=

# A model its provider serves ([model] provider), or gepa's reflection_model
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=

# Sandboxes: only the ones you use; Harbor reads these. Modal, Beam and Blaxel also take
# their own CLI's login, and Daytona a JWT with an organization id.
# MODAL_TOKEN_ID=
# MODAL_TOKEN_SECRET=
# DAYTONA_API_KEY=
# E2B_API_KEY=
# RUNLOOP_API_KEY=
# BEAM_TOKEN=
# BL_WORKSPACE=
# BL_API_KEY=

# The proxy's two tokens; unset, each run makes its own. Never in [rollout] env.
# SHIPYARD_PROXY_TOKEN=
# SHIPYARD_CONTROL_TOKEN=

# Where runs live; unset: ./runs
# SHIPYARD_RUNS=
```

`TINKER_BASE_URL` points shipyard at another backend that implements the Tinker API, such as Fireworks or Baseten. Set it to the URL that backend gives you, and `TINKER_API_KEY` to that backend's key. The run notes the URL in its `process.json`, and its costs go under `tinker@<host>` instead of `tinker`.

!!! warning
    Keep keys out of `run.toml`, `[rollout] env` included. Every run copies its `run.toml`, byte for byte, into its record.

## A workspace

```
my-workspace/
├── .env
├── blueprints/
│   └── 02-eval-tinker-docker/
│       └── run.toml
├── tasks/
│   └── hello-world/        a dataset
│       └── hello-world/    a Harbor task
├── runs/                   one directory per run, written by shipyard
└── jobs/                   one Harbor job directory per batch
```

- `blueprints/` is a convention. `check` and `run` take a path: the blueprint's directory, or its `run.toml`.
- `tasks/<dataset>/<task>/` is where `check` and `run` look for datasets, under the working directory. Harbor fills it: `uv run harbor datasets download hello-world -o tasks`.
- `runs/` is `--root`, else `SHIPYARD_RUNS`, else `./runs`.
- `jobs/` is `[rollout] jobs_dir`, `"jobs"` by default.

Keep `.env`, `runs/`, `jobs/` and `tasks/` out of version control:

```text title=".gitignore"
.env
runs/
jobs/
tasks/
```

## Check it works

`shipyard check` reads a blueprint and lists every problem at once, before anything is spent. `--verbose` adds the `ok` lines:

```
$ uv run shipyard check --verbose blueprints/02-eval-tinker-docker
...
ok  recipe evaluate
ok  model Qwen/Qwen3-8B, provider tinker (served by this run)
ok  dataset hello-world
ok  harness pi@0.85.1, sandbox docker
ok  dataset hello-world: 1 task under tasks/hello-world
ok  serving Qwen/Qwen3-8B on this machine for a docker sandbox
ok  tinker serves Qwen/Qwen3-8B
```

The `...` stands for the resolved config, printed first with every default filled in. `check` exits 1 when a line says `blocked`, and 0 otherwise.

Next: [Measure a policy](../tutorials/measure-a-policy.md).
