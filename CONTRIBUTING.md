# Contributing

## Set up

```sh
uv sync --extra dev
```

## The checks

CI runs these on every push and pull request. Run them before you open one:

```sh
uv run ruff check src tests
uv run ruff format src tests      # CI runs it with --check
uv run pytest -q
```

The suite needs no keys and no sandbox. `tests/conftest.py` clears the Tinker and proxy variables from the shell. Harbor trials, the Tinker training client and the proxy's sampling client are fakes under `tests/` (`trials.py`, `trainers.py`, `proxies.py`, `records.py`).

The docs build with:

```sh
uvx --with mkdocs-material mkdocs build --strict
```

## Where the code is

`src/shipyard/` keeps the shared pieces at the top and groups the rest by part: `sampling/`
(running trials), `proxy/` (serving the weights), `training/` (the trainer and its recipes) and
`cli/` (the commands, and the run they open). Roughly, by docs page:

| Docs page | Modules |
|---|---|
| Blueprints | `config` (the schema and `load`), `cli/preflight` (every `check` finding), `cli/resolved` (what `check` prints), `cli/main` (the commands) |
| Runs | `cli/run` (the run `shipyard run` opens around a trainer), `record` (the files' names, writers and readers), `logs` (the trainer's loggers), `cli/show` (a run read back), `cli/smoke`, `progress` (live bars) |
| Datasets | `data` |
| Rollouts | `sampling/engine` (one Harbor job), `sampling/endpoint` (the run's side of serving), `sampling/secrets` (what each trial holds alone), `harnesses` (per-harness wiring) |
| The proxy | `proxy/serve` (the process), `proxy/server` and the modules beside it, `proxy/wire` (shared by both ends), `proxy/client` and `proxy/tunnel` (the run's side) |
| Admission | `sampling/admit` (a trial's verdict: its reward, or why it is masked) |
| Recipes | `training/trainer` (the base `Trainer`: the loop, a step, a job), `training/recipes/<kind>` (one subclass per `[recipe] kind`), `training/credit` (rewards into advantages), `training/pack` (records into sequences and datums), `training/client` (the Tinker training client) |
| Modules | `modules` |
| gepa | `training/recipes/gepa/` (`cycle` is its loop) |
| Costs | `cost` |

`tests/test_<module>.py` tests `<module>`. A test file named for a behaviour instead (`test_train`, `test_lifecycle`, `test_sandboxes`) says so in its docstring. Fixtures are `tests/blueprints/`, `tests/modules/` and `tests/tasks/`; the fakes are `tests/proxies.py`, `tests/trainers.py`, `tests/trials.py` and `tests/records.py`.

## A new recipe

A gradient recipe is a `Trainer` subclass that overrides the rules it changes. Per rollout:
`shape` and `advantage` (rewards into one advantage each), and `privileged` (text the
teacher reads first). Per group: `trains_on` (whether it
carries a gradient) and `weights` (its factor on its tokens). Per token, in the order a step
runs them: `signals` (logprobs and other values each training sequence is scored with, the
teacher's from `teacher` among them), `token_advantages` (the rollout's advantage and the
signals combined at each target) and
`datum` (what Tinker receives). Per substep: `loss_fn` and `bounds` (a Tinker loss and its
config), or `forward_backward` for several losses or one of its own. Then `ADAM` (the
optimizer) and `rules` (what `check` prints about it). To add one called `my-recipe`:

1. In `src/shipyard/config.py`, add a class over `Gradient` with `kind: Literal["my-recipe"]` and its own knobs, each with a default and a bound. Add it to the `Recipe` union and its name to `KINDS`.
2. Add `src/shipyard/training/recipes/my_recipe.py` (a hyphen in the name becomes an underscore) defining `MyRecipeTrainer`, a subclass of `Trainer` or of the recipe it is closest to, such as `DAPOTrainer`. Register it in `TRAINERS` in `src/shipyard/training/recipes/__init__.py`, and export it from `src/shipyard/training/__init__.py`.
3. Add a fixture, `tests/blueprints/my-recipe/run.toml`. Test its rules in `tests/test_training_trainer.py`, and its defaults and `check` line in `tests/test_config.py`.
4. Document it in `docs/concepts/recipes.md` and `docs/reference/configuration.md`.

A recipe that only changes the run around the step (how a step samples, what it logs, a loop of its own) overrides `sample`, `log` or `train` the same way. A one-off of this kind can live in a trainer file passed to `shipyard run` instead of the package.

## A new module kind

To add a module kind:

1. In `src/shipyard/modules.py`, write a class with the `Kind` protocol: `marker` (the file that marks a directory as this kind), `alone` (whether the marker is the whole module), `home(name)` (the module's directory under the modules directory, `""` for the top), `check(module)` (why a module cannot be delivered, or None), `deliver(modules, into)` (write them under `into` and return the Harbor agent-config fields that carry them), and `reference()` (what a gepa reflector is told about the kind, or None).
2. Add it to `KINDS` by name. If its marker is in `RESERVED` (`server.py` for tools, `agent.py` for harnesses), take it out.
3. Test `seed`, `check` and `deliver` for it in `tests/test_modules.py`.
4. Document it in `docs/concepts/modules.md`.
