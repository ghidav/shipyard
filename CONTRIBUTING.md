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

## A new recipe preset

A gradient recipe is a name for a `Preset`: how the advantage is formed, which Tinker loss, which clipping. To add one called `my-preset`:

1. In `src/shipyard/config.py`, add a class over `Gradient` with `kind: Literal["my-preset"]` and its own knobs, each with a default and a bound. Add it to the `Recipe` union and its name to `KINDS`.
2. Add `src/shipyard/recipes/my_preset.py` (a hyphen in the name becomes an underscore). Like `dapo.py`, it imports `shipyard.recipes.train` as `loop` and defines two functions. `preset(recipe) -> Preset` builds the preset with `loop.shared(recipe)` and, for a clipped ratio, `loop.clipped(low, high)`. `async def run(run)` awaits `loop.train(run, preset(run.config.recipe))`.
3. Add a fixture, `tests/blueprints/my-preset/run.toml`. Test what the preset resolves to beside the other three in `tests/test_credit.py`, and its defaults and `check` line in `tests/test_config.py`.
4. Document it in `docs/concepts/recipes.md` and `docs/reference/configuration.md`.

## A new module kind

To add a module kind:

1. In `src/shipyard/modules.py`, write a class with the `Kind` protocol: `marker` (the file that marks a directory as this kind), `check(module)` (why a module cannot be delivered, or None), `deliver(modules, into)` (write them under `into` and return the Harbor agent-config fields that carry them), and `reference()` (what a gepa reflector is told about the kind, or None).
2. Add it to `KINDS` by name. If its marker is in `RESERVED` (`server.py` for tools, `agent.py` for harnesses), take it out.
3. Test `seed`, `check` and `deliver` for it in `tests/test_modules.py`.
4. Document it in `docs/concepts/modules.md`.
