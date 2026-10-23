# Shipyard

Post-training framework for AI agents on Harbor tasks.

## Installation

Requires **Python 3.12+** and [uv](https://github.com/astral-sh/uv):

```sh
uv tool install shipyard
```

## Quickstart

```sh
shipyard init                                     # copy the recipes and .env.example here
cp .env.example .env                              # then set your keys in .env
uvx harbor datasets download <dataset> -o tasks   # tasks land in tasks/<dataset>/<task>/
shipyard run recipes/dapo --data.train <dataset>  # the run writes runs/dapo__<id>/
```

> [!IMPORTANT]
> Add `--dry` to see every setting the run would use, with each default `run.toml` leaves marked, and to check the run without starting it.

## Customizing


- **Settings:** a recipe's settings can be updated in its `run.toml` (model, harnesses, data, learning rate, LoRA rank, ...); you can overwrite them using `--table.key VALUE` on `shipyard run`.
- **Algorithm:** a recipe defines a `trainer.py` where it subclasses `Trainer` and writes `step` from the primitives `draw`, `rollout` and `update`, giving the sequences their advantages in between; the core holds no algorithm of its own (see [`src/shipyard`](src/shipyard/README.md)).
- **Recipes:** GRPO, DAPO, on-policy distillation and GEPA, in [`recipes/`](recipes/README.md).

The trainer updates on exactly what the agent sampled: `harness-proxy` serves the model to the harnesses, and records every call's token ids and logprobs.

## Layout

```
src/shipyard/            the Trainer API, with models/, jobs/ and run/
packages/harness-proxy/  serves the agent to its harnesses, and records every call
recipes/                 the built-in recipes
```

## License

Apache-2.0. See [LICENSE](LICENSE).
