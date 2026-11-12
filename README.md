# Shipyard

Post-training framework for AI agents on Harbor tasks.

## Installation

Requires **Python 3.12+** and [uv](https://github.com/astral-sh/uv):

```sh
uv tool install shipyard
```

## Quickstart

The recipes train on [Tinker](https://thinkingmachines.ai/tinker/) and run their trials in [Modal](https://modal.com)'s sandboxes, whose calls reach the model through a Cloudflare tunnel this machine opens: install [`cloudflared`](https://github.com/cloudflare/cloudflared), or keep Docker running.

```sh
shipyard init                                                    # copy the recipes and .env.example here
cp .env.example .env                                             # then set TINKER_API_KEY in .env
uvx modal setup                                                  # log in to Modal, or set its keys in .env
uvx harbor@0.24.0 datasets download reasoning-gym-easy -o tasks  # the recipes' tasks, in tasks/reasoning-gym-easy/
shipyard run recipes/dapo                                        # the run writes runs/dapo__<id>/
```

> [!IMPORTANT]
> Add `--dry` to see every setting the run would use, with each default `run.toml` leaves marked, and to check the run without starting it.

## Customizing


- **Settings:** a recipe's settings can be updated in its `run.toml` (model, harnesses, data, learning rate, LoRA rank, ...); you can overwrite them using `--table.key VALUE` on `shipyard run`.
- **Algorithm:** a recipe defines a `trainer.py` where it subclasses `Trainer` and writes `step` from the primitives `draw`, `rollout` and `update`, giving the sequences their advantages in between; the core holds no algorithm of its own (see [`src/shipyard`](src/shipyard/README.md)).
- **Recipes:** GRPO, DAPO (also with its rollouts overlapping its updates), rejection-sampling fine-tuning, on-policy distillation and GEPA, in [`recipes/`](recipes/README.md).

The trainer updates on exactly what the agent sampled: `harness-proxy` serves the model to the harnesses, and records every call's token ids and logprobs.

## Layout

```
src/shipyard/            the Trainer API, with models/, jobs/ and run/
packages/harness-proxy/  serves the agent to its harnesses, and records every call
recipes/                 the built-in recipes
```

## License

Apache-2.0. See [LICENSE](LICENSE).
