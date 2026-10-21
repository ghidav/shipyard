# Recipes

Each folder is a recipe: a `run.toml` with every setting, and a `trainer.py` whose `Trainer` subclass holds the algorithm. `shipyard init` copies them into `./recipes`.

| Recipe | Algorithm | Overrides | Loss |
| --- | --- | --- | --- |
| [`dapo`](dapo/) | DAPO: an overlong penalty, rewards normalized in the group, every prompt weighing the same, dynamic sampling | `step`, `score` | `ppo`, clipped asymmetrically |
| [`opd`](opd/) | On-policy distillation: each token's advantage is the teacher's logprob minus the policy's | `score` | `importance_sampling` |
| [`gepa`](gepa/) | GEPA: a search over the prompt and skills, with no weight update; needs `shipyard[gepa]` | `step` | none |

Each folder's `README.md` says what its recipe needs. A folder without a `trainer.py` runs the base `Trainer`: GRPO with `importance_sampling`.

## Writing one

Subclass `Trainer` and override `score`, or `step`, from [the primitives](../src/shipyard/README.md). Declare the recipe's own keys as a `Recipe`, and what a resume must restore as a `State`:

```python
class MyTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        coef: float  # no default, so run.toml must set it

    async def score(self, groups: Sequence[Group]) -> None: ...
```

```bash
shipyard run recipes/opd                                  # the folder: its run.toml and trainer.py
shipyard run my.toml recipes/dapo/trainer.py --root runs  # or both, named
shipyard resume runs/opd__a1b2c3d --tag step-100          # a new run from that checkpoint
```
