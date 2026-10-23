# Recipes

Each folder is a recipe: a `run.toml` with every setting, and a `trainer.py` whose `Trainer` subclass holds the algorithm. `shipyard init` copies them into `./recipes`.

| Recipe | Algorithm | Models | Loss |
| --- | --- | --- | --- |
| [`grpo`](grpo/) | GRPO: each reward less its group's mean, groups of equal rewards skipped | `agent`, trained | `importance_sampling` |
| [`dapo`](dapo/) | DAPO: an overlong penalty, rewards normalized in the group, every prompt weighing the same, dynamic sampling | `agent`, trained | `ppo`, clipped asymmetrically |
| [`opd`](opd/) | On-policy distillation: each token's advantage is the teacher's logprob minus the agent's | `agent`, trained; `teacher` | `importance_sampling` |
| [`gepa`](gepa/) | GEPA: a search over the prompt and skills, with no weight update; needs `shipyard[gepa]` | `agent`; `reflector` | none |

Each folder's `README.md` says what its recipe needs. A recipe is its `run.toml` and its `trainer.py` together: the core holds no algorithm of its own.

## Writing one

Subclass `Trainer` and write `step` from [the primitives](../src/shipyard/README.md), and `evaluate` if the config has a test set. Read the models the config names in `__init__`, and mark the ones the recipe trains. Declare the recipe's own keys as a `Recipe`, such as the harness it runs and its group size, and what a resume must restore as a `State`:

```python
class MyTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        harness: Harness  # no default, so run.toml must set it
        group_size: int
        coef: float

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]  # [models.agent]
        self.agent.training = True

    async def step(self) -> None:
        groups = await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )
        ...  # give the sequences their advantages
        await self.update(self.agent, groups)
```

A recipe that wants several harnesses or renderers runs several rollouts, at once with `asyncio.gather`, and puts their groups together as it means them.

```bash
shipyard run recipes/opd                                  # the folder: its run.toml and trainer.py
shipyard run my.toml recipes/dapo/trainer.py --root runs  # or both, named
shipyard resume runs/opd__a1b2c3d --tag step-100          # a new run from that checkpoint
```
