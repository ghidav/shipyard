# Example blueprints

Ready-made blueprints for `thinkingmachines/Inkling-Small`, one per recipe. Copy one into your
workspace's `blueprints/`, fetch its dataset into `tasks/`, then check and run it:

```
harbor datasets download reasoning-gym-easy -o tasks
./shipyard check blueprints/dapo-reasoning-gym
./shipyard run blueprints/dapo-reasoning-gym
```

`harbor datasets download` nests the tasks one level deeper than a run reads them. Move them up so
the layout is `tasks/<dataset>/<task>/` (see [Datasets](../docs/concepts/datasets.md)).

| Blueprint | Recipe | Dataset | Sandbox |
|---|---|---|---|
| `smoke` | `evaluate` | `hello-world` | docker |
| `measure-aime` | `evaluate` | `aime` | modal |
| `dapo-reasoning-gym` | `dapo` | `reasoning-gym-easy` | modal |
| `cispo-reasoning-gym` | `cispo` | `reasoning-gym-easy` | modal |
| `gepa-terminal-bench` | `gepa` | `terminal-bench-sample` | modal |
| `fst-reasoning-gym` | `fst` | `reasoning-gym-easy` | modal |
| `feedback-dapo-reasoning-gym` | `dapo` with Claude's feedback, through its `trainer.py` | `reasoning-gym-easy` | modal |

Every blueprint needs `TINKER_API_KEY`. The Modal ones need Modal's credentials and Harbor's
extra (`uv add "harbor[modal]"`). `gepa` and `fst` also need `ANTHROPIC_API_KEY` for the reflector.

## A blueprint with a trainer file

`feedback-dapo-reasoning-gym` trains with DAPO, and asks Claude for feedback on each failed rollout.
The policy reads that feedback before the task and scores, as its own teacher, the tokens it
sampled without it (see [Privileged text](../docs/concepts/recipes.md#privileged-text)). The
behavior is in the directory's `trainer.py`, which `run` takes after the blueprint:

```
uv add anthropic
./shipyard check blueprints/feedback-dapo-reasoning-gym blueprints/feedback-dapo-reasoning-gym/trainer.py
./shipyard run blueprints/feedback-dapo-reasoning-gym blueprints/feedback-dapo-reasoning-gym/trainer.py
```

It needs `ANTHROPIC_API_KEY` as well. Claude's tokens are counted under `anthropic` in the run's
`costs.json`, and each feedback is kept in its trial's directory as `privileged.txt`.
