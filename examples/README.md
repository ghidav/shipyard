# Example blueprints

Ready-made blueprints for `thinkingmachines/Inkling-Small`, one per recipe. Copy one into your
workspace's `blueprints/`, fetch its dataset into `tasks/`, then check and run it:

```
harbor datasets download reasoning-gym-easy -o tasks
./shipyard check blueprints/dapo-reasoning-gym
./shipyard run blueprints/dapo-reasoning-gym
```

`harbor datasets download` nests the tasks one level deeper than a run reads them; move them up so
the layout is `tasks/<dataset>/<task>/` (see [Datasets](../docs/concepts/datasets.md)).

| Blueprint | Recipe | Dataset | Sandbox |
|---|---|---|---|
| `smoke` | `evaluate` | `hello-world` | docker |
| `measure-aime` | `evaluate` | `aime` | modal |
| `dapo-reasoning-gym` | `dapo` | `reasoning-gym-easy` | modal |
| `cispo-reasoning-gym` | `cispo` | `reasoning-gym-easy` | modal |
| `gepa-terminal-bench` | `gepa` | `terminal-bench-sample` | modal |
| `fst-reasoning-gym` | `fst` | `reasoning-gym-easy` | modal |

Every blueprint needs `TINKER_API_KEY`; the Modal ones need Modal's credentials and Harbor's
extra (`uv add "harbor[modal]"`); `gepa` and `fst` also need `ANTHROPIC_API_KEY` for the reflector.
