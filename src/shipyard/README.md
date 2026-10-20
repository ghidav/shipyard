# shipyard

Recipes are implemented as subclasses of `Trainer`. Each step calls `step`, made of four primitives: `draw`, `rollout`, `score` and `update`. A recipe usually overrides `score`, `step`, or both. The trainer does the rest: it binds the policies, opens the session and the proxy, evaluates, checkpoints, logs and resumes.

```mermaid
flowchart LR
    draw["draw()"] -->|tasks| rollout["rollout(tasks)"]
    rollout -->|groups| score["score(groups)"]
    score -->|advantages| update["update(groups)"]
    update -.->|new weights| rollout
```

| Module | Holds |
| --- | --- |
| `trainer.py` | `Trainer`: the four primitives, the default `step`, the run loop, evaluation and checkpoints |
| `policy.py` | `Policy`, a model the trainer samples or scores, and `INITIAL` |
| `groups.py` | `Group` → `Trial` → `Sequence`: what a rollout returns, and where advantages go |
| `state.py` | `State`: what a checkpoint saves and a resume restores |
| `config.py` | `Blueprint`, `Recipe`, `load`: a `run.toml`, checked against the recipe |
| `cli.py` | `shipyard init`, `run` and `resume` |
| [`models/`](models/README.md) | the policies' backends, the training clients and the trial proxy |
| [`jobs/`](jobs/README.md) | Harbor jobs, harnesses, secrets and verdicts |
| [`run/`](run/README.md) | the run's directory, streams, costs and tasks |