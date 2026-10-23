---
name: custom-recipes
description: Shipyard recipes: the run.toml config and the Trainer subclass whose step a run takes, written from draw, rollout and update and the recipe's own scoring. Use when post-training a model with shipyard, or writing or changing a recipe.
---

# Custom recipes

A **recipe** is a folder of two files: `run.toml`, the **config** holding every setting of a run, and `trainer.py`, defining one `Trainer` subclass that holds the algorithm. `shipyard run <folder>` checks the config against the trainer before anything opens, runs it, and writes the run to `runs/<folder>__<id>/`.

Start from the shipped recipe closest to your algorithm; `shipyard init` copies them into `./recipes`, each a worked example:

| Recipe | Algorithm | Shows |
| --- | --- | --- |
| `grpo` | RL on the tests' rewards: each reward less its group's mean | the plain `step` and `score` |
| `dapo` | RL with rewards normalized per group, every prompt weighing the same | `substeps`, a clipped loss, scaling advantages per group |
| `opd` | on-policy distillation: each token's advantage is the teacher's logprob less the agent's | a second model, scored with `logprobs` |
| `gepa` | a search over the prompt and skills, no weight update | modules, a model asked for text through a Harbor task (`reflect.py`) |

## run.toml

| Table | Holds |
| --- | --- |
| `modules` (a top-level key) | a folder of `PROMPT.md` and `skills/<name>/SKILL.md`, relative to the config, that every trial carries |
| `[models.<role>]` | one table per model the recipe reads, by **role** (`agent`, `teacher`, ...): `model`; `provider`, `tinker` (default) or `together`, which a run can train, or an API such as `anthropic`, which it only calls; `from_checkpoint`; `renderers`; for a model the recipe trains, `learning_rate` (required), `warmup`, `lora_rank`; sampling: `temperature`, `max_tokens`, `max_context`, ... |
| `[data]` | `train` and `test`, dataset names: folders of Harbor tasks under `./tasks`; `batch_size`, tasks a step; `seed` |
| `[rollout]` | how trials run: `sandbox` (`docker` on this machine, or a cloud one such as `modal`), `concurrency`, `timeout`, `setup_timeout`, `secrets` |
| `[recipe]` | `steps`, `substeps`, `evaluate_every`, `grade_timeouts`, and the trainer's own keys, such as its `harness` and `group_size` |
| `[checkpoints]` | `every` this many steps; `ttl_hours` a save lives |
| `[logging.wandb]` | `project`, `entity` |

- Every table refuses keys it does not know, and a key that moved says where to. `shipyard run` lists every problem in the config before anything opens, so its message answers what a key may be. Every key, with a comment on each, is in `shipyard/config.py` in the installed package.
- Any key can be set for one run: `shipyard run recipes/mine --data.batch_size 2 --recipe.steps 1`. The run's own `run.toml` keeps the overrides, so a resume does too.
- A harness is `"pi@0.85.1"`, a name pinned at a version, or a table with `name`, `env` and `kwargs`.
- `[rollout.secrets]` leases per-trial credentials: `NAME = "POOL"` gives each trial in flight one value of this machine's variable `POOL` (values separated by commas or whitespace) as `NAME`, in its sandbox and its verifier, so no more trials run at once than `POOL` holds.

## Trainer

```python
from collections.abc import Sequence
from pathlib import Path

from shipyard import Config, Group, Harness, Trainer


class MyTrainer(Trainer):
    class Recipe(Trainer.Recipe):
        harness: Harness  # no default: run.toml must set it, under [recipe]
        group_size: int
        coef: float = 1.0

    class State(Trainer.State):
        solved: dict[str, int] = {}  # each field has a default

    def __init__(self, config: Config, directory: Path) -> None:
        super().__init__(config, directory)
        self.agent = self.models["agent"]  # [models.agent]
        self.agent.training = True  # this run trains it
        self.teacher = self.models["teacher"]  # [models.teacher], only scored with

    async def step(self) -> None:
        groups = await self.rollout(
            self.draw(),
            model=self.agent,
            harness=self.recipe.harness,
            attempts=self.recipe.group_size,
            modules=self.state.modules,
        )
        await self.score(groups)
        await self.update(self.agent, groups)

    async def score(self, groups: Sequence[Group]) -> None: ...
```

- `trainer.py` defines one `Trainer` subclass. Its folder loads as a package, so helper files beside it import relatively: `from .helpers import shaped`.
- `Recipe` declares the recipe's own `[recipe]` keys. A nested table such as `[recipe.reflector]` is a field typed by a pydantic model, as in gepa's `reflect.py`.
- The recipe writes `step`, and `evaluate(self, tasks) -> list[Group]` when the config has `[data] test`; `score` is the recipe's own method, called from its `step`. The run refuses a trainer missing either.
- Read every model the config names in `__init__`, as `self.models[role]`, and set `training = True` there on each one the recipe trains: the run refuses a model the recipe never reads, a model in training with no `learning_rate`, and `training` set once the run has opened the model.
- On the trainer: `self.recipe`, the `[recipe]` table as your `Recipe`; `self.config`, the whole `run.toml`; `self.state`, your `State`; `self.models`; `self.generator.train` and `.test`, every task path.
- The run does the rest. It opens each model and the proxy, calls `step` `[recipe] steps` times, evaluates the test set every `evaluate_every` steps from step 0 and after the last, checkpoints every `[checkpoints] every` steps and at the end, and writes a row per step, and one per evaluation at the step whose weights it measured, to `metrics.jsonl`. An evaluation runs beside the next step's `update`, which serves its new weights once the evaluation has finished, so `update` is where a recipe's step lets it run.

## The primitives

### draw

`self.draw()` returns the next batch: `[data] batch_size` task paths, shuffled once per pass over the tasks with `[data] seed`. It counts batches in `self.state.drawn`, so a resumed run goes on from the batch after.

Override it for a curriculum: return any list of task directories, such as some of `self.generator.train`, and keep what the choice reads (counts, scores) in your `State`. A task's instruction is readable before any rollout, at `task / "instruction.md"`.

### rollout

`await self.rollout(tasks, model=..., harness=..., attempts=..., modules=...)` runs `model` on each task `attempts` times through `harness`, in Harbor sandboxes, as one job, and returns a `Group` per task, in order. Each group holds `attempts` **trials**, each graded by the task's tests.

- `modules=`: `self.state.modules` for the run's own prompt and skills, `None` for none, or any `Modules`.
- `renderer=`: one of the model's `renderers`; a model with several must be given one.
- `purpose=`: the job's label. Only training rollouts, the default outside `evaluate`, count in the step's row.
- A recipe that wants several harnesses or renderers runs several rollouts, at once with `asyncio.gather`, and puts their groups together as it means them.
- Any model the run holds can be rolled out, a provider's model through its harness with the harness's own credentials. To ask a model for text, write a Harbor task (`instruction.md`, `task.toml`, `environment/`, `tests/`) whose tests copy what the model wrote into the trial's `verifier/`, roll it out, and read it back from `trial.dir / "verifier"`: gepa's `reflect.py`.

Roll a batch out in one call: shipyard runs up to `[rollout] concurrency` trials at once, and raises only when no trial of the call reached the model, so a one-task call has few trials, and a broken sandbox or two can end the run.

A trial offers `reward`, `graded`, `sequences`, `sampled_tokens`, `instruction`, `transcript`, `steps` (Harbor's trajectory), `test_output`, `result` (Harbor's `result.json`, every key of the verifier's rewards under `verifier_result.rewards`) and `dir`. Its `transcript` includes the prompt and skills it carried, and `test_output` can print the tests' expected answers: filter both before a model reads them.

### score

Scoring sets the **advantages**: a number per target token of each sequence, which `update` trains on. A trial's **sequences** are the chains of model calls it made, as the token ids the model read and sampled; `seq.targets` are the positions it sampled. A sequence trains only once it has advantages.

- Per trial: `trial.set_advantage(x)` gives every target token of every sequence `x`. It rewrites them all, so call it before adding advantages of your own.
- Per token: `seq.advantages = tensor`, one value per target.
- From a model: `await model.logprobs(seqs)` returns each sequence's logprob at every target, as a tensor, for any model on Tinker or Together the run holds; it reads the agent's token ids, so it shares the agent's tokenizer.
- From a group: `group.graded`, `group.rewards`, `group.reward_variance`, `group.centered_rewards`.
- `self.metrics["name"] = value` adds a number to this step's row.

### update

`await self.update(model, groups, loss=..., loss_config=...)` trains `model` on every sequence with advantages, then serves its new weights to the next rollout. It returns what it trained, or None when nothing was scored.

- `loss`: `importance_sampling` (the default); `ppo` and `cispo`, which take `clip_low_threshold` and `clip_high_threshold`, bounds on the ratio itself such as 0.8 and 1.2; `dro`, which takes `beta`; `cross_entropy`, which weighs each target by its advantage: supervised fine-tuning, with advantage 1 on what to imitate.
- `[recipe] substeps`: optimizer steps per update, each on a share of the groups, whole groups each; the model's `learning_rate` and `warmup`; `ADAM = tinker.AdamParams(...)` as a class attribute.

Update each model in training once a step: each call takes its optimizer steps, serves new weights and advances the warmup. The loss sums over tokens, so a long sequence weighs more than a short one; to weigh every group the same, scale its advantages, as dapo's `step` does. The ratio is taken against the sampler's logprobs, which the proxy recorded as the model sampled.

## Additional information

### Graded and masked trials

A trial is **masked** when its outcome says nothing about the agent: the sandbox, the grader or the model's server failed, or Harbor cut it at its timeout (`[recipe] grade_timeouts = true` grades those instead). A masked trial has no reward and no sequences, and `group.graded`, `group.rewards` and the row's `graded` leave it out. A group can hold fewer graded trials than its attempts, or none, so compute group statistics, and counts per task, over its graded trials. The row's `trials`, `graded` and `masked` show how many each step lost.

### Recipe state and resuming

What a recipe learns across steps (a bank, a curriculum's counts, a search's best) lives in its `State` fields. Every checkpoint saves `self.state`, and `shipyard resume runs/<run>` starts a new run from the last checkpoint, `--tag step-<n>` from another. A field needs a default and a JSON value. A plain attribute starts empty again on a resume.

### Models

Every model comes from the config by role, as `self.models[role]`, and does nothing until the run starts and binds it; using one before raises "not connected yet". Its role names its lane in the proxy and its rows in `usage.jsonl`.

| Model | Runs on | `logprobs` | Trains |
| --- | --- | --- | --- |
| on Tinker or Together | a session on its training service, served through its lane of the run's proxy | yes | while `training` |
| behind a provider's API, such as `provider = "anthropic"` | the provider's API, with its harness's own credentials | no | no |

- `trainable` says whether a run could train it: a base model or a training state on Tinker or Together. Weights saved for sampling (`.../sampler_weights/...`) are served and scored as they are.
- `budget` is the most tokens one trial may sample, once the model is served.

### Modules

`modules` names a folder of `PROMPT.md` and `skills/<name>/SKILL.md`, which every trial carries into its harness. `Modules.read(folder)` reads one; `modules.texts` and `modules.with_texts({...})` read and replace their texts, raising `Inadmissible` for text a harness cannot be given; `self.keep(modules, "best")` writes them into the run; setting `self.state.modules` changes what the next rollouts carry.

### Task hooks

A task can carry a `hooks.py` beside its `task.toml`: code the run executes on this machine, outside the sandboxes, at events of each of the task's trials, such as setting up an outside system before the agent starts and tearing it down once the trial is done. Every recipe runs a task's hooks; the config names none.

- A hook is `async def <event>(event: TrialHookEvent)`, named after one of Harbor's trial events: `start`, `environment_start`, `agent_start`, `agent_end`, `verification_start`, `end` or `cancel`. Helpers start with `_`. Any other public function, or a hook that is not `async`, stops the run before anything opens.
- `event.config` is the trial's own Harbor config: its leased secrets are in `event.config.environment.env`, its folder is `Path(event.config.trials_dir) / event.config.trial_name`, and a value set in `event.config.verifier.env` reaches that trial's verifier alone, never its agent.
- Harbor's rules apply. A hook that raises at `start` skips `end`; one that raises at `environment_start` masks its trial, and `end` still runs, so set-up belongs there; one that raises at `end` leaves the grade as it was. A hook has no time limit of its own, so bound what it waits on.
- The run logs each hook it runs and how it ended.

### The step's row

`metrics.jsonl` holds a row per step: `trained`, `jobs`, `groups`, `trials`, `graded`, `masked`, `uniform` (groups with graded trials and nothing to train), `sequences`, `reward_mean`, `reward_spread`, `sampled_mean`, and the update's `train_tokens`, `kl_v1`, `kl_v2`, `entropy`, `learning_rate`, `substeps`, `loss` and `update_seconds`; a step that trained several models prefixes each one's update fields with its role. After an evaluation it holds `eval_` fields, and then `self.metrics`, which starts empty each step; a key that matches one of these is written as `recipe_<key>`.

### Checking a recipe

1. `shipyard run recipes/mine` refuses a wrong config, trainer or model before anything opens, listing every problem. Fix each until the run starts.
2. Run one small step on this machine: `shipyard run recipes/mine --recipe.steps 1 --data.batch_size 2 --recipe.group_size 4 --rollout.sandbox docker`.

The recipe is checked when that run's `metrics.jsonl` holds one row with `graded` above 0 and `trained: true`. A row with `trained: false` and `uniform` equal to `groups` had nothing to learn, every group's rewards being equal: raise `group_size` or pick harder tasks, and run it again.
