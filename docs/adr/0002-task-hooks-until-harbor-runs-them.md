# Task hooks live in the task, and shipyard runs them until Harbor does

Some tasks need work outside their sandboxes around each trial: Bauplan's create a lakehouse branch for the trial's user and record the baseline its verifier compares against before the agent starts, and tear the user's branches down once the trial is done. Such a task carries that work as hooks, async functions in a `hooks.py` beside its `task.toml`, each named after one of Harbor's trial events, and shipyard attaches them to every Harbor trial of the task in `_run_trial`. Harbor has the per-trial events but no way for a task to declare hooks, so this is shipyard's stand-in until Harbor supports task hooks natively; then shipyard's loader goes, and plain `harbor run` runs them as well. Decided 22 Oct 2026.

## Considered Options

- **The lease does it.** `[rollout.secrets]` already acquires something for each trial before Harbor starts it and releases it after, and could have prepared the lakehouse and given each trial its own copy of the task. Rejected: the lease is admission, when a trial starts and with which identity, while set-up and teardown belong to the task, not to shipyard.
- **Hooks on the engine, registered by a recipe** (`engine.add_hook`, a pass-through of Harbor's own). Rejected: the recipe would carry each task family's set-up, and a run mixing tasks with different needs would dispatch between them by hand.
- **Hooks declared in `run.toml`.** Rejected: the config names no hooks.
- **Inside the task's sandboxes**: `prepare` as the environment's healthcheck before the agent, the baseline handed over by a collect hook from a file a non-root agent cannot read, and `teardown` at the end of the verifier. Nothing would run on the run's machine, and plain `harbor run` would run it all. Set aside: every task's image and grader would change, and a non-root agent is untested with pi on Modal.
- **A rollout engine of our own.** Owning the trial would make set-up simple, but means rewriting the 20k or so lines of Harbor shipyard uses, and losing grading parity with Harbor's evaluations.

## Consequences

- Hooks run on the run's machine, in its process, with its credentials. A task without a `hooks.py` has none, so downloaded datasets run as before, and no run has to opt in.
- Harbor's rules apply as they are: a hook that raises at `start` skips `end`; one that raises at `environment_start` masks its trial and `end` still runs, so set-up belongs there; one that raises at `end` leaves the grade as it was. Hooks have no time limit of their own.
- A hook fires after Harbor has loaded the trial's task, so it cannot give a trial its own task folder. Per-trial values reach the verifier through `event.config.verifier.env`, which works because Harbor hands a hook the trial's own config rather than a copy; `tests/test_hooks.py` pins that. Bauplan's grader reads its baseline from there first, and from `/tests` otherwise.
- Stand-in trial runners run no hooks: tests reach them through their loading, and through `_run_trial` with Harbor's trial stubbed.
- When Harbor supports task hooks natively, delete `shipyard/jobs/hooks.py` and its calls in `_run_trial` and `Trainer.__init__`, and move each task's hooks to Harbor's form.
