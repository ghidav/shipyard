# Datasets

A *task* is one Harbor task directory: an instruction, a container to run it in, and tests
that grade the result. A *dataset* is a directory of tasks under `tasks/` in the working
directory:

```text
tasks/<dataset>/<task>/
```

This is Harbor's hello-world dataset, with its one task:

```text
tasks/hello-world/
└── hello-world/
    ├── environment/
    │   └── Dockerfile
    ├── instruction.md
    ├── solution/
    │   └── solve.sh
    ├── task.toml
    └── tests/
        ├── test.sh
        └── test_state.py
```

A blueprint names a dataset by its directory name:

```toml
[data]
dataset = "hello-world"
batch_size = 1
group_size = 2
```

Run `shipyard` from the directory that holds `tasks/`: `check` and `run` both look for it
there.

## Getting one

`harbor datasets download` fills the directory. Point `-o` at `tasks`, not at the dataset:

```sh
harbor datasets download hello-world -o tasks
```

Harbor writes `<output dir>/<dataset name>/<task name>/`, so `-o tasks` gives exactly the
layout above. For a dataset named `org/name`, the directory is `name`.

`-o tasks/hello-world` nests one level too deep: Harbor adds its own `hello-world/` inside,
and the tasks land at `tasks/hello-world/hello-world/<task>/`. Then no task sits directly
under `tasks/hello-world/`, and `check` blocks:

```text
blocked  no dataset at tasks/hello-world: no valid Harbor task directory under it; `harbor datasets download` fills it
```

Flatten it by moving the inner directory up one level:

```sh
mv tasks/hello-world tasks/hello-world.nested
mv tasks/hello-world.nested/hello-world tasks/hello-world
rmdir tasks/hello-world.nested
```

## What counts as a task

A directory directly under `tasks/<dataset>/` is a task when:

- its name does not start with a dot;
- Harbor accepts it: a `task.toml` that parses, an `environment/` directory, and the tests
  its config asks for.

Anything else there is skipped without a word: files, hidden directories, a directory Harbor
would refuse. Tasks are taken in name order. A dataset with no task at all blocks `check`,
since a run over zero batches would finish and look like a success. With `--verbose`, `check`
counts the tasks it found:

```text
ok  dataset hello-world: 1 task under tasks/hello-world
```

## Batches and epochs

A *batch* is `batch_size` tasks. Each batch is one Harbor job, and each task in it runs
`group_size` times, so a job holds `batch_size × group_size` trials.

The order is made once per epoch:

1. the tasks of every named dataset, in the order the datasets are listed;
2. shuffled with the seed `seed + epoch`, the epoch counted from 0;
3. cut into batches of `batch_size`, keeping a short last batch.

`epochs` repeats this. The same tasks and the same `seed` give the same batches in any
process; another `seed` gives another order. A blueprint with one task, `batch_size = 1`,
`group_size = 4` and `epochs = 2` makes two batches, each a job of four trials:

```toml
[data]
dataset = "hello-world"
batch_size = 1
group_size = 4
epochs = 2
```

`evaluate` and the gradient recipes go through the batches in order. `gepa` does not use
them: it measures each candidate on a minibatch of the tasks, or on all of them
([gepa](gepa.md)). Of `[data]`, it reads only `dataset` and `group_size`.

## A list of datasets

`dataset` takes a list. The tasks are pooled before the shuffle, so a batch can mix them:

```toml
[data]
dataset = ["hello-world", "terminal-bench"]
batch_size = 4
```

`check --verbose` looks for each one, and a missing one blocks the run:

```text
ok  datasets hello-world, terminal-bench
ok  dataset hello-world: 1 task under tasks/hello-world
blocked  no dataset at tasks/terminal-bench: no valid Harbor task directory under it; `harbor datasets download` fills it
```

## Which tasks a run saw

A run uses what is under `tasks/` when it starts. To know later which tasks a run saw, keep
`tasks/` under your own version control, or read the trial names in its job directories.
