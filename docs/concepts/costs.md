# Costs

Every run keeps `costs.json` in its [run directory](runs.md). It holds counts: tokens for
each party, and seconds for each sandbox. For the bill in money, see each backend's own
dashboard.

## The shape

This is `costs.json` for a run whose model was served through the proxy:

```json
{
  "parties": {
    "tinker": {
      "trials": 2,
      "input_tokens": 6216,
      "cache_tokens": 4224,
      "output_tokens": 727
    }
  },
  "sandbox": {
    "docker": {
      "trials": 2,
      "seconds": 131.392899
    }
  }
}
```

And this is `costs.json` for a run whose harness called a provider directly:

```json
{
  "parties": {
    "anthropic": {
      "trials": 1,
      "input_tokens": 30901,
      "cache_tokens": 25837,
      "output_tokens": 116
    }
  },
  "sandbox": {
    "docker": {
      "trials": 1,
      "seconds": 75.140128
    }
  }
}
```

The run rewrites the whole file after every job, and whenever a training step spends tokens. `shipyard
show` prints it under `costs`.

## Parties

A *party* is whoever bills the tokens. A run can have several. In a `gepa` run, a reflector
that calls a provider adds that provider as a party beside the policy's.

| Tokens from | Party |
|---|---|
| a served model | `tinker`, or `tinker@<host>` when `TINKER_BASE_URL` is set |
| a provider model | the provider Harbor recorded for the trials, lower-cased |
| a `gepa` reflection | the provider Harbor recorded, else the part of `reflection_model` before `/` |

The full `TINKER_BASE_URL` is in `process.json` as `tinker_base_url`.

A provider party falls back to `[model] provider` when Harbor recorded none, or when the
trials disagree. A reflection with neither is billed to `unknown`.

## Token counts

`trials` counts the trials that left a result. `output_tokens` counts what the model wrote.
`input_tokens` counts the whole prompt, and `cache_tokens` counts the part of it the backend
had cached. The counts come from different sources:

- **A served model.** The proxy counts from its own records. Above, 6,216 is the sum of
  `prompt_tokens` over the run's four `requests.jsonl` rows, 4,224 of them cached.
- **A provider model.** The counts are what the harness reported to Harbor. Above, 30,901
  input tokens of which 25,837 were cached.

A trial that reported nothing adds no tokens.

## Training counts

A gradient recipe adds up to three counts to the served party, each only once it is spent:

| Key | Counts the tokens sent |
|---|---|
| `train_tokens` | through the training step, `forward_backward` |
| `reference_tokens` | through the trainer's forward pass for the reference logprobs; none with `reference = "sampler"` |
| `anchor_tokens` | to the starting weights, for the KL term; none with `kl_coef = 0` |

## Sandbox seconds

`sandbox` has one entry per sandbox type. `trials` counts the trials that left a result, and
`seconds` sums their container time. A trial's time runs from the first of Harbor's timed
phases starting (environment setup, agent setup, agent run, verifier) to the last one
finishing.

The seconds are summed over trials, and trials run in parallel. The served run above spent
131 sandbox seconds in 90 seconds of wall clock, two trials at a time.

## Per job

Each `jobs.jsonl` row carries the token counts of its own job, with its `party`, `sandbox`
and `sandbox_seconds`. `costs.json` adds them up. A row's `trials` counts every trial the job
planned, where `costs.json` counts those that left a result. To split a run's costs by batch,
or to tell the reflector's tokens from the policy's, read the rows:

```json
{"at": "2026-10-07T20:57:50.871956+00:00", "job": "02-eval-tinker-docker__eTNXY25-0000", "purpose": "rollout", "trials": 2, "batch": 0, "tasks": 1, "graded": 2, "served": "tinker", "bridged": 2, "party": "tinker", "input_tokens": 6216, "cache_tokens": 4224, "output_tokens": 727, "sandbox": "docker", "sandbox_seconds": 131.392899}
```

The job rows carry no training counts. `train_tokens` is on each step's row in `metrics.jsonl`,
and `reference_tokens` and `anchor_tokens` are in `costs.json` only.
