# Admission

Admission decides what each trial counts for. Its answer is the trial's
**verdict**: a reward or a **mask**, exactly one of the two. A **masked** trial
is left out of every mean and every gradient. A trial with a reward counts,
even when the reward is 0.

The verdict is read from Harbor's `result.json` and, for a served model, from
the proxy's records of the trial (see [The proxy](proxy.md)).

## Masked or zero

The rules run in this order, and the first that matches decides. Rules 1 to 7
read the records, so they apply to a served model only; a provider-served
trial starts at rule 8.

| # | the trial | verdict |
|---|---|---|
| 1 | left no record at all | mask `env_error` |
| 2 | took more turns, by its harness's own log, than the proxy recorded (only under `check_turns`) | mask `env_error` |
| 3 | had its first call refused as too long for the context, and no call served | mask `context_overflow` |
| 4 | had a call refused for the token budget | reward 0 |
| 5 | had a call refused as too long while others were served, and the harness then ended with an error that is neither Harbor's clock nor the grader's | reward 0 |
| 6 | ended on a failed call: its last record carries an error other than the two refusals, or its harness's log says the last call failed and the last record carries none | mask `api_error` |
| 7 | sent an image | mask `multimodal` |
| 8 | left no `result.json` | mask `env_error` |
| 9 | was cut by Harbor's clock (`AgentTimeoutError`) | mask `timeout` |
| 10 | ended with the harness giving up on the endpoint (below) | mask `api_error` |
| 11 | has a number at `verifier_result.rewards.reward` | that reward |
| 12 | anything else | mask `grading_error` |

Rule 6 reads the harness's log where its profile knows how. pi's is
`agent/pi.txt`: its last assistant message stopped on `error`. A failed call
with no failed record never reached the proxy, or the proxy refused it before
recording, as with a dropped connection. pi exits cleanly either way, so
without this rule such a trial would be graded as the policy's work.

Rule 10 reads Harbor's `exception_info.exception_type`. The endpoint failures
are `ApiError`, `UnknownApiError`, `ApiInternalServerError`,
`ApiOverloadedError`, `ApiConnectionClosedError`, `ApiResponseStalledError`,
`ApiRateLimitError`, `ApiUsageLimitError`, `ApiProviderResourceNotFoundError`,
and any other name ending in `ApiError`. Harbor's three endings that are the
policy's own (`OutputTokenExceededError`, `ContextWindowExceededError`,
`AgentSafetyRefusalError`) are not: such a trial is graded as usual. Nor are
`NetworkConnectionError`, `AgentAuthenticationError` and `ModelNotFoundError`,
which Harbor raises when the harness's own command fails on the network, a key
or a model name. Such a trial goes on to rule 11.

A mask is for a trial that measured something other than the policy: a
container that never started, a grader that wrote nothing, an endpoint that
failed, a clock that ran out, or images, which training cannot read. A zero is
for an ending the policy caused: it wrote until its token budget was gone, or
filled its context and its harness gave up. Masking those would hide the
failure, and the policy would never learn to stop. A harness that filled its
context and still finished keeps the verifier's reward (rule 11).

The verdict is the same for every recipe. `dapo` and `cispo` then dock a
rollout that sampled near or up to its budget, as DAPO does, so a budget cut
trains as −0.5 there ([The overlong term](recipes.md#the-overlong-term-dapo-and-cispo));
`evaluate` and `gepa` count it as 0.

## Why a masked rollout enters no mean

A zero for a container that never started would read as a policy that
failed. Averaged in, it pulls the mean down for a reason the policy had no
part in; in training, it pushes the policy away from whatever it did.

So a masked rollout enters no mean. `evaluate` averages the graded trials
only, and logs the masked ones as a count beside them. A batch with nothing
graded logs a `mean` of `null`, never 0. A gradient recipe builds each
group's baseline from its graded members only, and a masked rollout carries
no gradient. The live run's metrics row:

```json
{"seq": 1, "batch": 0, "tasks": 1, "rollouts": 2, "graded": 2, "masked": 0, "mean": 1.0}
```

## Completeness by construction

The served rules trust that a trial's records are complete. They are, by how
they are made:

- The proxy records every call it serves, under the trial named in the
  call's address. A call refused for the budget or the context, or one that
  fails at the sampler, leaves a record too, with `error` set.
- The run takes a trial's records as one object, in call order, and drains
  them from the proxy. A fetch lost on the way is asked again for the same
  answer.
- A trial that never reached the proxy has no records, which rule 1 masks.

A call that never reached the proxy, a harness talking to some other
endpoint, leaves nothing to see. The optional per-harness turn count covers
it. With `[rollout] check_turns = true`, the run counts the turns the harness
says it took, from its own log, `agent/<harness>.txt`, and masks the trial
when that count is more than the proxy's records (rule 2). Two profiles carry
a counter: `claude-code` counts distinct request ids, and `opencode` counts
`step-start` lines. A harness without a counter, or a trial without the log,
is not checked. The check is off by default.

## The job row counts

Every rollout job writes one row to `jobs.jsonl`. Its counts show
admission's verdicts:

| key | what it counts | when present |
|---|---|---|
| `trials` | every planned trial | always |
| `graded` | trials with a reward | always |
| `masked` | masked trials, by mask | when any were masked |
| `ended` | Harbor's `exception_type` of every trial that named one, graded or masked | when any did |
| `turned_away` | calls the proxy refused, for the budget or the context | served, when any |
| `cut` | volatile lines the proxy cut from system prompts (`[rollout] cut_volatile`) | served, when any |
| `bridged` | calls whose prompt the bridge built | served, when any |

`trials` is always `graded` plus every count under `masked`. The live served
run's row:

```json
{"job": "02-eval-tinker-docker__eTNXY25-0000", "purpose": "rollout", "trials": 2, "batch": 0, "tasks": 1, "graded": 2, "served": "tinker", "bridged": 2, "party": "tinker", "input_tokens": 6216, "cache_tokens": 4224, "output_tokens": 727, "sandbox": "docker", "sandbox_seconds": 131.392899}
```

Both trials were graded, so the row has no `masked` and no `ended`.
