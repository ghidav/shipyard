# Glossary

**Backend**: what a `Model` runs on once the run binds it (`shipyard.models.backends.Backend`): a `Session` on the agent's client, which trains it or serves it as it is; `Sampling`, for any other model on Tinker; or a `Provider`, whose API serves the model. `connect` gives each model the trainer holds its backend, by its provider. _Avoid_: client (the training client is one part of a `Session`), agent (every model has a backend, not only the agent).

**Records**: what the proxy recorded of one trial's model calls, or that it recorded none because a provider's model served the trial (`shipyard.records.Records`, held as `Trial.records`). It is the one place that reads the proxy's record format: each call was *answered*, *turned away* (for the budget or the context), or *failed* at the sampler. _Avoid_: calls (reads as tool or function calls), served (`Record.served` already names the weights that answered a call).
