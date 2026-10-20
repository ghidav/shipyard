"""OpenAI- and Anthropic-compatible model server for harnesses, backed by Tinker.

The proxy samples from a Tinker sampling client and records every call as token ids.
`serve` is the process entry point (`python -m harness_proxy.serve`) and `server` the HTTP
server. A call is read by its wire's module in `dialects`, completed by `chat` (rendered through
`rendering` and the `bridge`, sampled through the `recorder`) and written back by its dialect.
`client` and `tunnel` are what the run uses to start and reach the proxy, and `wire` holds the
definitions both sides share. Submodules are imported by name; `cookbook` is the only one that
imports `tinker_cookbook`."""
