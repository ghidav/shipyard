"""OpenAI- and Anthropic-compatible model server for harnesses, backed by Tinker.

The proxy samples from a Tinker sampling client and records every call as token ids.
`serve` is the process entry point (`python -m shipyard.proxy.serve`), `server` and the
modules it imports implement the server, `client` and `tunnel` are what the run uses to
start and reach it, and `wire` holds the definitions both sides share. Submodules are
imported by name; `cookbook` is the only one that imports `tinker_cookbook`."""
