"""The proxy: the model server a harness talks to, backed by a Tinker sampling client,
recording every call as token ids. `serve` is the process (`python -m shipyard.proxy.serve`),
`server` and the modules it imports are the server, `client` and `tunnel` are the run's side,
and `wire` is what both ends share. Submodules are imported by name; `cookbook` is the only one
that imports `tinker_cookbook`."""
