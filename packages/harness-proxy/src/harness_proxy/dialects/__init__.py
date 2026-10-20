"""The wires a harness speaks, one module each.

A dialect has `PATH`, the route under `/v1`; `read`, which turns a request body into a
`chat.Call`; `answer` and `events`, which write a `chat.Reply` as one JSON body or as SSE chunks;
and `error`, a refusal in the wire's own shape."""
