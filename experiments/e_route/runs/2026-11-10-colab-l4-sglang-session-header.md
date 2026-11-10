# SGLang: does a session id reaching the worker change turn 2's cache hit? (2026-11-10)

Question, from sgl-project/sglang#47763 item 1: sgl-router reads a session header but does not pass it
to the worker, so the worker never registers the session. Does that cost cache hits?

Setup: Colab L4, SGLang 0.5.23, sglang-router 0.3.2, `Qwen/Qwen2.5-1.5B-Instruct`, worker started with
`--enable-cache-report --mem-fraction-static 0.35 --max-total-tokens 16384`. Today's worker reads the
session only from the body (`session_id`), so "header reaches the worker" is the same as "body
`session_id` set". One conversation per cell, a fresh worker per cell: turn 1 (a system prompt of about
4,000 tokens, user "hi"), then 10 unrelated requests of about 3,000 tokens each (about twice the pool)
with no `session_id`, then turn 2 (turn 1 + the reply + "and then?"). `temperature` 0. Run by hand from a
runbook (no script in this repo); numbers as reported, per request `usage.prompt_tokens_details.cached_tokens`.

| Worker | body `session_id` | turn 2 cached / prompt tokens |
|---|---|---|
| control, default, no flood | none | 3,999 / 4,011 (hit) |
| default | `s-da` | 3 / 4,012 (miss) |
| default | none | 3 / 4,012 (miss) |
| `--enable-session-radix-cache` | `s-sa` | 4,000 / 4,012 (hit) |
| `--enable-session-radix-cache` | none | 3 / 4,012 (miss) |

Router: `python -m sglang_router.launch_router --help` (0.3.2) shows no policy that routes by a session
header. With `--policy cache_aware`, each conversation's two turns stayed on one worker (it matches the
prompt); with `--policy round_robin` they were split across the two workers.

What it means:
- In the default configuration a session id makes no difference: both cells miss after the flood, by the
  same amount. The session id is attribution only there, as SGLang's source says.
- With `--enable-session-radix-cache`, the session id decides what survives memory pressure: KV a session
  holds is evicted after KV no session holds. A session that reaches the worker only as a header is not
  held, so its prefix is evicted first. The miss is an eviction; the dropped header is why it was this
  conversation's prefix that went.

What Prefixwise does with it: `session_header_dropped`, a detail under `evicted` (src/prefixwise/details.py),
claimed only when SGLang's `/server_info` shows `--enable-session-radix-cache`, the client sent
`x-session-id`, and no body `session_id` reached the worker. Before this, Prefixwise itself removed a
client's body `session_id` on its way to SGLang unless its own KV-event join was on (never the case with
this flag), which is the "none" row above; with the flag on it now passes it through.
