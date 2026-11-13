# Demos

## `hallway.py`: break the cache three ways, then fix it

A four-minute, keypress-driven demo for showing someone on a laptop. One scripted coding agent, nine
turns, through Prefixwise; then the same agent with the fix on.

```
python experiments/demo/hallway.py            # mock provider: no key, no network, no cost
python experiments/demo/hallway.py --live     # the real API: ANTHROPIC_API_KEY (or OPENAI_API_KEY)
python experiments/demo/hallway.py --auto     # no pauses, for a recording
```

| act | what the agent does | what Prefixwise names |
|---|---|---|
| 1 | three clean turns | `first_write`, then hits |
| 2 | a plugin reload sends the same eight tools in another order (turns 4 and 6) | `tool_reserialize`, changed at `tools[0]`, priced |
| 3 | the current time in the system prompt ticks (turns 7 and 8) | `volatile_system`, changed at the system block |
| 4 | an earlier reply is shortened in place (turn 9) | `history_edit`, changed at that message |
| | the report: every miss priced, the fixes ranked, the ceiling if fixed | |
| 5 | the same agent with `--optimize` on the proxy, and the two changes the report asked for in the agent (no time in the prefix; append, don't rewrite) | every turn after the first hits; before and after side by side |

In the live run nothing is simulated: the cached-token counts come from the provider, and the run
costs under a cent on Claude Haiku. The mock provider reports a hit when the tools, system prompt and
earlier messages are unchanged since the session's previous turn, with 2,000-token prompts; a live
provider that keeps older entries can turn the return to the first tool order (turn 6) back into a
hit. Ledgers hold hashes and token counts, never prompt text. `--keep DIR` keeps the two ledgers; the
closing lines print the `prefixwise report --session` command for them.

What this does not show: engine evictions and routing misses confirmed from vLLM or SGLang KV events,
which need a GPU ([E-evict](../e_evict/runs/), [E-route](../e_route/runs/)). The measured proof that
the fix saves what was predicted is [E4](../e4/runs/).

## `2026-11-10-optimize-real-run.md`

`--optimize` on the Anthropic API, before and after, on an eight-turn scripted agent: 74.6% to 87.1%
served from cache, avoidable tokens 3,375 to 0.
