# `--optimize` on the Anthropic API, before and after: 2026-11-10

Run by Muse on prefixwise 0.11.0 (from PyPI), Claude Sonnet 4.6, in Colab notebook cells. Total spend
$0.0966 across a dry run and the two runs below. Transcribed by Claude from Muse's report; the four
outputs are as printed (the two tables' column spacing was restored from the pasted text; every value is
Muse's). This is a demonstration on one small scripted agent, not an experiment with a pass rule: the
measured proof of the fix is [E4](../e4/runs/2026-11-06-live.md).

## The agent

Eight turns. Its tool list comes back in another order on turns 4 and 5, the way a plugin reload does,
and returns to the first order on turn 6. It ignores the model's replies and appends a fixed line each
turn, so both runs send the same prompts; each run tags its own system prompt so one run cannot read the
other's cache.

```python
"""A small scripted agent: 8 turns against whatever ANTHROPIC_BASE_URL points at. Its tool list
comes back in another order on turns 4 and 5, the way a plugin reload does. The replies are not
used: each turn appends a fixed assistant line, so both runs send the same prompts."""
import json, os, time, urllib.request

BASE = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
KEY = os.environ["ANTHROPIC_API_KEY"]
CC = {"cache_control": {"type": "ephemeral"}}
# the run's own tag, so a second run within five minutes cannot read the first run's cache
SYSTEM = "You are a coding agent. " + "Rules and repo map. " * 500 + f" Run tag {int(time.time())}."
tool = lambda n: {"name": n, "description": n + " tool",
                  "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
TOOLS = [tool(n) for n in ("bash", "edit", "glob", "grep", "read", "write")]

msgs = [{"role": "user", "content": "In one short sentence, how would you add a --json flag?"}]
for i in range(8):
    tools = TOOLS[3:] + TOOLS[:3] if i in (3, 4) else TOOLS
    last = {**msgs[-1], "content": [{"type": "text", "text": msgs[-1]["content"], **CC}]}
    body = {"model": "claude-sonnet-4-6", "max_tokens": 64, "tools": tools, "tool_choice": {"type": "none"},
            "system": [{"type": "text", "text": SYSTEM, **CC}], "messages": msgs[:-1] + [last]}
    req = urllib.request.Request(BASE + "/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "x-api-key": KEY,
                                  "anthropic-version": "2023-06-01"})
    urllib.request.urlopen(req).read()
    msgs += [{"role": "assistant", "content": f"Step {i} is done."},
             {"role": "user", "content": f"And step {i + 1}, in one short sentence?"}]
```

## Before: `prefixwise run --ledger before.db -- python agent.py`

```
prefixwise: proxy on http://127.0.0.1:37971 (anthropic -> https://api.anthropic.com, openai -> https://api.openai.com); ledger before.db

-- prefixwise: this run --
8 requests, 26,932 prompt tokens, 75% served from cache. Cost $0.0356.
Avoidable: 3,375 tokens ($0.0116), 12.5% of prompt tokens. Fix these, biggest first:
  1. The same tools were sent in a different order or format (2 requests, 3,375 tokens, $0.0116).
     Fix: Send tools in a fixed order with deterministic serialization (or let the proxy do it: prefixwise serve --fix canonical_tools).
     Prefixwise can do it for you: prefixwise run --fix canonical_tools (proven)
Expected misses (not waste): 1 x first request of a session: nothing to reuse yet.
Cache writes never read: 46 tokens were written to the cache, and the next turn of the same conversation replaced the prefix before reading them (2 requests); the write surcharge on them cost $0.0000 above sending them uncached. Fixing the misses above saves this too; it is not counted twice.
All runs: prefixwise report --ledger before.db
```

`prefixwise report --ledger before.db --session latest`:

```
Session run-20261110-010636: 8 requests, 26,932 prompt tokens, cache hit rate 74.6%, 3,375 avoidable, cost $0.0356   (claude-sonnet-4-6)

turn  prompt  cached    hit             cause  first changed     cost  avoidable  avoidable $
----  ------  ------  -----  ----------------  -------------  -------  ---------  -----------
1      3,286       0   0.0%       first_write              -  $0.0125          0      $0.0000
2      3,309   3,283  99.2%              none    messages[8]  $0.0017          0      $0.0000
3      3,332   3,306  99.2%              none   messages[10]  $0.0016          0      $0.0000
4      3,355       0   0.0%  tool_reserialize       tools[0]  $0.0132      3,329      $0.0115
5      3,378   3,352  99.2%              none   messages[14]  $0.0019          0      $0.0000
6      3,401   3,329  97.9%  tool_reserialize       tools[0]  $0.0016         46      $0.0002
7      3,424   3,398  99.2%              none   messages[18]  $0.0020          0      $0.0000
8      3,447   3,421  99.2%              none   messages[20]  $0.0013          0      $0.0000
```

## After: `prefixwise run --optimize --ledger after.db -- python agent.py`

```
prefixwise: proxy on http://127.0.0.1:45301 (anthropic -> https://api.anthropic.com, openai -> https://api.openai.com); ledger after.db

-- prefixwise: this run --
8 requests, 26,932 prompt tokens, 87% served from cache. Cost $0.0250.
No avoidable waste found: every miss had a legitimate cause.
Expected misses (not waste): 1 x first request of a session: nothing to reuse yet.
All runs: prefixwise report --ledger after.db
```

`prefixwise report --ledger after.db --session latest`:

```
Session run-20261110-010817: 8 requests, 26,932 prompt tokens, cache hit rate 87.1%, 0 avoidable, cost $0.0250   (claude-sonnet-4-6)

turn  prompt  cached    hit        cause  first changed     cost  avoidable  avoidable $
----  ------  ------  -----  -----------  -------------  -------  ---------  -----------
1      3,286       0   0.0%  first_write              -  $0.0125          0      $0.0000
2      3,309   3,283  99.2%         none    messages[8]  $0.0016          0      $0.0000
3      3,332   3,306  99.2%         none   messages[10]  $0.0017          0      $0.0000
4      3,355   3,329  99.2%         none   messages[12]  $0.0016          0      $0.0000
5      3,378   3,352  99.2%         none   messages[14]  $0.0015          0      $0.0000
6      3,401   3,375  99.2%         none   messages[16]  $0.0020          0      $0.0000
7      3,424   3,398  99.2%         none   messages[18]  $0.0021          0      $0.0000
8      3,447   3,421  99.2%         none   messages[20]  $0.0021          0      $0.0000
```

## Reading it

| | before | with `--optimize` |
|---|---:|---:|
| Served from cache | 74.6% | 87.1% |
| Cost of the run | $0.0356 | $0.0250 |
| Avoidable tokens | 3,375 | 0 |

- **Turn 4 is nearly all of the waste** (3,329 of the 3,375 avoidable tokens). The reordered tool list read nothing and wrote the prompt again: $0.0132
  where a hit costs about $0.0017.
- **Turn 6 was not a second full miss, and the mock had predicted one.** The list went back to its first
  order and read 3,329 of 3,401 tokens: exactly what turn 4 reads in the run with the fix, the prefix
  turn 3 had written under that order. So the older entry was still cached and within reach. Prefixwise still names the cause (the prompt changed at
  `tools[0]` against turn 5) and counts 46 tokens as avoidable, not 3,375. In this run, flipping back to an order whose entry
  was still cached cost 46 tokens, not a second full miss. That is one observation, not a rule: it
  depends on the older entry still being alive and within reach of the breakpoint.
- **With `--optimize` every turn after the first is a hit.** The fix at work is `canonical_tools`; the
  agent sets its own `cache_control`, so `auto_breakpoints` placed nothing.
- No screen recording was possible from the runtime, so the output is text.
