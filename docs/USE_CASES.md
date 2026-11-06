# Prefixwise use cases

Eight situations where Prefixwise answers a question your provider dashboard cannot. Every command here runs as written; the `examples/` files are in this repository.

Setup for all of them:

```bash
pip install "git+https://github.com/tejachavali/prefixwise"
```

---

## 1. "Our agent's token bill jumped and the cache hit rate fell"

The provider tells you *how many* tokens missed the cache. Prefixwise tells you *why*.

```bash
prefixwise serve --upstream https://api.openai.com      # your agent's base URL -> http://127.0.0.1:8080/v1
# ... let the agent run for a while ...
prefixwise report
```

If `volatile_system` tops the table, something like a timestamp or request id is in your system prompt. Every turn rewrites the start of the prompt, so nothing after it can be reused. Confirm with one request:

```bash
prefixwise request <X-Prefixwise-Request-Id from any response>
```

**Fix:** move the volatile value to the end of the prompt (or into the latest user message), restart the agent, and watch `avoidable` for that cause fall to zero in the next `prefixwise report`.

## 2. Diagnose one miss from your logs, without a proxy

You have two consecutive request bodies from your logs and want to know why the second one did not hit the cache:

```bash
prefixwise explain examples/turn1.json examples/turn2.json
```

```text
examples/turn1.json -> examples/turn2.json: VOLATILE_SYSTEM  [WASTE]
  first difference: block 2 (system): only volatile values changed (dates, times, ids, numbers)
```

It names the first block that changed and how: bytes only (a serialization difference), volatile values only, content, tools reordered, history appended, or history truncated. No API key and no network.

## 3. Gate prompt and harness changes in CI

Cache-breaking regressions come from innocent-looking changes: a `dict` iterated in a new order, a new field in a tool result, a refactor that rebuilds the system prompt per turn. Record one representative session as JSONL. Each line is a request body, or `{"request": body, "usage": {...}, "gap_seconds": n}` if you have usage. Then fail the build when a turn turns wasteful:

```bash
prefixwise explain examples/session.jsonl --fail-on-waste    # exit code 2 on any waste turn
```

```yaml
# .github/workflows/cache-gate.yml
- run: pip install "git+https://github.com/tejachavali/prefixwise"
- run: python scripts/record_golden_session.py > session.jsonl   # your harness, offline or mocked
- run: prefixwise explain session.jsonl --fail-on-waste
```

`examples/session.jsonl` deliberately contains two regressions, a re-rendered test result and a reordered tool list, so you can see what a failure looks like.

## 4. A/B-test prompt or agent versions on tokens per *successful* task

A shorter prompt that saves 30% per call but fails more tasks is not cheaper. Label each version, tag each task, record outcomes:

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1",
                default_headers={"X-Prefixwise-Agent-Id": "prompt-v2",
                                 "X-Prefixwise-Task-Id": task_id,
                                 "X-Prefixwise-Session-Id": task_id})
```

```bash
prefixwise task "$TASK_ID" --success     # or --failure, from your grader / test run
prefixwise report --by agent_id
```

The `tokens/success` column divides *all* tokens a version spent, failures included, by the tasks it completed. That is the number to compare.

See `examples/openai_agent.py` for a complete runnable loop (`prefixwise serve --mock`, then `python examples/openai_agent.py`).

## 5. Anthropic: is `cache_control` set, and is the 5-minute TTL right?

Two Anthropic-specific failure modes show up directly:

- `breakpoint_misplaced`: requests without any `cache_control` breakpoint. Nothing is cached. Add one after your stable prefix (tools and system).
- `ttl_expired`: the gap between turns exceeded the cache lifetime: 5 minutes by default, 1 hour with `"cache_control": {"type": "ephemeral", "ttl": "1h"}`.

```bash
prefixwise serve --upstream https://api.anthropic.com
prefixwise report            # look at ttl_expired vs none
```

If many turns land in `ttl_expired` (humans in the loop, slow tools, long CI steps), the 1-hour TTL is usually worth it. 1-hour cache writes cost more than 5-minute ones, so check current pricing against how often you would otherwise re-write the prefix. The proxy chains Anthropic's own cache diagnostics too, and stores the provider's reason beside Prefixwise's (`prefixwise request <id>` shows `native_reason`).

## 6. Coding agents: tool output and compaction

Coding agents resend tool output (test runs, file reads, shell output) every turn. Two patterns are common:

- `tool_result_mutation`: the harness re-renders an earlier tool result, with a fresh timing line, a truncation that moved, or ANSI codes stripped differently. Everything after it misses. **Fix:** store the tool output once, exactly as first sent.
- Compaction: when your harness summarises or clears old context, send `X-Prefixwise-Compaction-Event: 1` on the next request. It is then counted as `compaction_rewrite`, which is legitimate, and not as `history_edit` waste. If `compaction_rewrite` is frequent, compact at task boundaries rather than every few turns.

## 7. Self-hosted vLLM: is prefix caching actually working?

```bash
vllm serve Qwen/Qwen3-8B --enable-prefix-caching --enable-prompt-tokens-details
prefixwise serve --upstream http://127.0.0.1:8000
```

The proxy forwards `X-Request-Id` and `X-Session-ID` (and `session_id` in the body for vLLM), so requests can be joined with vLLM's KV-cache events. In the report:

- Mostly `none` means prefix caching works.
- `routing_miss` with identical prompts across several replicas means your router has no session affinity: the same session lands on a replica that has never seen its prefix.
- `--enable-prompt-tokens-details` is required. Without it vLLM reports no cached tokens, and every turn looks like a miss.

The repository's E1-open experiment (`experiments/e1_open/`) runs 120 controlled turns against vLLM to validate the classifier itself.

## 8. Measure an existing tool or framework by changing one URL

Anything that lets you set a base URL can be measured without code changes:

```bash
prefixwise serve --upstream https://api.anthropic.com --port 8080
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 your-anthropic-based-tool
```

```python
# LangChain / LangGraph
ChatOpenAI(base_url="http://127.0.0.1:8080/v1", default_headers={"X-Prefixwise-Session-Id": run_id})
# Anthropic SDK
Anthropic(base_url="http://127.0.0.1:8080", default_headers={"X-Prefixwise-Session-Id": run_id})
```

Without a session header, Anthropic traffic is grouped by `metadata.user_id` if the client sends one. Otherwise everything is one session, and parallel conversations will look like edits of each other. Set `X-Prefixwise-Session-Id` whenever you can.

---

### What Prefixwise stores

Hashes of each prompt block (raw, canonical, and with volatile values masked), token counts, the cause, and your session, task and agent labels. Never prompt text. Set `PREFIXWISE_HMAC_KEY` so hashes of short blocks are keyed and cannot be reversed by guessing common strings.
