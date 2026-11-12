# How Prefixwise decides why a request missed the cache

This is the specification of the decision procedure: what Prefixwise compares, the order of its
rules, the evidence each rule needs, and the invariants that must hold. The code is the
implementation of this document (`src/prefixwise/classifier.py`, `details.py`, `collector.py`,
`engines.py`); `tests/test_invariants.py` checks the invariants on generated conversations. If
the two disagree, one of them has a bug.

The procedure is deterministic. It uses no statistics and no learned model: every verdict follows
from the two requests, the provider's usage report and, behind a self-hosted engine, the engine's
own KV-cache events. Where that evidence does not decide, the verdict is `unknown`.

## 1. What is compared

**A request** is the body the client sent, as the provider will render it. **A block** is one tool
definition, the system prompt (each system or developer message), one message, or one tool result.
Cache markers (`cache_control`) are removed before fingerprinting: they say where to cache, not what
the prompt is. Equivalent encodings of one Anthropic prompt (string or text-block content) are
fingerprinted in one form.

**Prefix order** is the order in which the provider renders the prompt, so the order in which a
prefix cache matches it:

| Where | Order |
|---|---|
| Anthropic, OpenAI and OpenAI-compatible APIs | tools, system, conversation |
| A self-hosted engine (vLLM, SGLang) | system, tools, conversation (chat templates render the system prompt first) |

The conversation is the messages and tool results **in the order they were sent**.

**Each request is compared with one earlier request: its previous turn.** That is the latest request
of the same conversation, in the same session, on the same model (a *lane*). A conversation is known
by its opening message, so parallel subagents in one session and an agent's small side calls (titles,
summaries) each have their own previous turn. A request whose opening matches no conversation in the
lane is a new conversation when its system prompt differs beyond dates and ids; otherwise it is
compared with the lane's latest turn. The session is named by the client
(`X-Prefixwise-Session-Id`, a body `session_id`, Codex's session headers, `x-session-id`) or, failing
those, by an end-user id (`metadata.user_id`, OpenAI `user`).

**Each block has three hashes** (SHA-256; HMAC with a per-install key for blocks under about 256
bytes, which a bare hash could give away):

| Hash | Over | Equal means |
|---|---|---|
| raw | the bytes as sent | the provider saw the same thing |
| canonical | the content with keys sorted at every depth, fixed separators, NFC unicode, and JSON Schema set-arrays (`required`, a list `type`) sorted | the same content, possibly serialized differently |
| masked | the canonical form with dates, times, UUIDs, hex ids and numbers replaced by placeholders | the same words: only volatile values differ |

No prompt text is kept. The hashes live in memory for the lane's next turn and are never exported.

**Usage** is the provider's report: `prompt` tokens, `cached` tokens read from the cache, and,
where reported, tokens `written` to it. From these:

- **Reusable prefix** `R = min(prompt_prev, prompt_curr)`. Where the previous turn's cache content is
  known (Anthropic, or any provider that reports writes), `R` is capped at what that turn left in the
  cache: `cached_prev + written_prev`. What was never cached cannot be expected back.
- **Served all**: `cached_curr >= R`, or short of `R` by less than one reporting unit (the read
  slack): reads are reported in whole blocks (OpenAI in 128 tokens, an engine in its block size plus
  16 tokens of generation tail), never more than an eighth of `R`, and exactly on Anthropic.

## 2. The decision procedure

The rules are applied in this order; the first that matches decides.

### 2.1 Before any comparison

| Rule | Verdict | Why it is proof |
|---|---|---|
| No previous turn | `first_write` (or `fork_boundary` for a declared fork) | nothing to reuse yet |
| Another model | `model_switch` | each model has its own cache |
| A new conversation in the lane | `none` if at least half the prompt was read (a shared prefix served), else `first_write` | its own previous turn does not exist |
| The previous request was below the provider's minimum cacheable length, and nothing was read | `below_min_prefix` | nothing was cached to reuse |

### 2.2 The first change

Walk the blocks in prefix order: the tools and system section first (in the order the provider
renders them), then the conversation in the order it was sent. The previous turn's conversation
must be an unchanged prefix of this one's: blocks appended after it are new content, never a change.
**The first block that differs is where the cache stopped matching, and which hash differs names
the cause:**

| First difference | Cause | Waste? | Why it is proof |
|---|---|---|---|
| tools: raw differs, the multiset of canonical tool hashes is equal | `tool_reserialize` | yes | the same tools reached the provider in another order or other bytes |
| tools: the canonical multiset differs | `tool_catalog_change` | no | a tool was added, removed or edited |
| system: raw differs, masked equal | `volatile_system` | yes | every changed character was inside a date, time, id or number |
| system: masked differs | `system_edit` | depends | the wording changed |
| a message | `history_edit` (`compaction_rewrite` with a compaction event) | yes (no) | an earlier message was rewritten, not appended to |
| a tool result | `tool_result_mutation` (`compaction_rewrite` with a compaction event) | yes (no) | a tool's output was re-rendered or removed after it was sent |

One exception: when the previous request was a single message and this one replaces it, it is a
new conversation on the same tools and system prompt (an agent's side call), not an edit; it is
judged by what the cache read, and the edit is kept as a secondary cause.

A change is then weighed against what the cache actually did:

| Condition | Verdict |
|---|---|
| This request is below the provider's minimum and nothing was read | `below_min_prefix`: no version of it could have been read; the change cost nothing |
| The gap since the previous turn exceeds the cache TTL and nothing was read | `ttl_expired`: the entry was gone; the change cost nothing more |
| The cache served all of `R` | `none`: the change came after the reusable prefix or never reached the rendered prompt |
| Otherwise | the change's cause |

In the first three the change is kept as a secondary cause, and the provider's own reason (if any)
is not scored against the verdict, since it answers another question.

### 2.3 The prompt did not change

| Rule (in order) | Verdict |
|---|---|
| The cache served all of `R` | `none`, except on Anthropic when the cache served everything it held and a stable part of at least the minimum cacheable size was sent uncached after the last breakpoint, turn after turn: `breakpoint_misplaced` |
| A prompt-affecting parameter changed (thinking, tool choice, reasoning, output format, chat-template arguments, LoRA adapter, multimodal processing options, RAG documents) | `param_rendered` |
| Below the provider's minimum cacheable length | `below_min_prefix` |
| Anthropic with no `cache_control` anywhere, or a breakpoint after a volatile block | `breakpoint_misplaced` |
| No usage at all | `unknown` |
| The server reports no cached-token count (Ollama, most llama.cpp builds), and no engine evidence | `unknown` |
| At least half of `R` read | `none` |
| Anthropic: this is the first turn carrying a breakpoint and nothing was read | `first_write` |
| Engine evidence: the block where the read stopped was removed | `evicted`, confirmed |
| Engine evidence: another cache served this request while the previous one still held the prefix | `routing_miss`, confirmed |
| An OpenAI-compatible API that is not OpenAI, prompt under 1,024 tokens | `unknown` (its minimum is not known) |
| Identical bytes within 60 seconds, nothing read | `retry_replay` |
| Behind one engine, events show the stop block still resident | `unknown`: not an eviction (the token prefix changed, e.g. template rendering) |
| Behind several caches without deciding events | `unknown` |
| Behind one engine | `evicted`, inferred |
| The gap exceeds the TTL | `ttl_expired` |
| Nothing read, gap within the TTL (or within the shortest provider TTL when unknown) | `routing_miss`, inferred |
| Otherwise | `unknown` |

### 2.4 Overrides from facts outside the two requests

Applied after 2.1 to 2.3, from facts the proxy saw in flight:

| Fact | Effect |
|---|---|
| The request was sent before the previous turn of its conversation finished, and read less than half its prompt | `unknown`: it could not have read that turn's write |
| `previous_response_id` names a response this proxy never saw | `unknown` |
| The engine keeps no prefix cache (SGLang `--disable-radix-cache`) | any miss becomes `unknown`, legitimate, no detail |
| Anthropic: the previous request asked for a 1-hour cache, got a 5-minute write, and the gap exceeded 5 minutes | `ttl_expired`, detail `ttl_shortened` |
| Anthropic: prompt unchanged, the nearest breakpoint is beyond the 20-block lookback from the previous write | `breakpoint_misplaced`, detail `lookback_exceeded` |

Every other **detail** (for example `thinking_stripped`, `sent_before_cache_written`,
`volatile_prefix_at_start`, `output_rerendered`, `host_restored`, `session_header_dropped`) explains a
cause without changing it, and is claimed only from facts in hand. The full list is in the guide.

## 3. Engine evidence

Behind vLLM or SGLang with KV events on, Prefixwise subscribes to the engine's event stream (blocks
stored and removed, with hashes, the parent block, the memory tier and, for blocks the request stored,
its session id) and keeps, per session, the chain of blocks the previous request stored. When the
response arrives, the join locates the block where the cache read stopped
(`cached_tokens / block size`):

| Join's judgement | Becomes |
|---|---|
| The read was a hit (at least half this prompt): blocks the read covered had been removed from the GPU, the offload tier's own stores have been seen, no other session stored them meanwhile, one cache, a clean stream | a `Restored` verdict: the hit gets the detail `host_restored` |
| A miss, and the stop block was removed | evidence for `evicted`, confirmed |
| A miss, and the stop block is still resident | evidence for "not an eviction" |
| Several caches: another cache served this request while the first still held the stop block | evidence for `routing_miss`, confirmed |
| Anything else (a concurrent request in the session, an overflowed index, a cut stream, an unplaced or ambiguous chain) | nothing: the classifier keeps its inference |

Only a `Restored` verdict can become `host_restored`. The join and the classifier use different hit
thresholds (half of this prompt, half of `R`), so a turn the classifier calls a hit may be one the join
judged a miss; its removed blocks were never restored and must not be labelled so.

## 4. Evidence tiers

Every verdict is one of:

- **confirmed**: named from direct evidence: the hashes of both requests, the provider's own usage,
  or the engine's events (`evicted`, `routing_miss` from events).
- **inferred**: the evidence fits one cause best and the evidence text says so (`evicted` behind one
  engine without events, `routing_miss` on a hosted API).
- **unknown**: the evidence does not decide. Prefixwise never guesses.

## 5. What a miss costs

- **Avoidable tokens** (waste only): `max(0, R - cached_curr)`. Appended content is never counted, a
  partial hit counts only what was not read, and on Anthropic nothing beyond what the previous turn
  left in the cache. Legitimate misses and undecided ones count zero.
- **Avoidable dollars**: those tokens at the difference between the rate actually paid for them and
  the cache-read rate, from a dated price table.
- **The bill** of every priced request is split into cache reads, new input, cache writes and output.
- The prediction is checked: when a fix is turned on, the saving realised is measured against the
  saving predicted (97.5% on vLLM and 99.9% on Claude for `canonical_tools`; 106.8% for
  `auto_breakpoints`; experiments/e4).

## 6. Invariants

These must hold for every input; `tests/test_invariants.py` checks them on generated conversations
(Anthropic and OpenAI request shapes, hosted and engine prefix orders, real tool schemas and tool
results):

1. **One change is named exactly.** A single change of each kind (tools reordered, keys reordered, a
   schema's `required` reordered, a tool added, removed or edited, a timestamp in the system prompt,
   a reworded system prompt, an edited earlier message, an edited tool result) produces its cause and
   its legitimacy when nothing was read, and when half was read.
2. **A full read is never a miss.** When the cache served all of `R`, the verdict is `none` (or the
   Anthropic early-breakpoint case), and any change is kept as a secondary cause.
3. **Appending is never a change.** A turn that only appends never yields a change cause, primary or
   secondary.
4. **The earliest change wins.** With two changes, the cause is the one in the block earliest in
   prefix order.
5. **Where and why agree.** The ledger's "first changed" block lies in the section the cause names.
6. **Avoidable is bounded.** `0 <= avoidable <= R - cached`, and avoidable is positive exactly when
   the verdict is waste and less than `R` was read.
7. **Engine evidence decides only an unchanged prompt.** An eviction or a routing miss from events
   never replaces a cause named by a changed block.
8. **Model switch and first turn win.** They are decided before any comparison.
9. **Deterministic.** The same two requests and usage give the same verdict.
10. **Served-all is monotonic.** If a read counts as full, every larger read does.
11. **The hashes ignore exactly what they should.** Key order at any depth leaves canonical and masked
    equal; any content change changes canonical and raw; volatile values leave masked equal; a word
    change does not.

Elsewhere in the suite: only a `Restored` verdict becomes `host_restored`
(`tests/test_host_restored.py`); Prefixwise forwards the fields routers and gateways route by
(`tests/test_routing_headers.py`, `tests/test_openrouter_session.py`).

## 7. Known limits

- **Per block, not per token.** A change at the end of a long system prompt is reported for the
  whole block; the avoidable count comes from usage, so it is still exact.
- **Thresholds.** "At least half of `R` read" separates a hit from a miss where the evidence is only
  usage; the read slack absorbs block rounding. A read between the two is a partial hit.
- **Inferred causes can be wrong.** `routing_miss` on a hosted API and `evicted` behind one engine
  without events are the best fit, marked inferred; events or diagnostics confirm them.
- **Lanes need a conversation id.** Without one, one user's conversations share a session, and the
  first turn of a later conversation with the same system prompt can read as a history edit.
- **Not measured at scale yet.** Accuracy is measured on injected causes (120 of 120 on vLLM, LMCache
  and SGLang), against Anthropic's own reasons (32 of 32) and by realised savings, not yet on large
  volumes of real agent traffic.
