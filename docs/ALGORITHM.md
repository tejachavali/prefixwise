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
definition, the system prompt (`system`, or each system or developer message that opens an OpenAI
chat conversation; the Responses API's `instructions`, and each system or developer item that opens
its input when the turn does not continue a stored response), one message, or one tool result. A system or
developer message sent later in a conversation (an agent's reminder) is a message where it was sent.
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
summaries) each have their own previous turn (the last 8 conversations of a lane are remembered). A
request whose opening matches no remembered conversation is a new conversation when its system prompt
is not empty and differs beyond dates and ids, and it carries no compaction event; otherwise it is
compared with the lane's latest turn, which may be another conversation's: 2.2 says what is then
still claimed (a change in the tools or the system prompt, which the lane shares) and what is not.
A model with no lane yet in the session is compared with the session's last turn on any model, which
makes it a `model_switch`. The session is named by the client (`X-Prefixwise-Session-Id`, a body
`session_id`, Codex's session headers, `x-session-id`) or, failing those, by an end-user id
(`metadata.user_id`, OpenAI `user`).

**A session that names no conversation** is one whose id did not come from the client as a
conversation id: an edge's default session (`prefixwise serve`, the middleware, an SDK recorder with
no session given), or an end-user id. It holds every conversation that named none. The default
session of `prefixwise run` (one per run) and of `prefixwise check` (one per scenario) is not such a
session: it names that run or scenario.

**Judged by its own read.** Where a request has no turn of its own to be compared with (2.2, 2.4), no
change is claimed: the verdict is `none` if at least half of its own prompt came from the cache, else
`unknown`. Such a row carries no detail, no "first changed" block and no evidence of how long the
cache lasts, and the provider's own reason is not scored against it.

**Each block has three hashes** (SHA-256 over the block as parsed and re-serialized, so whitespace
and escaping in the original bytes are not seen). With a per-install key set (`PREFIXWISE_HMAC_KEY`),
blocks of 256 bytes or less, which a bare hash could give away, are keyed hashes (HMAC) instead;
without one they are plain SHA-256 like the rest:

| Hash | Over | Equal means |
|---|---|---|
| raw | the content as sent, serialized | the provider saw the same content in the same form |
| canonical | the content with keys sorted at every depth, fixed separators, NFC unicode, and JSON Schema set-arrays (`required`, a list `type`) sorted | the same content, possibly serialized differently |
| masked | the canonical form with dates, times, UUIDs, hex ids, numbers and any token containing a digit (`v2`, `gpt4`) replaced by placeholders | the same words: only volatile values differ |

No prompt text is kept. The hashes live in memory for the lane's next turn; when a remote collector
(`--collector`) or a record log (Redis) is configured, they travel with the turn's record to it,
so set `PREFIXWISE_HMAC_KEY` there too if short blocks must not be guessable.

**Usage** is the provider's report: `prompt` tokens, `cached` tokens read from the cache, and,
where reported, tokens `written` to it. From these:

- **Reusable prefix** `R = min(prompt_prev, prompt_curr)` (this prompt when the previous turn has no
  usage). Where the previous turn's cache content is known (Anthropic, or any provider that reports
  writes) and is not zero, `R` is capped at what that turn left in the cache: `cached_prev +
  written_prev`. What was never cached cannot be expected back.
- **Served all**: `cached_curr >= R`, or short of `R` by less than one reporting unit (the read
  slack): reads are reported in whole blocks (an engine whose events showed its block size: that size
  plus 16 tokens of generation tail; any other provider but Anthropic: 128 tokens), never more than an
  eighth of `R`; Anthropic's counts are exact.
- **Half read**: where only usage decides between a hit and a miss (2.3), the test is
  `cached_curr >= min(prompt_prev, prompt_curr) / 2` (half of this prompt when the previous turn has
  no usage), without the cap on `R`. The engine join's own hit test uses half of this prompt
  (section 3).

## 2. The decision procedure

The rules are applied in this order; the first that matches decides.

### 2.1 Before any comparison

| Rule | Verdict | Why it is proof |
|---|---|---|
| No previous turn | `first_write` (or `fork_boundary` for a declared fork) | nothing to reuse yet |
| Another model | `model_switch` | each model has its own cache |
| A new conversation in the lane | `none` if at least half its prompt was read (a shared prefix served), else `first_write` | its own previous turn does not exist |
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
| system: masked differs | `system_edit` | not counted (a deliberate edit may or may not be worth its cost) | the wording changed |
| a message | `history_edit` (`compaction_rewrite` with a compaction event) | yes (no) | an earlier message was rewritten, not appended to |
| a tool result | `tool_result_mutation` (`compaction_rewrite` with a compaction event) | yes (no) | a tool's output was re-rendered or removed after it was sent |

One exception: **another conversation is not an edit of this one.** It is decided only when the walk
reaches the conversation, so a change in the tools or the system prompt is named first, against
whichever turn of the lane the request was compared with. When the opening message differs and no
compaction event came with the request (with one, it is a `compaction_rewrite`, always):

| The request | Is |
|---|---|
| The previous request was a single message (an agent's side call, a one-shot classification) | a **new conversation** on the same tools and system prompt. The edit is kept as a secondary cause and the walk goes on to the tool results (a changed tool result is still named) |
| This request is a single message, in a session that names no conversation (section 1) | a **new conversation**. The previous conversation's tool results are not compared |
| This request has several messages, and the blocks after its opening are the compared turn's (at least one to compare, all that both have equal) | the same conversation with its first message edited: `history_edit` |
| This request has several messages, the blocks after its opening are not the compared turn's, and there is evidence that the lane mixes conversations: the session names no conversation, or the lane is full (it remembers 8 conversations, each with a turn in the hour before this request was sent) | **another conversation whose own previous turn is not at hand** (never seen, or dropped to make room): judged by its own read (section 1) |
| Anything else (a named session with room to remember) | `history_edit`: a history rewritten from its first message is still reported |

**A new conversation** is judged by 2.3 up to and including the first-marker rule: a hit is a hit, a
missing `cache_control` is still named. Below that it shares only the tools and system prompt before
its opening message with the compared turn, and one case is known to have been in the cache: on
Anthropic's own API (not behind a self-hosted engine), the previous turn carried `cache_control` on a
block of tools+system and read or wrote something, this request carries it on that same block (the
last such block of the previous turn), and nothing was read. That prefix is then judged by the rest
of 2.3 like any prefix that did not change (`ttl_expired`, `routing_miss`). In every other case the
size of the shared part, and whether it was cached, is not known: the turn is the conversation's
`first_write`, with its legitimacy left open (neither expected nor waste), never an inferred
`routing_miss` or expiry, and it is not evidence of how long the cache lasts.

A change is then weighed against what the cache actually did:

| Condition | Verdict |
|---|---|
| This request is below the provider's minimum and nothing was read | `below_min_prefix`: no version of it could have been read; the change cost nothing |
| The previous turn left nothing in the cache (below), and nothing was read | `below_min_prefix`: nothing was there to lose, or to expire; the change cost nothing |
| The entry's age exceeds the cache TTL and nothing was read | `ttl_expired`: the entry was gone; the change cost nothing more |
| The cache served all of `R` | `none`: the change came after the reusable prefix or never reached the rendered prompt |
| Otherwise | the change's cause |

In the first four the change is kept as a secondary cause, and the provider's own reason (if any)
is not scored against the verdict, since it answers another question.

**A previous turn that left nothing in the cache.** Claimed only where a marked prefix below the
model's minimum cacheable size is the one explanation left. All of these hold:

- Anthropic's own API, not behind a self-hosted engine;
- the previous turn's usage carried a cache-write count (`cache_creation_input_tokens`, or the
  `cache_creation` split): a server that leaves the field out says nothing by its 0;
- that turn carried `cache_control`, had a prompt, and read 0 and wrote 0 tokens;
- its last block was not marked. A marker there covers the whole prompt: one below the minimum is
  the rule of 2.1, and one at or above it that wrote nothing had its markers ignored;
- this request does not meet the conditions of the `markers_ignored` detail (2.4), which then stands
  with the cause the other rules give.

After such a turn a request that read nothing lost nothing: with a change it is the row above, and
with none it is the row of 2.3. So no waste verdict follows such a turn, except for a request that
carries no `cache_control` at all (2.3, section 5).

**Age, not idle time.** A stated TTL is compared with the cache entry's age: the time since the
previous turn was *sent*. A provider counts an entry's lifetime from the request that wrote or read
it, so that turn's own generation time is part of the age (measured on Anthropic: two 5-minute entries
read nothing 318 s after the request that wrote them, 153 s and 292 s after its response ended). The
ledger's `gap_seconds` stays the idle time (from the previous turn's end), which the retry rule and
the observed lifetimes use; when the two differ, the evidence gives both.

On Anthropic (not behind a self-hosted engine, which caches without markers), when a change is named
in a prompt that carries no `cache_control`, nothing was read, and the previous turn read 0 and wrote
0 tokens, the evidence says so and `breakpoint_misplaced` is added as a secondary cause: the change
is the first thing to fix (a marker on a prefix that changes every turn buys writes nobody reads),
and nothing is cached until the markers are there too.

### 2.3 The prompt did not change

| Rule (in order) | Verdict |
|---|---|
| The cache served all of `R` | `none`, except on Anthropic when the cache served everything the previous turn held and a stable part of that turn, at least the minimum cacheable size, lay after its last breakpoint and was sent uncached again: `breakpoint_misplaced` |
| A prompt-affecting parameter changed (thinking, tool choice, reasoning, output format, chat-template arguments, LoRA adapter, multimodal processing options, RAG documents) | `param_rendered` |
| Below the provider's minimum cacheable length | `below_min_prefix` |
| Anthropic with no `cache_control` anywhere, the entry's age past 1 hour (the longest a marker can ask for, whatever lifetime the edge recorded for the request: with no marker that is only the 5-minute default), and nothing read | `ttl_expired`: no marker would have kept it; `breakpoint_misplaced` is kept as a secondary cause |
| A breakpoint after a volatile block (a fact the classifier has a rule for, but the proxy never sets today), the entry's age past the request's own TTL, and nothing read | `ttl_expired`, likewise |
| Anthropic with no `cache_control` anywhere, or a breakpoint after a volatile block | `breakpoint_misplaced` |
| No usage at all | `unknown` |
| The server reports no cached-token count (Ollama, most llama.cpp builds), and no engine evidence | `unknown` |
| Half read (section 1) | `none` |
| Anthropic: this is the first turn carrying a breakpoint and less than half was read | `first_write` |
| A new conversation (2.2) whose shared tools and system prompt are not known to have been cached and missed | `first_write`, legitimacy left open |
| The previous turn left nothing in the cache (2.2), and nothing was read | `below_min_prefix`: what its marker covered was too small to cache, so there was nothing for a router to miss, nothing to expire, and nothing a retry could have read |
| Engine evidence: the block where the read stopped was removed | `evicted`, confirmed |
| Engine evidence: another cache served this request while the previous one still held the prefix | `routing_miss`, confirmed |
| An OpenAI-compatible API that is not OpenAI, prompt under 1,024 tokens | `unknown` (its minimum is not known) |
| Identical bytes within 60 seconds, less than half read | `retry_replay` |
| Behind an engine, events show the stop block still resident in the cache that served it | `unknown`: not an eviction (the token prefix changed, e.g. template rendering) |
| Behind several caches without deciding events | `unknown` |
| Behind one engine | `evicted`, inferred |
| The entry's age exceeds the TTL | `ttl_expired` |
| No TTL known (not Anthropic, not behind an engine), and this ledger has seen the cache on this provider and model still read an unchanged prefix at least 5 times after idle gaps of *A* or longer (each more than a minute): less than half read within *A* | `routing_miss`, inferred, naming the observed lifetime |
| No TTL known, at least 5 such idle hits seen, and less than half read after a gap at least *C*: the shortest gap, longer than every gap that was read and than 5 minutes, after which at least 5 unchanged prefixes read less than half (a changed cache key excluded) | `ttl_expired`, inferred, naming the observed lifetime |
| Less than half of the reusable prefix read (nothing, or only part), the entry's age within the TTL (or within the shortest provider TTL, 5 minutes, when none is known): too soon to have expired. On Anthropic a part read counts only within 5 minutes: a 1-hour breakpoint early in the prompt can still read after a later 5-minute one expired | `routing_miss`, inferred |
| Otherwise | `unknown` |

### 2.4 Facts outside the two requests

Three facts about the turn at hand are checked before the comparison, in this order. The first and
the third replace 2.1 to 2.3 altogether, the second replaces 2.2 and 2.3; no detail is attached to
their verdicts:

| Fact | Verdict |
|---|---|
| `previous_response_id` names a response this proxy never saw | `unknown`: the prefix cannot be reconstructed |
| **An earlier point of its own conversation.** The request's conversation blocks (messages and tool results, in order) are a proper prefix of the compared turn's: every block it carries opens that turn, which goes on after them. It carries no compaction event, and is not a stitched Responses turn | 2.1 still applies (a compared turn below the minimum is `below_min_prefix`). Otherwise nothing it sent changed, and its predecessor is the turn before that point, not the one at hand: it is judged by its own read (section 1), and is also `none` when the cache served all of `R`, since the turn at hand is its own conversation's and what that turn left is what it could read |
| The request was sent more than the clock skew (1 s, `PREFIXWISE_CLOCK_SKEW`) before the turn it is compared with finished, it is not a new conversation in the lane (2.1) or a stitched Responses turn, and it read less than half of the shorter of the two prompts | `unknown`: it could not have read that turn's write |

An earlier point is told by the blocks alone, never by the clock. What the clock then decides is only
what the record leaves behind. Sent more than the clock skew before the compared turn was sent, it is
the record of an earlier turn delivered late: it does not become the conversation's latest turn (the
later one stays), though a Responses turn that continues it by `previous_response_id` still finds it.
Otherwise it is a rewind to, or a replay of, an earlier turn, and becomes the latest turn like any
other. The evidence says which.

Two details can change the cause 2.1 to 2.3 gave:

| Fact | Effect |
|---|---|
| Anthropic, cause `ttl_expired`, `routing_miss` or `unknown`, less than a tenth of the prompt read, the entry's age (since the previous request was sent) over 5 minutes; the previous request asked for a 1-hour cache on every breakpoint, used no server tools, and its write was kept for 5 minutes | `ttl_expired`, detail `ttl_shortened` |
| Anthropic, cause `routing_miss` or `unknown`, less than a tenth read, the entry's age at most 5 minutes, the previous turn wrote, the conversation was appended to, and the nearest breakpoint is at least 22 positions past the previous write (Anthropic looks back 20; a run of tool-use or tool-result blocks counts as one position) | `breakpoint_misplaced`, detail `lookback_exceeded` |

Applied last: when the engine keeps no prefix cache (SGLang `--disable-radix-cache`), any miss becomes
`unknown`, legitimate, with no detail.

Every other **detail** (for example `markers_ignored`, `thinking_stripped`, `sent_before_cache_written`,
`volatile_prefix_at_start`, `output_rerendered`, `host_restored`, `session_header_dropped`) explains a
cause without changing it, and is claimed only from facts in hand. The full list is in the guide.
`markers_ignored`: Anthropic, `cache_control` on the last block, a prompt of at least 8,192 tokens,
nothing read and nothing written, on any cause but `none` and `below_min_prefix`. The windows of the
details that look at another session's request (`shared_prefix_not_reused`, `volatile_prefix_at_start`)
are counted from when that request was sent.

## 3. Engine evidence

Behind vLLM or SGLang with KV events on, Prefixwise subscribes to the engine's event stream (blocks
stored and removed, with hashes, the parent block, the memory tier and, for blocks the request stored,
its session id) and keeps, per session, the chain of blocks the previous request stored. When the
response arrives, the join locates the block where the cache read stopped
(`cached_tokens / block size`):

| Join's judgement | Becomes |
|---|---|
| The read was a hit (at least half this prompt): blocks the read covered had been removed from the GPU, the offload tier's own stores have been seen, no other session stored them meanwhile, one cache, no concurrent request in the session, a clean stream | a `Restored` verdict: the hit gets the detail `host_restored` |
| A miss, and the stop block was removed | evidence for `evicted`, confirmed (a removal seen is a removal, whatever else was in flight) |
| A miss, the stop block still resident, the removals settled, no concurrent request, a clean stream, a full-attention model | evidence for "not an eviction" |
| Several caches: another cache served this request while the first still held the stop block | evidence for `routing_miss`, confirmed |
| Anything else (an overflowed index, a cut stream, an unplaced or ambiguous chain, or a negative claim the conditions above do not allow) | nothing: the classifier keeps its inference |

On a model with sliding-window or Mamba layers, which engines free without events, "not an eviction"
is never claimed; a confirmed eviction still is.

Only a `Restored` verdict can become `host_restored`. The join and the classifier use different hit
thresholds (the join: half of this prompt; the classifier: half of the shorter prompt, section 1), so a
turn the classifier calls a hit may be one the join judged a miss; its removed blocks were never
restored and must not be labelled so.

## 4. Evidence tiers

Every verdict is one of:

- **confirmed**: named from direct evidence: the hashes of both requests, the provider's own usage,
  or the engine's events (`evicted`, `routing_miss` from events; the evidence text says "confirmed").
- **inferred**: the evidence fits one cause best, and the evidence text begins "inferred"
  (`evicted` behind one engine without events, `routing_miss` on a hosted API).
- **unknown**: the evidence does not decide. Prefixwise never guesses.

The tier is in the evidence text; it is not a separate field.

## 5. What a miss costs

- **Avoidable tokens** (waste only): `max(0, R - cached_curr)`. Appended content is never counted, a
  partial hit counts only what was not read, and where the previous turn's cache content is known and
  is not zero, nothing beyond it. Legitimate misses and undecided ones count zero. A previous turn
  known to have left nothing (2.2) needs no cap of zero: a request that read nothing after it gets no
  waste verdict. Where the previous turn held nothing for any other reason (it carried no
  `cache_control`, its markers were ignored, the server reports no writes), `R` is not capped: what a
  marked, stable prompt would have reused stands, whichever of the two turns lacked the marker. One
  exception: an Anthropic
  breakpoint left early (2.3) counts the stable part it left uncached, `min(prompt_prev, prompt_curr)
  - cached_curr`, since that part is exactly what a later breakpoint would have cached.
- **Avoidable dollars**: those tokens at the difference between the rate actually paid for them and
  the cache-read rate, from a dated price table (the input rate when the request's usage does not say
  what was paid). With no usage on the previous turn, nothing is avoidable: how much was reusable is
  not known.
- **The bill** of every priced request is split into cache reads, new input, cache writes and output.
- The prediction is checked: when a fix is turned on, the saving realised is measured against the
  saving predicted (97.5% on vLLM and 99.9% on Claude for `canonical_tools`; 106.8% for
  `auto_breakpoints`; experiments/e4).

## 6. Invariants

These must hold for every well-formed request (a valid Anthropic, OpenAI Chat or Responses body;
roles the wire formats do not define are not compared); `tests/test_invariants.py` checks them on
generated conversations
(Anthropic and OpenAI request shapes, hosted and engine prefix orders, real tool schemas and tool
results):

1. **One change is named exactly.** A single change of each kind (tools reordered, keys reordered, a
   schema's `required` reordered, a tool added, removed or edited, a timestamp in the system prompt,
   a reworded system prompt, an edited earlier message, an edited tool result) produces its cause and
   its legitimacy when nothing was read, and when half was read.
2. **A full read is never a miss.** For a turn of a known conversation (not a new one, not a first
   write), when the cache served all of `R`, the verdict is `none` (or the Anthropic early-breakpoint
   case), and any change is kept as a secondary cause.
3. **Appending is never a change.** A turn that only appends (including a system or developer reminder
   mid-conversation) never yields a change cause, primary or secondary.
4. **The earliest change wins.** With two changes, the cause is the one in the block earliest in
   prefix order.
5. **Where and why agree.** The ledger's "first changed" block lies in the section the cause names, in
   the hosted and the engine prefix order, with one change or two.
6. **Avoidable is bounded.** `0 <= avoidable <= min(prompt_prev, prompt_curr) - cached`, and no more than
   the previous turn's known cache content (where it is not zero) minus what was read; it is positive
   exactly when the verdict is waste and less than `R` was read. So there is no waste verdict with
   nothing avoidable: after a turn known to have left nothing (2.2), a request that read nothing is
   never waste, whatever changed. The test computes avoidable tokens with the cap the collector
   uses. (The early-breakpoint case in section 5 is the one waste that is bounded by the shorter
   prompt and not by `R`.)
7. **Engine evidence decides only an unchanged prompt.** An eviction or a routing miss from events
   never replaces a cause named by a changed block.
8. **Model switch and first turn win** over every rule of 2.1 to 2.3 (2.4's two in-flight facts can
   still leave a turn unjudged, and an engine that keeps no prefix cache still turns a miss into
   `unknown`).
9. **Deterministic.** The same two requests and usage give the same verdict.
10. **Served-all is monotonic.** If a read counts as full, every larger read does.
11. **The hashes ignore exactly what they should.** Key order at any depth leaves canonical and masked
    equal; any content change changes canonical and raw; volatile values leave masked equal; a word
    change does not.

Elsewhere in the suite: only a `Restored` verdict becomes `host_restored`
(`tests/test_host_restored.py`); Prefixwise forwards the fields routers and gateways route by
(`tests/test_routing_headers.py`, `tests/test_openrouter_session.py`); a gateway log with start and end
times (or latency) replayed with `prefixwise import gateway` gives the live proxy's verdict on every
request (without them, requests are replayed one after another and none is taken as in flight), and a parallel subagent of
another type is a new conversation, not an unjudged one (`tests/test_gateway_import.py`).

Not covered by the generated cases yet: the Responses API request shape, the engine order with
Anthropic's request shape, and the Anthropic early-breakpoint case (each has hand-written tests).

## 7. Known limits

- **Observed lifetimes are per conversation, not per cache entry.** The idle gap is measured within a
  conversation. Where many conversations share a long prefix (one system prompt across a fleet), others
  can keep it warm, so a hit after a long gap can overstate how long an unused entry lasts. The window
  uses the 5th-longest idle hit, not the longest, to limit that, and it is shown as observed, never as
  the provider's TTL.
- **Per block, not per token.** A change at the end of a long system prompt is reported for the
  whole block; the avoidable count comes from usage, so it is still exact.
- **Masking is broad.** Any token containing a digit is treated as volatile, so a system prompt that
  changes `v2` to `v3` reads as `volatile_system`, not `system_edit`. Dates with and without a year
  (`Nov 12`, `12. November`) are masked; without a year only a capitalized month and a day of 1 to 31
  count, so `Step 2 May fail` still reads as a date (rare) while `Rule 1 may be skipped` does not. Text is serialized unescaped (`café` stays `café`), but a
  control character is escaped (`\u001b`) and that escape is masked, so a change inside a terminal
  colour code is not seen as wording.
- **Thresholds.** "At least half of `R` read" separates a hit from a miss where the evidence is only
  usage; the read slack absorbs block rounding. A read between the two is a partial hit.
- **Inferred causes can be wrong.** `routing_miss` on a hosted API and `evicted` behind one engine
  without events are the best fit, marked inferred; events or diagnostics confirm them.
- **Lanes need a conversation id.** Without one, every conversation shares a session, and what is told
  apart is told from the messages (2.2):
  - A conversation whose own previous turn is not at hand (it opens with several messages, or the
    lane dropped it) is judged by its own read: a miss is `unknown`, never named.
  - A compaction sent with no compaction event is not reported there. Rewritten into a single
    message it reads as a new conversation; rewritten into several, as another conversation.
    Send the event (`X-Prefixwise-Compaction-Event: 1`) or a session id.
  - In a session that does name a conversation, a first message unlike the latest turn's still reads
    as a `history_edit` when it is really another conversation's (a second subagent of the same
    type starting after the first one's second turn), unless the lane is full of recent
    conversations and the request has several messages.
- **Judged by its own read means half of its own prompt.** A request with no turn of its own to be
  compared with is a hit when at least half of its prompt came from the cache. One that read only a
  large shared system prompt is then a hit though its own history was not read; one that read all of
  its history and appended a larger tool result is `unknown`. So is a request taken for another
  conversation (2.2) that read all the compared turn left but under half of its own prompt; before
  that rule it was compared with that turn and came out `none`.
- **A new conversation's shared prefix is sized in one case only** (2.2: Anthropic, the same marked
  block of tools+system in both requests, the previous turn read or wrote something). Elsewhere a new
  conversation that read less than half of the shorter of the two prompts is a `first_write` with its
  legitimacy left open, even if the tools and system prompt it shares with the previous conversation
  were cached and were not read: on OpenAI and other automatic caches that miss is never claimed. In
  the one case, the marked part is taken to have been cached because the previous turn read or wrote
  something; if that part was itself below the model's minimum and the write came from a later
  marker, a `routing_miss` or `ttl_expired` (neither is waste) is named for an entry that never was.
- **An Anthropic request with no `cache_control` is `breakpoint_misplaced` even when it is a one-shot.**
  Independent requests that share only a short system prompt (a classifier with a different document
  each time) are each waste, with the shorter of the two prompts counted as avoidable, though a
  marker would have cached only the shared part. The rule is older than the new-conversation rule
  and comes before it.
- **Not measured at scale yet.** Accuracy is measured on injected causes (120 of 120 on vLLM, LMCache
  and SGLang), against Anthropic's own reasons (32 of 32) and by realised savings, not yet on large
  volumes of real agent traffic.
