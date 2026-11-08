# Review: Prefixwise at fleet scale (2026-11-08)

A review of the 0.7.0 code for 10,000+ requests/s, several gateway pods, several vLLM/SGLang
replicas and a no-prompt-storage rule. Each item says what the code does now (file and
function), and what changes. Three defects were found and fixed with tests
(`tests/test_hyperscale_review.py`); the other items were already built, or are designs below.
The rule throughout: a guess is worse than no answer.

## Verdicts

| # | Claim | Finding |
|---|---|---|
| 1.1 | Prompt-change rules run before the hit check, so partial hits are mispriced | **Already right.** `classify` (`classifier.py`) names the first change, then overrides it: a cache that served all of the reusable prefix is `none`; nothing read after the TTL is `ttl_expired` (`score=False`). A partial hit keeps the cause, and `avoidable_for_turn` (`ledger.py`) prices only `reusable - cached`, the diverged part. |
| 1.2 | `below_min_prefix` must run before prompt-change rules | **Half right; fixed.** A *previous* request below the minimum was already checked first. A *current* request that shrank below it, with a system or history change, was priced as waste. Now `below_min_prefix`, with the change kept as secondary. |
| 1.3 | Replace previous-turn comparison with an LCP ring buffer of K=16 | **Already solved, differently.** `_best_parent` (`proxy.py`) finds a turn's own conversation by its opening message, up to `PREFIXWISE_LANE_THREADS` (8) per session and model. An LCP choice was built and dropped: it matched a reverted edit to the turn before the edit and hid it (E1). |
| 1.4 | Per-prefix last-hit time for TTL; in-flight write races labelled `routing_miss` | **Already built.** The gap runs from the parent turn's response to this request's send. Only that conversation reads its own history, so this is the entry's last use. Races: `prefix_inflight` (`proxy.prepare`) tracks same-prefix requests in flight until the upstream starts answering, and `detail_for` (`details.py`) names `sent_before_cache_written`. A mid-conversation race cannot happen: the next turn needs the previous response. |
| 1.5 | `prefixwise import` skips fingerprinting and classification | **Wrong premise.** `importers._classify` classifies every logged request. Logs do not hold the request as sent: no tools, no rendered system prompt, and history the agent later rewrote. Hashing them would invent `history_edit`. It names only what the log shows (version change, system hash, compaction, effort, TTL, Anthropic's own reason), otherwise `unknown`. |
| 2.1 | Dual hash and chunk sketch | Three hashes per block already (`fingerprint.py`): `raw`, `canonical`, `masked`. Equal masked hashes with different raw hashes prove `volatile_system` without text. Chunk sketches: design below, not built. |
| 2.2 | Write surcharge, long-context tiers, 5m vs 1h | **Partly missing; fixed.** Request cost already had both write rates and `long_prompt` tiers (`pricing.py`). Avoidable cost used the input rate; it now uses the rate actually paid (the write rate on a rewrite). New: `ttl_advice` (`findings.py`). `unamortized_write_waste` is not added: see below. |
| 2.3 | `include_usage`, Anthropic usage merge, cancelled streams | **Already built.** The proxy adds `stream_options.include_usage` and strips the extra usage chunk if the client did not ask for it. `StreamAccumulator` merges `message_start` and `message_delta`. A client hang-up records status 499 / `client_disconnected` with no verdict and leaves session history untouched. |
| 2.4 | `SessionStateStore` (Redis), `LedgerSink` (ClickHouse...) | Design below: session affinity first; a store interface only with a measured need. |
| 2.5 | KV-event watermark | **Already built.** See below. |
| 3 | Flowchart items | **Already done** in 0.7.0 (#52): the import route, the arrowhead, the gap, the label mask, hero fills, slate decisions, blue verdicts, emerald hit, `unknown` in the verdict column. `tests/test_figures.py` checks the drawings against the classifier's rule order, so the 1.2 change needed no change to the drawing: its first decision already reads "too short to cache?". |
| 4 | `check` and `--optimize` specs | **Exist** (not built): [check-and-optimize.md](check-and-optimize.md). The spec's cost projection now specifies the rate paid. |

## 2.2 Why there is no `unamortized_write_waste`

Avoidable cost is a difference between two worlds: this request missed, against this request
hit. Turn N's cache write is paid in both worlds, so adding its surcharge to turn N+1's miss
counts money that a fix would not save. What a fix saves is turn N+1's own rewrite: those
tokens billed at the write rate instead of the read rate. That is now the price
(`PriceTable.avoidable_cost` with the request's usage). With the built-in Claude rates, the
true figure is about 26-28% above the old input-rate price for 5-minute writes, and about twice
it for 1-hour writes. Rows recorded before this change keep their old price.

The 5m vs 1h choice is a separate question, answered per model from the ledger by `ttl_advice`.
It needs the TTL each turn was judged against (ledger column `cache_ttl_seconds`, schema 5), so
rows from older ledgers are left out.

- **Kept by 1 hour:** `ttl_expired` misses judged against a 5-minute cache, with a gap within
  the hour (not `ttl_shortened`: those asked for 1 hour already).
- **Saved, at most:** what those misses wrote again, read instead: 5-minute write rate minus
  read rate.
- **Extra:** every other write on that model on the 5-minute cache, at the 1-hour rate minus the
  5-minute rate. Writes already on 1 hour cost nothing more.

Rates are taken at the long-prompt tier where a row's prompt is in it. Part of each rewrite is
new tail that would be written anyway, so saved minus extra is an upper bound on what 1 hour
gains. When it is not positive, 1 hour cannot pay and the report says to keep 5 minutes.
Otherwise it says 1 hour is worth trying and measuring, with both numbers.

## 2.4 Several gateway pods

**Measured first.** One proxy process handles about 185 requests/s, and it is CPU-bound in
fingerprinting (README, "Capacity"). So 10,000 requests/s takes about 55 processes, whatever
the state store. The design question is where a session's turns are matched.

**Recommended: session affinity.** Route each session to one pod by hashing
`X-Prefixwise-Session-Id` (consistent hashing at the ingress, e.g. Envoy/NGINX `hash_on`
header; or Kubernetes with a header-hash load balancer). It needs no new code and no network hop
per request. It also keeps the stream-recording wait (`begin_recording` / `_await_recording`)
local. That wait is what stops a fast client's next turn being compared with the wrong one.
When a pod is lost, its sessions' next turns are `first_write` once. That is a known, bounded,
labelled error, not a wrong verdict, and it is what the README already advises.

**A shared store, only if affinity is impossible.** The interface, so it can be added without
touching the classifier:

```python
class SessionStateStore(Protocol):
    def get_lane(self, session_id: str, model: str) -> LaneState | None: ...
    def put_turn(self, session_id: str, model: str, opening: str, turn: TurnState,
                 *, max_threads: int) -> None: ...              # LRU per lane
    def begin_recording(self, key: tuple) -> None: ...
    def end_recording(self, key: tuple) -> None: ...
    def await_recording(self, key: tuple, timeout: float) -> bool: ...
```

`TurnState` holds what `_Lane` holds now: block hashes (64 hex characters each, no text), token
counts and times. A Redis implementation stores one hash per lane: up to 8 openings, each with a
snapshot of about 200 bytes per block (three 64-character hashes and counts). `put_turn` runs as a Lua script, so the LRU and the
write are atomic. The recording wait becomes a key with a short expiry plus a pub/sub wake-up.
Costs, to weigh against affinity:
- two round trips per request, about 0.3 ms each in-zone, on the proxy's hot path;
- a snapshot of a 200-block prompt is about 40 KB on the wire per turn;
- the recording wait is now distributed, so a lost wake-up means a 10 s stall, not a missed
  comparison.

Build it when a user shows a deployment that cannot hash on a header.

**Ledger.** One SQLite file per pod, WAL with `synchronous=NORMAL` and a 10 s busy timeout
(`sqlite3.connect(..., timeout=10)`, `ledger.py`), one commit per request. That suits a process
at 185 requests/s. The write happens after the client's response is complete (the stream is
half-closed first), so it never adds latency. For the fleet view, the existing sinks are the
answer:
- OpenTelemetry spans with the verdict (`otel.py`), to any OTLP backend, ClickHouse included;
- Prometheus metrics (`prom.py`), aggregated across pods by the usual `sum by`.

A `LedgerSink` interface (`record(entry)`, `flush()`) with an async batch writer is worth adding
only for a direct ClickHouse/PostgreSQL writer, which OTLP already covers.

## 2.1 Chunk sketches (designed, not built)

**What they would add.** The position of a change inside one large block (for example, the
system prompt changed at about character 41,000 of 52,000), where today only the block is named
(`first_divergence`).

**Design.** For `system` and `tools` blocks only, an HMAC-SHA-256 of each 512-byte window of the
canonical bytes, truncated to 64 bits, kept in memory with the turn and never in the ledger. On a
`system_edit` or `volatile_system`, the first differing window gives the byte offset. The ledger
records the offset only (`divergence_offset`, an integer).

**Why not now.**
- **Privacy:** a hash of a short window of predictable text can be guessed by trying candidates,
  far more easily than a hash of a whole block. So it must be keyed (HMAC with the per-install
  key), and never exported.
- **Cost:** about 100 window hashes for a 50 KB system prompt, every request. That can be cached
  by the block's raw hash, as masking is.
- **Value:** small. `volatile_system` is already proven by the masked hash, and the usual
  culprit (a date) is named by the masked comparison.

Build it when users ask where in a long prompt an edit landed.

## 2.5 KV events and requests

What `EngineEvents` (`engines.py`) does now:

- **Ordering:** each batch carries vLLM's sequence number. A gap counts as dropped messages,
  which blocks negative claims: a removal may have been among them. A restart (the sequence
  number going back) clears the windows.
- **Late events, the watermark:** before judging a request, `_settle` waits up to
  `settle_seconds` (50 ms) until the stream has caught up with the response time. That is, until
  a batch stamped later has been read (corrected by the measured clock offset), or the socket is
  empty. If it has not caught up in time, no negative claim is made.
- **Correlation:**
  - `request_sent` freezes the session's previous block chain;
  - `request_done` looks up the block where the cache read stopped (`cached_tokens // block_size`)
    in that chain;
  - session-tagged events (vLLM `kv_cache_report_mode`) say which cache served the request.
- **Not judged:**
  - concurrent requests of one session;
  - hybrid models;
  - a change in the error count in the meantime.
- **Several caches:** `EngineFleet` waits only when a routing claim is possible, and a claim
  needs the same block hash on the previous cache.
- **Token-ID block hashes:** they are compared as opaque keys. Prefixwise never recomputes them,
  so the engine's hash scheme does not matter.
