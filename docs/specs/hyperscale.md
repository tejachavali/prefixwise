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
| 2.4 | `SessionStateStore` (Redis), `LedgerSink` (ClickHouse...) | **Redesigned and built** (section 2.4): edges emit turn records; collectors own state and the ledger; HTTP or a session-partitioned Redis Stream between them; JSONL and ClickHouse sinks. |
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
It needs each request's own cache lifetime (ledger column `cache_ttl_seconds`, schema 5), so
rows from older ledgers are left out.

- **Kept by 1 hour:** `ttl_expired` misses on the 5-minute cache, with a gap within
  the hour (not `ttl_shortened`: those asked for 1 hour already).
- **Saved, at most:** what those misses wrote again, read instead: 5-minute write rate minus
  read rate.
- **Extra:** every other write on that model on the 5-minute cache, at the 1-hour rate minus the
  5-minute rate. Writes already on 1 hour cost nothing more; writes of unknown lifetime are not
  counted.

Rates are taken at the long-prompt tier where a row's prompt is in it. Part of each rewrite is
new tail that would be written anyway, so saved minus extra is an upper bound on what 1 hour
gains. When it is not positive, 1 hour cannot pay and the report says to keep 5 minutes.
Otherwise it says 1 hour is worth trying and measuring, with both numbers.

## 2.4 Fleet scale: an edge and a collector

**What was wrong with the first answer.** The 0.7.0 design put everything in one process on
the request path: hashing, classification, conversation state and the ledger. "Hash the session
header at the ingress" was offered for several pods. That fails in practice: most clients carry
the session in the body (`session_id`, `metadata`, LiteLLM's id) which an ingress cannot hash
on; every deploy wiped conversation state (a burst of false `first_write`s per pod); 10,000
requests/s would need about 55 CPU-bound processes doing SHA-256 inline on a thread-per-request
server; the stream-recording wait could stall a conversation under load; and there was no fleet
ledger, so `prefixwise report` did not exist across pods.

**The shape (built in 0.8: `prefixwise.records`, `prefixwise.collector`).** Two halves, one boundary.

- **Edge (data plane).** Wherever the request already is: the proxy, `prefixwise.wrap`, the
  LiteLLM callback, later a gateway plugin. It fingerprints the request (three hashes per block,
  no text), reads the response's usage and timing, and emits one **turn record** per request
  (`TurnRecord`, a few KB: hashes, positions, counts, digests of the prompt-affecting parameters,
  the provider's usage, the engine join's verdict). It keeps only what a request needs on its way
  out: requests in flight and entries written by prefix (for `sent_before_cache_written` and
  `shared_prefix_not_reused`), streams ended but not yet recorded, fields an upstream rejected,
  and the response id of each conversation's last turn (to chain the provider's diagnostics).
  With several edges those are per edge: the two fan-out sub-reasons and the recording wait see
  only the requests that passed through the same edge, which is stated here rather than solved.
  Hashing is its whole CPU cost, and it is independent of the number of edges.
- **Collector (control plane).** `Collector.ingest(record)`: compares the record with the previous
  turn of its own conversation, names the cause, prices it, writes the ledger. It owns all
  conversation state (lanes, Responses snapshots, per-session facts). One lock serializes
  `ingest`, which is the ordering a conversation needs: turn N is recorded before N+1 is compared
  with it. The edge asks it two things at request time (`chain_parent`: the previous turn's
  response id for the provider's diagnostics; `stitched_opening`: which conversation a stitched
  Responses turn continues); in one process those are a lock away. Measured here (`Collector.ingest` on the realistic three-turn record set, this VM): about 5,800
  records/s into a file-backed SQLite ledger (WAL, `synchronous=NORMAL`), about 170 µs per record,
  of which the ledger commit is most; one collector takes the records of some thirty 185-request/s
  edges before it is the bottleneck, and the partitioned log below is the way past that.
- **In one process** (`prefixwise serve`, `prefixwise.wrap`, the callback) the edge calls the
  collector directly. Same code, same verdicts, no configuration: the single-user product is the
  fleet product with a zero-length wire.

**The record is the contract.** `TurnRecord.to_dict()` is its wire form; `version` changes when a
field's meaning does, and a collector refuses a version it does not read rather than guess.
`tests/test_records.py` pins that a record holds no prompt text, that a record that crossed JSON
yields the identical ledger row, and that malformed records are refused.

**Transport (built: `prefixwise.transport`, `prefixwise collect`).** `prefixwise collect` runs a
collector as a service; an edge given `--collector URL` (or `PREFIXWISE_COLLECTOR`) sends records
instead of ingesting them:
- one FIFO sender thread per edge, a bounded queue (10,000 records), batches of up to 200 to
  `POST /api/v1/ingest`, up to six attempts with backoff on 5xx/429/connection errors; when the
  queue is full or a batch has failed its last attempt, records are dropped and counted, never
  blocking a request; a 401/400 is not retried (the first is printed once);
- a bearer token on the collector (`PREFIXWISE_AUTH_TOKEN`; the edge sends
  `PREFIXWISE_COLLECTOR_TOKEN`); a collector listening beyond localhost refuses to start without
  one unless told otherwise; TLS at the ingress;
- ingest is idempotent by `request_id` while the collector runs (the last 200,000 are remembered in
  memory; a restart mid-retry can admit a duplicate, which the log transport below removes), malformed records are
  reported back by index and skipped, and a record the collector itself fails on becomes a
  `not_recorded` failure row, never a lost request;
- ordering: a conversation's turns are sequential, and an edge emits turn N's record at N's
  response end, before turn N+1 is even sent, so with a FIFO sender N arrives before N+1 unless
  the sender is backed up by more than one turn gap. The collector keeps each turn's own
  `sent_at`/`done_at` (not its arrival time), and a turn sent before the compared turn's response
  ended is `unknown` with the reason ("sent N s before the previous turn had finished"), counted
  as `prefixwise_ingest_concurrent_total`: the failure mode is visible, not a wrong cause. The same
  rule names parallel sampling of one conversation, which could not have read the other sample's
  write either. Two consequences are stated, not solved, here: a conversation's turns should take
  one edge (across edges, a backed-up sender can deliver N after N+1: N+1 is then a `first_write`
  and N `unknown`), and edge clocks must agree within `PREFIXWISE_CLOCK_SKEW` (1 s) for the rule
  to see sequential turns as sequential. The partitioned log below removes both;
- sizing: the queue holds 10,000 records, about 54 s of a 185-request/s edge, while a batch to a
  black-holed collector is given up after about 75 s (six 10 s timeouts and 15.5 s of backoff), so
  an outage of a minute already drops records, counted;
- the collector serves the existing `/api/v1`, `/metrics` and `prefixwise report` over the fleet's
  one ledger; an edge answers `/api/v1`, `/ledger` and `/tasks` with 409 and the collector's URL;
- the engine join (vLLM/SGLang KV events) stays with the edge that fronts the engine; its verdict
  travels in the record.

**Several collectors (built: `prefixwise.log`, `prefixwise collect --log`).** Records go on a
Redis Stream partitioned by session id (`crc32(session_id) % PREFIXWISE_LOG_PARTITIONS`, 16 by
default, the same on every edge and collector). Each collector owns a set of partitions
(`--partitions 0-7`) and reads each in order through a consumer group, so:
- one collector sees all of a session's turns, in the order they reached the log, and no two
  collectors ever see one session. What the log fixes is which collector a session reaches and the
  consumption order; an edge still appends in its own send order, so a backed-up edge can append
  turn N after another edge appended N+1, and that record is named `unknown` as under HTTP. One
  edge per conversation remains the way to never see that;
- a record is acknowledged only after its ledger row is written, and the row is written before
  the collector's state changes, so a refused row leaves nothing behind and the redelivery is
  compared with the same predecessor (at-least-once); a redelivery of a written record is a
  duplicate by request id, with the duplicate set seeded from the ledger at start, so a restart
  mid-batch writes no row twice and loses none (records delivered but not written are not marked
  seen by the replay);
- at start a collector replays its partitions' records of the last day into its conversation state
  (no rows, no metrics; pending records left out; a record that breaks the classifier is skipped
  and counted), in each partition's own thread, so `/health` answers meanwhile and shows
  `replaying`; the first turn after a deploy is then compared with its real predecessor;
- a starting collector takes over everything a dead predecessor left unacknowledged at once; when
  another consumer is still active on the partition (a rolling restart), it leaves that consumer's
  batch alone (taking it would write those rows twice: XACK is per group, not per owner) and claims
  only what stays idle for a minute, and it reports the conflict in stderr and `/health`. Deploy
  collectors so that the old one stops before the new one starts (`strategy: Recreate`, or
  `maxSurge: 0`); two live collectors on one partition split its sessions between them;
- Redis 7.0 or later (XAUTOCLAIM drops trimmed entries from the pending list only from 7.0);
- the stream is capped per partition (`PREFIXWISE_LOG_MAXLEN`, 1,000,000 records); records that
  fall off before a collector read them are lost, visibly a capacity decision.
Tested here against a real Redis (`tests/test_log.py`): two collectors split sessions and together
produce exactly one proxy's rows; a restarted collector finds its state and adds no duplicate;
a dead consumer's record is processed once; an unwritable ledger leaves records unacknowledged
until it works again. Kafka would fit the same consumer with a partitioner; it is not built.

**Ledger sinks (built: `prefixwise.sinks`).** The SQLite ledger stays what `prefixwise report`
reads; `--sink` sends a copy of every committed row to a JSONL file or to ClickHouse (HTTP,
`INSERT ... FORMAT JSONEachRow`, batched on the bounded sender). A sink never sits on the ledger's
write path: a failing one is reported once and skipped. Postgres is a `LedgerSink` with two
methods that an operator can add with their driver; it is not built in.

**Engine join.** The KV-event subscription stays on the collector side conceptually, but the
request-time half (`request_sent` freezes the previous chain before the request goes out) needs
the edge: today the proxy subscribes and puts the verdict in the record. With a remote collector,
the edge that fronts the engine keeps that role; it is the one that sees the engine's requests.

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
