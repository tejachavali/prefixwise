# Prefixwise on the FreeInference agentic trace

The question: **on 1.19M real agent requests, why were prompt-cache reads missed?** The data is the FreeInference
agentic trace (Harvard MADSys, [paper](https://data.freeinference.org/pdf/paper/from-requests-to-sessions),
[code](https://github.com/HarvardMadSys/freeinference_agentic_trace),
[data](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)): 12,002 sessions and 16 weeks
of coding and assistant agents. It holds no text. Per request it holds the input as chained 16-token block ids,
the provider's own `prompt_tokens` and `cache_read_tokens`, times, model, an anonymous provider letter, o200k
tokens per role, and the authors' own label against step − 1 (`transition`: append, mutation or switch, and a
`cause` for a mutation). The authors computed that label from the text before anonymising it.

`prefixwise_trace.py` compares each request with its session's previous step, the way Prefixwise compares a turn
with its previous turn (`docs/ALGORITHM.md`, sections 1 to 3). It names a verdict and counts the tokens lost. It
then checks the block comparison against the authors' labels. Nothing under `src/` is used or changed: the trace
has no request bodies, so the classifier's rules are applied to what the trace does hold.

## Run it

```bash
# 1. Get one week of the Parquet tables, with the trace repository's own downloader.
git clone https://github.com/HarvardMadSys/freeinference_agentic_trace fi && cd fi
make setup                                   # .venv with the package and huggingface_hub (needs python3.12)
.venv/bin/python -m release download --tables --week 2026-07-19
#   -> data/v1/release/2026-07-19_2026-07-25/{manifest.json,sessions.parquet,requests.parquet,tool_calls.parquet,tables.json}
#   (the Hub copy is kept under data/hub/ and hard-linked into place)

# 2. Run the analysis (pyarrow, pandas and numpy; the trace's .venv has them).
.venv/bin/python /path/to/prefixwise/experiments/freeinference/prefixwise_trace.py \
    data/v1/release/2026-07-19_2026-07-25 --verify \
    --out week-2026-07-19.json --per-request week-2026-07-19.parquet
```

On Colab, `pip install huggingface_hub` and an older Python work too. `make setup` asks for `python3.12`; with
another version, run `python -m venv .venv && .venv/bin/pip install -e '.[huggingface]'`.

**Size.** The 16 weeks' tables are 9 GB together (the README of the trace repository; `--tables` says the same).
The weeks differ greatly in size. Request counts per week, from the simulation outputs that ship with the
repository, run from 6,859 (2026-06-07) to 186,844 (2026-08-30). If size grows with request count, one week is
roughly 0.05 to 1.4 GB, and 2026-07-19 (76,513 requests) is about 0.6 GB. That is an estimate: the per-week file
sizes are only on the Hub.

Other forms:

```bash
prefixwise_trace.py data/v1/release                      # every week in the release folder
prefixwise_trace.py data/v1/release --week 2026-07-19 --week 2026-07-26
prefixwise_trace.py WEEK --scope agentic                 # leave out the SWE-bench scaffold (eval:swe-bench)
prefixwise_trace.py --selftest [--trace-repo fi] [--workdir DIR]
```

`--verify` checks each table's sha256 against `tables.json`, as `release/reader.py` does. `--batch-rows`
(default 2,000) sets how many requests' block ids are decoded at a time. pyarrow decodes a row group (10,000
requests, about 100M ids according to `release/convert.py`) at a time, so expect about 1 GB of peak memory. On
synthetic data the scan ran at about 28M block ids per second.

## What is compared

Each session, a subagent's included, is its own conversation. A subagent has its own `session_id`, with steps
from 0, and a `parent_session_id` in `sessions.parquet`. The authors' `transition` is also computed within one
`session_id` against step − 1 (`export/request_inputs/transitions.py`). `requests.parquet` is sorted by
`(session_id, step)`, as `release/trace_lines.py` writes it and `simulation/cheaper_reads.py` relies on; the
script checks the order and stops if it is broken. Block ids are week-local, so comparisons never cross a week.
The script only ever compares within one session, which lives in one week.

| Quantity | Definition |
|---|---|
| shared blocks | the leading `block_ids` the request has in common with step − 1 (the same count as `cheaper_reads.shared_prefixes`, checked in the self-test) |
| `reusable_o200k` | 16 × shared blocks, in o200k tokens; never more than step − 1's full blocks |
| prefix | `intact` when every full block of step − 1 starts this input; `truncated` when this input is a strict prefix of step − 1's; `diverged` otherwise |
| `R` | Prefixwise's reusable prefix, `min(prompt_prev, prompt_curr)` in provider tokens (ALGORITHM 1); `prompt_curr` when step − 1 has no count |
| `reusable` | `reusable_o200k` × (step − 1's provider tokens ÷ its o200k tokens), capped at `R`; equal to `R` when the prefix is intact |
| read | `cache_read_tokens`. **null** means the provider reports no cache count and is never read as 0 (the paper's `is_cache_reporting` is `notna()`). **0** means nothing was read |
| gap | this request's `start_ms` − step − 1's `end_ms` |

**The partial last block.** `export/request_inputs/digests.py` hashes only full 16-token blocks, so the last
`o200k mod 16` tokens of an input never get an id. Up to 15 tokens at the end of step − 1 are invisible. A change
there leaves the prefix "intact". `prev_tail_o200k` records the size of that tail, and the validation counts the
mutations it could hide. For an intact prefix the reusable prefix is `R`, and the 128-token read slack covers the
tail.

**Units.** Block ids count o200k tokens. `prompt_tokens` and `cache_read_tokens` count tokens in the model's own
tokenizer. The script scales the shared o200k tokens by step − 1's ratio, as `simulation/cheaper_reads.py` scales a
step's shared share by its prompt tokens. The trace's simulation never uses the provider's reads: it replays a
cache at 16-token block granularity with no reporting unit. The read slack below therefore comes from
Prefixwise, not from the trace.

**Regions.** The input order is fixed by `export/log_records/messages.py`: the tool definitions entry, then the
top-level system prompt, then the prompt's messages. `role_tokens` sums the o200k tokens of each entry by role, so
the boundaries are at `tool_definitions` and at `tool_definitions + system`, and everything after them is the
conversation. A 16-token block that crosses a boundary is labelled `a~b`, because the first differing token can
be anywhere in the block. Both step − 1's layout (`region`) and this step's (`region_curr_layout`) are recorded.

## The verdict (first rule that matches)

| # | Verdict | Rule | ALGORITHM |
|---|---|---|---|
| 1 | `first_write` | step 0 of a session | 2.1 |
| 2 | `unknown` | step − 1 is missing from the table | — |
| 3 | `switch` | model **or provider letter** differs from step − 1 | 2.1 (`model_switch`) |
| 4 | `unknown` | sent more than 1 s before step − 1 ended, and read < R/2 | 2.4 |
| 5 | `unknown` | no `prompt_tokens` | 2.3 |
| 6 | `hit` (`served_all`) | read ≥ R − min(128, R/8) | 1, served all |
| 7 | `hit` (`half_read`) | prefix intact and read ≥ R/2 | 1 and 2.3, half read |
| 8 | `below_min` | R < 1,024 and nothing read (0 or null) | 2.1, 2.2, 2.3 |
| 9 | `prompt_changed` | prefix diverged or truncated; detail = region of the first differing block, or `truncated` | 2.2 |
| 10 | `unknown` (`no_cache_count`) | read is null | 2.3 |
| 11 | `not_read` | prefix intact, read < R/2; detail = `nothing_read`/`partial_read` and the gap `<5m`, `5-60m`, `>60m` | 2.3 |

Tolerance: the hit test is Prefixwise's own, in two parts. The read slack is one 128-token reporting unit, capped
at an eighth of `R`. Anthropic reports exact counts, but the letters are anonymous, so the slack is applied to
every provider. On an intact prefix, the half-read rule also counts as a hit. On a changed prefix, only a full read
counts, because a read that stops at the change is exactly what a changed prompt produces. The 4th rule follows
Prefixwise's in-flight rule. It is placed after `switch` so that the authors' switches line up with the script's.

**Tokens lost**, in provider tokens. `lost_tokens = max(0, reusable − read)`: the part of the prefix that was still
shared and was not read. `invalidated_tokens = max(0, R − max(reusable, read))`: what the change put beyond reach.
It is non-zero only for `prompt_changed`. Their sum is `max(0, R − read)`, Prefixwise's bound on avoidable tokens
(ALGORITHM 5). A null read gives a null `lost_tokens`, counted as unmeasured. The summary gives both measures as
shares of all prompt tokens.

**Prefixwise's names and legitimacy** (`prefixwise_cause`, `legitimacy`). Each block has a single hash, so
Prefixwise's raw, canonical and masked comparison is not available. A tool change cannot be split into
`tool_reserialize` (waste) and `tool_catalog_change` (legitimate), and a system change cannot be split into
`volatile_system` (waste) and `system_edit` (not counted). Both are `undetermined`. A conversation change is
`history_edit|tool_result_mutation` (avoidable). It becomes `compaction_rewrite` (legitimate) when the authors' cause
is `compaction`, the trace's only record of a compaction event. The `not_read` buckets map as follows:

- `<5m` with nothing read: `routing_miss` (inferred, undetermined, as in classifier.py).
- `5-60m`: `ttl_expired` under a 5-minute TTL, or `routing_miss` under a 1-hour TTL. The TTLs of the anonymous
  providers are unknown, so this bucket is undetermined.
- `>60m`: `ttl_expired` (legitimate).

## Validation against the authors' labels

`validation` in the JSON, and its main lines on stdout:

- `transition_by_prefix`: whether `append` coincides with an intact prefix. It should, exactly: the export
  tokenises each entry on its own and concatenates the tokens, so an unchanged run of entries gives an unchanged
  run of tokens.
- `cause_by_prefix`, `cause_by_region`, `cause_region_fit`: where each mutation cause first diverges. The expected
  fit is `tool_definitions` → tool definitions (in either layout), `system_prompt` → system, and every other cause →
  conversation. A boundary block that includes the expected region counts as `ambiguous`.
  `divergence_share_quartiles_by_cause` gives the position as a share of step − 1's input.
- `disagreements`, with counts and up to 20 `(session_id, step)` examples each:
  - `append_with_diverged_prefix`;
  - `mutation_with_intact_prefix`, by cause, and how many had a partial last block that could hide the change;
  - mutations whose region does not fit their cause;
  - switch label against model/provider;
  - null transitions after step 0, and mutations without a cause.
- `read_beyond_reusable_by_cause`: changed prompts where the provider read more than the shared prefix (plus
  slack and 2%). This happens when the tokenizer scaling is off, when the change never reached the provider's
  rendered prompt, or when the cache was warmed by another request.

## Output

stdout shows the verdict table (requests, lost, invalidated, and the share of prompt tokens), the details of
`prompt_changed`/`not_read`/`unknown`/`hit`, the top causes of lost tokens, the validation lines, and the top 10
providers, harnesses and models. The JSON (`--out`) adds:

- `diagnostics`: null and zero reads, reads above the prompt (this would flag a provider whose `prompt_tokens`
  excludes cached tokens), rows whose block count is not `floor(sum(role_tokens)/16)`, and the tokenizer ratio;
- `prefixwise_cause`, `lost_by_legitimacy`, `top_loss_causes`;
- `prompt_changed_region_by_gap_nothing_read`;
- `by_provider`, `by_model`, `by_harness`, `by_subagent`: verdict counts, lost tokens and their share of prompt
  tokens per value.

`--per-request` writes one row per request with every column above.

## Self-test

`--selftest` writes a synthetic week in exactly the release schema to a temporary folder, removed after. The schema
is copied from `release/schema.py`, along with `manifest.json` and `tables.json` with real sha256s. It builds the
inputs as token lists in the export's order and hashes 16-token blocks with chained BLAKE2b, as `digests.py` does.
It numbers the hashes as week-local ids, as `requests_table.py` does. It then checks 29 verdicts and 8 token values:

- append hit;
- `not_read` in each gap bucket;
- system, tool-definition, tool-result and compaction mutations at known positions;
- a model switch and a provider switch;
- the half-read rule;
- a subagent below the minimum;
- a null-count provider with a dropped turn and a truncation;
- a provider whose tokenizer counts 1.25× o200k (scaling checked);
- a change in a block across the tool/system boundary;
- an in-flight request;
- one `append` label with a diverged prefix and one `mutation` hidden in the partial last block. The validation must
  count exactly one of each.

With `--trace-repo` pointing at a clone of the trace repository, it also checks four things:

- the schema copy equals `release/schema.py`;
- `release/reader.py` loads the synthetic week and checks it;
- `simulation/cheaper_reads.shared_prefixes` counts the same shared blocks as this script;
- the lines above all pass.

Last run: `self-test: PASSED (29 verdicts, 8 token checks)`, with all five trace-repository checks ok (the three
schemas, the reader, the shared blocks), under pyarrow 26.0.1, pandas 3.0.5, numpy 2.5.3 and Python 3.13.

## Assumptions the trace code does not settle

1. **`role_tokens.system` is contiguous after the tool definitions.** It counts the top-level system prompt and every
   `system`/`developer` message (`rendering.role_of`). A system or developer message later in the conversation (an
   agent's reminder) is counted in `system` but sits in the conversation. In that case the system region is drawn
   too wide, and the conversation starts too late. The authors' `system_prompt` cause also covers such messages
   ("the earlier message at the break is the system prompt", which tests the role). Some `system_prompt` mutations
   may therefore fall in the conversation region. This is a known source of `disagrees`, not an error in either.
2. **Provider tokens scale evenly with o200k tokens across the input.** The reusable prefix in provider tokens is
   `reusable_o200k × prompt_prev / o200k_prev`. JSON tool definitions and prose tokenise differently, and provider
   overheads (chat-template tokens, Anthropic's tool-use preamble) are spread evenly by this scaling. That is only
   approximate. It matters for `lost_tokens` on changed prompts and for `read_beyond_reusable`. It does not matter
   for the hit test, which uses only provider counts.
3. **`prompt_tokens` includes the cached tokens** on every provider. The paper divides `cache_read_tokens` by
   `prompt_tokens` and clips at 1. The export takes `prompt_tokens` from the gateway record first, then `usage`. For
   Anthropic, `usage.input_tokens` excludes cache reads. `diagnostics.requests_read_over_prompt` (overall, and per
   provider in `by_provider`) shows whether some provider breaks this.
4. **The read slack is 128 tokens for every provider**, because the letters are anonymous. This is lenient by up to
   128 tokens for a provider with exact counts (Anthropic).
5. **The minimum cacheable prefix is 1,024 tokens for every provider.** Some Anthropic models have 2,048 or 4,096 and
   some engines have none. The letters do not say which is which.
6. **Each provider letter has a single cache**, so `not_read` within 5 minutes is read as routing or eviction. The
   trace cannot show where requests were routed. Each request is compared only with its own session's step − 1, so
   a read served by another session's (or a sibling subagent's) write shows up as `read_beyond_reusable` or as a
   `first_write` that read at least half (`diagnostics.first_write_reading_at_least_half`).
7. **A provider change with the same model is a switch.** Prefixwise names only `model_switch`, but different
   upstreams do not share a cache, and the authors also call this a `switch`. Provider letters come from one
   private map (`export/privacy/pseudonyms.py`), so they are consistent across weeks.
8. **Sessions at a week boundary.** The export routes each log record to the week of its completion time
   (`export/log_records/log_files.py`, `fields.week_of(end_ms)`) and rebuilds sessions inside each week. A
   conversation that runs across Sunday 00:00 UTC is therefore most likely split. Its later part would appear in
   the next week as a separate session with its own step 0. Its first request there is a `first_write` that may
   read a warm cache (counted in `first_write_reading_at_least_half`). The script never compares across weeks
   (block ids are week-local). The split itself is inferred from the code, not observed.
9. **`start_ms` is the gateway's arrival time**: `end_ms − latency_ms` (`requests_table.py`). The in-flight rule and
   the gap use it. Gateway queueing is not visible.
10. **Synthetic only.** The Hub returns 403 from this machine, so every number above comes from synthetic rows that
    follow the schema. None of it comes from the trace itself.
