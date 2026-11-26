# Dogfood 3: every verdict checked on a real Claude Code session of 10,517 calls

Run on 2026-11-26, code at `df379b3` (0.12.7), then again with the fixes this record led to.

**The data.** The Claude Code logs of the machine Prefixwise is developed on: one long session
(20 days, 2026-11-06 to 2026-11-26, a 254 MB log), 160 subagent logs, and a second copy of part of
the session under another project folder. 10,517 distinct API calls, 3.97 billion prompt tokens, on
Claude Opus 5.5, Fable 5.1, Fable 5.2 and Haiku 5.5. Claude Code 2.1.x. The logs are private: this
record holds counts, causes, token numbers and gaps only.

**The method.** An independent auditor wrote its own extractor for the raw logs, froze a copy,
ran `prefixwise import claude-code` on it, and decided for every row (not a sample) whether the
verdict was right, from the token numbers, the gaps and the events the log shows. It also compared
each verdict with Anthropic's own cache diagnosis, which Claude Code requests and logs.

## What held

- **Complete.** 21,723 log lines carry usage; they collapse to 10,517 distinct responses, and the
  ledger has 10,517 rows. Tokens and cost match the extractor on every row.
- **Fast.** 3 to 4 s for the whole import (14 s the first time, cold disk), 47 MB of memory at
  peak. No crash, nothing on stderr. Importing again adds nothing.
- **Subagents kept apart.** Each of the 160 has its own session; a fork that read all 327,722
  tokens its parent left is a hit.
- **No false alarm on a hit.** No row called a miss had read nearly everything: the most any miss
  read was 43% of what the previous call left.
- **Private.** The ledger holds the session id and no account, device or organisation id, no
  text, no paths.

## What was wrong, and is fixed

| Cause given (0.12.7) | Rows | Right | Wrong | What the wrong ones were |
|---|---:|---:|---:|---|
| `none` | 10,275 | 10,258 | 17 | 16 compactions and 1 model switch |
| `first_write` | 147 | 147 | 0 | |
| `ttl_expired` | 80 | 80 | 0 | |
| `system_edit` | 8 | 0 | 8 | expiries: each came after more than an hour idle |
| `model_switch` | 5 | 5 | 0 | |
| `unknown` | 2 | | 2 | both 5-minute expiries |

1. **The front of the prompt, read from the cache whatever the conversation did, hid 17
   misses.** About 60,000 tokens at the front of this session's prompt (its system prompt and
   tools, by position) were read on the very first call and after 21 hours idle. What kept them in
   the cache is not in these logs; other sessions with the same system prompt and tools would. The
   hit test asked whether a request read half of the smaller of the two prompts. After a
   compaction the prompt is about 100,000 tokens, so reading the 60,000 passed as a hit, though
   none of the 900,000 tokens of conversation came back (16 of 16 compactions). One model switch
   right after a compaction passed the same way. The test now asks whether the request read what
   the previous request on the same model left in the cache; if it read less, after a compaction,
   an expiry or a model switch, that is the cause.
2. **Eight expiries were named "Claude Code was upgraded".** The upgrade check ran before the
   lifetime check. In all eight, more than an hour had passed (3,788 s to 88,446 s), so the cache
   was gone either way. Three other upgrades fell inside the cache lifetime and were full hits:
   here an upgrade did not break the cache. An expiry now comes first, and names the upgrade as
   context.
3. **Two expiries were `unknown`, because the send time was read from the wrong entry.** Claude
   Code writes a `deferred_tools_record` entry when a response arrives (stamped within 50 ms of
   it, on all 2,905 in the main conversation), and the importer took the last entry before a
   response as the time the request was sent. For 4,388 of the 10,850 requests as logged (the 10,517 distinct ones, some in two files), 40%, the
   "sent" time was therefore the response time. Two subagent calls sent 318 s and 319 s after the
   previous one, past the 5-minute lifetime, read as 292 s and 153 s. The same reading gave 32
   rows a negative gap (a message queued while the previous response streamed is stamped when it
   was queued). The send time is now the latest input stamped before the response (a user entry,
   or another entry stamped at least 0.1 s before it), or the end of the previous response if that
   is later.

With the fixes, on the same frozen logs:

| Cause | Rows |
|---|---:|
| `none` | 10,258 |
| `first_write` | 147 |
| `ttl_expired` | 90 |
| `compaction_rewrite` | 16 |
| `model_switch` | 6 |
| `system_edit`, `unknown` | 0 |

Every row now matches the auditor's own reading. No gap is negative. The 10,258 `none` rows are
10,245 that read all the previous request left, and 13 first calls that read an entry written
elsewhere.

**The first version of the fix was itself reviewed, and was worse than before in three cases**,
all corrected before it was merged:

- A system prompt change, an effort change or an upgrade was named even when the read went far
  past the front of the prompt, where those sit. They are now named only when under half was read.
- The previous request was taken whatever its model, so switching back to a model whose cache
  was intact read as a miss. Each request is now compared with the previous one on its own model.
- One entry stamped after a response made the importer discard that request's real input.

## Anthropic's own diagnosis, on the same calls

88 calls carry one. It is never set on a hit (0 of 10,245), nor on a first call (0 of 160).

| Anthropic says | Calls | Prefixwise (fixed) | |
|---|---:|---|---|
| `previous_message_not_found` | 74 | `ttl_expired` | Prefixwise names the cause: every one came after more than an hour idle (3,687 s at least) |
| `model_changed` | 6 | `model_switch` | agree |
| `tools_changed` | 5 | `compaction_rewrite` | the log shows a compaction before each; Prefixwise is more specific |
| `system_changed` | 1 | `compaction_rewrite` | same |
| `messages_changed` | 1 | `compaction_rewrite` | same |
| `unavailable` | 1 | `compaction_rewrite` | Anthropic gives no cause |

- On this session, 75 of the 88 diagnoses name no cause (74 `previous_message_not_found`, 1
  `unavailable`). For the 74, the gap gives it.
- 24 misses carry no diagnosis at all: 16 subagent expiries on the 5-minute cache (the 2 that
  were `unknown` among them) and the 8 earliest compactions.
- `cache_missed_input_tokens` is not the tokens that were rewritten. On the `tools_changed` rows
  it is nearly the whole prompt, though about 62,000 tokens were read. It looks like the tokens
  after the point where the request differs.

## What this session lost, and why

98.2% of prompt tokens came from the cache (main conversation 98.6%, subagents 95.3%). At list
prices the bill is 45% cache reads, 40% cache writes and 15% output. Tokens that were written
again though they had been cached are about a quarter of the bill:

| True cause | Calls | Share of the bill |
|---|---:|---:|
| Main conversation resumed after more than an hour idle | 72 | about 21% |
| Model switches | 6 | about 1.4% |
| Subagents resumed after more than 5 minutes | 18 | about 1.4% |
| Compactions | 16 | about 0.3% |

The 72 resumes carried 519,000 tokens of context on average. None of this is a defect in a
prompt, so the report rightly shows no avoidable waste. But it shows no figure for these misses
either, and they are the one thing here a user can change (compact, or start a new session,
before a long break). Showing what expected misses cost is tracked as a separate change.

**After that change** (the same frozen logs, imported again): the report's expected-misses line reads
`90 x too long since the previous turn: the cache entry expired (36,080,062 tokens the cache had held
were sent again, [dollars] more than reading them, 23% of the bill)`. The 90 are the 72 and the 18 in
the table above, and the importer's 22.9% is close to their 21% and 1.4% (the table was worked out by
hand from the raw logs). The token count is what was not read: it is less than 90 contexts of the
average size above, because the front of the prompt was still read on many of these turns, as noted
earlier in this record. The figure is gross: what was paid again, not what a longer cache lifetime
would have saved (72 of the 90 came after more than an hour, longer than any stated lifetime). The
other rows of the table get no figure: after a compaction or a model switch, how much could have been
read is not known in tokens.

## Limits

- **One machine, one user, one kind of work.** Rules tuned on it may not hold elsewhere. The
  share that counts as a full reuse (95% of what the previous call left) rests on these 10,517
  calls: hits read 100% or more, misses 56% or less, nothing in between.
- **What the rules still cannot see.** A write with both lifetimes (a 1-hour prefix and a
  5-minute tail) is taken as 1 hour, so a tail that expired after ten minutes is not named. A
  first request on a model that reads its whole prompt is a hit, and one that reads less is the
  switch's, with no way to tell how much was reusable. A response that is the sum of several
  model calls is judged only by whether half was read.
- **Rows imported by an earlier version keep their verdicts** (their logs may be gone). The
  import says how many, and a new ledger classifies them again.
- **Claude Code does not log every call.** Its own counter reports more cache reads than the logs
  hold: 84% to 87% of reads are in the logs, by model, and 98.5% to 99.9% of writes. Sixteen
  compaction summaries are not logged, and likely one call after each finished turn. An import
  cannot see those.
- **`claude-fable-5-2` was priced as `claude-fable-5-1`**, 25% high, and said so. Fixed
  separately.
- The auditor's per-row files hold private content and are not kept in the repository.
