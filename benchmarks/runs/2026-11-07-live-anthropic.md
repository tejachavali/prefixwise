# Live overhead: Anthropic, direct vs. through Prefixwise

Run by Muse on 2026-11-07 from Colab, `main` at 0ff0c70, `python benchmarks/live_overhead.py
--provider anthropic`. Key from Colab Secrets; no code changed. Transcribed by Claude from
Muse's report.

provider anthropic, model claude-haiku-5-5, 20 pairs (each a ~4k-token cached prefix, max_tokens 16)

| | result |
|---|---|
| cache hit share | direct **98.1%**, proxy **98.1%**: **MATCH** |
| ledger | **20/20** proxied requests recorded |
| added latency | median **+29 ms**, p90 +397 ms (direct median 1,009 ms) |

**Reading it.** The proxy does not change what Anthropic can reuse: both sides were cached
identically. The latency figure is dominated by network variance, not by the proxy: each
pair is two separate round trips from Colab to Anthropic, about a second each, and the
order alternates. The proxy's own cost at this prompt size, measured without a network in
`2026-11-07-vm.md`, is 3-6 ms. A median of +29 ms on ~1 s calls is about 3%; the p90 is
single-request jitter. More pairs, or the same network path, would narrow it.

**First attempt** (same day) stopped at pair 5 on a bug in the script (cache breakpoints
accumulated in the session history; fixed in PR #26). Pairs 1-4 there: both sides 98% cached.

**OpenAI:** passed the same check after a script fix; see `2026-11-07-live-openai.md`.
