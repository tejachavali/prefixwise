# Live overhead: OpenAI, direct vs. through Prefixwise

Run by Muse on 2026-11-07 from Colab, `main` at 0ac3cf8, `python benchmarks/live_overhead.py
--provider openai` (about 48 s). Key from Colab Secrets; no code changed. Transcribed by Claude
from Muse's report.

provider openai, model gpt-5.6-luna, 20 pairs (each a ~4k-token prefix, `reasoning_effort: none`)

| | result |
|---|---|
| cache hit share | direct **97.9%**, proxy **97.9%**: **MATCH** |
| ledger | **20/20** proxied requests recorded |
| added latency | median **+43 ms**, p90 +347 ms (direct median 1,093 ms) |

**Reading it.** As on Anthropic, the proxy does not change what the provider can reuse. The
latency figure is dominated by network variance between the two separate round trips from
Colab, about a second each; the proxy's own cost at this size, measured without a network in
`2026-11-07-vm.md`, is 3-6 ms. The median is about 4% of a call; the p90 is single-request
jitter.

**First attempt** (same day) was rejected by OpenAI on the direct request: GPT-5.6 Luna on
`/v1/chat/completions` refuses function tools unless `reasoning_effort` is `none`. A script
bug, fixed in PR #27.
