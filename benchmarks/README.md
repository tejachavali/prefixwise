# Benchmarks

`bench.py` measures what Prefixwise costs you in production: added latency,
throughput, memory, behaviour over time, and what happens when things break.
No API keys and no network: the real `prefixwise serve` runs as its own process
in front of a fake model API (`upstream.py`, a third process), with an on-disk
ledger, and is driven with realistic agent sessions: a long system prompt and
eight tools, then one turn (user message, assistant reply) appended per request.

```bash
python benchmarks/bench.py --quick                 # ~3 minutes
python benchmarks/bench.py                         # ~10 minutes
python benchmarks/bench.py --soak 1800             # with a 30-minute soak
python benchmarks/bench.py --only faults           # one part
```

It writes `benchmarks/runs/<date>-<host>.md` (tables) and `.json` (every
sample), and exits non-zero if any request errored or any fault case failed.
CI runs the fault suite on every push.

| part | what it measures | pass means |
|---|---|---|
| overhead | proxy latency minus direct latency to the same upstream, same request, at 2k, 30k and 120k-token prompts, streaming and not, OpenAI and Anthropic shapes; also time to first byte | small next to model latency |
| throughput | requests/s and p50/p99 at concurrency 1-128, with an instant upstream (the proxy's own ceiling) and a 300 ms one | 0 errors, every request in the ledger |
| memory | RSS over 6,000 sessions, with the default session cap and a cap of 1,000 | RSS plateaus at the cap |
| soak | mixed traffic for a fixed time: both providers, a third streamed, sessions of 5-25 turns, task outcomes posted; RSS, threads and fds sampled | 0 errors, RSS flat once warm, no thread or fd growth |
| faults | upstream 500 / 429 / dropped connection / HTML body / no usage / slow / drop mid-stream; client invalid JSON / non-object / oversize / disconnect mid-stream / 300 idle or half-sent sockets | the right status, and the proxy healthy after each |

What it does not measure: real provider latency (the overhead is additive, so
it carries over), or the classifier's accuracy. Accuracy is E1:
`experiments/e1_open` (vLLM, injected causes) and `experiments/e1_hosted`
(agreement with Anthropic's own diagnostics).

Results are machine-dependent; the fake upstream and the load generator share
the CPUs with the proxy, so throughput figures are a floor.
