# E-template, end to end: vLLM, real agents, SGLang. Colab L4, 2026-11-09

Run by Muse on a fresh L4 runtime, main at e49afcd (PRs #73 and #74 merged: the block-rounded hit
and the engine block size on the record), Prefixwise 0.9.0 plus those, vLLM 0.33.0, SGLang 0.5.23,
opencode 1.19.5, Claude Code 2.1.319, Qwen3-4B. Each engine run: 5 variants x 6 turns, KV events on.
No code changed, nothing pushed from Colab, no key printed. Transcribed by Claude from Muse's
report. The first run, on the code before those PRs, is [2026-11-09-colab-l4.md](2026-11-09-colab-l4.md).

## 1. vLLM 0.33.0 + KV events (block size 16)

| variant | turn | prompt | cached | completion | prev_prompt | prev_full | reuse | lost tail | Prefixwise |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| kept | 1 | 403 | 0 | 573 |  |  | first | 0 | `first_write` |
| kept | 2 | 510 | 400 | 348 | 403 | 976 | prompt | 576 | `none` |
| kept | 3 | 545 | 496 | 1500 (truncated) | 510 | 858 | prompt | 362 | `none` |
| kept | 4 | 2062 | 2032 | 1500 (truncated) | 545 | 2045 | full | 0 | `none` |
| kept | 5 | 3580 | 3552 | 802 | 2062 | 3562 | full | 0 | `none` |
| kept | 6 | 3658 | 3568 | 245 | 3580 | 4382 | prompt | 814 | `none` |
| stripped | 1 | 404 | 0 | 597 |  |  | first | 0 | `first_write` |
| stripped | 2 | 497 | 400 | 254 | 404 | 1001 | prompt | 601 | `none` |
| stripped | 3 | 542 | 496 | 474 | 497 | 751 | prompt | 255 | `none` |
| stripped | 4 | 599 | 528 | 269 | 542 | 1016 | prompt | 488 | `none` |
| stripped | 5 | 662 | 592 | 557 | 599 | 868 | prompt | 276 | `none` |
| stripped | 6 | 741 | 656 | 231 | 662 | 1219 | prompt | 563 | `none` |
| nothink | 1 | 408 | 0 | 60 |  |  | first | 0 | `first_write` |
| nothink | 2 | 482 | 400 | 26 | 408 | 468 | prompt | 68 | `none` |
| nothink | 3 | 522 | 464 | 33 | 482 | 508 | prompt | 44 | `none` |
| nothink | 4 | 571 | 512 | 26 | 522 | 555 | prompt | 43 | `none` |
| nothink | 5 | 614 | 560 | 30 | 571 | 597 | prompt | 37 | `none` |
| nothink | 6 | 661 | 608 | 55 | 614 | 644 | prompt | 36 | `none` |
| toggle | 1 | 407 | 0 | 67 |  |  | first | 0 | `first_write` |
| toggle | 2 | 484 | 400 | 195 | 407 | 474 | prompt | 74 | `none` |
| toggle | 3 | 528 | 480 | 38 | 484 | 679 | prompt | 199 | `none` |
| toggle | 4 | 578 | 512 | 275 | 528 | 566 | prompt | 54 | `none` |
| toggle | 5 | 645 | 576 | 27 | 578 | 853 | prompt | 277 | `none` |
| toggle | 6 | 685 | 640 | 209 | 645 | 672 | prompt | 32 | `none` |
| tools | 1 | 546 | 0 | 160 |  |  | first | 0 | `first_write` |
| tools | 2 | 597 | 544 | 85 | 546 | 706 | prompt | 162 | `none` |
| tools | 3 | 621 | 592 | 184 | 597 | 682 | prompt | 90 | `none` |
| tools | 4 | 668 | 608 | 179 | 621 | 805 | prompt | 197 | `none` |
| tools | 5 | 703 | 656 | 225 | 668 | 847 | prompt | 191 | `none` |
| tools | 6 | 747 | 688 | 272 | 703 | 928 | prompt | 240 | `none` |

25 turns with a previous turn: reuse full 2, prompt-only 23, break 0, cold 0. Previous output
re-prefilled (lost tail): **5,679 tokens, 25% of those turns' prompt tokens**. Prefixwise:
`first_write` x5, `none` x25; `unknown` ("not an eviction") **0**; truncated answers 2. KV stream:
619 batches, 2,117 blocks stored, 0 removed, 0 dropped, 0 decode errors.

## 2. Real agents against the same vLLM, through `prefixwise run`

opencode (config: an OpenAI-compatible provider whose base URL is `{env:OPENAI_BASE_URL}`, model
`Qwen/Qwen3-4B`) and Claude Code (`ANTHROPIC_BASE_URL` at vLLM, which serves `/v1/messages`;
`ANTHROPIC_MODEL` and the small-model variables set to `Qwen/Qwen3-4B`), each run twice with
"write a function that reverses a string, and a test for it" in an empty git repository. Claude
Code warned that it does not know the model and assumed a 200k context; both agents ran.

```
8 requests, 53,405 prompt tokens, 64% served from cache.
No avoidable waste found among the misses whose cause is known.
Cause not determined for 1 request.
Expected misses (not waste): 4 x first request of a session.

cause         requests  prompt   cached  avoidable
first_write          4  23,366   49.9%          0
none                 3  22,817   98.7%          0
unknown              1   7,222    0.0%          0
```

`grep -c "not an eviction"` on the export: **0**. Per agent: opencode run 1, 3 requests, 15,509
prompt tokens, 49% cached; run 2, 3 requests, 15,626 tokens, 99% cached (the engine's cache is
global: the second session read the first's prefix). Claude Code: 1 request per run, 11,135 tokens,
0% then 100% cached. The `unknown` is an opencode request that read nothing: the proxy `run` started
had no engine join (it did not probe vLLM as `serve` does), so the KV events that would have named
it as an eviction or ruled one out were not read. Fixed after this run (`prefixwise run` now
detects the engine behind the agent's base URL); the row's evidence is requested from Muse.

## 3. SGLang 0.5.23 + KV events (page size 1, `--full-attention`)

| variant | turn | prompt | cached | completion | prev_prompt | prev_full | reuse | lost tail | Prefixwise |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| kept | 1 | 403 | 0 | 546 |  |  | first | 0 | `first_write` |
| kept | 2 | 547 | 403 | 241 | 403 | 949 | prompt | 546 | `none` |
| kept | 3 | 587 | 547 | 444 | 547 | 788 | prompt | 241 | `none` |
| kept | 4 | 641 | 587 | 230 | 587 | 1031 | prompt | 444 | `none` |
| kept | 5 | 684 | 641 | 350 | 641 | 871 | prompt | 230 | `none` |
| kept | 6 | 747 | 684 | 258 | 684 | 1034 | prompt | 350 | `none` |
| stripped | 1 | 404 | 4 | 507 |  |  | first | 0 | `first_write` |
| stripped | 2 | 526 | 404 | 160 | 404 | 911 | prompt | 507 | `none` |
| stripped | 3 | 568 | 526 | 1500 (truncated) | 526 | 686 | prompt | 160 | `none` |
| stripped | 4 | 2085 | 2067 | 1500 (truncated) | 568 | 2068 | full | 0 | `none` |
| stripped | 5 | 2130 | 2085 | 1500 (truncated) | 2085 | 3585 | prompt | 1500 | `none` |
| stripped | 6 | 2160 | 2130 | 320 | 2130 | 3630 | prompt | 1500 | `none` |
| nothink | 1 | 408 | 4 | 60 |  |  | first | 0 | `first_write` |
| nothink | 2 | 482 | 404 | 26 | 408 | 468 | prompt | 64 | `none` |
| nothink | 3 | 522 | 478 | 33 | 482 | 508 | prompt | 30 | `none` |
| nothink | 4 | 571 | 518 | 25 | 522 | 555 | prompt | 37 | `none` |
| nothink | 5 | 613 | 567 | 29 | 571 | 596 | prompt | 29 | `none` |
| nothink | 6 | 659 | 609 | 50 | 613 | 642 | prompt | 33 | `none` |
| toggle | 1 | 407 | 3 | 59 |  |  | first | 0 | `first_write` |
| toggle | 2 | 476 | 403 | 165 | 407 | 466 | prompt | 63 | `none` |
| toggle | 3 | 514 | 476 | 27 | 476 | 641 | prompt | 165 | `none` |
| toggle | 4 | 553 | 510 | 236 | 514 | 541 | prompt | 31 | `none` |
| toggle | 5 | 615 | 553 | 45 | 553 | 789 | prompt | 236 | `none` |
| toggle | 6 | 673 | 611 | 225 | 615 | 660 | prompt | 49 | `none` |
| tools | 1 | 551 | 4 | 179 |  |  | first | 0 | `first_write` |
| tools | 2 | 602 | 551 | 85 | 551 | 730 | prompt | 179 | `none` |
| tools | 3 | 626 | 602 | 185 | 602 | 687 | prompt | 85 | `none` |
| tools | 4 | 669 | 626 | 241 | 626 | 811 | prompt | 185 | `none` |
| tools | 5 | 703 | 669 | 240 | 669 | 910 | prompt | 241 | `none` |
| tools | 6 | 751 | 703 | 259 | 703 | 943 | prompt | 240 | `none` |

25 turns: reuse full 1, prompt-only 24, break 0, cold 0. Previous output re-prefilled: **7,145
tokens, 36% of those turns' prompt tokens** (three 1,500-token truncations inflate it). Prefixwise:
`first_write` x5, `none` x25; `unknown` **0**; truncated 3. KV stream: 34 batches, 12,971 blocks
stored, 0 removed, 0 dropped. The 3 or 4 tokens cached on first turns are the chat template's
opening, shared with the other variants' sessions in SGLang's global radix tree.

## What it shows

1. **"Not an eviction" did not occur: 0 of 50 harness turns on two engines, 0 of 8 agent
   requests.** On Qwen3 the token prefix is stable; the divergence always sits at the boundary
   between the previous prompt and its output, which Prefixwise never expected back.
2. **Prompt-only reuse is the rule on a reasoning model: 47 of 50 turns.** The previous output is
   prefilled again every turn, in the form the template renders it (reasoning dropped): 25% of the
   prompt tokens on vLLM, 36% on SGLang. Prefixwise said `none` on all of them, which is right
   (nothing the client did, no eviction), but said nothing about the tail. That is now the detail
   `output_rerendered`, added after this run and sized from it (claimed from 160 output tokens,
   where the rounding of reads cannot hide a reused output).
3. **Truncated reasoning stays in history.** The 3 turns that reused prompt *plus* output follow a
   1,500-token truncation: with no closing `</think>`, Qwen3's template keeps the whole unfinished
   thinking as content, so the prompt jumps by 1,500 tokens and is then reused in full. A session
   that truncates a reasoning model's output pays for that thinking on every later turn.
4. **The exact shortfall, from SGLang's page size of 1.** A full read stops exactly at the
   previous prompt (`kept`: 403 of 403, 547 of 547) or 4 tokens before it when thinking was off
   (`nothink`, `toggle`: the `<think>\n\n</think>\n\n` stub, which the rendered history never
   repeats). The 25-token shortfall in the first run's table (`tools` turn 6) does not recur; every
   vLLM read here is the previous prompt minus that stub, rounded down to a 16-token block.
5. **The two fixes hold live.** `toggle` is `none` on all 10 turns (it was `param_rendered` on the
   first run); the engine's block size was on every record.
6. **`prefixwise run` needed the engine join.** Its proxy did not probe the upstream, so the one
   agent miss it could not name was `unknown` where `serve` would have read the KV events. Fixed.
