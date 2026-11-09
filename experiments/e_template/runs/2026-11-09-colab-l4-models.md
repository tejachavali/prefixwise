# E-template on three more templates: Llama 3.2, Qwen2.5, DeepSeek-R1 distill. Colab L4, 2026-11-09

Run by Muse on main at 19c9a2e (0.10.0), vLLM 0.33.0, KV events on, 5 variants x 6 turns per model (Llama 3.2: template alone), one
server at a time. No code changed, nothing pushed from Colab, no key printed. Transcribed by Claude from
Muse's report. The Qwen3-4B runs this extends: [2026-11-09-colab-l4-rerun.md](2026-11-09-colab-l4-rerun.md).

| model | template keeps history as generated? | reuse (25 turns) | previous output re-prefilled | `output_rerendered` |
|---|---|---|---|---|
| Qwen/Qwen3-4B (earlier run) | no: drops `<think>` from earlier turns | prompt-only 23-24, full 1-2 | 25% (vLLM), 36% (SGLang) | would name 29 of 47 (added after) |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | no: drops everything up to `</think>`, puts `<think>\n` in the generation prompt | **prompt-only 25 of 25** | **57%** (17,987 tokens) | **25 of 25**, live |
| Qwen/Qwen2.5-3B-Instruct | yes | **full 25 of 25** | 0 | 0 of 25, live |
| unsloth/Llama-3.2-3B-Instruct | yes (template alone; vLLM did not start, below) | full (template alone) | 0 | none (template alone) |

"Not an eviction" (`unknown`): 0 on every model that ran. No block removed on either stream.

## DeepSeek-R1-Distill-Qwen-1.5B (vLLM, block 16)

| variant | turn | prompt | cached | completion | prev_prompt | prev_full | reuse | lost tail | Prefixwise |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| kept | 1 | 395 | 0 | 572 |  |  | first | 0 | `first_write` |
| kept | 2 | 529 | 384 | 206 | 395 | 967 | prompt | 583 | `none` [output_rerendered] |
| kept | 3 | 556 | 512 | 711 | 529 | 735 | prompt | 223 | `none` [output_rerendered] |
| kept | 4 | 935 | 544 | 467 | 556 | 1267 | prompt | 723 | `none` [output_rerendered] |
| kept | 5 | 1074 | 928 | 520 | 935 | 1402 | prompt | 474 | `none` [output_rerendered] |
| kept | 6 | 1286 | 1072 | 811 | 1074 | 1594 | prompt | 522 | `none` [output_rerendered] |
| stripped | 1 | 396 | 0 | 608 |  |  | first | 0 | `first_write` |
| stripped | 2 | 546 | 384 | 256 | 396 | 1004 | prompt | 620 | `none` [output_rerendered] |
| stripped | 3 | 574 | 544 | 405 | 546 | 802 | prompt | 258 | `none` [output_rerendered] |
| stripped | 4 | 728 | 560 | 1500 (truncated) | 574 | 979 | prompt | 419 | `none` [output_rerendered] |
| stripped | 5 | 2239 | 720 | 1500 (truncated) | 728 | 2228 | prompt | 1508 | `none` [output_rerendered] |
| stripped | 6 | 3750 | 2224 | 1500 (truncated) | 2239 | 3739 | prompt | 1515 | `none` [output_rerendered] |
| nothink | 1 | 396 | 0 | 484 |  |  | first | 0 | `first_write` |
| nothink | 2 | 510 | 384 | 447 | 396 | 880 | prompt | 496 | `none` [output_rerendered] |
| nothink | 3 | 547 | 496 | 585 | 510 | 957 | prompt | 461 | `none` [output_rerendered] |
| nothink | 4 | 634 | 544 | 357 | 547 | 1132 | prompt | 588 | `none` [output_rerendered] |
| nothink | 5 | 720 | 624 | 1500 (truncated) | 634 | 991 | prompt | 367 | `none` [output_rerendered] |
| nothink | 6 | 2231 | 704 | 1500 (truncated) | 720 | 2220 | prompt | 1516 | `none` [output_rerendered] |
| toggle | 1 | 395 | 0 | 504 |  |  | first | 0 | `first_write` |
| toggle | 2 | 496 | 384 | 292 | 395 | 899 | prompt | 515 | `none` [output_rerendered] |
| toggle | 3 | 527 | 480 | 1500 (truncated) | 496 | 788 | prompt | 308 | `none` [output_rerendered] |
| toggle | 4 | 2037 | 512 | 1500 (truncated) | 527 | 2027 | prompt | 1515 | `none` [output_rerendered] |
| toggle | 5 | 3548 | 2032 | 1500 (truncated) | 2037 | 3537 | prompt | 1505 | `none` [output_rerendered] |
| toggle | 6 | 5059 | 3536 | 481 | 3548 | 5048 | prompt | 1512 | `none` [output_rerendered] |
| tools | 1 | 404 | 0 | 583 |  |  | first | 0 | `first_write` |
| tools | 2 | 485 | 400 | 271 | 404 | 987 | prompt | 587 | `none` [output_rerendered] |
| tools | 3 | 519 | 480 | 265 | 485 | 756 | prompt | 276 | `none` [output_rerendered] |
| tools | 4 | 598 | 512 | 458 | 519 | 784 | prompt | 272 | `none` [output_rerendered] |
| tools | 5 | 709 | 592 | 755 | 598 | 1056 | prompt | 464 | `none` [output_rerendered] |
| tools | 6 | 855 | 704 | 799 | 709 | 1464 | prompt | 760 | `none` [output_rerendered] |

25 turns: prompt-only 25, full 0, break 0, cold 0. Previous output re-prefilled: 17,987 tokens, 57% of
those turns' prompt tokens. Prefixwise: `first_write` x5, `none` x25, every `none` with
`output_rerendered`; `unknown` 0; truncated answers 8. KV stream: 1,456 batches, 3,517 blocks stored, 0
removed, 0 dropped.

Template alone (CPU): turn 2's prompt shares 66 tokens of turn 1's 68-token prompt (and none of its output), for
`kept` and `stripped` alike: the template renders earlier assistant turns as `<｜Assistant｜>` + content +
`<｜end▁of▁sentence｜>`, and the generation prompt's `<think>\n` (2 tokens) is never repeated. It ignores
`enable_thinking` (tail empty) and renders no date. The harness's template-alone mode read that as
`break` (no slack); it now allows the generation tail, as the live classification does.

What is specific to this template: the model's answer begins after `<think>\n`, so the content the
client appends never contains the opening tag, and when a truncated answer (no `</think>`) is kept whole
as content (`stripped` 4 to 6, `toggle` 3 to 6: prompts of 2,000 to 5,000 tokens), it is prefilled once
more as history (lost tail about 1,500) before it is part of the reused prefix. R1-style answers are
long: the 57% is the share of every prompt that was the previous answer, prefilled again.

## Qwen2.5-3B-Instruct (vLLM, block 16)

| variant | turn | prompt | cached | completion | prev_prompt | prev_full | reuse | lost tail | Prefixwise |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| kept | 1 | 403 | 0 | 52 |  |  | first | 0 | `first_write` |
| kept | 2 | 469 | 448 | 25 | 403 | 455 | full | 0 | `none` |
| kept | 3 | 508 | 480 | 28 | 469 | 494 | full | 0 | `none` |
| kept | 4 | 552 | 528 | 23 | 508 | 536 | full | 0 | `none` |
| kept | 5 | 592 | 560 | 38 | 552 | 575 | full | 0 | `none` |
| kept | 6 | 647 | 624 | 25 | 592 | 630 | full | 0 | `none` |
| stripped | 1 | 404 | 0 | 50 |  |  | first | 0 | `first_write` |
| stripped | 2 | 468 | 448 | 18 | 404 | 454 | full | 0 | `none` |
| stripped | 3 | 500 | 480 | 27 | 468 | 486 | full | 0 | `none` |
| stripped | 4 | 543 | 512 | 32 | 500 | 527 | full | 0 | `none` |
| stripped | 5 | 592 | 560 | 19 | 543 | 575 | full | 0 | `none` |
| stripped | 6 | 628 | 608 | 31 | 592 | 611 | full | 0 | `none` |
| nothink | 1 | 404 | 0 | 50 |  |  | first | 0 | `first_write` |
| nothink | 2 | 468 | 448 | 18 | 404 | 454 | full | 0 | `none` |
| nothink | 3 | 500 | 480 | 27 | 468 | 486 | full | 0 | `none` |
| nothink | 4 | 543 | 512 | 32 | 500 | 527 | full | 0 | `none` |
| nothink | 5 | 592 | 560 | 19 | 543 | 575 | full | 0 | `none` |
| nothink | 6 | 628 | 608 | 31 | 592 | 611 | full | 0 | `none` |
| toggle | 1 | 403 | 0 | 50 |  |  | first | 0 | `first_write` |
| toggle | 2 | 467 | 448 | 25 | 403 | 453 | full | 0 | `none` |
| toggle | 3 | 506 | 480 | 22 | 467 | 492 | full | 0 | `none` |
| toggle | 4 | 544 | 512 | 30 | 506 | 528 | full | 0 | `none` |
| toggle | 5 | 591 | 560 | 16 | 544 | 574 | full | 0 | `none` |
| toggle | 6 | 624 | 592 | 43 | 591 | 607 | full | 0 | `none` |
| tools | 1 | 546 | 0 | 21 |  |  | first | 0 | `first_write` |
| tools | 2 | 602 | 560 | 16 | 546 | 567 | full | 0 | `none` |
| tools | 3 | 632 | 608 | 36 | 602 | 618 | full | 0 | `none` |
| tools | 4 | 684 | 656 | 27 | 632 | 668 | full | 0 | `none` |
| tools | 5 | 728 | 704 | 24 | 684 | 711 | full | 0 | `none` |
| tools | 6 | 769 | 736 | 20 | 728 | 752 | full | 0 | `none` |

25 turns: full 25. Previous output re-prefilled: 0. Prefixwise: `first_write` x5, `none` x25, no detail;
`unknown` 0. KV stream: 77 batches, 1,070 blocks stored, 0 removed. Template alone: history rendered as
generated (`kept` shares 106 of 76+30), no thinking clause, no date, tool calls through `tojson`.

## Llama-3.2-3B-Instruct (template alone)

vLLM did not start on the L4: the launch script's `--max-model-len 131072` needs 14.0 GiB of KV cache and
12.97 GiB were free after the 6.21 GiB of weights (`ValueError: ... the estimated maximum model length is
121392`). Skipped per the runbook; `VLLM_ARGS="--max-model-len 32768"` would have started it. Template
alone: history rendered as generated (`kept` shares 125 of 97+28, `stripped` 112 of 97+15: reuse `full`),
`enable_thinking` ignored, tool calls through `tojson`, and the date in the system header
(`renders_date: true`, `where: prefix`, 19 of 97 tokens shared across two dates), as the 3.2-1B template
showed on the first run.

## What it shows

1. **The detail follows the template, not the model family.** It named all 25 DeepSeek turns, none of the
   25 Qwen2.5 turns; on Qwen3 it would have named 29 of 47. Where a template keeps history as generated
   (Qwen2.5, Llama 3.x), the previous answer is reused in full and the detail stays silent.
2. **Reasoning templates differ in degree.** Qwen3 recomputes the previous answer every turn (25-36% of
   prompt tokens); the R1 distill, with longer answers and `<think>\n` in its generation prompt, 57%.
3. **A truncated answer is paid for again.** On both reasoning templates the unfinished thinking is kept
   whole as history; on DeepSeek it is prefilled once more as history before it is part of the reused
   prefix (prompts of 2,000 to 5,000 tokens here). A client that truncates a reasoning model's output
   should drop the unfinished thinking before resending.
4. **The generation tail is 0 to 4 tokens**, template by template: 0 on Qwen2.5 and Llama, 4 on Qwen3 with
   thinking off, 2 on DeepSeek. The template-alone diff allows 8, the live classification a block plus 8, and the classifier's
   read slack a block plus 16: all of them cover it.
