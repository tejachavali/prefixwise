# Changelog

## Unreleased

- **A tool whose schema lists `required` in another order is the same tool.** JSON Schema's `required` (and a list `type`, such as `["string", "null"]`) is a set, and a schema built from a Python set lists it in a different order after a restart. Prefixwise called that `tool_catalog_change`, a legitimate change, and `--fix canonical_tools` sent it as it came, so the miss was neither counted as waste nor fixed. Now the detector names it `tool_reserialize` and the fix sorts those two arrays, at any depth; `enum`, `anyOf` and every other array keep their order, and default, example and const values are never touched. Key order was already sorted at every depth. From a Reddit question (r/LLMDevs) about nested tool schemas. An edge and a collector on different versions hash such a tool differently, so upgrade them together, or expect one tool change per conversation across the upgrade.

## 0.11.3: 2026-11-10

One fix for anyone running Prefixwise in front of a router that routes by session: the client's
`x-session-id` now reaches the router unchanged, and Prefixwise sends no session header of its own.

- **The client's `x-session-id` is forwarded, and Prefixwise sends none of its own.** The proxy replaced it with its own session id on every request, `default` for traffic with no Prefixwise session, so behind a router that routes on that header (sgl-router, llm-d, AIBrix) every session shared one routing key and landed on one worker. Now the client's value goes through unchanged, groups turns into sessions when no Prefixwise id is given, and is the session SGLang's KV events are labelled with; with no client value, none is sent, as docs/ENGINES.md always said.

## 0.11.2: 2026-11-10

For local models. On Ollama and most llama.cpp builds, which do not report cached tokens, Prefixwise no
longer calls every unchanged turn a routing miss: it says the count is missing and still names every
change to the prompt. Prefixwise inside vLLM with LMCache is validated live, and `prefixwise` on its
own shows how to start.

- **A server that reports no cached tokens is `unknown`, not a routing miss.** Ollama and many llama.cpp builds send `prompt_tokens` with no `prompt_tokens_details`, so their 0 cached means "not reported". Prefixwise read it as nothing reused and called every unchanged turn a `routing_miss` ("reached a machine without the cache"). Now such a turn is `unknown`, with that evidence; the report says the server leaves the count out and that the hit rate is not known rather than zero; the session view shows `?`. Changes to the prompt (tool order, a timestamp, an edited message) are still named, from the prompt alone. A server that reports 0 is unchanged. Found by running a new user's first session against an Ollama-style server.
- **`prefixwise` with no command shows how to start**: the five commands most people need, instead of an argument error. A report before any run says to run an agent through Prefixwise first.

- **Prefixwise inside vLLM works with LMCache, live.** One flag on `vllm serve`, no proxy: after a GPU flood every next turn was a hit labelled `host_restored`, 5 of 5, as through the proxy; the `canonical_tools` replay on the same server matched the vLLM-alone run to the token; and `prefixwise serve` detected vLLM and its event stream from the URL alone, its first live record ([record](experiments/e_evict/runs/2026-11-10-colab-l4-middleware-lmcache.md)). docs/LMCACHE.md gains the one-flag setup.

## 0.11.1: 2026-11-10

Documentation only: no change to what the package does. The README is a third of its length and shows a
real run, so the page on PyPI says what Prefixwise does in the first screen.

- **The README shows a real run.** `--optimize` on the Anthropic API, before and after, on one small agent: 74.6% to 87.1% served from cache, $0.0356 to $0.0250, 3,375 avoidable tokens to none ([record](experiments/demo/2026-11-10-optimize-real-run.md)). It replaces the mock output. In that run, the turn that flipped the tool list back to an order still in the cache read almost all of it.
- **A shorter README.** About 160 lines where there were 510: what Prefixwise does (identify, fix, prevent, auto-detect), the three commands, real output, the results with their records, the ways in, and the limits. Everything it held before is kept whole in [docs/GUIDE.md](docs/GUIDE.md).

## 0.11.0: 2026-11-10

`--optimize` does more. `auto_breakpoints` passed its live test on Anthropic, so `--optimize` now places
`cache_control` markers on an Anthropic request that carries none, once its tools and system prompt were
seen within the cache window: 106.8% of the predicted savings realised and the prompt-side cost down 85.6%,
write surcharge included, with tool choices unchanged. A request Prefixwise marked is never the reason for
an error, and nothing is placed for a self-hosted engine. A minor version, not a patch, because a flag you
may already use changes what it sends; `--fix canonical_tools` keeps the old behaviour. The GitHub Action's
moving tag is now `v0.11` (`v0.10` stays at 0.10.7).

- **`auto_breakpoints` is proven, and `--optimize` turns it on.** E4 live on Anthropic (Haiku 5.5), a client that sets no `cache_control`, 60 paired requests: 106.8% of the predicted savings realised, prompt-side cost down 85.6% with the write surcharge included, tool choices 59 of 60 right without the fix and 60 of 60 with it, no marker rejected ([record](experiments/e4/runs/2026-11-09-auto-breakpoints-live.md)). **What changes for you:** with `--optimize` (or `PREFIXWISE_OPTIMIZE=1`, or `optimize` in `PREFIXWISE_FIXES`), an Anthropic request that carries no marker now gets them once its tools and system prompt were seen within the cache window; requests with their own markers, one-shot requests, other providers and self-hosted engines (the vLLM middleware included) are untouched. To keep the old behaviour, name the fix you want: `--fix canonical_tools`. The limits are the run's: one model, multi-turn sessions seconds apart, Anthropic's own API.
- **A request Prefixwise marked is never the reason for an error.** Any 400 or 422 on a request that carries placed markers is resent as the client sent it, whether or not the error names `cache_control` (a gateway may say only "extra inputs are not permitted"), together with any added field the error names; the markers count as refused only when the resend goes through, and after two such they are no longer placed. Before, only an error naming the field was resent.
- **`prefixwise check` ignores `PREFIXWISE_FIXES`.** The gate judges the agent as it is; a fix exported in the CI shell no longer hides a finding.
- **E4 covers a second fix**: `experiments/e4/e4.py --fix auto_breakpoints` replays a client without markers and scores it in tokens and in dollars (prompt side, surcharge included); one request sent without its markers fails the run.

## 0.10.7: 2026-11-09

One correction to the session view from 0.10.5: what the number under "first changed" means.

- **`report --session`: what the number in `messages[8]` means, said correctly.** It is the block's position in the whole prompt, from 0: tools, then system, then the conversation, one position per tool, system prompt and message. It is not the message's own index, as the footer, the API doc and the 0.10.5 notes said. The footer and docs now say so, and that on an append-only hit it is the first block of the new turn. A `model_switch` or `first_write` row no longer shows a first-changed block: it was a comparison with another model's or another conversation's request.

## 0.10.6: 2026-11-09

Three small ones from the first day of the session view. `prefixwise fixes` says whether the markers
`auto_breakpoints` placed have paid for themselves on your traffic. `report --session` shows each request's
cost, with the dollar column it had named as avoidable dollars. And opencode through `prefixwise run` is
verified, with the record.

- **`prefixwise fixes` says whether `auto_breakpoints` paid.** Over the requests it rewrote: the cache writes the placed markers caused, the reads they enabled, and the reads' saving against the writes' surcharge in input-token equivalents, from Anthropic's multipliers (5-minute write 1.25x, read 0.1x). A model, not a bill: a read of an entry another client's own markers wrote counts, and a read of these writes by such a client does not. Also in `GET /api/v1/fixes` as `auto_breakpoints`.

- **`report --session` shows each request's cost**, and names the dollar column it had as avoidable dollars. The column read as the request's cost and showed $0.0000 on legitimate misses next to a run total of $0.034 (Muse, opencode run). The header now carries the session's total cost and how many requests were priced.
- **opencode through `prefixwise run` is verified**: it reads `ANTHROPIC_BASE_URL`, so `prefixwise run -- opencode run "..."` routes every request ([record](experiments/compat/2026-11-09-opencode-run.md)). The README says so again, with the record.

## 0.10.5: 2026-11-09

Four additions. `--optimize` turns on every proven fix in one flag, so a fix that passes its replay later
joins without a flag change. `--fix auto_breakpoints` places `cache_control` markers on an Anthropic request
that carries none, only once the same prefix was seen within the cache window, so a one-shot request never
pays for a write nobody reads; it stays opt-in until a replay shows the writes pay. A changed `response_format`
is named as the cause it is. And `prefixwise report --session latest` shows one session turn by turn: which
turn stopped reading the cache, why, and the block where the prompt first changed.

- **`prefixwise report --session ID|latest`**: one session turn by turn: prompt and cached tokens, hit rate, cause and detail, the block where the prompt first differs from the previous request (`system[0]`, `tools[2]`, `messages[5]`), avoidable tokens and dollars; failed requests in their place. A compaction or a tool change shows as the turn it broke the cache on. The same as data: `GET /api/v1/sessions/{id}/turns` (`latest` works there too). Hashes and counts only, as everywhere. From r/opencodeCLI (Individual-Quote-958): "the first point where two consecutive requests stop sharing a prefix, plus which part changed".

- **`--fix auto_breakpoints`** (opt-in, Anthropic): `cache_control` markers for a request that carries none, on the last system block (covering the tools), the last message, and the previous user turn when more than 20 blocks follow it. Placed only once this proxy saw the same tools and system prompt within the cache window, from this session or another, so a one-shot request never pays for a write nobody reads; never when the client set a marker; never below the model's minimum. A provider 400 that names `cache_control` resends the client's request as it was and, on the second, stops. Rows it rewrote carry `auto_breakpoints` in `fixes`, and the record says what the provider received, markers included. Not in `--optimize`: it changes the bill, and no E4 run has shown the writes pay yet.
- **The first turn that carries a `cache_control` marker after turns without one is `first_write`**, not `routing_miss`: Anthropic writes only at a breakpoint, so the unmarked turn before it left nothing to read. Seen with `auto_breakpoints` on, and true of a client that adds markers mid-session.
- **`prefixwise serve --optimize` and `prefixwise run --optimize`**: every proven fix in one flag (today `canonical_tools`), so a fix that passes its E4 run later joins without a flag change. `--fix NAME` still names one; `optimize` works in `PREFIXWISE_FIXES`, and `PREFIXWISE_OPTIMIZE=1` turns it on for the vLLM middleware. Off by default, as before. The spec's idea of sorting the keys of `response_format` and `tool_choice` too was built and dropped: providers parse those, so wrapper key order never reaches the prompt, and the order that does (a schema's `properties`) must not change.
- **A changed `response_format` is `param_rendered`.** Chat Completions renders the structured-output schema into the prompt, so a session that changes it mid-way now sees a named cause with its fix instead of an unexplained miss. Records carry one more parameter digest, never the schema.

## 0.10.4: 2026-11-09

The dollar line is always there, and the result fits in a message. Every current Claude model is priced,
a point release the table does not list is priced at its family's nearest version and the report says so,
and a model no rule prices is named instead of a dash. `prefixwise report --share` prints a block of counts
and dollars for a chat or a thread.

- **The dollar line is always there.** The table now holds every current Claude model on Anthropic's pricing page (added: Opus 5, 4.8, 4.7, 4.6, 4.5, 4.1; Sonnet 5, 4.5; Haiku 4.5; Fable 5; Mythos 5.1 and 5, next to the 5.5 models and Sonnet 4.6 already there), and a bare Claude id the table does not list (`claude-opus-4-9`) is priced at the nearest version of its family, the newest at or below it; the report says "Priced at a related model's rate: claude-opus-4-9 as claude-opus-4-8". A variant such as `-fast` stays unpriced, and a model no rule prices is now named: "Not priced: gpt-5.6" instead of a silent dash.
- **`prefixwise report --share`**: a short block for a chat or a thread: requests, tokens, hit rate, dollars (with a note when partial or approximate), the top cause and its fix, the cache writes never read back. Counts and dollars only. `report --prices FILE` now reaches the report's pricing, as the flag said.
- README opens with the bill and the three commands, not the validation table.

## 0.10.3: 2026-11-09

One addition: `prefixwise explain --diff` shows what changed between two requests, not only where. It
reads the request files you give it and prints the differing block before and after; the proxy and the
ledger still never see prompt text.

- **`prefixwise explain --diff` shows what changed.** The hashes say where two requests diverge; with `--diff`, `explain` also prints the first differing block before and after, as a unified diff of the request files you gave it, cut at 40 lines. It reads your own files on your machine: the proxy and the ledger still never see prompt text. From a Reddit thread (r/opencodeCLI): "hashes won't explain what changed".

## 0.10.2: 2026-11-09

Two more from this week's feedback, both about costs that were hidden. An agent that trims its history
every turn while its system prompt still hits now sees the cache writes that are never read back; before,
any partial hit zeroed that count. And a session restarted with a fresh timestamp in its system prompt is
told that another session had the same prefix moments earlier, with the fix, on any provider.

- **Unread cache writes are counted on partial hits.** The previous turn's write sits after what that turn read, so a turn that replaces history while its system prompt and tools still hit leaves the whole write unread; before, any read at all zeroed `unread_write_tokens`, so an agent that trims history every turn (Reddit, Specialist-Can2889: 535k tokens written and 334k read in one 32-turn Claude Agent SDK session) showed no unread-write cost. Now the unread share is the write minus what this turn read beyond the previous turn's read, the 1-hour part of a mixed write counted as read first (it comes first in the prompt). The collector (behind the proxy and the middleware) and the Claude Code importer pass the previous read.
- **A restart that recomputes a shareable prefix is named.** The first request of a session that reads nothing, within the cache window of another session's request with the same model, credential and tools and a system prompt that differs only in a date, time or id (a timestamp set at session start, a per-user id), carries the detail `volatile_prefix_at_start`, says which kind differed, and the report prints its fix under the expected misses. Before, the proxy compared prefixes across sessions as sent, so a fresh timestamp hid the match, and only on Anthropic. The comparison uses masked prefix keys (no text kept), memoised per prefix. From Reddit (Character_Fix_5317, r/DeepSeek). Turn facts gain the keys: a 0.10.1 collector refuses such records as malformed, so upgrade the collector first.

## 0.10.1: 2026-11-09

Two fixes from the first week's feedback. A timestamp in the system prompt is a volatile value whatever
its format, so a format change between turns is `volatile_system` with its fix, not an unexplained edit,
and the verdict now says what the prompt carries (a date, an id, a path). The Codex importer tells the
same tools sent in another order (`tool_reserialize`, priced as waste) from a changed catalog. Also: the
PyPI page's links work, and the template experiment ran on three more models.

- **Codex import tells a reordered tool list from a changed catalog.** On a `dynamic_tools_updated` event the importer also hashes the tools as a set; when only the order changed the miss is `tool_reserialize` (waste, with avoidable tokens and cost, as the proxy and the Claude Code importer price it) instead of `tool_catalog_change` (a legitimate change). From Reddit (kantorcodes1, r/LLMDevs). Codex rows now carry `classifier_version` `codex-log-2`.
- **A timestamp in any format is a volatile value, and the verdict says what the system prompt carries.** From Reddit feedback (r/DeepSeek, r/LLMDevs): a timestamp whose format or precision changed between turns (`2026-11-09T04:10:00Z` one turn, `Nov 9, 2026 04:10:05` the next) read as `system_edit`, an expected miss with no fix, because only ISO dates were masked. Every common date form (ISO, slash, `Nov 9, 2026`, `09 Nov 2026`, RFC 2822 with a weekday, the `date` command's `Sun Nov  9 04:10:00 UTC 2026`, with or without a time and a zone) now masks to one `<DATE>`, so the miss is `volatile_system` with its fix and its avoidable tokens; month and weekday names are whole words and dotted numbers (`3.11.12`, `192.168.1.10`) are numbers, so no edit turns volatile. Each block fingerprint now carries the *kinds* of volatile value it holds (`date`, `time`, `id`, `path`, `number`; never the values), and the `volatile_system` evidence names them: "the system prompt carries a date or timestamp: move it to the end of the prompt, or drop it; it also carries a file path, which other sessions and machines cannot share". Paths are recognised, not masked (an edited instruction naming a path stays an edit); `prefixwise check` warns about a path in the system prompt as it does about a date. A 0.10.0 collector refuses records with the new field; upgrade the collector first.
- **The PyPI page's links work.** The project description is README.md with its relative links (docs/, experiments/, benchmarks/, the top-level .md files, in-page anchors) rewritten to GitHub at build time (`hatch-fancy-pypi-readme`); on the 0.10.0 page they still point nowhere. `Documentation` and `Changelog` added to the project URLs. WORKLOG.md brought up to date.
- **E-template on three more templates** (Colab L4, vLLM 0.33.0; `experiments/e_template/runs/2026-11-09-colab-l4-models.md`): `output_rerendered` live on a DeepSeek-R1 distill, 25 of 25 turns (the previous answer was 57% of prompt tokens); silent on Qwen2.5, which keeps history as generated and reused it in full on 25 of 25 turns; Llama 3.2's template reuses in full too and renders the date into the prefix. "Not an eviction": 0 on every model. The harness's template-alone diff now allows the generation tail (0 to 4 tokens, template by template) instead of calling it a break.

## 0.10.0: 2026-11-09

Prefixwise now runs inside vLLM with no proxy, recognises the engine behind any URL you give it (`serve`
and `run` alike), and names what a reasoning model's chat template costs: on Qwen3, the previous answer is
prefilled again every turn (`output_rerendered`), measured live on vLLM and SGLang with real agents. The
block-rounded hit is read correctly (no more `param_rendered` on a thinking toggle), the engine's block
size travels on the record, and `host_restored` is validated live on LMCache and HiCache. The GitHub
Action's moving tag is `tejachavali/prefixwise@v0.10` (`@v0.9` stays at 0.9.x).

- **`output_rerendered`:** a new detail on `none` behind a self-hosted engine (vLLM, SGLang): the read covered the previous prompt and stopped there, so none of the previous turn's output (160+ tokens) was reused although the earlier messages are unchanged; the template renders history differently from how it was generated (Qwen3 drops reasoning) or the client did not resend it. Engines only: a hosted API counts reasoning it never returns in the output, and may not cache output at all. Sized from the live E-template runs, where the previous output was recomputed on 47 of 50 turns (25-36% of prompt tokens); the detail would have named 29 of them. Turn records now carry each turn's prompt and output counts in their facts (`prompt_tokens`, `completion_tokens`): a 0.9.0 `prefixwise collect` refuses such a record as malformed, so upgrade the collector first.
- **`prefixwise run` detects the engine** behind the agent's base URLs, as `serve` does for `--upstream`: `--engine` and KV events (`auto`) when vLLM or SGLang answers on the OpenAI or the Anthropic one (vLLM serves `/v1/messages`); `--engine none` opts out, `--full-attention` for SGLang. The engine applies to the providers whose URL it is, so traffic to another host through the same proxy is not judged as the engine's. Live: an opencode miss against vLLM was `unknown` for want of the join.
- **E-template end to end** (vLLM, SGLang, real agents; Colab L4): "not an eviction" 0 of 50 harness turns and 0 of 8 agent requests; the block-rounded hit fix and the block size on the record verified live (`toggle` is `none` 10/10); a truncated Qwen3 answer stays in history whole (`experiments/e_template/runs/2026-11-09-colab-l4-rerun.md`).
- **The engine's block size on the record.** The KV-event watcher remembers the block (page) size the engine reports (and forgets it when the engine restarts), the proxy and the vLLM middleware put it on each turn record (`engine_block`; None until the first store is seen), and the classifier uses it as the read slack: a full read behind vLLM with 16-token blocks may stop a block and a generation-prompt stub short of the previous prompt, no more, where without the events the slack is the 128-token ceiling. A divergence 112 tokens before the end of a 2,000-token prompt is now caught on vLLM. In-process SGLang has no KV join, so its records carry none. The record version is unchanged (an optional field was added): a 0.10.0 `prefixwise collect` reads 0.9.0 records and replays its log across the upgrade; a 0.9.0 collector refuses 0.10.0 records (`unknown record fields: ['engine_block']`), so upgrade the collector first.
- Fixed: a cache read reported in whole blocks (vLLM and OpenAI round `cached_tokens` down to a block; SGLang to its page size) that is less than a block short of the previous prompt is a hit; before, a prompt-affecting parameter change on such a turn (`enable_thinking` toggled on Qwen3, which renders only a 4-token tail stub) was reported as `param_rendered` where the same turn without the change was `none`. Seen live on vLLM (`experiments/e_template/runs/2026-11-09-colab-l4.md`); Anthropic's exact counts keep the exact check.
- **A chat template's arguments are compared parameters:** `chat_template_kwargs` (Qwen3's `enable_thinking`) and a per-request `chat_template` join `tool_choice`, `thinking` and the rest in `PARAM_KEYS`, so toggling them behind a self-hosted engine is `param_rendered` (or `none` with it as a secondary cause when the cache absorbed it), not an `unknown` "not an eviction". `experiments/e_template`: what the engine's chat template does to a conversation's cached prefix, per turn (reuse of the previous prompt alone, of prompt plus output, or a prefix break), with a CPU mode that renders the template and names the clause; Qwen3's template re-renders every earlier assistant turn without its `<think>` block, so the previous output is re-prefilled each turn (measured on CPU; the live run follows).

- **`host_restored` run live:** E-evict with KV events on vLLM 0.33.0 + LMCache 0.5.7 and on SGLang 0.5.23 + HiCache (Colab L4, Qwen3-4B): 5 of 5 restored hits on each carry the detail, `offload_tier_seen` on both streams, no lost messages (`experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md`). The guides' tables carry the rows.

- **Prefixwise inside vLLM, no proxy:** `vllm serve <model> ... --middleware prefixwise.PrefixwiseMiddleware` loads the edge as ASGI middleware in vLLM's own API server (`prefixwise.asgi`; raw ASGI 3, no framework dependency; also `app.add_middleware(PrefixwiseMiddleware, ...)` in a FastAPI or Starlette app). It prepares each inference request as the proxy does (session id and `vllm_xargs` for the KV join, `--fix` rewrites, `stream_options.include_usage`), relays streams chunk by chunk, records the turn from the response, and returns the `X-Prefixwise-*` headers on non-streaming responses; classification and the ledger write run off the event loop. KV events are found from the engine itself on the first request (`app.state.engine_client.get_kv_event_sources()`, every data-parallel rank; `PREFIXWISE_KV_EVENTS` overrides; vLLM's `--kv-events-config` on the command line is the last resort). Configured by environment: `PREFIXWISE_LEDGER`, `PREFIXWISE_ENGINE`, `PREFIXWISE_FIX`, `PREFIXWISE_COLLECTOR` / `PREFIXWISE_LOG`, `PREFIXWISE_FULL_ATTENTION`. With `--api-server-count` above 1 it warns once that a collector is needed (`PREFIXWISE_COLLECTOR`), since each process would otherwise keep its own ledger. A client that leaves mid-stream is a `client_disconnected` row (499), as at the proxy; bodies over `PREFIXWISE_MAX_BODY_BYTES` pass through unexamined; SGLang in-process declines KV events with a note (its join needs `/server_info`: run the proxy). Tested against a stand-in ASGI engine, and run live inside `vllm serve` 0.33.0 (Colab L4, Qwen3-4B): the proxy's flood gives the same five confirmed evictions, the KV events found from the engine itself (`experiments/e_evict/runs/2026-11-08-colab-l4-middleware.md`).

- **`prefixwise serve` recognises the engine behind `--upstream`:** at startup it asks the server what it is (SGLang's `/server_info`, the deprecated `/get_server_info` on older versions; vLLM's `/version`, then `/kv_event_sources`; never `/v1/models`, which routers relay) and sets `--engine` itself, plus `--kv-events auto` when the engine publishes KV events over tcp and the engines extra is installed. It prints what it detected, warns when SGLang runs with `--enable-cache-report` off or answers 401 (set `PREFIXWISE_API_KEY`), and reminds you to add `--full-attention` (never guessed). Explicit flags always win, and `--engine none` skips detection for a router that relays its worker's `/server_info` (the SGLang model gateway); a hosted API or an upstream that does not answer within 1.5 s detects as no engine and changes nothing (`prefixwise.engines.probe_upstream`).

- `GET /engine` and the engine-check line of `experiments/e_evict` show `offload_tier_seen` (the stream has carried a store on a non-GPU medium, so `host_restored` can be claimed); E-evict records each turn's detail.

## 0.9.0: 2026-11-08

Prefixwise now stops cache regressions before they ship. `prefixwise check -- pytest
tests/agent_scenarios` runs an agent's scenarios through a local mock provider and fails the build
on the causes that are code bugs (a tool list in a drifting order, a timestamp in the system
prompt, a dropped breakpoint, a mutated tool result), with a GitHub Action, a pytest plugin that
names each test as the scenario, and a monthly projection from a production baseline. Behind
self-hosted engines, a hit that an offload tier rescued after a GPU eviction is told from a GPU
hit (`host_restored`), and new guides cover SGLang, HiCache and LMCache end to end. One verdict
got more honest: an Anthropic breakpoint left early while the stable prompt grows past it is a
misplaced breakpoint, not a hit. Every change was reviewed independently; each confirmed finding
is a test.

**Upgrading:** nothing to change; no ledger migration (schema 6). What moves:
- **Verdicts, Anthropic:** when the breakpoint stays early (on the system block) while the stable
  prompt grows, turns that were `none` are now `breakpoint_misplaced` ("breakpoint too early") with
  the stable tail as avoidable tokens. A report spanning the upgrade shows more
  `breakpoint_misplaced` and more avoidable cost on Anthropic sessions with a system-only
  breakpoint. The fix text says to add a breakpoint on the last message.
- **New detail:** `none` rows can carry `host_restored` (one self-hosted engine with
  `--kv-events`, once the offload tier's own events have been seen).
- **Mock numbers:** `prefixwise run --mock` and `prefixwise serve --mock` now read all 2,000
  synthetic prompt tokens on a continuing request and write all 2,000 on a miss (before: 1,000 of
  2,000, a shape that was itself an early breakpoint). Avoidable tokens and dollars in mock
  reports roughly double; verdicts are unchanged.
- **pytest:** installing Prefixwise registers a `pytest11` plugin, `prefixwise`, that does nothing
  unless run under `prefixwise check` (`PREFIXWISE_CHECK` and `PREFIXWISE_URL` set); `-p no:prefixwise`
  disables it.
- **Release tags:** the release workflow now runs for `vX.Y.Z` tags only, so the moving `v0.9` tag
  the GitHub Action is used by (`uses: tejachavali/prefixwise@v0.9`) is not a release.
New: `prefixwise check` (`--fail-on`, `--strict`, `--allow`, `--baseline`, `--bodies`, `--json`,
`--first-turn-delay`), `prefixwise export --baseline`, `action.yml`, `POST /api/v1/check/scenario`,
`PriceTable.miss_rate`; `docs/SGLANG.md`, `docs/LMCACHE.md`.

- **Docs:** `docs/SGLANG.md` (RadixAttention, HiCache, data-parallel ranks) and `docs/LMCACHE.md` (which hits the offload tier rescued, which misses were the agent's) join `docs/VLLM.md`, which now covers `--kv-events auto`, replicas behind a router and `host_restored`; all three linked from the README.

- **`prefixwise check`, a CI gate for cache regressions:** `prefixwise check -- pytest tests/agent_scenarios` runs the agent's scenarios through a local mock provider and exits 2 when a turn misses its cache for a cause that is a code bug (`tool_reserialize`, `volatile_system`, `breakpoint_misplaced`, `tool_result_mutation`; `--strict` adds `history_edit`, `system_edit`, `param_rendered`; `--fail-on` names any set), 1 when the command itself fails, 0 otherwise. Each turn is compared with its conversation's previous turn as in production; nothing is compared with another branch and no prompt is stored. The mock answers a user turn that offers tools with a call to the first tool (placeholder arguments by the schema's types; all three APIs, streams included), so an agent's loop runs the second turn the gate judges, and holds each conversation's first response (`--first-turn-delay`, 1.1 s) so a per-second timestamp in the prompt changes between turns; a date or time in a system prompt that did not change is a warning, as is a run with no multi-turn conversation. Under pytest the bundled plugin (`pytest11` entry point, inert outside `check`) names each test's node id as the scenario, so findings and the allow-list (`--allow FILE`, keyed scenario, cause, first changed block) name the test. `prefixwise export --baseline` writes `.prefixwise-baseline.json` (per model: turns and eligible turns per month, mean reusable tokens; counts only) and `check` reads it from the working directory for a monthly projection over eligible turns, labelled an estimate; without one, tokens. `--bodies FILE...` judges saved request bodies offline; `--json` / `--json-file` for machines. `action.yml`: the GitHub Action (`uses: tejachavali/prefixwise@v0.9`) that installs the pinned version, runs the gate and writes the report to the job summary.
- **`host_restored`:** behind one self-hosted engine with `--kv-events`, a hit whose prefix blocks had been evicted from the GPU (`BlockRemoved` on the GPU medium) yet were read back without recomputing prefill (vLLM CPU offload, LMCache, SGLang HiCache) is `none` with the detail `host_restored`: a hit that cost a restore, and a sign the GPU cache alone is too small for the session. The engine join now reports, on a hit, how many of the previous request's blocks the read covered had been removed; removals on the offload tiers' own media stay ignored, and nothing is claimed until the tier's own stores have been seen, when another session stored a removed block meanwhile (vLLM), behind several caches, or from a contended or cut event stream.

- **A breakpoint left early is a misplaced breakpoint, not a hit:** on Anthropic, when the blocks are unchanged, the cache served everything the previous turn left in it, and the prefix both turns share runs on past that by at least the model's minimum cacheable size, the verdict is `breakpoint_misplaced` ("breakpoint too early") and the avoidable tokens are that stable tail, which is paid uncached on every turn. Before, the held-prefix cap made it `none`. A breakpoint that moves forward each turn, a tail below the minimum, a turn that did not read all the cache held, a one-shot side call replaced by another, or a rewritten or shorter last turn are unaffected.
- **The mock provider is a faithful cache:** a request that continues the previous one (same tools, system prompt, messages prefix and prompt-affecting parameters, with a breakpoint) reads all 2,000 of its synthetic prompt tokens, as a provider's cache with a breakpoint on the last message would; one that does not writes all of them. Before, a hit read 1,000 of 2,000 and the first write was 1,000, which, by its own numbers, was a breakpoint left early. `prefixwise run --mock` and the README's example report show the new numbers.
- Checked, with tests: a miss that wrote nothing to the cache is priced at the input rate it paid, not the write rate (the rate is what the request's usage shows); the write surcharge on an unread write is shown apart from avoidable cost and never added to it.

## 0.8.0: 2026-11-08

Prefixwise now has the shape a fleet needs. The proxy is two halves with one boundary: the **edge**,
wherever the request already is (the proxy, `prefixwise.wrap`, the LiteLLM callback), fingerprints
it and reads its usage into a turn record of hashes and counts; the **collector** (`prefixwise
collect`) owns conversation state, the verdicts, the dollars and the one ledger. Between them, an
HTTP sender that never makes a request wait, or a session-partitioned Redis Stream so several
collectors share a fleet with every conversation's turns reaching one of them in order. Every
committed row can also go to a JSONL file or ClickHouse. One process with nothing configured
behaves as before. The verdicts got more honest too: a full hit is a hit whatever else changed, a
turn sent while its predecessor was in flight claims nothing, avoidable tokens stop at what the
cache held, and avoidable cost is priced at the write rate the request actually paid. Every change
was reviewed independently; each confirmed finding is a test.

**Upgrading:** nothing to change for one process. The ledger moves to schema 6 in place on first
open (new columns `cache_ttl_seconds`, `unread_write_tokens`, `unread_write_cost`; rows recorded
before have none there and are left out of the TTL advice and the unread-writes line). Verdicts and
dollars change in these cases, so a report that spans the upgrade can shift:
- a full cache hit is `none` even when a parameter changed, there was no `cache_control`, or the
  prompt is below the minimum in the table (before: `param_rendered`, `breakpoint_misplaced`,
  `below_min_prefix`); a prompt that shrank below the minimum is `below_min_prefix`, not waste;
- a turn sent before its conversation's previous turn had finished, and that read less than half
  its prompt, is `unknown`, counted as `prefixwise_ingest_concurrent_total` (before: `routing_miss`
  or `retry_replay`);
- on Anthropic, avoidable tokens are capped at what the previous turn left in the cache, and
  avoidable cost includes the cache-write surcharge (about 26-28% higher for 5-minute writes, about
  2x for 1-hour writes, on the built-in Claude rates); existing rows keep the cost they were
  recorded with;
- `overhead_ms` is now the edge's own time, before classification;
- the LiteLLM callback records a conversation's calls in the order they were sent, waiting up to 2 s
  for an earlier call of the same conversation.
New: `prefixwise collect` and `prefixwise serve --collector URL` (`PREFIXWISE_COLLECTOR`,
`PREFIXWISE_COLLECTOR_TOKEN`), `--log redis://` with `pip install 'prefixwise[log]'`
(`PREFIXWISE_LOG_PARTITIONS`, 16), `--sink` (`PREFIXWISE_SINKS`), `PREFIXWISE_CLOCK_SKEW` (1 s).

- **Docs:** the architecture figure and `docs/ARCHITECTURE.md` show the two halves: the edge where the request is, the turn record, the collector where the ledger is, the HTTP and Redis Stream wires between them, and sinks as a way out.
- **Several collectors, one log:** `prefixwise collect --log redis://...` reads a session-partitioned Redis Stream (`prefixwise:records:<partition>`; `pip install 'prefixwise[log]'`), and edges started with `PREFIXWISE_COLLECTOR=redis://...` write to it. Each collector owns a set of partitions (`--partitions 0-7`), so all of a session's turns reach one collector, in the order they reached the log, and no two collectors see one session. A record is acknowledged only after its row is written, and the row before the collector's state changes, so a restart mid-batch writes no row twice and loses none; a restarted collector takes over a dead predecessor's unacknowledged records at once (a live one, as in a rolling restart, keeps its batch: deploy collectors with `strategy: Recreate` or `maxSurge: 0`) and replays the last day of its partitions to rebuild its conversation state (the first turn after a deploy is not a `first_write`), seeding its duplicate check from the ledger; a second collector on the same partition is reported. Needs Redis 7.0 or later. Partition count: `PREFIXWISE_LOG_PARTITIONS` (16), the same on every edge and collector. What the log orders is which collector a session reaches and the consumption order; an edge still appends in its own send order, so keeping a conversation's turns on one edge remains the way to never see a late record.
- **Ledger sinks:** `--sink file:///path.jsonl` and `--sink clickhouse://user:pass@host:8123/db.table` (repeatable; `PREFIXWISE_SINKS`) send a copy of every committed row to a file or to ClickHouse's HTTP interface, on the same bounded, never-blocking sender as the record transport; a sink that fails is reported once and the ledger is unaffected. `docs/DATA.md` has the ClickHouse table.
- **A collector for a fleet:** `prefixwise collect` runs the control plane as a service with one ledger; edges (`prefixwise serve --collector URL`, or `PREFIXWISE_COLLECTOR=URL` for `prefixwise.wrap` and the LiteLLM callback) fingerprint requests where they are and send turn records to it (`prefixwise.transport`). The sender is one thread per edge with a bounded queue: a request never waits on the collector, and when it is down records are retried with backoff, then dropped and counted (`/health`, `/metrics`: `prefixwise_records_*`). Ingest is idempotent by request id while the collector runs, refuses malformed records by index, and needs the collector's `PREFIXWISE_AUTH_TOKEN` when one is set (`PREFIXWISE_COLLECTOR_TOKEN` on the edge; a collector listening beyond localhost refuses to start without one). An edge answers with `X-Prefixwise-Queued` instead of a verdict (the verdict is made at the collector) and points `/api/v1`, `/ledger`, `/requests/<id>` and `/tasks` to the collector (409), which serves them all. A conversation's turns should take one edge to arrive in order (across edges a late turn is `unknown`, never mis-classified); edge clocks within `PREFIXWISE_CLOCK_SKEW` (1 s) of each other count as sequential (so a parallel request sent less than a second before the previous turn ended is judged as before). One process with no collector configured behaves exactly as before.
- **A turn sent while its conversation's previous turn was still in flight is `unknown`** ("sent N s before the previous turn had finished"): it could not have read that turn's cache write, so neither a hit nor a miss cause can be claimed; before it was `routing_miss` or `retry_replay`. Counted as `prefixwise_ingest_concurrent_total`. Parallel sampling of one conversation and a record that arrived late both land here, on purpose.
- **Edge and collector:** the proxy is now two halves with one boundary. The edge (the proxy, `prefixwise.wrap`, the LiteLLM callback) fingerprints a request and reads its usage into a `TurnRecord` (`prefixwise.records`: hashes, positions and counts, never text; versioned, with a wire form). The collector (`prefixwise.collector`) owns conversation state, classification, pricing and the ledger, and `ingest`s records. In one process the edge calls it directly, as before: no verdict or dollar changes, no configuration (`overhead_ms` now measures the edge's own time, before classification; `prefixwise explain` compares the same prompt-affecting parameters as the proxy, `text` and `extra_key` included). This is the shape a fleet needs (several edges, one collector, one ledger: `docs/specs/hyperscale.md`); the transport between machines comes next.
- **A full cache hit is `none`, whatever else differs:** when the prompt blocks are unchanged and the cache served everything the previous turn left in it (what it read plus what it wrote, so a turn's uncached tail after the last breakpoint is not expected back), the verdict is `none` even if a request parameter changed (kept as secondary), the request carried no `cache_control`, or the prompt is below the minimum in Prefixwise's table. Before, such hits were `param_rendered`, `breakpoint_misplaced` or `below_min_prefix`: a hit reported as a miss (no dollars, but a wrong cause and a counted wasteful request).
- **Avoidable tokens stop at what the cache held:** on Anthropic, a waste turn's avoidable tokens are capped at what the previous turn left in the cache (its cache reads plus its writes, which end at its last breakpoint). Before, the previous turn's whole prompt counted, so with the breakpoint on the system block a history edit was charged for the uncached history after it. Same in `prefixwise import claude-code`.
- **Cache writes never read:** a miss of the same conversation that replaced the prefix (or let it expire) and read nothing right after the previous turn wrote it left that write unread; a retry, a routing miss or an eviction leave the entry for a later turn and are not counted, nor is a side call that fell back to another conversation's turn. `prefixwise report` now says how many such tokens were written and what the write surcharge on them cost above sending them uncached (new ledger columns `unread_write_tokens`, `unread_write_cost`, schema 6, added in place). It is shown apart from avoidable cost, not added to it: the cause named for the miss is what to fix, and fixing it saves this too.
- **A request too short to cache is not waste, even if its prompt changed:** a prompt below the provider's minimum cacheable prefix, with nothing read, is `below_min_prefix` even when its system prompt, tools or history also changed (the change stays as a secondary cause). Before, only a previous request below the minimum was caught, so a prompt that shrank below it was priced as `volatile_system` or `history_edit` waste.
- **Avoidable cost includes the cache-write surcharge:** a miss's avoidable tokens are priced at the rate the request actually paid for its uncached prompt tokens (on Anthropic usually the cache-write rate, 1.25x input for 5 minutes or 2x for 1 hour, as they are written again) minus the cache-read rate. Before, the plain input rate was used, which understated Anthropic waste: the true figure is about 26-28% higher for 5-minute writes and 2.0-2.1x for 1-hour writes (built-in Claude rates). Without the request's usage, the input rate is still used.
- **5-minute or 1-hour cache:** `prefixwise report`, the end of `prefixwise run` and `findings` in the API say, per Anthropic model, whether the 1-hour cache could have paid on this ledger. Kept by 1 hour: the `ttl_expired` misses on the 5-minute cache with a gap within the hour (not `ttl_shortened`); their rewrites would have been reads. Extra: every other 5-minute write at the 1-hour rate; writes on 1 hour, or of unknown lifetime, are not counted. When even the best case does not pay, the report says to keep 5 minutes. New ledger column `cache_ttl_seconds` (schema 5, added in place): each request's own cache lifetime. Rows recorded before it have none and are left out of this advice.
- Existing rows keep the avoidable cost they were recorded with; only new rows use the write rate.
- **Fixed, LiteLLM callback:** LiteLLM runs success callbacks on a pool of threads, so a session's calls could be recorded out of order, and a turn compared with the one after it read as `history_edit`. Calls are now recorded in the order they were sent (a turn waits at most 2 s for earlier turns of its own conversation, never for streams or other conversations; failed, cancelled or abandoned calls release the turns after them).

## 0.7.0: 2026-11-08

Sharper verdicts where agents actually run: parallel subagents in one session are each compared
with their own conversation, SGLang servers with several data-parallel ranks get confirmed routing
misses (as vLLM did in 0.6), and the proxy no longer holds a stream open while it classifies it. Every
change was reviewed independently, twice for the classifier changes; each confirmed finding is a test.

**Upgrading:** nothing to change; no ledger migration. Verdicts change in these cases, so a report
that spans the upgrade can shift:
- interleaved conversations in one session: fewer `system_edit` / `history_edit`, more `none`, and
  `first_write` for a subagent's first turn;
- a prompt change the cache absorbed anyway is `none`; one after the entry had expired is
  `ttl_expired` (the change stays as a secondary cause);
- near the TTL, fewer `ttl_expired`: the gap no longer counts the request's own generation time;
- OpenAI `developer` messages are part of the system prompt: changes to them are `volatile_system` /
  `system_edit`, no longer `history_edit`; when tools and the system prompt both change, the tools
  change is named (the providers' prefix order).
New setting: `PREFIXWISE_LANE_THREADS` (conversations remembered per session and model, default 8).

- **Interleaved conversations in one session** (parallel subagents on one model): each request is compared with its own conversation's latest turn (known by its opening message; up to `PREFIXWISE_LANE_THREADS`, default 8, per session and model), however long the others ran, and the provider's diagnostics are chained to that same turn. A conversation none of them shares, whose system prompt differs beyond dates and ids, is a new one (`first_write`, or `none` if the cache served most of it). Before, the subagents' turns were compared with each other and named `system_edit` or `history_edit`. One conversation still costs one snapshot. A Responses turn sent with `previous_response_id` stays in its parent's conversation and is compared with the turn it names. When a stream ends, the next request of that conversation waits (up to 10 s, normally milliseconds) until the streamed turn is recorded, so it is compared and chained with it; other conversations never wait. System and developer messages are all part of the system prompt. Limits: conversations opening with the identical message count as one; past 8 per lane the least recently used is forgotten.
- **Fixed:** the gap behind `ttl_expired` and `routing_miss` included the request's own generation time; it now runs from the previous response to this request's send.
- **A change that cost nothing is not a miss:** when the cache still served all of the reusable prefix, the turn is `none` (the change kept as secondary); when the entry had expired anyway (nothing read, gap over the TTL), it is `ttl_expired`. Partial hits stay priced for the part the change cost only.
- Hosted providers' prefix order is respected: tools are checked before the system prompt.
- **Streams are no longer held open while Prefixwise classifies them:** the client sees the end of the stream before the ledger write and any KV-event wait. SQLite runs with `synchronous=NORMAL` in WAL mode.
- Flowcharts: neutral decisions, one verdict column (a hit in green, `unknown` with the rest), the new rule order, and agent logs drawn through their own rules (the logs hold no prompt to fingerprint).
- Specs (not built yet): `prefixwise check`, a CI gate for cache regressions, and `prefixwise serve --optimize` ([docs/specs](docs/specs/check-and-optimize.md)).
- **SGLang with several data-parallel ranks.** `prefixwise serve --engine sglang --kv-events auto` follows every rank's KV-event stream (from `/server_info`: `endpoint_port_base + rank`, batches checked against their rank stamp) and, with `--full-attention`, confirms a `routing_miss` when a turn lands on another rank while the rank that served the previous turn still holds the very block it needed. SGLang reports only new blocks, so the serving rank is the one that stored the session's blocks from exactly where the read stopped (if the previous turn's late finish store starts there too, nothing is claimed), and SGLang's block hashes are the same on every rank, so the match is exact. With several ranks known, an eviction is never just inferred; a changed `extra_key` (left out of SGLang's hashes) is a parameter change. The wait for a store published after the response happens only when a claim is possible. Refused, with the reason: prefill/decode disaggregation, ranks across several nodes, session radix cache. Previously several ranks turned the join off. Tested over real sockets; not yet run live.
- **GitHub Releases are created automatically:** after PyPI accepts a release, the workflow creates the GitHub Release from that version's CHANGELOG section with the same files, so the repository's Releases page no longer lags PyPI.
- `docs/ARCHITECTURE.md`: how Prefixwise is built today, with two flowcharts (how a request flows, how a cause is decided) in it and in the README. CI keeps them true: a figure that differs from its script, or a decision chart missing a cause, fails the tests.
- Docs: which routers pass Prefixwise's `session_id` and `vllm_xargs` through to vLLM, checked against each router's source (Nov 2026): llm-d, production-stack, AIBrix and the current SGLang model gateway do; Dynamo and the older SGLang gateway copy in the sglang repo do not, so behind them no routing claim is made.

## 0.6.0: 2026-11-08

Prefixwise where people already are: in their own code (`prefixwise.wrap`), behind LiteLLM, in
their OpenTelemetry traces, and in front of several vLLM replicas, where it now confirms routing
misses. Each integration was reviewed independently against the source it plugs into (the SDKs,
LiteLLM, OpenTelemetry's GenAI conventions, vLLM) before release; every confirmed finding is a
test. Routing misses confirmed live: 5/5 on two vLLM 0.33.0 replicas.

**Upgrading:** nothing to change; no ledger migration. New, all opt-in: `prefixwise.wrap`,
`prefixwise.litellm_callback`, `PREFIXWISE_OTEL` / `--otel` (`pip install 'prefixwise[otel]'`),
`--kv-events auto` or several endpoints. One verdict change: with several KV caches known
(`--kv-events auto` or several endpoints), an unchanged prefix that missed is no longer inferred
as an eviction; one engine behaves as before.

- **Several vLLM KV caches: routing misses, confirmed.** `--kv-events auto` subscribes to every data-parallel rank (vLLM's `GET /kv_event_sources`); comma-separated endpoints cover replicas behind a router that passes the request body through (not Dynamo, which takes the session from headers). Which cache served each request comes from the KV events themselves (vLLM reports a request's blocks, with its session, on the stream of the cache that ran it), so the router need not say. A turn served by another cache, while the one that served the previous turn still held the very block this turn needed, is a confirmed `routing_miss`; a prefix rendered differently, a side call served elsewhere, a replica that went away, a stream behind or two caches reporting one request claim nothing; the same cache twice keeps the one-engine verdicts. **Fixed:** with several caches known, an unchanged prefix that missed is no longer inferred as an eviction (a request sent to another cache looks the same); it is `unknown` unless the events decide. SGLang stays one data-parallel rank for now. Live on two vLLM 0.33.0 replicas behind a router (Colab L4): routing misses confirmed 5/5, sticky-routing control 5/5 hits ([record](experiments/e_route/runs/2026-11-08-colab-l4.md)).
- **`prefixwise.wrap(client)`: Prefixwise in your own code, no proxy.** Wrap an `Anthropic`, `AsyncAnthropic`, `OpenAI` or `AsyncOpenAI` client (`messages.create`, `beta.messages.create`; `chat.completions.create` / `.parse`, `responses.create` / `.parse`; `with_raw_response` too). Calls go to the provider unchanged and each completed one is classified and recorded exactly as the proxy would, into the same ledger. `prefixwise.session(id, task_id=...)` groups calls. The request recorded is what went on the wire: SDK objects carried into the next turn (a response's message, Anthropic content blocks) count as the dicts the SDK sent, `extra_body` is included, `NOT_GIVEN`, timeouts and extra headers are not. Tested with the real SDKs over HTTP.
- **LiteLLM callback.** `litellm.callbacks = [PrefixwiseCallback()]`, or `callbacks: prefixwise.litellm_callback.prefixwise_callback` in the LiteLLM proxy server's config, records every completed chat call. It classifies the request body LiteLLM actually sent the provider: after translation (tool calls as `tool_use` / `tool_result` blocks), with every parameter (`thinking`, `prompt_cache_key`, ...) and the breakpoints LiteLLM injects (`cache_control_injection_points`). Claude through Anthropic, Vertex AI or Bedrock is judged by Anthropic's rules and prices, with the 5-minute / 1-hour split of cache writes. Answers from LiteLLM's own response cache reached no provider and are not recorded. Tested with LiteLLM 1.108 against a mock server, including its proxy server's config loader.
- **OpenTelemetry export.** `PREFIXWISE_OTEL=1` or `prefixwise serve --otel` (also `run`): a span per request, from the proxy, `prefixwise.wrap` and the LiteLLM callback alike, following the GenAI semantic conventions as of Nov 2026: `chat {model}`, CLIENT; `gen_ai.provider.name` (the well-known value for the actual provider: `aws.bedrock`, `groq`, `azure.ai.openai`, ...); `gen_ai.usage.input_tokens` including cache reads and writes, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` (the current name for cache writes), only when the provider reported usage; `gen_ai.conversation.id` only for a session id the client gave, never a made-up one or a user id; `server.address`, `max_tokens`, `finish_reasons`, `previous_response.id`, `openai.response.service_tier`. Plus `prefixwise.cache.cause`, `.legit`, `.detail`, `.avoidable_tokens`, `.avoidable_cost_usd` and cost. Failed requests are error spans with `error.type`; a call Prefixwise could not classify is not. A `traceparent` header puts the proxy's span in the caller's trace and is now passed on upstream (with `tracestate`). Inside an app, spans go to the app's tracer provider, which Prefixwise never replaces; `serve` and `run` export OTLP themselves, honouring `OTEL_EXPORTER_OTLP_*` (HTTP or gRPC), `OTEL_TRACES_EXPORTER` and `OTEL_SERVICE_NAME` / `OTEL_RESOURCE_ATTRIBUTES` (`pip install 'prefixwise[otel]'`). `PREFIXWISE_OTEL=cause` emits only Prefixwise's attributes, for apps whose own instrumentation already reports the call. Never carries prompt text; an export failure or a bad setting never affects a request. Tested end to end against a stand-in OTLP collector.
- `wrap`, the LiteLLM callback and `Recorder` (the shared machinery for other integrations) record non-streamed calls for now, never apply fixes or the proxy's engine join, and can never break a call: a failure is reported once and leaves a `not_recorded` row.
- Reviewed independently before release (SDKs, LiteLLM, OpenTelemetry against the spec's source); every confirmed finding is now a test.

## 0.5.0: 2026-11-07

Built from research into what model labs, engine maintainers and agent users report about
prompt caching, and the gaps they leave: Prefixwise now reaches Claude Code and Codex subscribers through their own logs, explains the
2026 ways agents break the cache, and puts its numbers in Prometheus and Grafana. Each new rule
was reviewed independently against the providers' docs before release.

**Upgrading:** ledgers migrate in place (schema 4 adds `detail`). The published Anthropic
agreement figure is restated as 32/32 named reasons (it counted 5 `unavailable` answers as
agreement; see below). Nothing else to change.

- **`prefixwise import codex`.** The same for Codex's rollout logs (`~/.codex/sessions` and `archived_sessions`), into `prefixwise-codex.db`: per-response usage from `token_usage_record` (older logs: `token_count` events), subagent threads (`parent_thread_id`) and forks labelled. Causes from what changes OpenAI's cached prefix: model switch, compaction, a changed tool catalog; expiry is never claimed (OpenAI's cache lifetime is not fixed), so other misses are `unknown`. Built from Codex's source (rollout format, Nov 2026) and checked on real Codex 0.167 logs. The import names models it has no prices for.
- **`prefixwise import claude-code`: for Pro/Max subscribers, no proxy.** Reads Claude Code's own session logs (`~/.claude/projects`, subagents included), read-only, into `prefixwise-claude-code.db`, and reports the same way plus a line per agent (main conversation, each subagent type). A cause is named only from what the log shows changed in the cached prefix: model switch, compaction, Claude Code upgrade, a changed system prompt (compared by hash), an effort change where it breaks the cache, expiry against the TTL Anthropic actually granted (`ttl_shortened`), or Anthropic's own `cache_miss_reason`. Context Claude Code appends mid-conversation (deferred tools, MCP and skill updates, the date) is never blamed, per its docs; anything else is `unknown`. A first request or a switch back that read the cache is a hit. Errors (`<synthetic>`) are skipped, old in-log subagent entries form their own stream, unreadable files are skipped. Reviewed independently against Claude Code's docs and real logs before release. Stores counts and event types, never message text.
- **The why behind the cause (schema 4, `detail`).** Seven sub-reasons for the ways agents break the cache in practice, each claimed only from facts in hand: `ttl_shortened` (asked for 1 hour, Anthropic kept the write 5 minutes; such misses, which the classifier timed against the hour asked for, are now `ttl_expired`), `lookback_exceeded` (more than 20 blocks since the last cached point; now `breakpoint_misplaced`), `markers_ignored` (breakpoints sent, nothing written or read: a gateway stripping markers), `sent_before_cache_written` (parallel requests with the same prefix), `shared_prefix_not_reused` (another session had just cached it), `thinking_stripped`, `cache_key_changed`. Shown under each cause in `prefixwise report`; a dimension and filter in `/api/v1`. Ledgers migrate in place. Each detail has strict preconditions (README table), set after an independent review reproduced false claims in a first version against Anthropic's docs; each of those scenarios is now a test that must produce no claim.
- **Fixed: dropping earlier thinking on older models is no longer a history edit.** Haiku 4.5 and earlier Opus/Sonnet models strip earlier turns' thinking themselves, so sending or omitting it renders the same prompt; it was counted as `history_edit` waste.
- **Prometheus and Grafana:** `GET /metrics` exports requests, prompt, cached, cache-write, avoidable and completion tokens, cost and avoidable cost by provider, model and miss cause; failures by error; histograms of upstream latency and of Prefixwise's own overhead; and the KV-event stream's health. Labels are bounded (no session or agent ids). It honours `PREFIXWISE_AUTH_TOKEN`, also as `Authorization: Bearer` (how Prometheus sends it). A Grafana dashboard (`deploy/grafana/prefixwise.json`) and alert rules (`deploy/prometheus/prefixwise-rules.yml`) ship with it; every PromQL expression in them is parse-checked and every metric they use is checked against the exporter. Guide: `docs/prometheus.md`.
- **Prefixwise never turns a working request into an error.** It adds a few fields to requests (Anthropic `diagnostics`, OpenAI `prompt_cache_options` and `stream_options.include_usage`, the engine join's `session_id` and `vllm_xargs`). If an upstream answers 400/422 naming one of them (a gateway or compatible server that refuses unknown fields), the request is resent exactly as the client sent it, the client's own values restored; after a second rejection the field is no longer added, with a one-line notice.
- **"No verdict" is no evidence.** Anthropic's and OpenAI's `unavailable`, `previous_message_not_found` and `comparison_response_not_found` are no longer scored as agreement or disagreement, nor counted in `native_compared`. `unavailable` also covers long conversations and unready comparisons, so scoring it only ever added hits. Restated: Prefixwise agreed with Anthropic on **32 of 32** misses Anthropic named (6 causes); the 5 parameter changes it called `unavailable` are reported separately (previously counted as 37 of 37).
- A client's own `diagnostics` object is no longer overwritten.

## 0.4.1: 2026-11-07

SGLang's KV events now confirm or rule out evictions, as vLLM's do, validated live. Nothing to
change when upgrading: SGLang users add `--kv-events` (and `--full-attention` for a dense model)
to opt in.

- **SGLang KV events:** `prefixwise serve --engine sglang --kv-events tcp://host:5557` confirms evictions on SGLang 0.5.20 or later, as on vLLM. SGLang writes a request's body `session_id` on the blocks it stores (an attribution label; it does not change the prompt), but reports only new blocks, so Prefixwise keeps a compact copy of SGLang's tree of GPU blocks (about 80 bytes per cached token) and walks each session's chain through parent links. A removed block where the cache read stopped is a confirmed eviction on any model; "not an eviction" needs `--full-attention`, because SGLang frees sliding-window and Mamba layers without events. At startup Prefixwise reads SGLang's `/server_info` and judges only when it is safe (0.5.20 or later, events on, one data-parallel rank, no `--enable-session-radix-cache`); otherwise it says why and keeps the inference. "Not an eviction" waits for the request's own store event, which SGLang publishes after every eviction made to admit it. Validated live on SGLang 0.5.23 (Qwen3-4B, L4): evictions confirmed 5 of 5, E1-open unchanged at 120/120. **Correction:** 0.4.0 said SGLang rejects session ids it did not open; that is true only of `session_params`, SGLang's stateful sessions.
- **README leads with measured results:** a table of live results with a link to each record, and charts (a fix's predicted vs realised saving, evictions per engine, overhead by prompt size) drawn from the records by `docs/figures/make_figures.py`. The original design notes moved to `docs/DESIGN.md`.
- **E-evict** reports Prefixwise's engine check and KV-event stream stats. SGLang 0.5.23 needs Python 3.12 or lower (its `outlines_core` has no 3.13 wheel); the experiment README shows the install.

## 0.4.0: 2026-11-07

`prefixwise run`, a report that says what to do, every metric as data (API and export), and
self-hosted engines (vLLM, SGLang, LMCache) validated live.

**Python 3.10 is now supported** (the test suite passes on 3.10 to 3.13); vLLM users on 3.10 can install it.

**Upgrading:**
- Existing ledgers migrate in place (schema 3); old rows count as succeeded.
- `prefixwise report` now prints plain-English findings by default. The totals and tables
  are behind `--detail` (or `--by`). Scripts should use `--json`, which keeps its keys and
  adds `findings`.
- Failed requests (upstream errors, cut-off streams) are now recorded as rows without a
  verdict. Token and cost metrics exclude them; `summary` adds `failed_requests`.

- **Self-hosted engines (vLLM, SGLang, LMCache):** `prefixwise serve --engine vllm|sglang` declares one self-hosted engine. With no routing possible, an unchanged-prefix miss is named `evicted` (inferred, and the evidence says so) instead of `routing_miss`. On vLLM, `--kv-events tcp://host:5557` reads the engine's KV-cache events: Prefixwise tags requests with its session id and asks vLLM to report every block a request uses, then checks the block where the cache read stopped (removed: confirmed eviction; still there: not an eviction). The verdict becomes `evicted` confirmed with a block count, or `unknown` ("not an eviction: the engine still holds this session's blocks"). Library callers: `handle("vllm", ...)` sends `session_id` only with `engine="vllm"`. Prefixes restored by LMCache or HiCache count as hits. `GET /engine` shows the stream. Optional extra `prefixwise[engines]`. Design and limits: `docs/ENGINES.md`.
- **SGLang and LMCache:** E1-open takes `--engine vllm | vllm-lmcache | sglang | sglang-hicache`, with launch scripts `sglang_server.sh` and `vllm_lmcache_server.sh` (flags checked against each project's source). Prefixwise sends a body `session_id` only to vLLM; SGLang reads it as its own session handle and rejects unknown ones.
- **E-evict** (`experiments/e_evict`): floods an engine past its GPU KV capacity and checks whether the session's prefix survives, with and without an offload tier (LMCache, HiCache), and with and without KV events.
- **E1-open report:** the misses list compares against the expected cause, so `switch_back` hits no longer appear as misses.
- **Fixed:** `vllm_server.sh` enabled KV events with a field name current vLLM rejects (`enable_kv_cache_events` is the field).
- **`prefixwise run -- <agent>`**: zero setup. Starts the proxy on a free port, points the agent at it (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`; the agent keeps its own keys; existing base URLs become the upstreams), runs it, and prints a plain-English report on that run. The agent's exit code is passed through; Ctrl-C goes to the agent (it cancels a turn, it does not kill the agent) and SIGTERM is passed on. Gateway base URLs are kept exactly as set; a base URL that is already a Prefixwise proxy is refused rather than double-counted; `PREFIXWISE_API_KEY` never replaces the agent's keys.
- **A report that says what to do:** `prefixwise report` now leads with findings in plain English (cache share, cost, avoidable waste by cause with the fix for each, expected misses, failures); `--detail` (or `--by`) adds the tables; `--since 24h` narrows the window. The same findings are data at `GET /api/v1/findings`.
- **One proxy, two providers:** per-provider upstreams, so Anthropic and OpenAI requests each reach their own API.
- **Data first** ([docs/DATA.md](docs/DATA.md)): every request leaves a row of facts, failures included. New per-request fields: status and error (upstream errors and cut-off streams are rows now, without a verdict, so rates keep honest denominators), api, stream, uncached and cache-write tokens, upstream latency, Prefixwise's own overhead, the gap since the previous turn, the verdict's evidence and secondary causes, and where the prompt first diverged. Existing ledgers migrate in place.
- **Every metric defined once** (`prefixwise.metrics`): a catalog of names, units and definitions that the CLI, the API and exports all use. Ratios are ratios of sums. Token and cost metrics count requests with a verdict; failures count in `requests_failed` and `error_rate`.
- **HTTP API `/api/v1`** ([docs/API.md](docs/API.md)): `metrics` (the catalog), `summary`, `breakdown?by=`, `timeseries?bucket=`, `requests` (paginated), `requests/{id}`, `fixes`, `prices`, `POST tasks`, `POST explain`; filters and time windows; same token auth as the proxy.
- **`prefixwise export`**: raw rows as JSONL or CSV.
- **Summary** reports `failed_requests`.
- **Fixed (agent compatibility check):** a session mixing Chat Completions and the Responses API on one model had its Responses calls rejected (400): Prefixwise chained a `chatcmpl-` id into `prompt_cache_options`. Diagnostics now chain only to an id from the same API (`resp_`, `msg_`).
- **Fixed:** a short Anthropic prompt (below the model's minimum cacheable prefix: 512 tokens for Claude 5.x, up to 4,096 for older models) was reported as a misplaced cache breakpoint, with avoidable tokens and advice that could not help. It is now `below_min_prefix`, not waste.
- **Fixed (agent compatibility check):** under `prefixwise run`, cancelling a streamed turn with Ctrl-C printed proxy tracebacks (BrokenPipeError) into the agent's terminal. Client disconnects are now silent, other server errors are one line, and while the agent runs the proxy's own messages go to `prefixwise-run.log` next to the ledger.
- **Fixed:** when the previous request was below the provider's minimum cacheable prefix, a change since (e.g. a rewritten message) was reported as waste. Nothing could have been cached, so it is now `below_min_prefix`.
- **Codex works with `prefixwise run`:** Codex 0.167+ ignores `OPENAI_BASE_URL`, so its requests bypassed the proxy. `prefixwise run -- codex ...` now adds `-c` overrides for a custom provider (plain HTTP + SSE instead of WebSocket, key from `OPENAI_API_KEY`, no sign-in screen, nothing written to `~/.codex`). The proxy forwards Codex's headers, including the `x-codex-turn-state` sticky routing that keeps requests on the same backend, and uses its thread id as the session.
- Report wording: "1 request", not "1 requests".

## 0.3.1: 2026-11-07

Production hardening, found by a new benchmark; per-model lanes; OpenAI prices.
**Upgrading:** existing ledgers are migrated in place on first start. Rows written before
0.3.1 have no timestamp, so `prefixwise prune` keeps them unless you add `--include-undated`.

- **Production benchmark** (`benchmarks/bench.py`): added latency, throughput, memory, a soak and 13 failure cases, against the real `prefixwise serve` and a fake upstream. Results in `benchmarks/runs/`; README "Running in production" summarises them. CI runs the failure cases on every push.
- **3-4x lower overhead on long prompts:** masked block hashes are cached, since an agent resends the same prefix every turn. A 120k-token request went from +50 ms to +13 ms (p50).
- **Fails open:** a ledger write or classifier failure no longer turns a paid upstream response into a 500 (or, mid-stream, appends an error to the stream). The response passes through with `X-Prefixwise-Error: not recorded`.
- **Upstream headers relayed:** `retry-after`, `retry-after-ms`, `x-should-retry`, rate-limit headers and the provider's request id now reach the client, on success, errors and streams. They were dropped, so SDKs could not honour the provider's backoff.
- **Idle connections are closed** after `PREFIXWISE_CLIENT_TIMEOUT` (120 s). Before, 300 idle sockets held 300 threads forever.
- **Graceful shutdown:** SIGTERM drains in-flight requests and streams (up to `PREFIXWISE_DRAIN_SECONDS`, 8 s) and exits 0. In Docker the proxy is PID 1, where an unhandled SIGTERM was ignored and `docker stop` killed it mid-stream after 10 s.
- **Ledger retention:** entries record when they were written; `prefixwise prune --older-than 30d` deletes old ones.
- **Per-model lanes within a session.** Each request is compared with the last request to the same model, so small-model side calls no longer mask causes on the main thread. Found by dogfooding opencode, which interleaves Haiku calls with Sonnet. A first call to a new model is still `model_switch`; switching back to a model whose prefix is still cached is a hit. Provider diagnostics chain to the same model's last response.
- **Replacing a one-shot request is not a history edit:** a single-message request followed by a different one (side calls) is judged on its shared-prefix cache reuse.
- **E1-open harness fix:** Session B turns 8–10 now keep the compacted history. They had reverted to the pre-compaction history, hidden by the model switch until lanes exposed it.
- **CLI:** `prefixwise report | head` exits quietly.
- **OpenAI prices:** GPT-5.6 Luna, Terra and Sol are priced from OpenAI's own pricing page (read 2026-11-07; standard speed, short context). The cache-write column is OpenAI's 30-minute write; the 12-hour rate is recorded but not applied. Sol's price is promotional and expires after 2026-11-30, after which it shows `-` until updated. Plain `gpt-5.6` has no published row and stays unpriced.
- **Price matching is stricter:** a table id matches only itself plus a dated snapshot or cloud version suffix, so `gpt-5.6-sol-pro` no longer borrows Sol's price.

## 0.3.0: 2026-11-07

- **Fixes the proxy can apply:** `prefixwise serve --fix canonical_tools` sorts tools so a drifting tool list stops breaking the cache. It is opt-in and reversible, and the ledger records which requests it rewrote.
- **`prefixwise fixes`:** ranks fixes by the tokens your own ledger says they would save.
- **Found by dogfooding (opencode):** `/messages`-style paths without `/v1` are accepted. Query strings are forwarded. Endpoints Prefixwise does not analyse (`/v1/models`, `count_tokens`, ...) pass through unrecorded. `HEAD` is answered. Sonnet 4.6 is priced.
- **Dollar costs:** every request is priced from a dated price table. Built in: Claude Haiku, Sonnet and Opus 5.5 and Fable 5.1, verified from Anthropic's pricing page on 2026-11-06, including Haiku's over-100k tier and 1-hour cache writes. Add your own with `PREFIXWISE_PRICES`. `report` shows cost, avoidable cost and cost per successful task; `fixes` shows savings in dollars; `prefixwise prices` shows the table. Unknown rates stay empty, never estimated.
- **E4 harness** (`experiments/e4`): replays recorded sessions with a fix off and on, and measures predicted vs. realised savings and task success. **`canonical_tools` is proven:** 99.9% (Anthropic) and 97.5% (vLLM) of predicted savings realised, with task success flat. `prefixwise fixes` shows the evidence.
- **E1-hosted:** on Anthropic, Prefixwise agreed with Anthropic's own miss diagnosis on 37 of 37 misses. OpenAI's diagnostics returned `unavailable` (recorded as an open question).
- **Fixed:** the moving Anthropic `cache_control` breakpoint, and string vs. text-block content, were both misread as `history_edit`.
- **Fixed:** Anthropic's real diagnostics shape (`diagnostics.cache_miss_reason`) is now read.
- **Fixed:** a request with no usage is now `unknown`, not `below_min_prefix`.
- **Security:**
  - `PREFIXWISE_AUTH_TOKEN`, and the proxy refuses to expose an API key without one
  - input validation and body limits
  - the ledger file is owner-only
  - `SECURITY.md`

## 0.2.0: 2026-11-06

- **CLI:** `prefixwise serve | report | task | request | explain`.
- **Proxy:** a drop-in pass-through for OpenAI chat completions, the Responses API and Anthropic messages, streaming included. Verdicts come back in `X-Prefixwise-*` headers.
- **Tokens per successful task:** measured per task, not per request, with a `/tasks` endpoint.
- **Causes:** tool-result, TTL and fork causes now fire, and the timing rules no longer guess.
- **E1-open** passed live on vLLM: 120 turns, 100%.
- Docker image and CI on Python 3.11 to 3.13.

## 0.1.0: 2026-11-05

- First version: fingerprinting, the 19-cause classifier, the ledger and the proxy.
