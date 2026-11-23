# DeepSeek's own Anthropic-compatible API: how it caches (2026-11-23)

LiteLLM sends a `/v1/messages` call for a DeepSeek model to DeepSeek's own Messages API
(`https://api.deepseek.com/anthropic`, `DeepSeekAnthropicMessagesConfig` in LiteLLM 1.110.2), in Anthropic's
format. Until its cache rules were known, Prefixwise's LiteLLM callback did not record these calls. This probe
settles them.

## The probe

`anthropic_compat_probe.py` at `658d0715`, run by Muse in Colab on Nov 23, 11:22 to 11:31 PST, with LiteLLM 1.110.2
and anthropic 2.12.0. DeepSeek listed two models, `deepseek-flash` and `deepseek-v4-pro` (no `deepseek-chat`), and
the probe used `deepseek-flash`. It sent one conversation of about 1,500 tokens over 3 turns, four times: with
`cache_control` markers (on the system prompt and the last message) and without, each directly through the
Anthropic SDK and through LiteLLM's `anthropic.messages`. That is 12 calls. MiniMax was skipped because its key
was not set. The output rows, as printed:

| Markers | Route | Turn 1: uncached / read / written | Turn 2 | Turn 3 |
|---|---|---|---|---|
| yes | direct | 1,485 / 0 / 0 | 220 / 1,280 / 0 | 235 / 1,280 / 0 |
| yes | LiteLLM | 1,487 / 0 / 0 | 222 / 1,280 / 0 | 237 / 1,280 / 0 |
| no | direct | 1,487 / 0 / 0 | 222 / 1,280 / 0 | 237 / 1,280 / 0 |
| no | LiteLLM | 1,488 / 0 / 0 | 223 / 1,280 / 0 | 238 / 1,280 / 0 |

"Uncached" is `input_tokens`, "read" `cache_read_input_tokens`, and "written" `cache_creation_input_tokens`. Every
response also reported `output_tokens` 16 (all of them thinking tokens) and `service_tier` "standard". The file
(`ds-probe.json`) is kept by Muse; the rows above are its full content.

## What it shows

- **The cache is automatic.** Markers change nothing: turns 2 and 3 read the same 1,280 tokens with and without
  them, directly and through LiteLLM.
- **No write is reported.** `cache_creation_input_tokens` is 0 on every call, the first included.
- **Usage is in Anthropic's fields, with Anthropic's meaning.** `input_tokens` excludes the cached tokens: turn 2
  is 220 uncached + 1,280 read = 1,500 tokens.
- **Reads stop short of the prompt.** 1,280 tokens were read of a prefix of about 1,500, and turn 3 read the same
  1,280 although turn 2 had sent about 1,500. 1,280 is a multiple of 64 and of 128, so these rows alone do not tell
  the read unit or whether reads grow as a conversation grows (the longer probe below does), nor the minimum
  cacheable length.
- **LiteLLM passes it through unchanged.** The same fields and counts arrive through LiteLLM, give or take one or two
  tokens of request size.

So DeepSeek's Messages API is an automatic prefix cache, judged from what it read. That it caches exactly as
DeepSeek's chat completions do was then measured: see the longer probe below. Prefixwise now records these calls under those rules. It compares the body LiteLLM sent, converted to a chat
completion, takes `cache_read_input_tokens` as the cached tokens, and claims no minimum. MiniMax's own API
is still not recorded.

## Checked through LiteLLM

`deepseek_messages_check.py` sends five 5-turn conversations, streamed and not, through LiteLLM 1.110.2's DeepSeek
route (`deepseek/deepseek-flash`, `litellm.anthropic.messages.acreate`). The target is `cachemock.py` in its
DeepSeek mode: Anthropic's format, an automatic cache, markers ignored and no writes reported. The run used the
callback from this change; no model was called. Each row shows the verdicts and the cached/prompt tokens per turn:

```
unchanged         stream=False ['first_write', 'none', 'none', 'none', 'none'] ['0/1237', '1237/1274', '1274/1311', '1311/1348', '1348/1385'] [('openai', 'chat')]
unchanged         stream=True  ['first_write', 'none', 'none', 'none', 'none'] ['0/1237', '1237/1274', '1274/1311', '1311/1348', '1348/1385'] [('openai', 'chat')]
system_edit       stream=False ['first_write', 'none', 'system_edit', 'none', 'none'] ['0/1237', '1237/1274', '0/1386', '1386/1423', '1423/1460'] [('openai', 'chat')]
system_edit       stream=True  ['first_write', 'none', 'system_edit', 'none', 'none'] ['0/1237', '1237/1274', '0/1386', '1386/1423', '1423/1460'] [('openai', 'chat')]
volatile_system   stream=False ['first_write', 'none', 'volatile_system', 'none', 'none'] ['0/1238', '1238/1275', '0/1312', '1312/1349', '1349/1386'] [('openai', 'chat')]
volatile_system   stream=True  ['first_write', 'none', 'volatile_system', 'none', 'none'] ['0/1238', '1238/1275', '0/1312', '1312/1349', '1349/1386'] [('openai', 'chat')]
tool_reserialize  stream=False ['first_write', 'none', 'tool_reserialize', 'none', 'none'] ['0/1239', '1239/1276', '0/1313', '1313/1350', '1350/1387'] [('openai', 'chat')]
tool_reserialize  stream=True  ['first_write', 'none', 'tool_reserialize', 'none', 'none'] ['0/1238', '1238/1275', '0/1312', '1312/1349', '1349/1386'] [('openai', 'chat')]
history_edit      stream=False ['first_write', 'none', 'none', 'none', 'none'] ['0/1238', '1238/1275', '1224/1314', '1314/1351', '1351/1388'] [('openai', 'chat')]
history_edit      stream=True  ['first_write', 'none', 'none', 'none', 'none'] ['0/1237', '1237/1274', '1223/1313', '1313/1350', '1350/1387'] [('openai', 'chat')]
```

- Unchanged conversations: a first write, then hits.
- The system edit, the changed timestamp and the reordered tools: each named at turn 3, streamed or not.
- The edited first message sits after a 1,200-token system prompt, so the cache still served all but 51 tokens of
  what the previous turn sent (1,224 of 1,275 on turn 3). By Prefixwise's rule a read within 128 tokens of the
  whole is served in full, so it is `none`, with the edit kept as a secondary cause.

Before this change, none of these calls was recorded.

## The longer probe: reads keep up (2026-11-23, 12:20 to 12:31 PST)

`anthropic_compat_probe.py --turns 10 --grow 200 --stream --chat --ledger` at `efc888c3` (this change), run by Muse
in Colab with LiteLLM 1.110.2, Prefixwise 0.12.4 and anthropic 2.12.0. The model was `deepseek-flash`. The run sent
one conversation of 10 turns, growing by about 215 tokens a turn, seven ways: directly and through LiteLLM's Messages
API, each with and without markers; streamed through LiteLLM, with and without markers; and as DeepSeek's chat
completions through LiteLLM. Prefixwise's callback recorded every LiteLLM call. That was 70 calls in 208 seconds. The
output is kept by Muse as `ds-probe2.json`; the rows below are read from it.

The reads were the same on all seven ways, turn by turn. The prompt is as sent without markers; streamed and chat
completions ran 1 token shorter:

| Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Prompt | 1,487 | 1,702 | 1,917 | 2,132 | 2,347 | 2,562 | 2,777 | 2,992 | 3,207 | 3,422 |
| Read | 0 | 1,408 | 1,664 | 1,792 | 2,048 | 2,304 | 2,560 | 2,688 | 2,944 | 3,200 |

- **Each read is the previous turn's prompt rounded down to 128 tokens.** This holds on every turn: 1,487 gives 1,408,
  1,702 gives 1,664, and so on. The read is 2 to 125 tokens short of what the previous turn sent.
- **No write is reported** on any call.
- **DeepSeek's chat completions read exactly the same.** `prompt_tokens_details.cached_tokens` equals
  `prompt_cache_hit_tokens` and the Messages API read on every turn, and hit plus miss equals the prompt. So the
  Messages API caches as the chat completions do, now measured rather than inferred.
- **Prefixwise's verdicts:** each of the five LiteLLM ways, streamed or not and Messages API or chat completions,
  recorded all 10 calls: `first_write`, then `none` on turns 2 to 10, with no miss named. The two direct ways do not
  pass through LiteLLM, so its callback has nothing to record from them.

One thing the 128-token rule does not explain: in the first probe, turns that appended only about 15 tokens read
1,280 where it predicts 1,408, on turn 2 and again on turn 3. The two probes fit DeepSeek not reading into roughly
the last 100 to 200 tokens of the request being served, but two short conversations do not settle that.

## Limits

- One model (`deepseek-flash`); two conversations, of 3 and 10 turns.
- Reads are the previous prompt rounded down to 128 tokens, which fits the 128 tokens Prefixwise allows for
  rounding, when a turn appends a few hundred tokens, as agent turns usually do. When a turn appends very little,
  the read stopped one block shorter in the first probe. An unchanged turn is then still a hit by the
  half-read rule, but a change in that unread tail is named as the cause although it cost little.
- The read unit, the minimum cacheable length and how long DeepSeek keeps a prefix are not measured, so a miss on
  an unchanged prefix is judged as on DeepSeek's chat completions: no minimum is claimed, and a short prompt that
  read nothing is `unknown`.
- MiniMax's own Anthropic-compatible API was not probed (no key).
