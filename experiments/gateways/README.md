# Prefixwise behind an LLM gateway

`gateway_check.py` checks Prefixwise behind any gateway that lets you set a provider's base URL (Bifrost,
Envoy AI Gateway / Agent Router, Portkey's open-source gateway, Plano, ...). It needs no API keys: a cache
mock (`../litellm/cachemock.py`) stands in for the provider.

```
client (this script) -> the gateway -> prefixwise serve (port 4100) -> cache mock (port 4101)
```

It sends five 5-turn conversations per format (OpenAI chat completions, Anthropic Messages), one unchanged and
four with one change at turn 3 (tools reordered, a timestamp in the system prompt changed, the system prompt
edited, the first user message edited), and reports:

1. whether every call was recorded (counted from the ledger's rows, whatever session they went under);
2. which conversation id reached Prefixwise through the gateway. Each is sent with its own prefix, so the
   report names the one that got through: `prefixwise` (`x-prefixwise-session-id`), `forwarded` (a
   `--session-header`), `generic` (`x-session-id`), `body` (OpenAI `user`, Anthropic `metadata.user_id`),
   `none`, or `split:a+b` when it got through on some calls and not others;
3. whether each conversation got the expected verdicts: `first_write, none, <the change>, none, none`.

```bash
# 1. the mock and the Prefixwise proxy; leave this running
python gateway_check.py --backend --setup-only --proxy-port 4100 --mock-port 4101 --ledger gw.db
# 2. the check itself with no gateway in between: 50 of 50 recorded, 10 of 10 right
python gateway_check.py --gateway http://127.0.0.1:4100 --ledger gw.db --out self.json
# 3. configure the gateway's OpenAI and Anthropic base URLs as http://127.0.0.1:4100, then
python gateway_check.py --gateway http://127.0.0.1:8080 --ledger gw.db \
    --openai-path /openai/v1/chat/completions --anthropic-path /anthropic/v1/messages [--stream]
```

Other options: `--header K:V` (the gateway's own headers), `--session-header NAME` (also send the conversation
id under a gateway's forwarded-header name, e.g. Bifrost's `x-bf-eh-x-prefixwise-session-id`),
`--openai-model`, `--anthropic-model`, `--formats openai,anthropic`, `--chunked` (send request bodies with
`Transfer-Encoding: chunked` and no `Content-Length`, as Plano, built on Envoy, does). It writes `--out` (JSON) and exits 1 when
any conversation is wrong. With `--backend` it refuses ports already in use, and it reads the ledger
read-only, so a wrong `--ledger` path is an error rather than an empty new file.
Several checks can share one backend at a time, as long as each gateway passes some conversation id:
calls recorded under no id are counted by every check running then.

Results are under `runs/`, one record per round, with each gateway's version (where known) and how its upstream
was set. Round 1 (2026-11-24): Envoy AI Gateway passed every check, streamed and not; Portkey passed without
streaming and returned errors when streamed; Bifrost and Plano did not start. Round 2: Portkey's streaming errors
happen without Prefixwise too (its own); Bifrost needs a virtual key; Plano showed that `prefixwise serve` read
chunked request bodies as empty (fixed; `--chunked` checks it). Round 3: Plano and Bifrost pass every check on the
fix, streamed and not; behind Bifrost the conversation id arrives through the body's `user` / `metadata.user_id`
or `x-bf-eh-x-prefixwise-session-id`. Where the four stand: `runs/2026-11-24-round3.md`.
