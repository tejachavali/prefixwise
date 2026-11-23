"""How does a provider's own Anthropic-compatible API cache? Before Prefixwise records such calls (LiteLLM's
/v1/messages sends DeepSeek and MiniMax their own Messages API), its rules must be known: automatic prefix
caching like OpenAI's, or breakpoints like Anthropic's, and which usage fields report the reads.

For each provider whose key is set, the same conversation (about 1.5k tokens, plus --grow words per turn,
for --turns turns) is sent: with `cache_control` breakpoints (system prompt and last message) and without,
directly through the Anthropic SDK (the provider's own fields) and through LiteLLM's `anthropic.messages`
(what Prefixwise's callback sees). Each turn's raw usage is printed. Options:

  --stream   also through LiteLLM streamed (the usage is what Prefixwise's callback records)
  --chat     also DeepSeek's chat completions through LiteLLM, the same conversation in OpenAI's format
  --ledger P attach Prefixwise's LiteLLM callback (one session per conversation) and print its verdicts
  --base U   send to U instead of the provider (a local mock, to check this script)

  DEEPSEEK_API_KEY   https://api.deepseek.com/anthropic     model deepseek-chat
  MINIMAX_API_KEY    https://api.minimax.io/anthropic       model MiniMax-M2

Usage: python anthropic_compat_probe.py OUT.json [--deepseek-model M] [--minimax-model M] [--turns N]
       [--grow W] [--stream] [--chat] [--ledger P] [--pause S] [--base U]
Never prints a key. 3 turns, 4 ways: 12 calls per provider; --turns 10 --stream --chat: 60.
"""
import argparse
import asyncio
import json
import os
import sqlite3
import time
import uuid

PROVIDERS = {  # name: (key variable, Anthropic-compatible base URL, LiteLLM prefix)
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/anthropic", "deepseek"),
    "minimax": ("MINIMAX_API_KEY", "https://api.minimax.io/anthropic", "minimax"),
}
SYSTEM = ("You are a careful coding agent. Read before you edit, keep changes small, and explain each step "
          "in one sentence. ") * 60
FILLER = ("The test output shows the function returns the wrong value for negative inputs and the edge case "
          "at zero. ")


def conversation(tag, turn, markers, grow):
    system = [{"type": "text", "text": f"[{tag}] " + SYSTEM}]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Summarise the rules above in one line."}]}]
    for t in range(1, turn):
        note = (FILLER * max(1, grow // 20)) if grow else ""
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": f"Step {t} done."}]},
                 {"role": "user", "content": [{"type": "text", "text": f"{note}Continue with step {t + 1}."}]}]
    if markers:
        system[-1]["cache_control"] = {"type": "ephemeral"}
        last = msgs[-1]["content"][-1]
        msgs[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}
    return system, msgs


def chat_of(system, msgs):
    out = [{"role": "system", "content": system[0]["text"]}]
    for m in msgs:
        out.append({"role": m["role"], "content": "".join(b["text"] for b in m["content"])})
    return out


def plain(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {k: v for k, v in usage.items() if v not in (None, {}, [])} if isinstance(usage, dict) else str(usage)


async def run(name, model_direct, model_litellm, base, a, sessions):
    import anthropic
    import litellm
    key = os.environ[PROVIDERS[name][0]]
    client = anthropic.Anthropic(base_url=base, api_key=key)
    ways = [("direct", True), ("litellm", True), ("direct", False), ("litellm", False)]
    if a.stream:
        ways += [("litellm-stream", True), ("litellm-stream", False)]
    if a.chat and name == "deepseek":
        ways += [("litellm-chat", False)]
    extra = {"api_base": a.base} if a.base else {}
    out = []
    for via, markers in ways:
        tag = uuid.uuid4().hex[:8]
        sid = f"{name}-{via}-{'m' if markers else 'n'}-{tag}"
        sessions.append(sid)
        for turn in range(1, a.turns + 1):
            system, msgs = conversation(tag, turn, markers, a.grow)
            if via == "direct":
                r = client.messages.create(model=model_direct, max_tokens=16, system=system, messages=msgs)
                usage = plain(r.usage)
            elif via == "litellm-chat":
                r = await litellm.acompletion(model=model_litellm, max_tokens=16, messages=chat_of(system, msgs),
                                              api_key=key, litellm_session_id=sid, **extra)
                usage = plain(getattr(r, "usage", None))
            else:
                r = await litellm.anthropic.messages.acreate(
                    model=model_litellm, max_tokens=16, system=system, messages=msgs, api_key=key,
                    stream=via == "litellm-stream", litellm_session_id=sid, **extra)
                if via == "litellm-stream":
                    usage = {}
                    async for chunk in r:  # Anthropic SSE: usage in message_start and message_delta
                        text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                        for line in text.splitlines():
                            if line.startswith("data:"):
                                try:
                                    ev = json.loads(line[5:])
                                except ValueError:
                                    continue
                                u = (ev.get("message") or {}).get("usage") or ev.get("usage")
                                if isinstance(u, dict):
                                    usage.update({k: v for k, v in u.items() if v not in (None, {}, [])})
                else:
                    u = r.get("usage") if isinstance(r, dict) else getattr(r, "usage", None)
                    usage = plain(u)
            row = {"provider": name, "markers": markers, "via": via, "turn": turn, "usage": usage}
            out.append(row)
            print(json.dumps(row))
            time.sleep(a.pause)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--deepseek-model", default="deepseek-chat")
    p.add_argument("--minimax-model", default="MiniMax-M2")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--grow", type=int, default=0, help="words added to each later user turn")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--ledger")
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--base")
    a = p.parse_args()
    if a.ledger:
        import litellm
        from prefixwise.litellm_callback import PrefixwiseCallback
        litellm.callbacks = [PrefixwiseCallback(a.ledger)]
    models = {"deepseek": a.deepseek_model, "minimax": a.minimax_model}
    rows, sessions = [], []
    for name, (var, base, prefix) in PROVIDERS.items():
        if not os.environ.get(var):
            print(f"{name}: {var} not set, skipped")
            continue

        async def go(name=name, base=base, prefix=prefix):
            got = await run(name, models[name], f"{prefix}/{models[name]}", a.base or base, a, sessions)
            await asyncio.sleep(2)  # LiteLLM logs on a background task: let it finish before the loop closes
            return got
        try:
            rows += asyncio.run(go())
        except Exception as exc:  # noqa: BLE001 - report and go on to the next provider
            print(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
    verdicts = {}
    if a.ledger:
        time.sleep(1)
        c = sqlite3.connect(a.ledger)
        for sid in sessions:
            got = c.execute("SELECT cause, prompt_tokens, cached_tokens FROM entries WHERE session_id=? ORDER BY id",
                            (sid,)).fetchall()
            verdicts[sid] = [list(r) for r in got]
            print(f"prefixwise {sid}: {[r[0] for r in got]} {[f'{r[2]}/{r[1]}' for r in got]}")
    json.dump({"date": time.strftime("%Y-%m-%d"), "turns": a.turns, "grow": a.grow, "rows": rows,
               "verdicts": verdicts}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
