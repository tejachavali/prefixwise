"""How does a provider's own Anthropic-compatible API cache? Before Prefixwise records such calls (LiteLLM's
/v1/messages sends DeepSeek and MiniMax their own Messages API), its rules must be known: automatic prefix
caching like OpenAI's, or breakpoints like Anthropic's, and which usage fields report the reads.

For each provider whose key is set, the same 3-turn conversation (about 3k tokens, growing) is sent twice:
with `cache_control` breakpoints (system prompt and last message) and without any. Each turn's raw usage is
printed: directly through the Anthropic SDK (the provider's own fields), and through LiteLLM's
`anthropic.messages` (what Prefixwise's callback would see).

  DEEPSEEK_API_KEY   https://api.deepseek.com/anthropic     model deepseek-chat
  MINIMAX_API_KEY    https://api.minimax.io/anthropic       model MiniMax-M2

Usage: python anthropic_compat_probe.py OUT.json [--deepseek-model M] [--minimax-model M] [--pause S]
Never prints a key. About 20 calls per provider, a few cents.
"""
import argparse
import asyncio
import json
import os
import time
import uuid

PROVIDERS = {  # name: (key variable, Anthropic-compatible base URL, LiteLLM prefix)
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/anthropic", "deepseek"),
    "minimax": ("MINIMAX_API_KEY", "https://api.minimax.io/anthropic", "minimax"),
}
SYSTEM = ("You are a careful coding agent. Read before you edit, keep changes small, and explain each step "
          "in one sentence. ") * 60


def conversation(tag, turn, markers):
    system = [{"type": "text", "text": f"[{tag}] " + SYSTEM}]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Summarise the rules above in one line."}]}]
    for t in range(1, turn):
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": f"Step {t} done."}]},
                 {"role": "user", "content": [{"type": "text", "text": f"Continue with step {t + 1}."}]}]
    if markers:
        system[-1]["cache_control"] = {"type": "ephemeral"}
        last = msgs[-1]["content"][-1]
        msgs[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}
    return system, msgs


def plain(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {k: v for k, v in usage.items() if v not in (None, {}, [])} if isinstance(usage, dict) else str(usage)


async def run(name, model_direct, model_litellm, base, pause):
    import anthropic
    import litellm
    key = os.environ[PROVIDERS[name][0]]
    client = anthropic.Anthropic(base_url=base, api_key=key)
    out = []
    for markers in (True, False):
        for via in ("direct", "litellm"):
            tag = uuid.uuid4().hex[:8]
            for turn in range(1, 4):
                system, msgs = conversation(tag, turn, markers)
                if via == "direct":
                    r = client.messages.create(model=model_direct, max_tokens=16, system=system, messages=msgs)
                    usage = plain(r.usage)
                else:
                    r = await litellm.anthropic.messages.acreate(model=model_litellm, max_tokens=16, system=system,
                                                                 messages=msgs, api_key=key)
                    u = r.get("usage") if isinstance(r, dict) else getattr(r, "usage", None)
                    usage = plain(u)
                row = {"provider": name, "markers": markers, "via": via, "turn": turn, "usage": usage}
                out.append(row)
                print(json.dumps(row))
                time.sleep(pause)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--deepseek-model", default="deepseek-chat")
    p.add_argument("--minimax-model", default="MiniMax-M2")
    p.add_argument("--pause", type=float, default=2.0)
    a = p.parse_args()
    models = {"deepseek": a.deepseek_model, "minimax": a.minimax_model}
    rows = []
    for name, (var, base, prefix) in PROVIDERS.items():
        if not os.environ.get(var):
            print(f"{name}: {var} not set, skipped")
            continue
        try:
            rows += asyncio.run(run(name, models[name], f"{prefix}/{models[name]}", base, a.pause))
        except Exception as exc:  # noqa: BLE001 - report and go on to the next provider
            print(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
    json.dump({"date": time.strftime("%Y-%m-%d"), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
