"""Why did OpenAI read nothing from its cache in the live Messages API test? Sends the same 4-turn
conversation (live_messages.py's, nothing injected) to OpenAI in several ways and prints the cached
tokens per turn, so a miss can be put on OpenAI, on LiteLLM's translation, or on timing:

  A  OpenAI Responses API directly (openai SDK), the body LiteLLM sends for this conversation, 1 s apart
  B  the same, 5 s apart
  C  the same, 1 s apart, with prompt_cache_key set to the conversation
  D  OpenAI Chat Completions directly, the same conversation in chat form, 1 s apart
  E  LiteLLM's Messages API (as in the live test), 5 s apart
  F  LiteLLM's Messages API, 1 s apart, and each body LiteLLM logged replayed at once to OpenAI directly

Usage: python openai_cache_probe.py OUT.json [--model gpt-4.1-mini] [--api-base URL (a mock, to check the script)]
Never prints a key.
"""
import argparse
import asyncio
import json
import time
import uuid

import live_messages as lm


def responses_body(model, tag, turn):
    """The body LiteLLM 1.110-1.112 sends to the Responses API for live_messages' conversation."""
    system = lm.system_for(tag, turn, "none")[0]["text"]
    items = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": system}]}]
    for m in lm.messages_for(turn, "none"):
        kind = "input_text" if m["role"] == "user" else "output_text"
        items.append({"type": "message", "role": m["role"], "content": [{"type": kind, "text": m["content"][0]["text"]}]})
    tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"],
              "strict": False} for t in lm.TOOLS]
    return {"model": model, "input": items, "tools": tools, "max_output_tokens": 64}


def chat_body(model, tag, turn):
    system = lm.system_for(tag, turn, "none")[0]["text"]
    msgs = [{"role": "system", "content": system}]
    for m in lm.messages_for(turn, "none"):
        msgs.append({"role": m["role"], "content": m["content"][0]["text"]})
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in lm.TOOLS]
    return {"model": model, "messages": msgs, "tools": tools, "max_tokens": 64}


def cached_of(usage):
    d = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
    prompt = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    return (getattr(d, "cached_tokens", 0) or 0) if d else 0, prompt


def direct(client, kind, model, pause, cache_key=False):
    tag, out = uuid.uuid4().hex[:10], []
    for turn in range(1, 5):
        if kind == "responses":
            body = responses_body(model, tag, turn)
            if cache_key:
                body["prompt_cache_key"] = f"probe-{tag}"
            r = client.responses.create(**body)
        else:
            r = client.chat.completions.create(**chat_body(model, tag, turn))
        out.append(cached_of(r.usage))
        time.sleep(pause)
    return out


async def via_litellm(model, pause, api_base, replay_client=None):
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    logged = []

    class Keep(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = (kwargs.get("additional_args") or {}).get("complete_input_dict")
            if isinstance(body, dict):
                logged.append(json.loads(json.dumps(body)))

    keep = Keep()
    litellm.callbacks = [keep]
    tag, out, replayed = uuid.uuid4().hex[:10], [], []
    for turn in range(1, 5):
        kw = dict(model=f"openai/{model}", max_tokens=64, system=lm.system_for(tag, turn, "none"), tools=lm.TOOLS,
                  messages=lm.messages_for(turn, "none"))
        if api_base:
            kw.update(api_base=api_base, api_key="k")
        r = await litellm.anthropic.messages.acreate(**kw)
        u = r.get("usage", {}) if isinstance(r, dict) else getattr(r, "usage", {})
        u = u if isinstance(u, dict) else u.model_dump()
        out.append((u.get("cache_read_input_tokens") or 0, (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)))
        if replay_client is not None and logged:
            body = {k: v for k, v in logged[-1].items() if k != "stream"}
            replayed.append(cached_of(replay_client.responses.create(**body).usage))
        await asyncio.sleep(pause)
    keys = sorted({k for b in logged for k in b})
    varying = sorted(k for k in keys if k not in ("input",) and len({json.dumps(b.get(k), sort_keys=True) for b in logged}) > 1)
    return out, replayed, keys, varying


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--api-base")
    a = p.parse_args()
    from openai import OpenAI
    PAUSE5 = 0.2 if a.api_base else 5
    PAUSE1 = 0.05 if a.api_base else 1
    client = OpenAI(base_url=a.api_base, api_key="k") if a.api_base else OpenAI()
    res = {}
    res["A responses 1s"] = direct(client, "responses", a.model, PAUSE1)
    res["B responses 5s"] = direct(client, "responses", a.model, PAUSE5)
    res["C responses 1s prompt_cache_key"] = direct(client, "responses", a.model, PAUSE1, cache_key=True)
    res["D chat 1s"] = direct(client, "chat", a.model, PAUSE1)
    e, _, _, _ = asyncio.run(via_litellm(a.model, PAUSE5, a.api_base))
    res["E litellm messages 5s"] = e
    f, replayed, keys, varying = asyncio.run(via_litellm(a.model, PAUSE1, a.api_base, replay_client=client))
    res["F litellm messages 1s"] = f
    res["F' litellm's logged body replayed directly"] = replayed
    for k, v in res.items():
        print(f"{k:45s} " + "  ".join(f"{c}/{t}" for c, t in v))
    print("keys LiteLLM sent:", keys)
    print("keys that changed between turns (besides input):", varying)
    json.dump({"model": a.model, "mock": bool(a.api_base), "turns": res, "litellm_keys": keys,
               "litellm_varying_keys": varying, "date": time.strftime("%Y-%m-%d")}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
