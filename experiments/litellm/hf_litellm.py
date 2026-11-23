"""Prefixwise through LiteLLM on agent conversations from Hugging Face datasets (no model called).

Two tests on each dataset, as tests B and C in README.md do on the GitHub datasets:

  translate  Is LiteLLM's translation append-only? Each model call's request is translated for
             OpenAI, Anthropic, Bedrock Converse, Bedrock invoke and Gemini (the body LiteLLM builds,
             sent to a closed port): step N's prompt must be the exact start of step N+1's.
  replay     Each conversation replayed through LiteLLM with Prefixwise's callback, against a provider
             that caches by content (cachemock.py, in this process), unchanged and with one change
             injected at a random step (tools reordered, a timestamp in the system prompt, a system
             edit, a history edit, a tool result edited). Four routes:
               chat-openai         litellm.acompletion, openai/...
               chat-anthropic      litellm.acompletion, anthropic/..., breakpoints injected
               messages-anthropic  litellm.anthropic.messages.acreate (Claude Code's route), Claude
               messages-openai     the same, an OpenAI model (LiteLLM sends the Responses API)
             Scored: on an unchanged turn any change cause is a false alarm; at the injected step the
             change must be named (or seen and shown as costing nothing; or the turn before it was
             below the model's minimum cacheable prefix, so nothing could be lost, counted apart).

Conversations are read with experiments/e5_scale's loader (any dataset whose rows hold a message
list: OpenAI chat, ShareGPT, Anthropic blocks). Needs `pip install datasets` and network to the Hub.

Usage:
  python hf_litellm.py translate OUT.json --dataset nvidia/Open-SWE-Traces --split claudecode --limit 100
  python hf_litellm.py replay OUT.json --dataset nvidia/Open-SWE-Traces --split claudecode --limit 30
  python hf_litellm.py replay OUT.json --source generated --limit 5      # no network: checks the script
Options: --config NAME  --max-chars N (per message, default 4000)  --max-steps N (per conversation,
default 16)  --routes chat-openai,chat-anthropic,messages-anthropic,messages-openai  --seed N

Live (real providers, keys from the environment; costs money): add --live, and set the models
(--anthropic-model, default anthropic/claude-haiku-4-5; --openai-model, default openai/gpt-4.1-mini), the
changes to inject (--kinds, default all), and a hard cap on model calls (--max-calls, default 400). On a turn
where nothing changed, a change cause is a false alarm; any other miss is counted as a provider miss.
"""
import argparse
import asyncio
import copy
import json
import os
import random
import sqlite3
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e5_scale"))

STAMP = " Current time: 2026-11-22T10:00:01Z."
CHANGES = ("tool_reserialize", "volatile_system", "system_edit", "history_edit", "tool_result_mutation")
LEGIT = {"first_write", "none", "below_min_prefix"}
# A cause that says the request changed. On a turn where nothing changed, naming one is a false alarm. Any
# other verdict on such a turn (routing_miss, ttl_expired, unknown, ...) says the provider did not read what
# it held: no claim about the request, counted apart as a provider miss (live providers do this; the mock
# does not).
CHANGE_CAUSES = {"tool_reserialize", "tool_catalog_change", "volatile_system", "system_edit", "history_edit",
                 "tool_result_mutation", "compaction_rewrite", "breakpoint_misplaced", "param_rendered",
                 "retry_replay", "cache_key_changed", "thinking_stripped", "model_changed"}
ROUTES = ("chat-openai", "chat-anthropic", "messages-anthropic", "messages-openai")


def conversations(args):
    """(name, OpenAI-format messages with assistant turns, tools) per usable conversation."""
    import e5_scale
    rng = random.Random(args.seed)
    convs = e5_scale.load(SimpleNamespace(source=args.source, dataset=args.dataset, config=args.config,
                                          split=args.split, limit=args.limit, max_chars=args.max_chars,
                                          turns=8), rng)
    out = []
    for n, conv in enumerate(convs):
        msgs, tools = to_openai(conv, rng, e5_scale)
        if sum(1 for m in msgs if m["role"] == "assistant") >= 4:
            out.append((f"{args.dataset or 'generated'}#{n}", msgs, tools))
    return out


def to_openai(conv, rng, e5):
    """e5's {role, text, tool_calls, tool_call_id} list as OpenAI chat messages; a tool result with no
    call in the trace gets one, so every result follows its call (as providers require)."""
    tools = copy.deepcopy(getattr(conv, "tools", None)) or e5._tools(rng)
    system = next((m["text"] for m in conv if m["role"] == "system"), "You are a coding agent.")
    msgs = [{"role": "system", "content": system}]
    open_calls, n = set(), 0
    for m in conv:
        role = m["role"]
        if role == "system":
            continue
        if role == "assistant" and m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": m["text"] or None, "tool_calls": copy.deepcopy(m["tool_calls"])})
            open_calls = {c["id"] for c in m["tool_calls"]}
        elif role == "tool" and m.get("tool_call_id") in open_calls:
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["text"]})
            open_calls.discard(m["tool_call_id"])
        elif role == "tool":
            n += 1
            cid = f"call_h{n}"
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": tools[0]["function"]["name"], "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": m["text"]})
        else:
            if open_calls:  # calls left unanswered in the trace: answer them, as the agent's runtime would
                for cid in sorted(open_calls):
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": "(no output)"})
                open_calls = set()
            msgs.append({"role": role, "content": m["text"]})
    return msgs, tools


def steps_of(msgs, max_steps):
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    return steps[:max_steps]


# -- translate ------------------------------------------------------------------------------------

def run_translate(args):
    import translate_b as tb  # registers its body-capturing callback
    results, t0, convs = {}, time.time(), conversations(args)
    for name, msgs, tools in convs:
        cut = steps_of(msgs, args.max_steps)
        tb.check(name, msgs[:cut[-1] + 1] if cut else msgs, tools, results)
    import importlib.metadata as md
    json.dump({"test": "translate", "dataset": args.dataset, "split": args.split, "config": args.config,
               "litellm": md.version("litellm"), "conversations": len(convs), "seconds": round(time.time() - t0),
               "results": results}, open(args.out, "w"), indent=1)
    print(f"{args.dataset} ({args.split}): {len(convs)} conversations")
    for t, r in results.items():
        print(f"  {t:18s} pairs={r['pairs']:5d} append-only={r['ok']:5d} broken={r['pairs'] - r['ok']:4d} "
              f"untranslated={r['untranslated']}")
        for err, n in (r.get("errors") or {}).items():
            print(f"      not translated x{n}: {err}")


# -- replay ---------------------------------------------------------------------------------------

def inject(kind, msgs, tools):
    msgs, tools = copy.deepcopy(msgs), copy.deepcopy(tools)
    if kind == "tool_reserialize":
        if len(tools) < 2:
            return None
        tools = tools[1:] + tools[:1]
    elif kind == "volatile_system":
        msgs[0]["content"] = msgs[0]["content"].replace(STAMP, " Current time: 2026-11-22T10:07:42Z.")
    elif kind == "system_edit":
        msgs[0]["content"] = "You must be concise. " + msgs[0]["content"]
    elif kind == "history_edit":
        j = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
        if j is None:
            return None
        msgs[j]["content"] = str(msgs[j]["content"]) + " (edited)"
    elif kind == "tool_result_mutation":
        j = next((i for i, m in enumerate(msgs) if m["role"] == "tool"), None)
        if j is None:
            return None
        msgs[j]["content"] = "[cleared] " + str(msgs[j]["content"])
    return msgs, tools


def anthropic_request(msgs, tools):
    """The same request as Claude Code would send it to /v1/messages: system blocks, Anthropic
    messages, Anthropic tools, a breakpoint on the system prompt and on the last message."""
    from prefixwise.litellm_callback import _anthropic_from_openai
    body = _anthropic_from_openai(copy.deepcopy(msgs), None, "m")
    system = body.get("system") or []
    if system:
        system[-1] = {**system[-1], "cache_control": {"type": "ephemeral"}}
    out = [copy.deepcopy(m) for m in body["messages"]]
    last = out[-1]
    content = last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
    content = [dict(b) for b in content]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    out[-1] = {**last, "content": content}
    atools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
               "input_schema": t["function"].get("parameters") or {"type": "object"}} for t in tools]
    return system, out, atools


CALLS = [0]


async def replay(base, route, name, msgs, tools, kind, rng, args):
    import litellm
    session = f"{route}|{kind}|{name}|{rng.random():.6f}"
    msgs = copy.deepcopy(msgs)
    msgs[0] = {"role": "system", "content": f"[{session}] " + str(msgs[0]["content"]) + STAMP}
    steps = steps_of(msgs, args.max_steps)
    if len(steps) < 4:
        return None
    k = rng.randrange(2, len(steps)) if kind != "unchanged" else None
    changed = inject(kind, msgs, tools) if kind != "unchanged" else None
    if kind != "unchanged" and changed is None:
        return None
    if kind == "tool_result_mutation":  # the edited result must already be in the history at step k
        j = next(i for i, m in enumerate(msgs) if m["role"] == "tool")
        later = [n for n, s in enumerate(steps) if s > j + 1]
        if not later:
            return None
        k = max(k, later[0])
    if kind == "history_edit":
        j = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        later = [n for n, s in enumerate(steps) if s > j]
        k = max(k, later[0])
    if args.live and CALLS[0] + len(steps) > args.max_calls:
        return None  # the cap would be passed: this replay is not run
    endpoint = {} if args.live else {"api_key": "k"}
    for n, i in enumerate(steps):
        m, t = changed if k is not None and n >= k else (msgs, tools)
        CALLS[0] += 1
        if route.startswith("chat-"):
            kw = dict(messages=m[:i], tools=t, litellm_session_id=session, num_retries=0, max_tokens=32, **endpoint)
            if route == "chat-openai":
                kw.update(model=args.openai_model)
                if not args.live:
                    kw["api_base"] = base + "/v1"
            else:
                kw.update(model=args.anthropic_model,
                          cache_control_injection_points=[{"location": "message", "role": "system"},
                                                          {"location": "message", "index": -1}])
                if not args.live:
                    kw["api_base"] = base
            await litellm.acompletion(**kw)
        else:
            system, amsgs, atools = anthropic_request(m[:i], t)
            model = args.anthropic_model if route == "messages-anthropic" else args.openai_model
            kw = dict(model=model, max_tokens=32, system=system, messages=amsgs, tools=atools,
                      litellm_session_id=session, **endpoint)
            if not args.live:
                kw["api_base"] = base if route == "messages-anthropic" else base + "/v1"
            await litellm.anthropic.messages.acreate(**kw)
        await asyncio.sleep(1.0 if args.live else 0.005)
    return session, len(steps), k


def run_replay(args):
    import cachemock
    import litellm
    from prefixwise.litellm_callback import PrefixwiseCallback
    from prefixwise.sdk import Recorder
    ledger = args.out.rsplit(".", 1)[0] + ".db"
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(ledger + suffix):
            os.remove(ledger + suffix)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cachemock.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    litellm.callbacks = [PrefixwiseCallback(recorder=Recorder(ledger, agent_id="hf"))]
    routes = [r for r in args.routes.split(",") if r]
    convs, rng, cases, t0 = conversations(args), random.Random(args.seed), [], time.time()

    async def main():
        for name, msgs, tools in convs:
            for route in routes:
                for kind in ("unchanged",) + tuple(c for c in CHANGES if c in args.kinds.split(",")):
                    try:
                        r = await replay(base, route, name, msgs, tools, kind, rng, args)
                    except Exception as exc:  # noqa: BLE001 - a conversation LiteLLM rejects is reported
                        cases.append((route, kind, None, 0, None, f"{type(exc).__name__}: {str(exc)[:160]}"))
                        continue
                    if r:
                        cases.append((route, kind) + r + (None,))
        await asyncio.sleep(5)

    asyncio.run(main())
    db = sqlite3.connect(ledger)
    summary, problems = {}, []
    for route, kind, session, n_steps, k, error in cases:
        s = summary.setdefault(f"{route}/{kind}", {"replays": 0, "calls": 0, "recorded": 0, "false_alarms": 0,
                                                   "injected_named": 0, "injected_seen_cost_nothing": 0,
                                                   "injected_below_minimum": 0, "injected_missed": 0,
                                                   "provider_misses": 0, "errors": 0})
        if error:
            s["errors"] += 1
            if len(problems) < 30:
                problems.append({"route": route, "kind": kind, "error": error})
            continue
        rows = list(db.execute("SELECT cause, secondary_causes, prompt_tokens, cached_tokens, evidence FROM entries "
                               "WHERE session_id=? ORDER BY id", (session,)))
        s["replays"] += 1
        s["calls"] += n_steps
        s["recorded"] += len(rows)
        for n, (cause, secondary, prompt, cached, evidence) in enumerate(rows):
            if k is not None and n == k:
                if cause == kind:
                    s["injected_named"] += 1
                    continue
                if cause == "none" and kind in (secondary or ""):
                    s["injected_seen_cost_nothing"] += 1
                    continue
                if cause == "below_min_prefix":  # nothing was cacheable before the change either
                    s["injected_below_minimum"] += 1
                    continue
                s["injected_missed"] += 1
            elif cause in LEGIT:
                continue
            elif cause in CHANGE_CAUSES or (k is not None and n > k and cause == kind):
                s["false_alarms"] += 1
            else:
                s["provider_misses"] += 1
                continue
            if len(problems) < 30:
                problems.append({"session": session, "step": n, "injected_at": k, "expected": kind if n == k else None,
                                 "got": cause, "secondary": secondary, "prompt": prompt, "cached": cached,
                                 "evidence": (evidence or "")[:240]})
    import importlib.metadata as md
    json.dump({"test": "replay", "live": args.live, "models": [args.anthropic_model, args.openai_model],
               "calls_sent": CALLS[0], "dataset": args.dataset, "split": args.split, "config": args.config,
               "litellm": md.version("litellm"), "prefixwise": md.version("prefixwise"), "seed": args.seed,
               "conversations": len(convs), "seconds": round(time.time() - t0), "summary": summary,
               "problems": problems}, open(args.out, "w"), indent=1)
    print(f"{args.dataset or 'generated'} ({args.split}): {len(convs)} conversations, {round(time.time() - t0)} s")
    total = {}
    for key, s in summary.items():
        print(f"  {key:42s} " + " ".join(f"{k}={v}" for k, v in s.items()))
        for k2, v in s.items():
            total[k2] = total.get(k2, 0) + v
    print("  total:", " ".join(f"{k}={v}" for k, v in total.items()))
    print("  problems:", len(problems))
    for p in problems[:10]:
        print("   ", json.dumps(p)[:400])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=("translate", "replay"))
    p.add_argument("out")
    p.add_argument("--source", choices=("hf", "generated"), default="hf")
    p.add_argument("--dataset")
    p.add_argument("--config")
    p.add_argument("--split")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--routes", default=",".join(ROUTES))
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--live", action="store_true")
    p.add_argument("--anthropic-model", default="anthropic/claude-haiku-4-5")
    p.add_argument("--openai-model", default="openai/gpt-4.1-mini")
    p.add_argument("--kinds", default=",".join(CHANGES))
    p.add_argument("--max-calls", type=int, default=400)
    args = p.parse_args()
    if args.source == "hf" and not args.dataset:
        sys.exit("--dataset is needed with --source hf")
    (run_translate if args.test == "translate" else run_replay)(args)


if __name__ == "__main__":
    main()
