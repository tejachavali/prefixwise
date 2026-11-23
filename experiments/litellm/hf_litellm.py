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
    out, invalid = [], []
    for n, conv in enumerate(convs):
        msgs, tools = to_openai(conv, rng, e5_scale)
        bad = pairing_problems(msgs)
        if bad:  # a live API would reject it: skipped, and reported (a converter gap, not the dataset's)
            invalid.append((n, bad[:3]))
            continue
        if sum(1 for m in msgs if m["role"] == "assistant") >= 4:
            out.append((f"{args.dataset or 'generated'}#{n}", msgs, tools))
    if invalid:
        print(f"skipped {len(invalid)} conversations a live API would reject, "
              f"e.g. row {invalid[0][0]}: {invalid[0][1]}")
    INVALID[0] = len(invalid)
    return out


INVALID = [0]  # conversations conversations() skipped as invalid for a live API


def to_openai(conv, rng, e5):
    """e5's {role, text, tool_calls, tool_call_id} list as OpenAI chat messages that a live API accepts:
    the conversation opens with a user message, tool-call ids are unique in it, every tool call is answered
    before the next message that is not a tool result, and every tool result answers a call. Agent logs
    break this in ways the API never saw: Claude Code logs each content block of one reply as its own
    assistant entry (merged back into one turn here); a dataset with no ids gets ids numbered per message
    (call_t0 in every turn, renamed here); a trace can miss a result (answered "(no output)", as the
    agent's runtime would) or hold a result with no open call (given a call of the first tool)."""
    tools = copy.deepcopy(getattr(conv, "tools", None)) or e5._tools(rng)
    system = next((m["text"] for m in conv if m["role"] == "system"), "You are a coding agent.")
    msgs = [{"role": "system", "content": system}]
    open_calls, used, n = [], set(), 0  # open_calls: (id in the trace, id sent), in call order

    def unique(cid):
        out, k = cid, 1
        while out in used:
            k += 1
            out = f"{cid}-{k}"  # [A-Za-z0-9_-], the ids Anthropic accepts
        used.add(out)
        return out

    def close_open():
        for _, sent in open_calls:
            msgs.append({"role": "tool", "tool_call_id": sent, "content": "(no output)"})
        open_calls.clear()

    started = False
    for m in conv:
        role = m["role"]
        if role == "system" or (not started and role != "user"):  # nothing before the first user message
            continue
        started = True
        if role == "tool":
            match = next((c for c in open_calls if c[0] == m.get("tool_call_id")), None)
            if match:
                msgs.append({"role": "tool", "tool_call_id": match[1], "content": m["text"]})
                open_calls.remove(match)
                continue
            close_open()
            n += 1
            cid = unique(f"call_h{n}")
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": tools[0]["function"]["name"], "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": m["text"]})
            continue
        calls = []
        for c in copy.deepcopy(m.get("tool_calls") or []):
            sent = unique(c["id"])
            open_calls.append((c["id"], sent))
            calls.append({**c, "id": sent})
        prev = msgs[-1]
        if role == "assistant" and prev["role"] == "assistant":  # one reply logged as several entries
            if m["text"]:
                prev["content"] = f"{prev['content']}\n{m['text']}" if prev["content"] else m["text"]
            if calls:
                prev.setdefault("tool_calls", []).extend(calls)
            continue
        if calls:  # the calls just opened belong to this message: close only the earlier ones
            earlier, open_calls[:] = open_calls[:-len(calls)], open_calls[-len(calls):]
            for _, sent in earlier:
                msgs.append({"role": "tool", "tool_call_id": sent, "content": "(no output)"})
        else:
            close_open()
        if role == "assistant" and calls:
            msgs.append({"role": "assistant", "content": m["text"] or None, "tool_calls": calls})
        else:
            msgs.append({"role": role, "content": m["text"]})
    return msgs, tools


def pairing_problems(msgs):
    """Where a live API would reject msgs: no user message first, a tool-call id used twice, a tool call not
    answered before the next message that is not a tool result, or a tool result that answers no open call.
    Empty when msgs is valid."""
    bad, open_calls, seen = [], set(), set()
    if len(msgs) < 2 or msgs[1]["role"] != "user":
        bad.append("the first message after the system prompt is not a user message")
    for i, m in enumerate(msgs):
        for c in m.get("tool_calls") or []:
            if c["id"] in seen:
                bad.append(f"messages[{i}]: tool-call id {c['id']} used twice")
            seen.add(c["id"])
        if m["role"] == "tool":
            if m.get("tool_call_id") not in open_calls:
                bad.append(f"messages[{i}]: tool result with no open call")
            open_calls.discard(m.get("tool_call_id"))
            continue
        if open_calls:
            bad.append(f"messages[{i}]: tool calls {sorted(open_calls)} not answered")
        open_calls = {c["id"] for c in m.get("tool_calls") or []}
    return bad


def steps_of(msgs, max_steps):
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    return steps[:max_steps]


# -- translate ------------------------------------------------------------------------------------

def run_translate(args):
    import e5_scale
    import translate_b as tb  # registers its body-capturing callback
    results, t0, convs = {}, time.time(), conversations(args)
    for name, msgs, tools in convs:
        cut = steps_of(msgs, args.max_steps)
        tb.check(name, msgs[:cut[-1] + 1] if cut else msgs, tools, results)
    import importlib.metadata as md
    json.dump({"test": "translate", "dataset": args.dataset, "split": args.split, "config": args.config,
               "litellm": md.version("litellm"), "conversations": len(convs), "seconds": round(time.time() - t0),
               "tools_with_nulls_dropped": e5_scale.TOOL_NULLS[0],
               "skipped_invalid": INVALID[0],
               "results": results}, open(args.out, "w"), indent=1)
    print(f"{args.dataset} ({args.split}): {len(convs)} conversations; tools read with null keys dropped: "
          f"{e5_scale.TOOL_NULLS[0]}")
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
               "skipped_invalid": INVALID[0], "problems": problems}, open(args.out, "w"), indent=1)
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


def selftest():
    """Agent-log shapes that break tool-call pairing, each converted and checked as a live API would: in
    OpenAI form (pairing_problems) and in the Anthropic body (each tool_use answered in the next message)."""
    import e5_scale

    def call(cid):
        return [{"id": cid, "type": "function", "function": {"name": "bash", "arguments": "{}"}}]

    def res(cid):
        return {"role": "tool", "text": "out " + cid, "tool_call_id": cid}
    u, done = {"role": "user", "text": "go"}, {"role": "assistant", "text": "done"}
    cases = {
        "one reply logged as two entries": [u, {"role": "assistant", "text": "", "tool_calls": call("a")},
                                            {"role": "assistant", "text": "", "tool_calls": call("b")},
                                            res("a"), res("b"), done],
        "text logged after the call": [u, {"role": "assistant", "text": "", "tool_calls": call("a")},
                                       {"role": "assistant", "text": "looking"}, res("a"), done],
        "a call with no result": [u, {"role": "assistant", "text": "", "tool_calls": call("a") + call("b")},
                                  res("a"), {"role": "assistant", "text": "more", "tool_calls": call("c")},
                                  res("c"), done],
        "a result with no call": [u, {"role": "assistant", "text": "hm"}, res("z"), done],
        "the same id in every turn (a dataset with no ids)": [
            u, {"role": "assistant", "text": "", "tool_calls": call("call_t0")}, res("call_t0"),
            {"role": "assistant", "text": "", "tool_calls": call("call_t0")}, res("call_t0"), done],
        "one reply as two entries, both id call_t0": [
            u, {"role": "assistant", "text": "", "tool_calls": call("call_t0")},
            {"role": "assistant", "text": "", "tool_calls": call("call_t0")}, res("call_t0"), res("call_t0"), done],
        "an assistant entry before the first user message": [
            {"role": "assistant", "text": "hello"}, res("q"), u,
            {"role": "assistant", "text": "", "tool_calls": call("a")}, res("a"), done],
    }
    failed = 0
    for name, conv in cases.items():
        try:
            bad = _selftest_case(conv, e5_scale)
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case, and the next cases still run
            bad = [f"{type(exc).__name__}: {exc}"]
        failed += bool(bad)
        print(f"{'ok' if not bad else 'FAIL':4s} {name}" + (f": {bad}" if bad else ""))
    print(f"self-test: {'PASSED' if not failed else 'FAILED'} ({len(cases)} cases)")
    sys.exit(1 if failed else 0)


def _selftest_case(conv, e5_scale):
    msgs, tools = to_openai(e5_scale.Conv(conv), random.Random(1), e5_scale)
    bad = pairing_problems(msgs)
    for i in steps_of(msgs, 99):
        _, out, _ = anthropic_request(msgs[:i], tools)
        for j, m in enumerate(out):
            ids = {b["id"] for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"} \
                if m["role"] == "assistant" and isinstance(m["content"], list) else set()
            nxt = out[j + 1]["content"] if j + 1 < len(out) and isinstance(out[j + 1]["content"], list) else []
            if ids - {b.get("tool_use_id") for b in nxt if isinstance(b, dict)}:
                bad.append(f"step {i}: Anthropic messages[{j}] tool_use not answered next")
        all_ids = [b["id"] for m in out if isinstance(m["content"], list) for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_use"]
        if len(all_ids) != len(set(all_ids)):
            bad.append(f"step {i}: Anthropic tool_use ids repeated")
        if not out or out[0]["role"] != "user":
            bad.append(f"step {i}: Anthropic messages do not start with a user message")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=("translate", "replay", "selftest"))
    p.add_argument("out", nargs="?")
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
    if args.test == "selftest":
        selftest()
    if not args.out:
        sys.exit("OUT is needed")
    if args.source == "hf" and not args.dataset:
        sys.exit("--dataset is needed with --source hf")
    (run_translate if args.test == "translate" else run_replay)(args)


if __name__ == "__main__":
    main()
