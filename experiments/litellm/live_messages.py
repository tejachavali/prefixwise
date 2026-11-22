"""Live test: LiteLLM's Anthropic Messages API (`/v1/messages`, Claude Code's route) against real
providers, with Prefixwise's callback attached. Each conversation is 4 turns; one change is injected
at turn 3 (or none). Scored against what the change must give: turn 1 first_write (or
below_min_prefix), turn 2 a hit, turn 3 the injected cause (a hit when nothing changed), turn 4 a hit.
Every row keeps the provider's own usage, so a verdict can be checked against what was really read.

Routes (each runs only when asked for, with its credentials in the environment):
  sdk     litellm.anthropic.messages.acreate, async, in this process
  proxy   the LiteLLM proxy server's /v1/messages through the Anthropic SDK, with Claude Code's
          session header (start the proxy yourself with live_proxy.yaml; its ledger is separate)

Usage:
  python live_messages.py OUT.json --route sdk --model anthropic/claude-haiku-4-5 [--stream both]
  python live_messages.py OUT.json --route proxy --proxy http://127.0.0.1:4000 --proxy-ledger PATH \\
      --model claude-haiku-4-5
Options: --changes none,tool_reserialize,volatile_system,system_edit,history_edit  --stream no|yes|both
         --api-base URL (a mock, for checking the script)  --pause SECONDS (between turns, default 1)

Prints a table and writes OUT.json: per conversation, each turn's cause, prompt tokens, cached tokens
and the provider's usage as recorded. Never prints a key.
"""
import argparse
import asyncio
import datetime
import json
import os
import sqlite3
import sys
import time
import uuid

CHANGES = ("none", "tool_reserialize", "volatile_system", "system_edit", "history_edit")
EXPECTED_AT_3 = {"none": {"none"}, "tool_reserialize": {"tool_reserialize"}, "volatile_system": {"volatile_system"},
                 "system_edit": {"system_edit"}, "history_edit": {"history_edit"}}
FIRST = {"first_write", "below_min_prefix"}
HIT = {"none"}

# A long, fixed system prompt and first message, so every turn is far above any provider's minimum
# cacheable prefix (4,096 tokens for some Claude models, 1,024 for OpenAI).
SYSTEM = ("You are a careful coding agent working in a Python repository. Read before you edit, keep changes "
          "small, explain each step in one sentence, and run the tests after every change. ") * 60
FILE = "".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(220))
TOOLS = [{"name": name, "description": f"{desc} " * 12,
          "input_schema": {"type": "object", "properties": {"arg": {"type": "string", "description": "the argument"}},
                           "required": ["arg"]}}
         for name, desc in (("bash", "Run a shell command and return its output."),
                            ("read_file", "Read a file from the repository and return its text."),
                            ("edit_file", "Replace a span of a file with new text."))]


def system_for(tag: str, turn: int, change: str) -> list:
    text = f"[run {tag}] " + SYSTEM
    if change == "volatile_system":
        stamp = "2026-11-22T10:00:00Z" if turn < 3 else "2026-11-22T10:07:42Z"
        text += f" Current time: {stamp}."
    elif change == "system_edit" and turn >= 3:
        text = "Answer in one short sentence. " + text
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def tools_for(turn: int, change: str) -> list:
    return TOOLS[1:] + TOOLS[:1] if change == "tool_reserialize" and turn >= 3 else TOOLS


def messages_for(turn: int, change: str) -> list:
    first = "Find the bug in this file and say which helper is wrong:\n\n" + FILE
    if change == "history_edit" and turn >= 3:
        first = "Look carefully. " + first
    msgs = [{"role": "user", "content": [{"type": "text", "text": first}]}]
    for t in range(1, turn):
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": f"Step {t}: I read the file."}]},
                 {"role": "user", "content": [{"type": "text", "text": f"Continue with step {t + 1}."}]}]
    last = msgs[-1]["content"][-1]
    msgs[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}  # as Claude Code marks it
    return msgs


async def run_sdk(args, tag, change, stream):
    import litellm
    session = f"live-{tag}"
    for turn in range(1, 5):
        kw = dict(model=args.model, max_tokens=64, system=system_for(tag, turn, change), tools=tools_for(turn, change),
                  messages=messages_for(turn, change), stream=stream, litellm_session_id=session)
        if args.api_base:
            kw.update(api_base=args.api_base, api_key="k")
        r = await litellm.anthropic.messages.acreate(**kw)
        if stream:
            async for _ in r:
                pass
        await asyncio.sleep(args.pause)
    return session


def run_proxy(args, tag, change, stream):
    import anthropic
    session = str(uuid.uuid4())
    client = anthropic.Anthropic(base_url=args.proxy, api_key=os.environ.get("LITELLM_PROXY_KEY", "sk-anything"),
                                 default_headers={"x-claude-code-session-id": session})
    for turn in range(1, 5):
        kw = dict(model=args.model, max_tokens=64, system=system_for(tag, turn, change), tools=tools_for(turn, change),
                  messages=messages_for(turn, change))
        if stream:
            with client.messages.stream(**kw) as s:
                for _ in s.text_stream:
                    pass
        else:
            client.messages.create(**kw)
        time.sleep(args.pause)
    return session


def score(change, causes):
    if len(causes) != 4:
        return "not recorded" if not causes else f"{len(causes)} of 4 recorded"
    ok = (causes[0] in FIRST and causes[1] in HIT and causes[2] in EXPECTED_AT_3[change] and causes[3] in HIT)
    return "right" if ok else "check"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--route", choices=("sdk", "proxy"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--changes", default=",".join(CHANGES))
    p.add_argument("--stream", choices=("no", "yes", "both"), default="both")
    p.add_argument("--api-base")
    p.add_argument("--proxy")
    p.add_argument("--proxy-ledger")
    p.add_argument("--pause", type=float, default=1.0)
    args = p.parse_args()
    changes = [c for c in args.changes.split(",") if c]
    streams = {"no": [False], "yes": [True], "both": [False, True]}[args.stream]
    if args.route == "sdk":
        import litellm
        from prefixwise.litellm_callback import PrefixwiseCallback
        from prefixwise.sdk import Recorder
        ledger = args.out.rsplit(".", 1)[0] + ".db"
        if os.path.exists(ledger):
            os.remove(ledger)
        litellm.callbacks = [PrefixwiseCallback(recorder=Recorder(ledger, agent_id="live"))]
    else:
        if not (args.proxy and args.proxy_ledger):
            sys.exit("--route proxy needs --proxy and --proxy-ledger (the proxy's PREFIXWISE_LEDGER_PATH)")
        ledger = args.proxy_ledger

    runs = []

    async def sdk_all():
        for change in changes:
            for stream in streams:
                tag = uuid.uuid4().hex[:10]
                runs.append((change, stream, await run_sdk(args, tag, change, stream)))
        await asyncio.sleep(5)  # LiteLLM logs async calls in the background

    t0 = time.time()
    if args.route == "sdk":
        asyncio.run(sdk_all())
    else:
        for change in changes:
            for stream in streams:
                tag = uuid.uuid4().hex[:10]
                runs.append((change, stream, run_proxy(args, tag, change, stream)))
        time.sleep(5)
    db = sqlite3.connect(ledger)
    cols = {r[1] for r in db.execute("PRAGMA table_info(entries)")}
    extra = ", secondary_causes" if "secondary_causes" in cols else ""
    out, tally = [], {}
    print(f"{'change':18s} {'stream':6s} {'verdicts (turns 1-4)':62s} {'cached / prompt tokens':36s} score")
    for change, stream, session in runs:
        rows = list(db.execute(f"SELECT cause, prompt_tokens, cached_tokens, evidence{extra} FROM entries "
                               "WHERE session_id=? ORDER BY id", (session,)))
        causes = [r[0] for r in rows]
        verdict = score(change, causes)
        tally[verdict] = tally.get(verdict, 0) + 1
        reads = " ".join(f"{r[2]}/{r[1]}" for r in rows)
        print(f"{change:18s} {str(stream):6s} {str(causes):62s} {reads:36s} {verdict}")
        out.append({"change": change, "stream": stream, "session": session, "score": verdict,
                    "turns": [{"cause": r[0], "prompt_tokens": r[1], "cached_tokens": r[2], "evidence": r[3],
                               **({"secondary": r[4]} if extra else {})} for r in rows]})
    try:
        import importlib.metadata as md
        versions = {k: md.version(k) for k in ("litellm", "prefixwise")}
    except Exception:  # noqa: BLE001
        versions = {}
    json.dump({"route": args.route, "model": args.model, "mock": bool(args.api_base), "versions": versions,
               "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t0),
               "tally": tally, "conversations": out}, open(args.out, "w"), indent=1)
    print("tally:", tally)


if __name__ == "__main__":
    main()
