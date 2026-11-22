"""Matrix A: each cache-breaking change, injected on turn 3 of a conversation sent through LiteLLM,
must be named by Prefixwise's LiteLLM callback. Providers x sync/async x stream/non-stream."""
import asyncio, copy, json, sqlite3, sys, threading, time, itertools
import litellm
from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cachemock  # content-addressed prompt cache, as providers have (see cachemock.py)
from http.server import ThreadingHTTPServer
srv = ThreadingHTTPServer(("127.0.0.1", 0), cachemock.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"
LEDGER = sys.argv[1]
cb = PrefixwiseCallback(recorder=Recorder(LEDGER, agent_id="matrix"))
litellm.callbacks = [cb]

SYSTEM = "You are a careful coding agent. Follow the repository conventions. " * 80
def tool(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
TOOLS = [tool("read_file"), tool("run_tests"), tool("search_code")]

def conversation(change, case=""):
    """Three turns; turn 3 carries the change. Returns (turns, expected cause on turn 3). Each case
    has its own system prompt text, as separate conversations would: a cache shared across cases
    could otherwise serve one case's changed prompt from another's write (correctly, but untested)."""
    sys1 = SYSTEM + f" Case {case}. Current time: 2026-11-22T10:00:01Z."
    t1 = dict(system=sys1, tools=TOOLS, msgs=[{"role": "user", "content": "Fix the failing test in utils.py"}])
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "utils.py"})}}]}
    result = {"role": "tool", "tool_call_id": "call_1", "content": "def add(a, b):\n    return a - b\n" * 30}
    t2 = dict(system=sys1, tools=TOOLS, msgs=t1["msgs"] + [call, result])
    nxt = t2["msgs"] + [{"role": "assistant", "content": "The bug is the minus sign."},
                        {"role": "user", "content": "Go ahead and fix it."}]
    t3 = dict(system=sys1, tools=TOOLS, msgs=copy.deepcopy(nxt))
    expected = "none"
    if change == "tool_reserialize":
        t3["tools"] = [TOOLS[2], TOOLS[0], TOOLS[1]]; expected = "tool_reserialize"
    elif change == "volatile_system":
        t3["system"] = SYSTEM + f" Case {case}. Current time: 2026-11-22T10:00:09Z."; expected = "volatile_system"
    elif change == "system_edit":
        t3["system"] = sys1.replace("careful coding agent", "fast coding agent"); expected = "system_edit"
    elif change == "history_edit":
        t3["msgs"][0]["content"] = "Fix the failing test in helpers.py"; expected = "history_edit"
    elif change == "tool_result_mutation":
        t3["msgs"][2]["content"] = t3["msgs"][2]["content"].replace("a - b", "a + b", 1)  # same length
        expected = "tool_result_mutation"
    return [t1, t2, t3], expected

def kwargs_for(model, turn, session, stream, markers="system"):
    kw = dict(model=model, api_key="sk-test", messages=[{"role": "system", "content": turn["system"]}] + turn["msgs"],
              tools=turn["tools"], litellm_session_id=session)
    if model.startswith("anthropic/"):
        points = [{"location": "message", "role": "system"}]
        if markers == "last":  # also on the latest message: the conversation itself is cached
            points.append({"location": "message", "index": -1})
        kw.update(api_base=BASE, cache_control_injection_points=points)
    else:
        kw.update(api_base=BASE + "/v1")
    if stream:
        kw.update(stream=True)
        if not model.startswith("anthropic/"):
            kw["stream_options"] = {"include_usage": True}
    return kw

def run_sync(kw):
    r = litellm.completion(**kw)
    if kw.get("stream"):
        for _ in r: pass

async def run_async(kw):
    r = await litellm.acompletion(**kw)
    if kw.get("stream"):
        async for _ in r: pass

CHANGES = ["none", "tool_reserialize", "volatile_system", "system_edit", "history_edit", "tool_result_mutation"]
# (model, where Claude's cache markers go). With markers on the system prompt only, nothing after it
# is cached, so an edit to the history or a tool result costs nothing: named as a secondary cause.
MODELS = [("openai/gpt-5.6", None), ("anthropic/claude-haiku-5-5", "system"), ("anthropic/claude-haiku-5-5", "last")]

def expectation(change, markers):
    if change in ("history_edit", "tool_result_mutation") and markers == "system":
        return "none", change
    return change, None
cases = []
GAP = 0.05  # seconds between turns, as an agent runs a tool between calls (the mock records each
           # request on a background thread after replying; a turn sent in the same instant can
           # reach it before the previous one is recorded, and the mock then serves nothing)
async def run_all_async(jobs):
    for kw in jobs:
        await run_async(kw)
        await asyncio.sleep(GAP)
    await asyncio.sleep(2)  # one event loop for every async call, kept alive for LiteLLM's logging tasks
async_jobs = []
for (model, markers), asynchronous, stream, change in itertools.product(MODELS, (False, True), (False, True), CHANGES):
    provider = model.split('/')[0] + (f"-{markers}" if markers else "")
    session = f"{provider}|{'async' if asynchronous else 'sync'}|{'stream' if stream else 'plain'}|{change}"
    turns, _ = conversation(change, session)
    expected = expectation(change, markers)
    for t in turns:
        kw = kwargs_for(model, t, session, stream, markers)
        if asynchronous:
            async_jobs.append(kw)
        else:
            run_sync(kw)
            time.sleep(GAP)
    cases.append((session, expected))
asyncio.run(run_all_async(async_jobs))
time.sleep(4)
db = sqlite3.connect(LEDGER)
ok = 0
out = []
for session, expected in cases:
    rows = list(db.execute("SELECT cause, secondary_causes FROM entries WHERE session_id=? ORDER BY id", (session,)))
    got = [r[0] for r in rows]
    primary, secondary = expected
    good = (len(got) == 3 and got[1] == "none" and got[2] == primary
            and (secondary is None or secondary in (rows[2][1] or "")))
    ok += good
    out.append((session, got, expected, "PASS" if good else "FAIL"))
for o in out:
    print(*o, sep="  ")
print(f"{ok}/{len(cases)} passed")
