"""DeepSeek's own Anthropic-compatible API through LiteLLM, recorded by Prefixwise, against cachemock.py in
its DeepSeek mode (Anthropic's format, an automatic cache, markers ignored, no writes reported): five
5-turn conversations, unchanged and with one change at turn 3 (system edit, timestamp changed, tools
reordered, an early message edited), streamed and not. No model is called.

Usage: python deepseek_messages_check.py PORT LEDGER.db   (needs litellm[proxy]; prints each session's verdicts)
"""
import asyncio
import os
import sqlite3
import subprocess
import sys
import time
import uuid

import litellm

from prefixwise.litellm_callback import PrefixwiseCallback

HERE = os.path.dirname(os.path.abspath(__file__))
PORT, DB = int(sys.argv[1]), sys.argv[2]
mock = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "cachemock.py"), str(PORT)])
time.sleep(1.5)
cb = PrefixwiseCallback(DB); litellm.callbacks = [cb]
SYS = "You are a careful coding agent. Read before you edit, keep changes small. " * 60
TOOLS = [{"name": n, "description": f"{n} a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
         for n in ("read", "write", "grep")]
async def convo(kind, stream):
    sid = f"{kind}-{stream}-{uuid.uuid4().hex[:6]}"
    base = f"[{sid}] " + SYS + " Current time: 2026-11-23T10:00:01Z."
    system, tools = [{"type": "text", "text": base, "cache_control": {"type": "ephemeral"}}], list(TOOLS)
    msgs = [{"role": "user", "content": "Fix the bug in calc.py."}]
    for turn in range(1, 6):
        if turn == 3 and kind == "system_edit":
            system = [{"type": "text", "text": base.replace("keep changes small", "keep every change small")}]
        if turn == 3 and kind == "tool_reserialize":
            tools = tools[1:] + tools[:1]
        if turn == 3 and kind == "history_edit":
            msgs[0] = {"role": "user", "content": "Fix the bug in calc.py (edited)."}
        if turn == 3 and kind == "volatile_system":
            system = [{"type": "text", "text": base.replace("10:00:01", "10:07:42")}]
        r = await litellm.anthropic.messages.acreate(model="deepseek/deepseek-flash", api_base=f"http://127.0.0.1:{PORT}",
                api_key="k", max_tokens=16, system=system, tools=tools, messages=msgs, stream=stream,
                litellm_session_id=sid)
        if stream:
            async for _ in r:
                pass
        msgs = msgs + [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{turn}", "name": "read", "input": {"path": "calc.py"}}]},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{turn}", "content": f"line {turn}"}]}]
    return sid
async def main():
    out = []
    for kind in ("unchanged", "system_edit", "volatile_system", "tool_reserialize", "history_edit"):
        for stream in (False, True):
            out.append((kind, stream, await convo(kind, stream)))
    await asyncio.sleep(2)
    return out
try:
    sessions = asyncio.run(main())
finally:
    mock.terminate()
time.sleep(1)
c = sqlite3.connect(DB)
for kind, stream, sid in sessions:
    rows = c.execute("select provider, api, cause, prompt_tokens, cached_tokens from entries where session_id=? order by id", (sid,)).fetchall()
    print(f"{kind:17s} stream={str(stream):5s} {[r[2] for r in rows]} {[f'{r[4]}/{r[3]}' for r in rows]} {sorted({(r[0], r[1]) for r in rows})}")
