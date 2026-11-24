"""Prefixwise behind an LLM gateway: is every call recorded, does the session reach Prefixwise, and is
each injected change named?

The gateway under test is configured to use `prefixwise serve` as its upstream for OpenAI and
Anthropic, so Prefixwise sees the bytes the gateway sends (after its own translation). With
--backend, this script starts that pair itself: the cache mock (../litellm/cachemock.py, a provider
that caches by content: OpenAI automatically, Anthropic at its cache_control markers) and
`prefixwise serve --upstream <mock>` on --proxy-port. The gateway then only needs its OpenAI and
Anthropic base URLs set to http://127.0.0.1:<proxy-port>.

Five 5-turn conversations per format, unchanged and with one change at turn 3: tools reordered, a
timestamp in the system prompt changed, the system prompt edited, the first user message edited. The
expected verdicts are first_write, none, <the change>, none, none. Each request names its
conversation three ways, so the report shows which one the gateway let through:
  header   x-prefixwise-session-id and x-session-id: "h-<id>"
  body     OpenAI `user` / Anthropic `metadata.user_id`: "u-<id>"
With neither, Prefixwise has no conversation to compare against and the run says so.

Usage:
  python gateway_check.py --backend --proxy-port 4100 --mock-port 4101 --ledger gw.db --setup-only
      (starts the mock and the Prefixwise proxy, and waits; point the gateway at port 4100)
  python gateway_check.py --gateway http://127.0.0.1:8080 --ledger gw.db \\
      --openai-path /v1/chat/completions --anthropic-path /anthropic/v1/messages \\
      --openai-model openai/gpt-4o-mini --anthropic-model anthropic/claude-haiku-4-5 [--stream] [--header K:V]
      [--session-header NAME]
  python gateway_check.py --backend --gateway self ...   (no gateway: the client talks to the proxy
      directly, to check this script)
Prints a table and writes --out (JSON). Exit code 1 when anything is wrong.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK = os.path.join(HERE, "..", "litellm", "cachemock.py")
KINDS = ("unchanged", "tool_reserialize", "volatile_system", "system_edit", "history_edit")
EXPECT = {"unchanged": None, "tool_reserialize": "tool_reserialize", "volatile_system": "volatile_system",
          "system_edit": "system_edit", "history_edit": "history_edit"}
SYSTEM = ("You are a careful coding agent. Read before you edit, keep changes small, and explain each step "
          "in one sentence. ") * 150  # over Claude Haiku 4.5's 4,096-token minimum, so every change can cost
TASK = ("Fix the failing test in calc.py. The test expects add(2, 3) to equal 5 and mul(2, 3) to equal 6, "
        "and both currently fail because the operators are swapped in the implementation. ") * 12
TOOLS = [("read", "Read a file"), ("edit", "Replace text in a file"), ("bash", "Run a shell command")]


def tools_for(fmt, order):
    out = []
    for name, desc in order:
        schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
                   if fmt == "openai" else {"name": name, "description": desc, "input_schema": schema})
    return out


def conversation(kind, turn, sid):
    """(system text, tools in order, messages as (role, text)) for this turn of this conversation."""
    system = f"[{sid}] " + SYSTEM + " Current time: 2026-11-24T10:00:01Z."
    tools, first = list(TOOLS), TASK
    if turn >= 3:
        if kind == "tool_reserialize":
            tools = tools[1:] + tools[:1]
        elif kind == "volatile_system":
            system = system.replace("10:00:01", "10:07:42")
        elif kind == "system_edit":
            system = system.replace("keep changes small", "keep every change small")
        elif kind == "history_edit":
            first = "(edited) " + TASK
    msgs = [("user", first)]
    for t in range(1, turn):
        msgs += [("assistant", f"Step {t} done."), ("user", f"Continue with step {t + 1}.")]
    return system, tools, msgs


def request(fmt, kind, turn, sid, args):
    system, tools, msgs = conversation(kind, turn, sid)
    if fmt == "openai":
        body = {"model": args.openai_model, "max_tokens": 16, "user": f"u-{sid}",
                "tools": tools_for("openai", tools),
                "messages": [{"role": "system", "content": system}] + [{"role": r, "content": c} for r, c in msgs]}
        if args.stream:
            body.update(stream=True, stream_options={"include_usage": True})
        path = args.openai_path
    else:
        content = [{"role": r, "content": [{"type": "text", "text": c}]} for r, c in msgs]
        content[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        body = {"model": args.anthropic_model, "max_tokens": 16, "metadata": {"user_id": f"u-{sid}"},
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "tools": tools_for("anthropic", tools), "messages": content}
        if args.stream:
            body["stream"] = True
        path = args.anthropic_path
    headers = {"content-type": "application/json", "x-prefixwise-session-id": f"h-{sid}",
               "x-session-id": f"h-{sid}", "authorization": "Bearer sk-test", "x-api-key": "sk-test",
               "anthropic-version": "2023-06-01"}
    for h in args.header or []:
        k, _, v = h.partition(":")
        headers[k.strip().lower()] = v.strip()
    for name in args.session_header or []:  # a gateway that forwards only its own prefixed headers
        headers[name.strip().lower()] = f"h-{sid}"
    return path, body, headers


def send(base, path, body, headers):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
            return r.status, ""
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - report and go on
        return 0, f"{type(e).__name__}: {e}"[:300]


def wait_for(url, seconds=30):
    for _ in range(seconds * 10):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.1)
    return False


def start_backend(args):
    mock = subprocess.Popen([sys.executable, "-I", MOCK, str(args.mock_port)])
    wait_for(f"http://127.0.0.1:{args.mock_port}/")
    env = dict(os.environ, PREFIXWISE_LEDGER_PATH=os.path.abspath(args.ledger))
    proxy = subprocess.Popen([sys.executable, "-m", "prefixwise", "serve", "--upstream",
                              f"http://127.0.0.1:{args.mock_port}", "--port", str(args.proxy_port),
                              "--ledger", os.path.abspath(args.ledger)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_for(f"http://127.0.0.1:{args.proxy_port}/health"):
        sys.exit("prefixwise serve did not start")
    return [mock, proxy]


def rows_for(ledger, sid):
    with sqlite3.connect(ledger) as c:
        for name in ("h-" + sid, "u-" + sid):
            rows = c.execute("SELECT cause, prompt_tokens, cached_tokens, provider FROM entries "
                             "WHERE session_id = ? ORDER BY id", (name,)).fetchall()
            if rows:
                return ("header" if name.startswith("h-") else "body"), rows
    return "none", []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gateway", default="self", help="the gateway's base URL, or 'self' (the proxy directly)")
    p.add_argument("--backend", action="store_true", help="start the cache mock and prefixwise serve here")
    p.add_argument("--setup-only", action="store_true", help="with --backend: start them and wait")
    p.add_argument("--proxy-port", type=int, default=4100)
    p.add_argument("--mock-port", type=int, default=4101)
    p.add_argument("--ledger", default="gateway-check.db")
    p.add_argument("--formats", default="openai,anthropic")
    p.add_argument("--openai-path", default="/v1/chat/completions")
    p.add_argument("--anthropic-path", default="/v1/messages")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--anthropic-model", default="claude-haiku-4-5")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--header", action="append", help="extra header for the gateway, K:V (its own auth)")
    p.add_argument("--session-header", action="append",
                   help="also send the session id under this header name (e.g. x-bf-eh-x-prefixwise-session-id)")
    p.add_argument("--out", default="gateway-check.json")
    args = p.parse_args()
    procs = start_backend(args) if args.backend else []
    try:
        if args.setup_only:
            print(f"mock on {args.mock_port}, prefixwise serve on {args.proxy_port}, ledger {args.ledger}; "
                  "Ctrl-C to stop", flush=True)
            while True:
                time.sleep(3600)
        base = f"http://127.0.0.1:{args.proxy_port}" if args.gateway == "self" else args.gateway
        results, bad = [], 0
        for fmt in args.formats.split(","):
            for kind in KINDS:
                sid = f"{fmt[:2]}-{kind[:6]}-{uuid.uuid4().hex[:6]}"
                errors = []
                for turn in range(1, 6):
                    path, body, headers = request(fmt, kind, turn, sid, args)
                    status, err = send(base, path, body, headers)
                    if status != 200:
                        errors.append(f"turn {turn}: {status} {err}")
                    time.sleep(0.2)
                time.sleep(1.0)
                via, rows = rows_for(args.ledger, sid)
                causes = [r[0] for r in rows]
                want = ["first_write", "none", EXPECT[kind] or "none", "none", "none"]
                ok = not errors and causes == want
                bad += not ok
                results.append({"format": fmt, "kind": kind, "session_via": via, "causes": causes,
                                "expected": want, "reads": [f"{r[2]}/{r[1]}" for r in rows],
                                "provider": sorted({r[3] for r in rows}), "errors": errors, "ok": ok})
                print(f"{'ok ' if ok else 'BAD'} {fmt:9s} {kind:16s} session via {via:6s} {causes}"
                      + (f"  errors: {errors[:2]}" if errors else ""), flush=True)
        recorded = sum(len(r["causes"]) for r in results)
        sent = 5 * len(results)
        print(f"\n{recorded} of {sent} calls recorded; {len(results) - bad} of {len(results)} conversations right")
        json.dump({"gateway": args.gateway, "stream": args.stream, "formats": args.formats,
                   "recorded": recorded, "sent": sent, "results": results}, open(args.out, "w"), indent=1)
        sys.exit(1 if bad else 0)
    finally:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main()
