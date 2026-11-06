"""The `prefixwise` command line.

    prefixwise serve      run the proxy (ledger on disk by default)
    prefixwise report     where tokens went: totals, causes, sessions, tasks
    prefixwise task       record a task outcome (success / failure)
    prefixwise request    show one request's verdict by request id
    prefixwise explain    offline: why does the cache miss between requests?
    prefixwise fixes      which fixes would save the most, from your own ledger
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__

DEFAULT_LEDGER = "prefixwise.db"


def _ledger_path(args) -> str:
    return args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER


def _open_existing_ledger(args):
    from .ledger import Ledger

    path = _ledger_path(args)
    if path != ":memory:" and not Path(path).exists():
        sys.exit(f"prefixwise: no ledger at {path} (run `prefixwise serve` first, or pass --ledger)")
    return Ledger(path)


def _fmt_int(n) -> str:
    return "-" if n is None else f"{int(n):,}"


def _fmt_pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    def line(cells):
        return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


# -- commands ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    from .proxy import serve

    ledger = args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER
    try:
        serve(host=args.host, port=args.port, ledger_path=ledger,
              upstream_url=args.upstream, mock=True if args.mock else None,
              fixes=args.fix or None)
    except ValueError as exc:  # unknown --fix name
        sys.exit(f"prefixwise: {exc}")
    return 0


def cmd_fixes(args) -> int:
    from .fixes import FIX_ADVICE

    led = _open_existing_ledger(args)
    summary = led.summary()
    rows = {r["cause"]: r for r in led.rollup("cause")}
    with led._lock:
        fixed_requests = dict(led.conn.execute(
            "SELECT fixes, COUNT(*) FROM entries WHERE fixes IS NOT NULL GROUP BY fixes").fetchall())
    recs = []
    for cause, advice in FIX_ADVICE.items():
        r = rows.get(cause)
        if not r or not r["avoidable_tokens"]:
            continue
        recs.append({"cause": cause, "requests": r["requests"], "predicted_savings_tokens": r["avoidable_tokens"],
                     "share_of_prompt_tokens": (r["avoidable_tokens"] / summary["prompt_tokens"])
                     if summary["prompt_tokens"] else None,
                     "fix": advice["fix"], "proxy_toggle": advice["toggle"],
                     "status": "recommended (predicted; prove it with experiments/e4)"})
    recs.sort(key=lambda x: -x["predicted_savings_tokens"])
    if args.json:
        print(json.dumps({"recommendations": recs, "requests_rewritten_by_fixes": fixed_requests}, indent=2))
        return 0
    if not recs:
        print("No avoidable waste with a known fix in this ledger.")
    for i, r in enumerate(recs, 1):
        toggle = f"   proxy can apply it: prefixwise serve --fix {r['proxy_toggle']}" if r["proxy_toggle"] else ""
        print(f"{i}. {r['cause']}: {_fmt_int(r['predicted_savings_tokens'])} avoidable tokens "
              f"({_fmt_pct(r['share_of_prompt_tokens'])} of prompt tokens) over {r['requests']} "
              f"request{'s' if r['requests'] != 1 else ''}")
        print(f"   fix: {r['fix']}")
        if toggle:
            print(toggle)
    if fixed_requests:
        print("\nAlready active: " + ", ".join(f"{k} rewrote {v} requests" for k, v in fixed_requests.items()))
    return 0


def cmd_report(args) -> int:
    led = _open_existing_ledger(args)
    summary = led.summary()
    rows = led.rollup(args.by)[: args.limit]
    if args.json:
        print(json.dumps({"summary": summary, "by": args.by, "rows": rows}, indent=2))
        return 0
    t = summary["tasks"]
    print(f"Requests            {_fmt_int(summary['requests'])}")
    print(f"Prompt tokens       {_fmt_int(summary['prompt_tokens'])}"
          f"   (cache hit rate {_fmt_pct(summary['cache_hit_rate'])})")
    print(f"Completion tokens   {_fmt_int(summary['completion_tokens'])}")
    print(f"Avoidable tokens    {_fmt_int(summary['avoidable_tokens'])}"
          f"   ({_fmt_pct(summary['avoidable_share'])} of prompt tokens)")
    tpst = summary["tokens_per_successful_task"]
    print(f"Tasks               {t['successful']} succeeded / {t['resolved']} resolved")
    print(f"Tokens per successful task  {_fmt_int(tpst) if tpst is not None else '- (no task outcomes recorded)'}")
    if not rows:
        return 0
    print()
    print(_table([[r[args.by] if r[args.by] is not None else "(none)", _fmt_int(r["requests"]),
                   _fmt_int(r["prompt_tokens"]), _fmt_pct(r["cached_tokens"] / r["prompt_tokens"])
                   if r["prompt_tokens"] else "-", _fmt_int(r["avoidable_tokens"]),
                   _fmt_int(r["waste_requests"]), _fmt_int(r["tokens_per_successful_task"])]
                  for r in rows],
                 [args.by, "requests", "prompt", "cached", "avoidable", "waste reqs",
                  "tokens/success"]))
    return 0


def cmd_task(args) -> int:
    from .ledger import Ledger

    Ledger(_ledger_path(args)).set_task_outcome(args.task_id, args.outcome == "success")
    print(f"task {args.task_id}: {args.outcome}")
    return 0


def cmd_request(args) -> int:
    row = _open_existing_ledger(args).get_request(args.request_id)
    if row is None:
        print(f"prefixwise: no request {args.request_id} in the ledger", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(row, indent=2))
    else:
        for k, v in row.items():
            print(f"{k:20} {v}")
    return 0


def _print_explanation(label: str, e) -> None:
    legit = {True: "legitimate", False: "WASTE", None: "undetermined"}[e.legit]
    if e.cause == "none":
        legit = "ok, prefix reused"
    print(f"{label}: {e.cause.upper()}  [{legit}]")
    if e.divergence:
        d = e.divergence
        print(f"  first difference: block {d['index']} ({d['role']}): {d['kind']}")
    print(f"  why: {e.evidence}")
    if e.prompt_tokens is not None:
        print(f"  usage: {_fmt_int(e.cached_tokens)} of {_fmt_int(e.prompt_tokens)} prompt tokens cached")


def cmd_explain(args) -> int:
    from .explain import explain_pair, explain_session, load_turns

    provider = None if args.provider == "auto" else args.provider
    try:
        if len(args.files) == 2:
            prev, curr = (load_turns(f) for f in args.files)
            if len(prev) != 1 or len(curr) != 1:
                sys.exit("prefixwise: with two files, each must hold exactly one request")
            results = [explain_pair(prev[0], curr[0], provider)]
            labels = [f"{args.files[0]} -> {args.files[1]}"]
        elif len(args.files) == 1:
            turns = load_turns(args.files[0])
            results = explain_session(turns, provider)
            labels = [f"turn {i + 1} ({t.session_id})" for i, t in enumerate(turns)]
        else:
            sys.exit("prefixwise: give one session file (JSON / JSONL) or two request files")
    except (OSError, ValueError) as exc:
        sys.exit(f"prefixwise: {exc}")
    waste = sum(1 for r in results if r.legit is False)
    fail = 2 if (args.fail_on_waste and waste) else 0
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return fail
    if results and results[0].offline:
        print("(offline: no usage given, so this explains what in the request breaks reuse;"
              " TTL, eviction and routing need real usage)\n")
    for label, r in zip(labels, results):
        _print_explanation(label, r)
    if len(results) > 1:
        print(f"\n{len(results)} turns, {waste} wasteful")
    return fail


# -- parser -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="prefixwise",
                                 description="Name why prompt caches miss, and price the miss.")
    ap.add_argument("--version", action="version", version=f"prefixwise {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def ledger_opt(p):
        p.add_argument("--ledger", help=f"SQLite ledger file (default: $PREFIXWISE_LEDGER_PATH or ./{DEFAULT_LEDGER})")

    p = sub.add_parser("serve", help="run the proxy")
    p.add_argument("--host", default=os.environ.get("PREFIXWISE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    p.add_argument("--upstream", help="upstream base URL, e.g. https://api.openai.com "
                                      "(default: $PREFIXWISE_UPSTREAM_URL; none = mock mode). "
                                      "API key: $PREFIXWISE_API_KEY, or the client's own auth header.")
    p.add_argument("--mock", action="store_true", help="synthetic responses, no upstream calls")
    p.add_argument("--fix", action="append", metavar="NAME",
                   help="apply a fix to every request before forwarding (repeatable). "
                        "Available: canonical_tools. Default: $PREFIXWISE_FIXES or none")
    ledger_opt(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="summarise the ledger")
    p.add_argument("--by", default="cause", choices=["cause", "session_id", "agent_id", "model",
                                                     "provider", "task_id"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    ledger_opt(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("fixes", help="fixes ranked by the tokens they would save, from your ledger")
    p.add_argument("--json", action="store_true")
    ledger_opt(p)
    p.set_defaults(func=cmd_fixes)

    p = sub.add_parser("task", help="record a task outcome")
    p.add_argument("task_id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--success", dest="outcome", action="store_const", const="success")
    g.add_argument("--failure", dest="outcome", action="store_const", const="failure")
    ledger_opt(p)
    p.set_defaults(func=cmd_task)

    p = sub.add_parser("request", help="show one request's ledger entry")
    p.add_argument("request_id", help="the X-Prefixwise-Request-Id response header")
    p.add_argument("--json", action="store_true")
    ledger_opt(p)
    p.set_defaults(func=cmd_request)

    p = sub.add_parser("explain", help="offline: why does the cache miss between requests?",
                       description="Explain cache misses offline. Give two request files (JSON), "
                                   "or one session file: a JSON array or JSONL of request bodies, "
                                   'or of {"request": body, "usage": {...}} records.')
    p.add_argument("files", nargs="+")
    p.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic", "vllm"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on-waste", action="store_true",
                   help="exit 2 if any turn is classified as waste (for CI gates on prompt changes)")
    p.set_defaults(func=cmd_explain)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
