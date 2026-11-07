"""The `prefixwise` command line.

    prefixwise serve      run the proxy (ledger on disk by default)
    prefixwise report     where tokens went: totals, causes, sessions, tasks
    prefixwise task       record a task outcome (success / failure)
    prefixwise request    show one request's verdict by request id
    prefixwise explain    offline: why does the cache miss between requests?
    prefixwise fixes      which fixes would save the most, from your own ledger
    prefixwise prices     the dated price table used for dollar costs
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


def _fmt_usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:,.4f}"


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
              fixes=args.fix or None, allow_unauthenticated=args.allow_unauthenticated,
              prices=args.prices)
    except ValueError as exc:  # unknown --fix name
        sys.exit(f"prefixwise: {exc}")
    return 0


def cmd_fixes(args) -> int:
    from .fixes import recommend

    rec = recommend(_open_existing_ledger(args))
    recs, fixed_requests = rec["recommendations"], rec["requests_rewritten_by_fixes"]
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
        if r["predicted_savings_usd"] is not None:
            unpriced = f" ({r['unpriced_requests']} requests unpriced)" if r["unpriced_requests"] else ""
            print(f"   worth: {_fmt_usd(r['predicted_savings_usd'])}{unpriced}")
        print(f"   fix: {r['fix']}")
        if toggle:
            print(toggle)
        if r["evidence"]:
            print(f"   {r['status']}: {r['evidence']}")
    if fixed_requests:
        print("\nAlready active: " + ", ".join(f"{k} rewrote {v} requests" for k, v in fixed_requests.items()))
    return 0


def cmd_report(args) -> int:
    from . import api
    from .findings import findings, render

    led = _open_existing_ledger(args)
    try:
        since = api._time(args.since, "--since")
    except api.ApiError as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    found = findings(led, since=since)
    detail = args.detail or args.by is not None
    args.by = args.by or "cause"
    summary = led.summary()
    rows = led.rollup(args.by)[: args.limit]
    if args.json:
        print(json.dumps({"findings": found, "summary": summary, "by": args.by, "rows": rows}, indent=2))
        return 0
    print(render(found))
    if since is not None or not detail:
        if since is None:
            print("\nDetails: prefixwise report --detail   Fixes: prefixwise fixes   Data: prefixwise export")
        return 0
    print()
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
    priced, n = summary["priced_requests"], summary["requests"]
    if priced:
        coverage = "" if priced == n else f"   ({priced:,} of {n:,} requests priced; see `prefixwise prices`)"
        print(f"Cost                {_fmt_usd(summary['cost'])}{coverage}")
        print(f"Avoidable cost      {_fmt_usd(summary['avoidable_cost'])}")
        cpst = summary["cost_per_successful_task"]
        print(f"Cost per successful task  {_fmt_usd(cpst) if cpst is not None else '- (needs task outcomes, all requests priced)'}")
    elif n:
        print("Cost                - (no request's model is in the price table; see `prefixwise prices`)")
    if not rows:
        return 0
    print()
    print(_table([[r[args.by] if r[args.by] is not None else "(none)", _fmt_int(r["requests"]),
                   _fmt_int(r["prompt_tokens"]), _fmt_pct(r["cached_tokens"] / r["prompt_tokens"])
                   if r["prompt_tokens"] else "-", _fmt_int(r["avoidable_tokens"]),
                   _fmt_int(r["waste_requests"]), _fmt_int(r["tokens_per_successful_task"]),
                   _fmt_usd(r["cost"]) if r["priced_requests"] else "-",
                   _fmt_usd(r["avoidable_cost"]) if r["priced_requests"] else "-"]
                  for r in rows],
                 [args.by, "requests", "prompt", "cached", "avoidable", "waste reqs",
                  "tokens/success", "cost", "avoidable $"]))
    return 0


def cmd_prices(args) -> int:
    from .pricing import PriceTable

    try:
        table = PriceTable.load(args.prices)
    except (OSError, ValueError) as exc:
        sys.exit(f"prefixwise: {exc}")
    if args.json:
        print(json.dumps({"as_of": table.as_of, "sources": table.sources, "models": table.models}, indent=2))
        return 0
    print(f"Price table as of {table.as_of} (USD per 1M tokens), from: {' + '.join(table.sources)}")
    rows = [[name, r.get("provider", "-"), _rate(r.get("input")), _rate(r.get("cache_read")),
             _rate(r.get("cache_write_5m", r.get("cache_write"))), _rate(r.get("cache_write_1h")),
             _rate(r.get("output")), "yes" if r.get("long_prompt") else "", r.get("valid_until", "")]
            for name, r in sorted(table.models.items())]
    print(_table(rows, ["model", "provider", "input", "cache read", "cache write", "write 1h", "output",
                        "tiered", "valid until"]))
    print("\ncache write: Anthropic's 5-minute write, OpenAI's 30-minute write (the defaults).")
    print("\nModels not listed are unpriced: their cost shows as '-', never estimated. Add rates with\n"
          "PREFIXWISE_PRICES=/path/prices.json (same format as the built-in table; see `--json`).")
    return 0


def _rate(v) -> str:
    return "-" if v is None else (v if isinstance(v, str) else f"{v:g}")


def cmd_task(args) -> int:
    from .ledger import Ledger

    Ledger(_ledger_path(args)).set_task_outcome(args.task_id, args.outcome == "success")
    print(f"task {args.task_id}: {args.outcome}")
    return 0


def _duration_seconds(text: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    t = text.strip().lower()
    if t and t[-1] in units:
        return float(t[:-1]) * units[t[-1]]
    return float(t) * 86400  # a bare number is days


def cmd_prune(args) -> int:
    from .ledger import Ledger

    try:
        seconds = _duration_seconds(args.older_than)
    except ValueError:
        print(f"prefixwise: --older-than takes a duration like 30d, 12h or 90 (days), not {args.older_than!r}",
              file=sys.stderr)
        return 2
    n = Ledger(_ledger_path(args)).prune(seconds, undated=args.include_undated)
    print(f"deleted {n} ledger entries older than {args.older_than}")
    return 0


def cmd_run(args) -> int:
    """Run an agent through Prefixwise and report on that run when it exits."""
    import subprocess
    import threading
    import time

    from .findings import findings, render
    from .proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler

    cmd = args.agent_command[1:] if args.agent_command[:1] == ["--"] else args.agent_command
    if not cmd:
        print("usage: prefixwise run [options] -- <agent command>, e.g. prefixwise run -- claude", file=sys.stderr)
        return 2
    # Where the agent would have gone: its own base URLs if set (a gateway), else the providers.
    anthropic = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    openai = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    openai = openai[:-3] if openai.endswith("/v1") else openai
    ledger = args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER
    try:
        proxy = PrefixwiseProxy(ledger_path=ledger, mock=True if args.mock else None, fixes=args.fix or None,
                                upstreams=None if args.mock else {"anthropic": anthropic, "openai": openai})
    except ValueError as exc:
        sys.exit(f"prefixwise: {exc}")
    proxy.default_session = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    server = PrefixwiseHTTPServer(("127.0.0.1", args.port), make_handler(proxy))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {**os.environ, "ANTHROPIC_BASE_URL": base, "OPENAI_BASE_URL": base + "/v1", "PREFIXWISE_URL": base}
    upstream = "mock" if args.mock else f"anthropic -> {anthropic}, openai -> {openai}"
    print(f"prefixwise: proxy on {base} ({upstream}); ledger {ledger}", file=sys.stderr, flush=True)
    start = time.time()
    try:
        code = subprocess.call(cmd, env=env)
    except FileNotFoundError:
        print(f"prefixwise: command not found: {cmd[0]}", file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130
    server.shutdown()
    server.drain(5)
    server.server_close()
    found = findings(proxy.ledger, since=start)
    print("\n-- prefixwise: this run --", file=sys.stderr)
    if not (found["summary"]["requests"] or found["summary"]["requests_failed"]):
        print(f"No requests reached Prefixwise. Does {cmd[0]} read ANTHROPIC_BASE_URL or OPENAI_BASE_URL? "
              "If not, set its base URL to the proxy by hand (see the README).", file=sys.stderr)
    else:
        print(render(found), file=sys.stderr)
        print(f"All runs: prefixwise report --ledger {ledger}", file=sys.stderr)
    return code


def cmd_export(args) -> int:
    """Raw ledger rows (facts, never prompt text) as JSONL or CSV, oldest first."""
    import csv

    from . import api, metrics

    led = _open_existing_ledger(args)
    try:
        since = api._time(args.since, "--since")
        filters = dict(f.split("=", 1) for f in args.where)
        where, params = metrics._where(filters, since, None)
    except (ValueError, api.ApiError) as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    if args.output:  # owner-only, like the ledger itself
        out = os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", newline="")
    else:
        out = sys.stdout
    with led.reader() as conn:
        cur = conn.execute(f"SELECT * FROM entries{where} ORDER BY id", params)
        names = [d[0] for d in cur.description]
        writer = csv.writer(out) if args.format == "csv" else None
        if writer:
            writer.writerow(names)
        n = 0
        for row in cur:
            n += 1
            if writer:
                writer.writerow(row)
            else:
                out.write(json.dumps(dict(zip(names, row))) + "\n")
    if args.output:
        out.close()
        print(f"wrote {n} rows to {args.output}", file=sys.stderr)
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
    p.add_argument("--prices", metavar="FILE",
                   help="extra or overriding rates (JSON, same format as `prefixwise prices --json`); "
                        "default: $PREFIXWISE_PRICES")
    p.add_argument("--allow-unauthenticated", action="store_true",
                   help="listen beyond localhost with PREFIXWISE_API_KEY set but no PREFIXWISE_AUTH_TOKEN "
                        "(anyone who reaches the port can spend the key)")
    p.add_argument("--fix", action="append", metavar="NAME",
                   help="apply a fix to every request before forwarding (repeatable). "
                        "Available: canonical_tools. Default: $PREFIXWISE_FIXES or none")
    ledger_opt(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="summarise the ledger")
    p.add_argument("--by", default=None, choices=["cause", "session_id", "agent_id", "model",
                                                  "provider", "task_id"],
                   help="print the per-group table, grouped by this (implies --detail)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    ledger_opt(p)
    p.add_argument("--since", metavar="WHEN", help="only requests since then: unix seconds or 24h, 7d, ...")
    p.add_argument("--detail", action="store_true", help="also print totals and the per-group table")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("prices", help="the dated price table used for dollar costs")
    p.add_argument("--prices", metavar="FILE", help="also load this rates file (default: $PREFIXWISE_PRICES)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_prices)

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

    p = sub.add_parser("run", help="run an agent through Prefixwise, then report on that run",
                       description="Starts the proxy on a free local port, points the agent at it with "
                                   "ANTHROPIC_BASE_URL and OPENAI_BASE_URL (the agent keeps its own keys), "
                                   "and prints what this run's cache did when the agent exits.")
    p.add_argument("--port", type=int, default=0, help="default: any free port")
    p.add_argument("--mock", action="store_true", help="synthetic responses, no API calls")
    p.add_argument("--fix", action="append", metavar="NAME", help="apply a fix (repeatable): canonical_tools")
    ledger_opt(p)
    p.add_argument("agent_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="raw ledger rows as JSONL or CSV (facts, never prompt text)")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p.add_argument("--since", metavar="WHEN", help="unix seconds, or a duration back from now: 24h, 7d")
    p.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                   help="filter (repeatable): cause, session, agent, model, provider, task, api, error")
    p.add_argument("-o", "--output", metavar="FILE", help="default: stdout")
    ledger_opt(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("prune", help="delete old ledger entries (retention)")
    p.add_argument("--older-than", required=True, metavar="AGE", help="e.g. 30d, 12h, 90 (days)")
    p.add_argument("--include-undated", action="store_true",
                   help="also delete entries recorded before 0.3.1 (they have no timestamp)")
    ledger_opt(p)
    p.set_defaults(func=cmd_prune)

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
    try:
        return args.func(args)
    except BrokenPipeError:
        # `prefixwise report | head`: the reader left early. Exit quietly, like
        # standard Unix tools, instead of printing a traceback.
        import os
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1  # as Python's docs recommend for SIGPIPE in CLIs


if __name__ == "__main__":
    raise SystemExit(main())
