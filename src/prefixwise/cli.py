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
from .check import BASELINE_FILE, DEFAULT_FAIL_ON, DEFAULT_FIRST_TURN_DELAY, STRICT_EXTRA

DEFAULT_LEDGER = "prefixwise.db"


def _ledger_path(args) -> str:
    return args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER


def _open_existing_ledger(args):
    from .ledger import Ledger

    path = _ledger_path(args)
    if path != ":memory:" and not Path(path).exists():
        sys.exit(f"prefixwise: no ledger at {path} yet. Run an agent through Prefixwise first: "
                 f"`prefixwise run -- <your agent>` (or `prefixwise serve`), or pass --ledger")
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

def _detect_engine(upstream: str | None, engine: str | None, kv_events: str | None, full_attention: bool | None,
                   mock: bool = False, probe=None) -> tuple[str | None, str | None, list[str]]:
    """Fill --engine and --kv-events from what the upstream says about itself when neither was given:
    a vLLM or SGLang server is one engine (`--engine`), and one that publishes KV events gets
    `--kv-events auto` when the engines extra is installed. Explicit flags always stand. Returns
    (engine, kv_events, lines to print)."""
    if engine == "none":  # a hosted API or a router: no engine, and no probe
        return None, kv_events, []
    if engine is not None or mock or not upstream:
        return engine, kv_events, []
    from .engines import probe_upstream
    info = (probe or probe_upstream)(upstream, api_key=os.environ.get("PREFIXWISE_API_KEY"))
    lines = [f"prefixwise: {n}" for n in info["notes"] if "did not answer" not in n]
    if info["engine"] == "sglang" and kv_events is None and info["publishes_events"]:
        lines.append("prefixwise: if this URL is a router in front of several SGLang servers (it relays their "
                     "/server_info), pass --engine none: one engine's verdicts do not hold behind a router")
    if not info["engine"]:
        return engine, kv_events, lines
    name = {"vllm": "vLLM", "sglang": "SGLang"}[info["engine"]]
    said = [f"--engine {info['engine']}"]
    if kv_events is None and info["publishes_events"]:
        try:
            import msgspec  # noqa: F401
            import zmq  # noqa: F401
            kv_events = "auto"
            said.append("--kv-events auto (it publishes KV events)")
        except ImportError:
            lines.append("prefixwise: the upstream publishes KV events; pip install 'prefixwise[engines]' to confirm "
                         "evictions from them (or pass --kv-events)")
    elif kv_events is None and info["publishes_events"] is False:
        lines.append(f"prefixwise: {name} publishes no KV events, so evictions are inferred; see docs/"
                     f"{'SGLANG' if info['engine'] == 'sglang' else 'VLLM'}.md to turn them on")
    lines.insert(0, f"prefixwise: upstream is {name}{' ' + info['version'] if info['version'] else ''} (detected): "
                    + ", ".join(said))
    if info["engine"] == "sglang" and kv_events and full_attention is None:
        lines.append("prefixwise: add --full-attention if the model has no sliding-window or Mamba layers, so the "
                     "events can also rule an eviction out (and confirm routing misses across ranks)")
    return info["engine"], kv_events, lines


def cmd_serve(args) -> int:
    from .proxy import serve

    ledger = args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER
    engine, kv_events, lines = _detect_engine(args.upstream or os.environ.get("PREFIXWISE_UPSTREAM_URL"), args.engine,
                                              args.kv_events, args.full_attention, mock=bool(args.mock))
    for line in lines:
        print(line, file=sys.stderr, flush=True)
    try:
        serve(host=args.host, port=args.port, ledger_path=ledger,
              upstream_url=args.upstream, mock=True if args.mock else None,
              fixes=_fixes(args), allow_unauthenticated=args.allow_unauthenticated,
              prices=args.prices, engine=engine, kv_events=kv_events,
              full_attention=args.full_attention, otel=args.otel, collector_url=args.collector, sinks=args.sink)
    except (ValueError, OSError) as exc:  # unknown --fix, --engine, --otel or --sink value
        sys.exit(f"prefixwise: {exc}")
    return 0


def cmd_collect(args) -> int:
    from .proxy import serve

    ledger = args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER
    try:
        serve(host=args.host, port=args.port, ledger_path=ledger, prices=args.prices,
              allow_unauthenticated=args.allow_unauthenticated, otel="", collect_only=True,
              log_url=args.log, log_partitions=args.partitions, sinks=args.sink)
    except (ValueError, ImportError, OSError) as exc:  # bad --partitions or --sink, or redis-py missing
        sys.exit(f"prefixwise: {exc}")
    except Exception as exc:  # noqa: BLE001 - an unreachable or refusing Redis
        if type(exc).__module__.startswith("redis"):
            sys.exit(f"prefixwise: the record log is not reachable ({type(exc).__name__}: {exc})")
        raise
    return 0


def cmd_fixes(args) -> int:
    from .fixes import recommend

    rec = recommend(_open_existing_ledger(args))
    recs, fixed_requests, paid = rec["recommendations"], rec["requests_rewritten_by_fixes"], rec.get("auto_breakpoints")
    if args.json:
        print(json.dumps(rec, indent=2))
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
    if paid:
        verdict = ("paid for itself" if paid["paid"] else "has not paid for itself yet") if paid["paid"] is not None \
            else "no cache writes or reads on those requests yet"
        print(f"   auto_breakpoints: wrote {_fmt_int(paid['written_tokens'])} tokens to the cache on "
              f"{paid['requests']} requests and read {_fmt_int(paid['read_tokens'])} back: {verdict} "
              f"(reads saved {_fmt_int(paid['read_saving_token_equivalents'])} input-token equivalents against a "
              f"write surcharge of {_fmt_int(paid['write_surcharge_token_equivalents'])}; a model: 5-minute write "
              "1.25x, read 0.1x, over the requests it rewrote)")
    return 0


def _report_session(led, session: str, as_json: bool, since: float | None = None) -> int:
    """`report --session ID`: the session turn by turn. Which turn stopped reading the cache, why, and
    the block where the prompt first changed, so a compaction or a tool change shows as the turn it
    broke the cache on. Hashes and counts only: the ledger holds no prompt text."""
    sid = led.latest_session() if session == "latest" else session
    turns = led.turns(sid, since=since) if sid else []
    if not turns:
        print(f"prefixwise: no requests recorded for session {session!r}" + (" in that window" if since else "")
              if sid else "prefixwise: the ledger has no sessions yet", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps({"session_id": sid, "turns": turns}, indent=2, default=str))
        return 0
    from .classifier import CACHE_NOT_REPORTED
    blind = {t["turn"] for t in turns if t.get("evidence") == CACHE_NOT_REPORTED}  # the server gave no count
    prompt = sum(t["prompt_tokens"] or 0 for t in turns)
    cached = sum(t["cached_tokens"] or 0 for t in turns)
    avoidable = sum(t["avoidable_tokens"] or 0 for t in turns)
    priced = [t for t in turns if t.get("cost") is not None]
    cost = sum(t["cost"] for t in priced)
    models = sorted({t["model"] for t in turns if t.get("model")})
    agents = sorted({t["agent_id"] for t in turns if t.get("agent_id")})
    print(f"Session {sid}: {len(turns)} requests, {_fmt_int(prompt)} prompt tokens, "
          f"cache hit rate {'not reported by the server' if blind else _fmt_pct(cached / prompt) if prompt else '-'}, "
          f"{_fmt_int(avoidable)} avoidable" + (f", cost {_fmt_usd(cost)}" if priced else "")
          + (f" ({len(priced)} of {len(turns)} priced)" if priced and len(priced) < len(turns) else "")
          + (f"   ({', '.join(models)})" if len(models) == 1 else ""))
    from .findings import BILL_PARTS, bill_line, floor_advice
    split = [t for t in turns if t.get("cost_cache_read") is not None]
    if split:
        b = {k: sum(t[f"cost_{k}"] or 0 for t in split) for k, _ in BILL_PARTS}
        b.update(total=sum(b.values()), requests=len(split), requests_priced=len(priced))
        for line in (bill_line(b), floor_advice(b)):
            if line:
                print(line)
    # The floor: a turn that hits re-reads the context already cached, and pays for it again at the
    # cache rate; it grows with the conversation.
    reread = [t["cached_tokens"] for t in turns if t.get("cached_tokens") and t["turn"] not in blind]
    if len(reread) >= 2:
        print(f"Re-read per turn: {len(reread)} turns read from the cache, {_fmt_int(sum(reread) // len(reread))} "
              f"tokens on average, at most {_fmt_int(max(reread))}: each paid for that context again, at the "
              f"cache rate.")
    print()
    # Each model has its own cache and each agent (a subagent) its own conversation: when a session
    # mixes them, say which row is which, since "first changed" compares within a model and conversation.
    extra = (["model"] if len(models) > 1 else []) + (["agent"] if len(agents) > 1 else [])
    rows = []
    for t in turns:
        tag = ([t.get("model") or "-"] if len(models) > 1 else []) + ([t.get("agent_id") or "-"] if len(agents) > 1 else [])
        if t.get("status") and t["status"] >= 400:
            rows.append([t["turn"], *tag, "-", "-", "-", f"failed: {t.get('error') or t['status']}", "-", "-", "-", "-"])
            continue
        cause = t.get("cause") or "-"
        if t.get("detail"):
            cause += f" ({t['detail']})"
        known = t["turn"] not in blind
        rows.append([t["turn"], *tag, _fmt_int(t["prompt_tokens"] or 0),
                     _fmt_int(t["cached_tokens"] or 0) if known else "?",
                     _fmt_pct(t["hit_rate"]) if t["hit_rate"] is not None and known else "?" if not known else "-", cause,
                     t["first_changed"] or "-", _fmt_usd(t["cost"]) if t.get("cost") is not None else "-",
                     _fmt_int(t["avoidable_tokens"] or 0),
                     _fmt_usd(t["avoidable_cost"]) if t.get("avoidable_cost") is not None else "-"])
    print(_table(rows, ["turn", *extra, "prompt", "cached", "hit", "cause", "first changed", "cost", "avoidable",
                        "avoidable $"]))
    if blind:
        print("\n?: the server reports no cached-token count, so a hit cannot be told from a miss "
              "(Ollama and many llama.cpp builds leave it out; vLLM needs --enable-prompt-tokens-details).")
    print("\nfirst changed: the block where this request's prompt first differs from this session's previous request "
          "on the same model and conversation. The number is the block's position in the whole prompt, from 0: "
          "tools, then system, then the conversation (one position per tool, system prompt and message; "
          "tool_results[n] is a tool result). On an append-only hit it is the first block of the new turn. "
          "One request: prefixwise request <id>")
    return 0


def cmd_report(args) -> int:
    from . import api
    from .findings import findings, render, render_share

    led = _open_existing_ledger(args)
    try:
        since = api._time(args.since, "--since")
    except api.ApiError as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    from .pricing import PriceTable
    try:
        table = PriceTable.load(getattr(args, "prices", None))
    except (OSError, ValueError) as exc:
        sys.exit(f"prefixwise: {exc}")
    if getattr(args, "session", None):
        return _report_session(led, args.session, args.json, since)
    found = findings(led, since=since, prices=table)
    detail = args.detail or args.by is not None
    args.by = args.by or "cause"
    summary = led.summary(since=since)
    rows = led.rollup(args.by, since=since)[: args.limit]
    if args.json:
        print(json.dumps({"findings": found, "summary": summary, "by": args.by, "rows": rows}, indent=2))
        return 0
    if args.share:
        print(render_share(found, args.since))
        return 0
    print(render(found))
    if not detail:
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


def _show_url(url: str) -> str:
    """scheme://host[:port] only: a gateway URL can carry credentials."""
    from urllib.parse import urlsplit

    u = urlsplit(url)
    return f"{u.scheme}://{u.hostname}{':' + str(u.port) if u.port else ''}" if u.hostname else "?"


def _fixes(args) -> list[str] | None:
    """--fix NAME... plus 'optimize' for --optimize; None when neither was given (the env decides)."""
    names = list(args.fix or [])
    if getattr(args, "optimize", False):
        names.append("optimize")
    return names or None


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit
    u = urlsplit(url)
    return f"{u.hostname}:{u.port or (443 if u.scheme == 'https' else 80)}"


def _is_prefixwise(url: str) -> bool:
    """Does this local base URL already point at a Prefixwise proxy?"""
    import urllib.request
    from urllib.parse import urlsplit

    u = urlsplit(url)
    if u.hostname not in ("127.0.0.1", "localhost", "::1"):
        return False
    try:
        with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/health", timeout=1) as r:
            body = json.loads(r.read())
        return isinstance(body, dict) and "ok" in body and "mock" in body
    except Exception:
        return False


def _codex_args(base: str) -> list[str]:
    """Codex ignores OPENAI_BASE_URL (codex-cli 0.167+). A custom model provider, set
    with -c overrides, sends its Responses calls to the proxy over plain HTTP + SSE
    (the built-in provider tries WebSocket first), takes the key from
    OPENAI_API_KEY, skips the sign-in screen and writes nothing to ~/.codex."""
    p = "model_providers.prefixwise."
    return ["-c", "model_provider=\"prefixwise\"", "-c", f'{p}name="Prefixwise"',
            "-c", f'{p}base_url="{base}/v1"', "-c", f'{p}env_key="OPENAI_API_KEY"',
            "-c", f'{p}wire_api="responses"', "-c", f"{p}supports_websockets=false"]


def cmd_run(args) -> int:
    """Run an agent through Prefixwise and report on that run when it exits."""
    import signal
    import subprocess
    import threading
    import time

    from .findings import findings, render
    from .proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler

    cmd = args.agent_command[1:] if args.agent_command[:1] == ["--"] else args.agent_command
    if not cmd:
        print("usage: prefixwise run [options] -- <agent command>, e.g. prefixwise run -- claude", file=sys.stderr)
        return 2
    # Where the agent would have gone: its own base URLs if set (a gateway), kept exactly
    # as given, else the providers. OPENAI_BASE_URL includes the API version path.
    anthropic = (os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
    openai = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    for name, url in (("ANTHROPIC_BASE_URL", anthropic), ("OPENAI_BASE_URL", openai)):
        if not args.mock and _is_prefixwise(url):
            print(f"prefixwise: {name} already points at a Prefixwise proxy ({_show_url(url)}). Unset it, "
                  "or use that proxy directly; chaining two would count every request twice.", file=sys.stderr)
            return 2
    ledger = args.ledger or os.environ.get("PREFIXWISE_LEDGER_PATH") or DEFAULT_LEDGER
    # A self-hosted engine behind the agent's base URLs gets the same join as `serve`: --engine, and KV
    # events when it publishes them. Both URLs are probed (vLLM also serves /v1/messages, so Claude Code
    # reaches it through the Anthropic one), and the engine applies to the providers whose URL is the
    # engine; traffic to another host through the same proxy is not judged as the engine's. Live:
    # without the join, an agent's miss against vLLM was `unknown` where the events would have named it.
    engine, kv_events, served = args.engine if args.engine != "none" else None, None, None
    if not args.mock and args.engine != "none":
        for name, url in (("openai", openai), ("anthropic", anthropic)):
            found, kv_events, lines = _detect_engine(url, args.engine, None, args.full_attention)
            for line in lines:
                print(line, file=sys.stderr, flush=True)
            if found:
                engine = found
                served = {n for n, u in (("openai", openai), ("anthropic", anthropic)) if _host_of(u) == _host_of(url)}
                break
    try:
        proxy = PrefixwiseProxy(ledger_path=ledger, mock=True if args.mock else None, fixes=_fixes(args),
                                upstreams=None if args.mock else {"anthropic": anthropic, "openai": openai},
                                otel=args.otel, engine=engine, kv_events=kv_events, full_attention=args.full_attention)
    except ValueError as exc:
        sys.exit(f"prefixwise: {exc}")
    if engine and served and len(served) < 2 and not args.engine:  # detected on one URL only
        proxy.engine_providers = served
    if proxy.spans is not None:  # `run` is its own process: it sets up the exporter
        from .otel import install_provider
        install_provider()
    proxy.versioned = {"openai"}
    proxy.api_key = None  # the agent's own keys pass through; never replace them here
    proxy.default_session = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    server = PrefixwiseHTTPServer(("127.0.0.1", args.port), make_handler(proxy))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {**os.environ, "ANTHROPIC_BASE_URL": base, "OPENAI_BASE_URL": base + "/v1", "PREFIXWISE_URL": base}
    if os.path.basename(cmd[0]) == "codex":
        if not os.environ.get("OPENAI_API_KEY"):
            print("prefixwise: Codex goes through Prefixwise with an API key: set OPENAI_API_KEY "
                  "(a ChatGPT sign-in cannot be proxied).", file=sys.stderr)
            server.shutdown()
            server.server_close()
            return 2
        cmd = [cmd[0], *_codex_args(base), *cmd[1:]]
    upstream = "mock" if args.mock else f"anthropic -> {_show_url(anthropic)}, openai -> {_show_url(openai)}"
    print(f"prefixwise: proxy on {base} ({upstream}); ledger {ledger}", file=sys.stderr, flush=True)
    start = time.time()
    try:
        child = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        child = None
        print(f"prefixwise: command not found: {cmd[0]}", file=sys.stderr)
    code = 127
    if child is not None:
        # While the agent runs it owns the terminal (often a full-screen UI), so the
        # proxy's own messages go to a log file. The agent writes to the terminal
        # directly, not through Python's sys.stderr, so it is unaffected.
        log_path = os.path.join(os.path.dirname(os.path.abspath(ledger)), "prefixwise-run.log")
        real_stderr = sys.stderr
        try:
            sys.stderr = open(log_path, "a", buffering=1)
        except OSError:
            log_path = None
        # Ctrl-C reaches the agent directly (same process group): agents use it to cancel a
        # turn, so Prefixwise must neither die nor kill the agent. SIGTERM is passed on.
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        old_term = signal.signal(signal.SIGTERM, lambda *_: child.terminate())
        try:
            code = child.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            server.shutdown()
            server.drain(5)
            if sys.stderr is not real_stderr:
                sys.stderr.close()
                sys.stderr = real_stderr
        code = 128 - code if code < 0 else code  # killed by a signal: shell convention
        if log_path and os.path.exists(log_path) and os.path.getsize(log_path):
            print(f"prefixwise: proxy messages from this run are in {log_path}", file=sys.stderr)
    else:
        server.shutdown()
    server.server_close()
    found = findings(proxy.ledger, since=start, prices=proxy.prices)
    print("\n-- prefixwise: this run --", file=sys.stderr)
    if not (found["summary"]["requests"] or found["summary"]["requests_failed"]):
        print(f"No requests reached Prefixwise. Does {cmd[0]} read ANTHROPIC_BASE_URL or OPENAI_BASE_URL? "
              "If not, set its base URL to the proxy by hand (see the README).", file=sys.stderr)
    else:
        if args.mock:
            print("(mock: synthetic responses and usage, so the numbers below are illustrative)", file=sys.stderr)
        print(render(found, command="run"), file=sys.stderr)
        print(f"All runs: prefixwise report --ledger {ledger}", file=sys.stderr)
    return code


def cmd_check(args) -> int:
    """A CI gate: run the agent's scenarios through a mock proxy (or judge saved trajectories) and
    fail on the causes that are code bugs."""
    import signal
    import subprocess
    import tempfile
    import threading
    import time

    from . import check
    from .pricing import PriceTable

    try:
        fail_on = check.parse_fail_on(args.fail_on, args.strict)
        allow = check.load_allow(args.allow) if args.allow else []
        baseline_path = args.baseline or (check.BASELINE_FILE if os.path.exists(check.BASELINE_FILE) else None)
        baseline = check.load_baseline(baseline_path) if baseline_path else None
        prices = PriceTable.load(args.prices) if getattr(args, "prices", None) else PriceTable.load()
    except (OSError, ValueError) as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    cmd = args.agent_command[1:] if args.agent_command[:1] == ["--"] else args.agent_command
    if bool(cmd) == bool(args.bodies):
        print("usage: prefixwise check [options] -- <command that runs the agent's scenarios>\n"
              "       prefixwise check [options] --bodies FILE...   (saved request bodies, explain's format)",
              file=sys.stderr)
        return 2

    if args.bodies:
        try:
            rows = check.rows_from_bodies(args.bodies)
        except (OSError, ValueError) as exc:
            print(f"prefixwise: {exc}", file=sys.stderr)
            return 2
        rep = check.judge(rows, fail_on, allow, mock=False)
        check.project(rep, baseline, prices)
        return _emit_check(rep, args)

    from .proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler

    tmp = None
    ledger = args.ledger
    if not ledger:
        tmp = tempfile.TemporaryDirectory(prefix="prefixwise-check-")
        ledger = os.path.join(tmp.name, "check.db")
    watch = check.VolatileWatch()
    try:
        # fixes=(): the gate judges the agent as it is, whatever PREFIXWISE_FIXES says in this shell
        proxy = PrefixwiseProxy(ledger_path=ledger, mock=True, upstreams=None, otel="", fixes=())
    except ValueError as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    proxy.versioned = {"openai"}
    proxy.api_key = None
    proxy.default_session = "default"
    proxy.check_mode = True
    proxy.mock_tool_calls = True
    proxy.mock_first_turn_delay = max(0.0, args.first_turn_delay)
    proxy.request_observer = watch
    server = PrefixwiseHTTPServer(("127.0.0.1", args.port), make_handler(proxy))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {**os.environ, "ANTHROPIC_BASE_URL": base, "OPENAI_BASE_URL": base + "/v1", "PREFIXWISE_URL": base,
           "PREFIXWISE_CHECK": "1"}
    if os.path.basename(cmd[0]) == "codex":
        env.setdefault("OPENAI_API_KEY", "prefixwise-check")  # the mock never sees it
        cmd = [cmd[0], *_codex_args(base), *cmd[1:]]
    print(f"prefixwise check: mock proxy on {base}; gate: {', '.join(fail_on)}", file=sys.stderr, flush=True)
    start = time.time()
    code = 127
    try:
        try:  # with --json, stdout is the report: the command's own output goes to stderr
            child = subprocess.Popen(cmd, env=env, stdout=sys.stderr if args.json else None)
        except FileNotFoundError:
            child = None
            print(f"prefixwise: command not found: {cmd[0]}", file=sys.stderr)
        if child is not None:
            old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            old_term = signal.signal(signal.SIGTERM, lambda *_: child.terminate())
            try:
                code = child.wait()
            finally:
                signal.signal(signal.SIGINT, old_int)
                signal.signal(signal.SIGTERM, old_term)
        server.shutdown()
        server.drain(5)
        server.server_close()
        code = 128 - code if code < 0 else code
        rows = check.rows_since(proxy.ledger, start)
        rep = check.judge(rows, fail_on, allow, mock=True)
        rep.agent_exit = code
        rep.warnings = watch.warnings(rep) + rep.warnings
        check.project(rep, baseline, prices)
    finally:
        proxy.close()
        if tmp is not None:
            tmp.cleanup()
    return _emit_check(rep, args)


def _emit_check(rep, args) -> int:
    from .check import render

    if args.json_file:
        with open(args.json_file, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=2)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(render(rep))
    return rep.exit_code


def cmd_export(args) -> int:
    """Raw ledger rows (facts, never prompt text) as JSONL or CSV, oldest first."""
    import csv

    from . import api, metrics

    led = _open_existing_ledger(args)
    if args.baseline:
        from .check import BASELINE_FILE, baseline_from_ledger
        if args.where or args.format != "jsonl":
            print("prefixwise: --baseline takes --since and -o only (it summarises every row per model)", file=sys.stderr)
            return 2
        try:
            since = api._time(args.since, "--since")
        except (ValueError, api.ApiError) as exc:
            print(f"prefixwise: {exc}", file=sys.stderr)
            return 2
        base = baseline_from_ledger(led, since)
        out = args.output or BASELINE_FILE
        with open(out, "w") as fh:
            json.dump(base, fh, indent=2)
            fh.write("\n")
        models = ", ".join(f"{m} ({v['eligible_turns_per_month']:,.0f} eligible turns/month)"
                           for m, v in base["models"].items()) or "no turns"
        print(f"wrote {out}: {base['window_days']:g} days of ledger scaled to a month; {models}", file=sys.stderr)
        return 0
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


def cmd_import(args) -> int:
    """Read agent session logs (read-only) into the ledger, then report on what was imported."""
    import time

    from . import api, findings, metrics
    from .importers import import_claude_code, import_codex
    from .ledger import Ledger

    try:
        since = api._time(args.since, "--since") if args.since else None
    except api.ApiError as exc:
        print(f"prefixwise: {exc}", file=sys.stderr)
        return 2
    codex = args.source == "codex"
    name, home = ("Codex", "~/.codex") if codex else ("Claude Code", "~/.claude/projects")
    root = os.path.expanduser(args.path or home)
    if not os.path.isdir(root):
        print(f"prefixwise: no {name} logs at {root} ({name} keeps them in {home}; pass --path)", file=sys.stderr)
        return 1
    # Its own ledger by default: someone who also proxies the agent keeps the two apart
    # (the proxy cannot know the log's response ids, so one request would count twice).
    path = args.ledger or f"prefixwise-{args.source}.db"
    led = Ledger(path)
    stats = (import_codex if codex else import_claude_code)(root, led, since=since)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    if stats["unreadable"]:
        print(f"prefixwise: skipped {stats['unreadable']} unreadable log file(s)", file=sys.stderr)
    if not stats["imported"]:
        if not stats["in_window"]:
            print(f"No requests in the {stats['logs']} logs" + (f" since {args.since}." if args.since else "."))
        else:
            print(f"Nothing new: the {stats['logs']} logs hold {stats['skipped_known']:,} requests already in {path}.")
        return 0
    day = lambda t: time.strftime("%Y-%m-%d", time.localtime(t))  # noqa: E731
    span = day(stats["first"]) if day(stats["first"]) == day(stats["last"]) else \
        f"{day(stats['first'])} to {day(stats['last'])}"
    print(f"Imported {stats['imported']:,} requests from {stats['logs']} {name} logs ({span})"
          + (f"; {stats['skipped_known']:,} were already in {path}" if stats["skipped_known"] else "")
          + (f"; {stats['duplicates']:,} logged twice were counted once" if stats["duplicates"] else "") + ".\n")
    window = {"since": stats["first"], "until": stats["last"] + 1}
    found = findings.findings(led, **window)
    # The headline first, in the short form made for pasting into a chat; the full report after it.
    print(findings.render_share(found, since_label=f"{name}, {span}"))
    print("\nDetails:\n" + findings.render(found, command="serve", headline=False))
    rows = metrics.query(led, ["requests", "prompt_tokens", "cache_hit_rate", "cost_usd", "avoidable_tokens"],
                         by="agent", order="-prompt_tokens", **window)
    if len(rows) > 1:
        print("\nBy agent:")
        for r in rows:
            cost = f"${r['cost_usd']:,.2f}" if r["cost_usd"] is not None else "-"
            hit = f"{r['cache_hit_rate']:.0%}" if r["cache_hit_rate"] is not None else "-"
            print(f"  {r['agent']:<30} {r['requests']:>6,} requests  {hit:>4} from cache  {cost:>10}")
    with led.reader() as conn:
        unpriced = [m for (m,) in conn.execute(
            "SELECT DISTINCT model FROM entries WHERE cost IS NULL AND recorded_at BETWEEN ? AND ? ORDER BY model",
            (window["since"], window["until"]))]
    if unpriced:
        print(f"\nNo prices for {', '.join(unpriced)}: add rates with PREFIXWISE_PRICES (see `prefixwise prices`).")
    print("\nAbout these numbers: they come from the logs, which hold each response's usage and what changed "
          "between requests, not the prompt itself, so causes are inferred and a miss the log cannot explain is "
          "`unknown`. Dollars are API prices: on a subscription (Claude Pro/Max, ChatGPT plans) they show what the "
          "usage is worth, not what you paid.")
    print(f"\nLedger: {path}  (prefixwise report --ledger {path} --detail, or prefixwise serve to browse /api/v1)")
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
    if e.diff:
        print("  what changed (as fingerprinted, from your files; nothing is stored):")
        for line in e.diff.splitlines():
            print(f"    {line}")


def cmd_explain(args) -> int:
    from .explain import explain_pair, explain_session, load_turns

    provider = None if args.provider == "auto" else args.provider
    try:
        if len(args.files) == 2:
            prev, curr = (load_turns(f) for f in args.files)
            if len(prev) != 1 or len(curr) != 1:
                sys.exit("prefixwise: with two files, each must hold exactly one request")
            results = [explain_pair(prev[0], curr[0], provider, diff=args.diff)]
            labels = [f"{args.files[0]} -> {args.files[1]}"]
        elif len(args.files) == 1:
            turns = load_turns(args.files[0])
            results = explain_session(turns, provider, diff=args.diff)
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
    sub = ap.add_subparsers(dest="command")

    def ledger_opt(p):
        p.add_argument("--ledger", help=f"SQLite ledger file (default: $PREFIXWISE_LEDGER_PATH or ./{DEFAULT_LEDGER})")

    p = sub.add_parser("collect", help="run a collector: one ledger for several edges (prefixwise serve "
                                        "--collector URL, or PREFIXWISE_COLLECTOR for wrap and LiteLLM)")
    p.add_argument("--host", default=os.environ.get("PREFIXWISE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8081")))
    p.add_argument("--prices", metavar="FILE", help="extra or overriding rates (default: $PREFIXWISE_PRICES)")
    p.add_argument("--allow-unauthenticated", action="store_true",
                   help="listen beyond localhost without PREFIXWISE_AUTH_TOKEN (anyone who reaches the port "
                        "can write records)")
    p.add_argument("--log", metavar="URL",
                   help="also read the record log (redis://host:6379/0; default $PREFIXWISE_LOG): edges started "
                        "with PREFIXWISE_COLLECTOR=<that URL> write to it; needs pip install 'prefixwise[log]'")
    p.add_argument("--partitions", metavar="SPEC",
                   help="the log partitions this collector owns, e.g. 0-7 or 0,3,5 (default: all "
                        "$PREFIXWISE_LOG_PARTITIONS, 16); every partition must have exactly one collector")
    p.add_argument("--sink", action="append", metavar="URL",
                   help="also send every row to a sink (repeatable): file:///path.jsonl, or "
                        "clickhouse://user:pass@host:8123/db.table (default: $PREFIXWISE_SINKS, comma-separated)")
    ledger_opt(p)
    p.set_defaults(func=cmd_collect)

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
                        "Available: canonical_tools, auto_breakpoints (Anthropic); both proven. Default: $PREFIXWISE_FIXES or none")
    p.add_argument("--optimize", action="store_true",
                   help="apply every proven fix (today: canonical_tools and, on Anthropic, auto_breakpoints), plus any --fix named")
    p.add_argument("--collector", metavar="URL",
                   help="send turn records to a `prefixwise collect` server (http://...) or to the record log it "
                        "reads (redis://...) instead of keeping a ledger here (default: $PREFIXWISE_COLLECTOR; "
                        "token: $PREFIXWISE_COLLECTOR_TOKEN)")
    p.add_argument("--sink", action="append", metavar="URL",
                   help="also send every row to a sink (repeatable): file:///path.jsonl, or "
                        "clickhouse://user:pass@host:8123/db.table (default: $PREFIXWISE_SINKS)")
    ledger_opt(p)
    p.add_argument("--engine", choices=["vllm", "sglang", "none"],
                   help="one self-hosted engine behind --upstream: no routing, so an unchanged prefix "
                        "that misses is an eviction (inferred; see --kv-events). Default: detected from the "
                        "upstream at startup (vLLM's /version, SGLang's /server_info), and --kv-events auto when "
                        "it publishes KV events; a hosted API detects as none. 'none': skip detection (a router "
                        "that relays a worker's /server_info, such as the SGLang model gateway, is not one engine)")
    p.add_argument("--kv-events", metavar="ENDPOINT",
                   help="the engine's KV-event stream, e.g. tcp://127.0.0.1:5557, to confirm evictions "
                        "or rule them out (vLLM, or SGLang >= 0.5.20; needs: pip install 'prefixwise[engines]'). "
                        "Several KV caches: 'auto' (every data-parallel rank, from the upstream: vLLM or SGLang), or "
                        "for vLLM comma-separated endpoints, one per replica behind a router; then a request sent "
                        "away from its cached prefix is a confirmed routing miss (on SGLang with --full-attention)")
    p.add_argument("--full-attention", action="store_true", default=None,
                   help="SGLang: the model has no sliding-window or Mamba layers, so KV events can also "
                        "rule an eviction out (vLLM detects this itself)")
    p.add_argument("--otel", action="store_const", const="genai", default=None,
                   help="export a span per request over OpenTelemetry (OTLP, configured by OTEL_* variables; "
                        "needs: pip install 'prefixwise[otel]'). Default: $PREFIXWISE_OTEL (1, or cause for "
                        "Prefixwise's attributes only), else off")
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
    p.add_argument("--session", metavar="ID",
                   help="one session turn by turn: hit rate per turn, the cause, and the block where the prompt "
                        "first changed, as role[position in the whole prompt] (tools[0], messages[9]); `latest` for the most "
                        "recent session")
    p.add_argument("--share", action="store_true",
                   help="a short block to paste into a chat or a thread: counts, dollars, the top cause and its fix; no text")
    p.add_argument("--prices", metavar="FILE", help="also load this rates file (default: $PREFIXWISE_PRICES)")
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
    p.add_argument("--engine", choices=("vllm", "sglang", "none"),
                   help="what is behind the agent's base URL (default: detected; none: a hosted API or a router)")
    p.add_argument("--full-attention", action="store_true", default=None,
                   help="SGLang: the model has no sliding-window or Mamba layers (lets its KV events rule an eviction out)")
    p.add_argument("--fix", action="append", metavar="NAME",
                   help="apply a fix (repeatable): canonical_tools, auto_breakpoints (Anthropic); both proven")
    p.add_argument("--optimize", action="store_true", help="apply every proven fix (today: canonical_tools and, on Anthropic, auto_breakpoints)")
    ledger_opt(p)
    p.add_argument("--otel", action="store_const", const="genai", default=None,
                   help="export a span per request over OpenTelemetry (OTLP, configured by OTEL_* variables; "
                        "needs: pip install 'prefixwise[otel]'). Default: $PREFIXWISE_OTEL (1, or cause for "
                        "Prefixwise's attributes only), else off")
    p.add_argument("agent_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="CI gate: fail when a code change breaks prompt-cache reuse",
                       description="Runs the agent's scenarios through a local mock proxy (no API calls) and "
                                   "exits 2 when a turn misses its cache for a cause that is a code bug: "
                                   "tool_reserialize, volatile_system, breakpoint_misplaced, tool_result_mutation. "
                                   "Each turn is compared with its conversation's previous turn, as in production; "
                                   "nothing is compared with another branch and no prompt is stored. The mock "
                                   "answers a turn that offers tools with a tool call, so the agent's loop runs a "
                                   "second turn. Exit 1 when the command itself fails.")
    p.add_argument("--fail-on", metavar="CAUSES",
                   help=f"comma-separated causes that fail (default: {','.join(DEFAULT_FAIL_ON)})")
    p.add_argument("--strict", action="store_true",
                   help=f"also fail on {', '.join(STRICT_EXTRA)}: real misses, but a scenario that edits "
                        "history or switches a parameter on purpose will trip them")
    p.add_argument("--allow", metavar="FILE",
                   help='known findings to pass: JSON list of {"scenario", "cause", "divergence_role"} ("*" = any)')
    p.add_argument("--baseline", metavar="FILE",
                   help=f"production baseline from `prefixwise export --baseline`, for the monthly projection "
                        f"(default: ./{BASELINE_FILE} when present; else tokens only)")
    p.add_argument("--bodies", nargs="+", metavar="FILE",
                   help="judge saved request bodies instead of running a command (prefixwise explain's format)")
    p.add_argument("--first-turn-delay", type=float, default=DEFAULT_FIRST_TURN_DELAY, metavar="SECONDS",
                   help="hold each conversation's first mock response, so a per-second timestamp in the prompt "
                        f"changes before the second turn (default {DEFAULT_FIRST_TURN_DELAY}; 0 = off)")
    p.add_argument("--port", type=int, default=0, help="mock proxy port (default: any free port)")
    p.add_argument("--prices", metavar="FILE", help="price table for the projection (default: built in)")
    p.add_argument("--json", action="store_true", help="the report as JSON on stdout")
    p.add_argument("--json-file", metavar="FILE", help="also write the JSON report here")
    p.add_argument("--ledger", help="keep this run's ledger here (default: a temporary file, deleted)")
    p.add_argument("agent_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("export", help="raw ledger rows as JSONL or CSV (facts, never prompt text)")
    p.add_argument("--baseline", action="store_true",
                   help=f"instead: write ./{BASELINE_FILE} for `prefixwise check` (per model: turns per month, "
                        "eligible turns per month, mean reusable prefix tokens; counts only, no ids or text)")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p.add_argument("--since", metavar="WHEN", help="unix seconds, or a duration back from now: 24h, 7d")
    p.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                   help="filter (repeatable): cause, session, agent, model, provider, task, api, error")
    p.add_argument("-o", "--output", metavar="FILE", help="default: stdout")
    ledger_opt(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="read agent session logs (no proxy needed) and report on them")
    p.add_argument("source", choices=["claude-code", "codex"],
                   help="whose logs: claude-code (~/.claude/projects) or codex (~/.codex)")
    p.add_argument("--path", metavar="DIR", help="where the logs are (default: the agent's own folder)")
    p.add_argument("--since", metavar="WHEN", help="only requests after this: unix seconds, or 24h, 7d")
    p.add_argument("--json", action="store_true", help="import counts as JSON, no report")
    p.add_argument("--ledger", help="SQLite ledger file (default: ./prefixwise-<source>.db, apart from a "
                                    "proxy's ledger so nothing is counted twice)")
    p.set_defaults(func=cmd_import)

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
    p.add_argument("--diff", action="store_true",
                   help="also print the first differing block before and after, as fingerprinted (cache markers "
                        "removed), from the files given: the hashes say where it changed, this says what. Read from "
                        "your files only: nothing is stored or sent")
    p.add_argument("--fail-on-waste", action="store_true",
                   help="exit 2 if any turn is classified as waste (for CI gates on prompt changes)")
    p.set_defaults(func=cmd_explain)
    return ap


GETTING_STARTED = """Prefixwise names why each LLM request missed the prompt cache, and what it cost.

  prefixwise run -- <your agent>        run an agent through it, get a report when it exits
  prefixwise report --session latest    that session, turn by turn
  prefixwise run --optimize -- <agent>  the same, with the proven fixes applied

  prefixwise serve --upstream <url>     a proxy for vLLM, SGLang or an API (detects the engine)
  prefixwise import claude-code         no API key? read the logs already on disk

All commands: prefixwise --help"""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        print(GETTING_STARTED)
        return 0
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
