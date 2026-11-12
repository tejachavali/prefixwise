"""What the ledger means, in plain English: the findings behind `prefixwise report`,
`prefixwise run` and GET /api/v1/findings.

`findings()` returns data (every number from prefixwise.metrics); `render()` turns it
into a few sentences a person can act on: how much was served from cache, what the
waste cost, the fix for each cause, and what was expected.
"""

from __future__ import annotations

from . import metrics
from .fixes import DETAIL_ADVICE, FIX_ADVICE, FIX_STATUS
from .pricing import PriceTable

# Anthropic cache lifetimes, seconds: the default, and the 1-hour option (cache_control ttl "1h").
TTL_5M, TTL_1H = 300.0, 3600.0

# One sentence per cause: what happened, in words that do not need the docs.
CAUSE_PLAIN = {
    "none": "served from cache",
    "first_write": "first request of a session: nothing to reuse yet",
    "model_switch": "switched model: each model has its own cache",
    "volatile_system": "the system prompt changed only in a date, time or id",
    "system_edit": "the system prompt was edited",
    "tool_reserialize": "the same tools were sent in a different order or format",
    "tool_catalog_change": "tools were added or removed",
    "compaction_rewrite": "the conversation was compacted (summarised)",
    "history_edit": "earlier messages were rewritten",
    "tool_result_mutation": "an earlier tool result was re-rendered differently",
    "fork_boundary": "a forked session started from its parent's history",
    "param_rendered": "a request parameter that is part of the cached prefix changed",
    "below_min_prefix": "the prompt was too short for the provider to cache",
    "breakpoint_misplaced": "the cache breakpoint is missing, after volatile content, or left early while the stable prompt grew past it",
    "ttl_expired": "too long since the previous turn: the cache entry expired",
    "evicted": "the engine evicted the cached prefix (memory full)",
    "routing_miss": "nothing changed, but the request reached a machine without the cache",
    "retry_replay": "an identical request was retried and missed",
    "unknown": "the cause could not be determined",
}


# The why behind a cause (prefixwise.details), in words, with what to do where it is clear.
DETAIL_PLAIN = {
    "ttl_shortened": "asked for a 1-hour cache but the entry was kept for 5 minutes",
    "lookback_exceeded": "more than 20 content blocks since the last cached point; add a breakpoint at the "
                         "start of each turn",
    "markers_ignored": "cache_control was sent but nothing was cached; check that no gateway strips the markers",
    "sent_before_cache_written": "sent in parallel before the first request had cached the prefix; stagger them "
                                 "or warm the prefix first",
    "shared_prefix_not_reused": "another session had just cached the same tools and system prompt, yet nothing "
                                "was read",
    "volatile_prefix_at_start": "another session (a restart, or another user of the same prompt) had the same "
                                "tools and system prompt moments earlier except for a date, time or id in it, so "
                                "this one recomputed a prefix it could have shared",
    "thinking_stripped": "thinking blocks were removed from earlier messages",
    "cache_key_changed": "prompt_cache_key changed between turns",
    "session_header_dropped": "the worker runs session radix cache, but the session was sent only in an "
                              "x-session-id header, which SGLang does not read, so its KV was evicted first",
}


# Details under an expected cause that still have something to fix: printed with their fix.
DETAIL_FIX = {
    "volatile_prefix_at_start": "Move the value that changes per session (a timestamp, an id) out of the system "
                                "prompt to the end of the prompt; a timestamp can also be set once per day. Then a "
                                "restarted or parallel session shares the cached tools and system prompt.",
    **DETAIL_ADVICE,
}


BILL_PARTS = (("cache_read", "cache re-reads"), ("input", "new input"), ("cache_write", "cache writes"),
              ("output", "output"))


def bill(ledger, since: float | None = None, until: float | None = None) -> dict | None:
    """Where the dollars went, over priced requests that carry the cost split (ledger schema 7
    or later): {"cache_read", "input", "cache_write", "output"} in USD, "total", "requests" (with a
    split) and "requests_priced". None when no request has a split. cache_read is the floor: on a
    long, well-cached session it is most of the bill, though no miss caused it."""
    row = metrics.query(ledger, ["cost_cache_read_usd", "cost_input_usd", "cost_cache_write_usd", "cost_output_usd",
                                 "requests_cost_split", "requests_priced"], since=since, until=until)
    r = row[0] if row else {}
    if not r.get("requests_cost_split"):
        return None
    out = {"cache_read": r["cost_cache_read_usd"] or 0.0, "input": r["cost_input_usd"] or 0.0,
           "cache_write": r["cost_cache_write_usd"] or 0.0, "output": r["cost_output_usd"] or 0.0}
    out["total"] = sum(out.values())
    out["requests"], out["requests_priced"] = r["requests_cost_split"], r["requests_priced"] or 0
    return out


def hit_rate_if_fixed(s: dict) -> float | None:
    """The share of prompt tokens served from the cache had every avoidable miss been fixed:
    (cached + avoidable) / prompt, over the same requests. Avoidable tokens are, by definition, the
    part of each wasteful request's reusable prefix the cache did not serve, so fixing the cause
    moves exactly them from uncached to cached; everything else (new content each turn, legitimate
    misses, unexplained ones) stays as it is. A prediction: measured, a fix realised 97.5% to 99.9%
    of what was predicted (experiments/e4), and a real run predicted at 87.1% read 87.1% with the
    fixes on (experiments/demo/2026-11-10-optimize-real-run.md). None without usage."""
    prompt, cached, avoidable = s.get("prompt_tokens") or 0, s.get("cached_tokens") or 0, s.get("avoidable_tokens") or 0
    if prompt <= 0:
        return None
    return min(1.0, (cached + avoidable) / prompt)


def ceiling_line(s: dict) -> str | None:
    """'Fixing these would take the share served from cache from 74.6% to 87.1%; ...', said after the
    list of avoidable causes. None when there is nothing avoidable, or too little to show."""
    pair = from_to(s.get("cache_hit_rate"), s.get("cache_hit_rate_if_fixed"), 1) if s.get("avoidable_tokens") else None
    if pair is None:
        return None
    return (f"Fixing these would take the share served from cache from {pair[0]} to {pair[1]}; "
            "the rest is new content each turn, misses with a legitimate cause and misses whose cause "
            "is not known.")


def from_to(now: float | None, fixed: float | None, decimals: int = 0) -> tuple[str, str] | None:
    """The current and the if-fixed share, printed together at one precision: the current share
    rounded (never up to 100% when it is not), the if-fixed one rounded down (a prediction is never
    rounded up into a promise), with as many decimals as it takes, up to two, for the second to be
    above the first. None when the gain is too small to show that way."""
    import math

    if now is None or fixed is None:
        return None
    for d in range(decimals, 3):
        scale = 10 ** d
        n = round(now * 100 * scale) / scale
        if now < 1 and n >= 100:
            n = math.floor(now * 100 * scale) / scale
        f = math.floor(fixed * 100 * scale + 1e-9) / scale
        if f > n and (f < 100 or fixed >= 1):
            return f"{n:.{d}f}%", f"{f:.{d}f}%"
    return None


def _share(v: float, total: float) -> str:
    p = v / total
    return "<1%" if 0 < p < 0.005 else f"{p:.0%}"


def bill_line(b: dict | None, short: bool = False) -> str | None:
    """One sentence on where the dollars went; None without a split or with nothing billed."""
    if not b or not b.get("total"):
        return None
    parts = [(label, b[key]) for key, label in BILL_PARTS if b[key]]
    partial = b["requests"] < b["requests_priced"]
    if short:
        return ("bill" + (f" ({b['requests']} of {b['requests_priced']} priced requests)" if partial else "") + ": "
                + ", ".join(f"{_share(v, b['total'])} {label}" for label, v in parts))
    text = "Where the bill went: " + ", ".join(f"{label} {_usd(v)} ({_share(v, b['total'])})" for label, v in parts)
    if partial:
        text += f" ({b['requests']} of {b['requests_priced']} priced requests; older rows have no split)"
    return text + "."


def floor_advice(b: dict | None) -> str | None:
    """When re-reading cached context is the biggest part of the bill, say where the lever is."""
    if not b or not b.get("total"):
        return None
    if b["cache_read"] >= max(b["input"], b["cache_write"], b["output"]) and b["cache_read"] > 0:
        return ("The largest part re-reads context the cache already holds, so a better hit rate would not lower it: "
                "the lever is how much context each turn re-reads (a new session for a new task, compacting at "
                "task boundaries, shorter tool results).")
    return None


def _details(ledger, cause: str, window: dict) -> list[dict]:
    rows = metrics.query(ledger, ["requests"], by="detail", filters={"cause": cause}, **window)
    return sorted(({"detail": r["detail"], "what": DETAIL_PLAIN.get(r["detail"], r["detail"]),
                    "requests": r["requests"]} for r in rows if r["detail"] and r["requests"]),
                  key=lambda d: -d["requests"])


def ttl_advice(ledger, since: float | None = None, until: float | None = None,
               prices: PriceTable | None = None) -> list[dict]:
    """Per Anthropic model: could the 1-hour cache have paid on this ledger?

    Each row's own cache lifetime is in `cache_ttl_seconds` (schema 5). Kept by 1 hour: the
    ttl_expired misses on the 5-minute cache whose gap was within the hour, except ttl_shortened
    (they asked for 1 hour already). Under 1 hour their rewrites would have been reads (saved);
    every other 5-minute write is billed at the 1-hour rate instead (extra). Writes already on
    1 hour cost nothing more; rows without a known lifetime (older ledgers, OpenAI) are left out.
    Part of each rewrite is new tail that would be written anyway, so saved - extra is an upper
    bound on what 1 hour gains: when it is not positive, 1 hour cannot pay. A model without both
    write rates is left out (never a guessed price). Rates at the long-prompt tier where the
    row's prompt is in it."""
    if prices is None:
        try:
            prices = PriceTable.load()
        except (OSError, ValueError):
            return []
    clauses, args = ["provider = 'anthropic'", "cause IS NOT NULL"], []
    if since is not None:
        clauses.append("recorded_at >= ?")
        args.append(since)
    if until is not None:
        clauses.append("recorded_at < ?")
        args.append(until)
    sql = (f"SELECT model, prompt_tokens, COALESCE(cache_write_tokens, 0), cause, gap_seconds, "
           f"cache_ttl_seconds, detail FROM entries WHERE {' AND '.join(clauses)}")
    with ledger.reader() as conn:
        rows = conn.execute(sql, args).fetchall()
    acc: dict = {}
    for model, prompt, writes, cause, gap, ttl, detail in rows:
        r = prices.rates_for(model)
        if r is None:
            continue
        r = prices._tier(r, prompt or 0)
        w5, w1, read = r.get("cache_write_5m"), r.get("cache_write_1h"), r.get("cache_read")
        if not all(isinstance(x, (int, float)) for x in (w5, w1, read)):
            continue
        a = acc.setdefault(model, {"misses": 0, "saved": 0.0, "extra": 0.0})
        if cause == "ttl_expired" and ttl == TTL_5M and gap is not None and gap <= TTL_1H \
                and detail != "ttl_shortened":  # asked for 1 hour already: the downgrade is the problem
            a["misses"] += 1
            a["saved"] += writes * (w5 - read) / 1e6
        elif ttl == TTL_5M and detail != "ttl_shortened":  # on 1 hour already, or unknown: not counted
            a["extra"] += writes * (w1 - w5) / 1e6
    out = [{"model": m, "misses": a["misses"], "saved_at_most_usd": round(a["saved"], 6),
            "extra_write_cost_usd": round(a["extra"], 6), "pays": a["saved"] > a["extra"]}
           for m, a in acc.items() if a["misses"]]
    return sorted(out, key=lambda x: -(x["saved_at_most_usd"] - x["extra_write_cost_usd"]))


def unread_writes(ledger, since: float | None = None, until: float | None = None) -> list[dict]:
    """Per model: cache writes that the next turn of the same conversation left unread (it replaced
    the prefix, or let it expire, and read nothing). The provider charged the write rate for tokens no one read: the surcharge
    above sending them uncached is money caching cost rather than saved. Not part of avoidable
    cost (fixing the miss saves it too); shown so the price of a freshly written, then invalidated,
    prefix is visible."""
    clauses, args = ["unread_write_tokens > 0"], []
    if since is not None:
        clauses.append("recorded_at >= ?")
        args.append(since)
    if until is not None:
        clauses.append("recorded_at < ?")
        args.append(until)
    sql = (f"""SELECT model, COUNT(*), SUM(unread_write_tokens), SUM(unread_write_cost),
                      SUM(CASE WHEN unread_write_cost IS NOT NULL THEN 1 ELSE 0 END)
               FROM entries WHERE {" AND ".join(clauses)} GROUP BY model ORDER BY 3 DESC""")
    with ledger.reader() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [{"model": m, "requests": n, "tokens": tok, "surcharge_usd": cost, "priced": priced}
            for m, n, tok, cost, priced in rows]


def findings(ledger, since: float | None = None, until: float | None = None,
             prices: PriceTable | None = None) -> dict:
    """The facts a person needs, for the whole ledger or a time window."""
    window = {"since": since, "until": until}
    s = metrics.summary(ledger, **window)
    by_cause = metrics.query(ledger, ["requests", "prompt_tokens", "cached_tokens", "avoidable_tokens",
                                      "avoidable_cost_usd", "requests_wasteful"], by="cause", **window)
    waste, expected, unexplained = [], [], []
    for r in by_cause:
        cause = r["cause"]
        if cause is None or not r["requests"] or cause == "none":
            continue
        item = {"cause": cause, "what": CAUSE_PLAIN.get(cause, cause), "requests": r["requests"],
                "avoidable_tokens": r["avoidable_tokens"], "avoidable_cost_usd": r["avoidable_cost_usd"],
                "details": _details(ledger, cause, window)}
        if cause == "unknown":  # not known to be expected, nor waste
            unexplained.append(item)
        elif r["avoidable_tokens"]:
            advice = FIX_ADVICE.get(cause, {})
            item["fix"] = advice.get("fix")
            item["proxy_fix"] = advice.get("toggle")
            item["fix_status"] = FIX_STATUS.get(advice.get("toggle") or "", {}).get("status")
            waste.append(item)
        else:
            expected.append(item)
    waste.sort(key=lambda x: -x["avoidable_tokens"])
    expected.sort(key=lambda x: -x["requests"])
    table = prices or PriceTable.load(None)
    unpriced, approximate = [], []
    for row in metrics.query(ledger, ["requests", "requests_priced"], by="model", **window):
        model = row["model"]
        if not model or not row["requests"]:
            continue
        if not row["requests_priced"]:  # from the rows, as they were priced: not from today's table
            unpriced.append(model)
            continue
        key, rates, approx = table.resolve(model)
        if approx:
            approximate.append((model, key))
    from .classifier import CACHE_NOT_REPORTED
    clauses, args = ["cause = 'unknown'", "evidence = ?"], [CACHE_NOT_REPORTED]
    for col, op, val in (("recorded_at", ">=", since), ("recorded_at", "<", until)):
        if val is not None:
            clauses.append(f"{col} {op} ?")
            args.append(val)
    with ledger.reader() as conn:
        not_reported = conn.execute("SELECT COUNT(*) FROM entries WHERE " + " AND ".join(clauses), args).fetchone()[0]
    errors = metrics.query(ledger, ["requests_failed"], by="error", **window)
    statuses = metrics.query(ledger, ["requests_failed"], by="status", **window)
    summary = {k: s[k] for k in ("requests", "requests_failed", "prompt_tokens", "cached_tokens",
                                 "cache_hit_rate", "avoidable_tokens", "avoidable_share",
                                 "cost_usd", "avoidable_cost_usd", "requests_priced",
                                 "tasks_successful", "tokens_per_successful_task")}
    summary["cache_hit_rate_if_fixed"] = hit_rate_if_fixed(summary)
    return {"summary": summary,
            "waste": waste, "expected": expected, "unexplained": unexplained,
            "rejected_credentials": sum(x["requests_failed"] or 0 for x in statuses if x["status"] in (401, 403)),
            "failures": [{"error": e["error"], "requests": e["requests_failed"]}
                         for e in errors if e["error"] and e["requests_failed"]],
            "ttl": ttl_advice(ledger, prices=table, **window),
            "unread_writes": unread_writes(ledger, **window),
            "unpriced_models": sorted(unpriced), "approximate_prices": sorted(approximate),
            "bill": bill(ledger, **window),
            "cache_not_reported": not_reported}


def _usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}" if abs(x) >= 1 or x == 0 else f"${x:.4f}"  # cents need precision


def _req(n: int) -> str:
    return f"{n:,} request{'s' if n != 1 else ''}"


def _n(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


def render(f: dict, command: str = "serve", headline: bool = True) -> str:
    """Plain sentences from findings(). `command` is how the user runs the proxy
    (serve or run), so the advice names a command they can type. `headline=False` leaves out the
    totals line, for a caller that has printed them already (render_share)."""
    s = f["summary"]
    if not s["requests"] and not s["requests_failed"]:
        return ("No requests recorded yet. Point your agent's base URL at the proxy, or run it with "
                "`prefixwise run -- <your agent>`.")
    lines = []
    if not s["requests"]:
        lines.append(f"No request got a response with usage: {_n(s['requests_failed'])} failed.")
    else:
        hit = s["cache_hit_rate"]
        head = f"{_req(s['requests'])}, {_n(s['prompt_tokens'])} prompt tokens"
        if hit is not None:
            head += f", {hit:.0%} served from cache"
        partial = s["requests_priced"] and s["requests_priced"] < s["requests"]
        if s["requests_priced"]:
            head += f". Cost {_usd(s['cost_usd'])}"
            if partial:
                head += f" ({s['requests_priced']} of {s['requests']} requests priced)"
        if headline:
            lines.append(head + ".")
        # Without the headline (import prints the short block first, with the split), the advice only.
        for extra in ((bill_line(f.get("bill")) if headline else None), floor_advice(f.get("bill"))):
            if extra:
                lines.append(extra)
        if f.get("approximate_prices"):
            lines.append("Priced at a related model's rate: " + "; ".join(f"{m} as {k}" for m, k in f["approximate_prices"])
                         + " (add the exact rates with --prices FILE).")
        if f.get("unpriced_models"):
            lines.append("Not priced: " + ", ".join(f["unpriced_models"])
                         + " (not in the price table; `prefixwise prices` lists it, --prices FILE adds rates).")
        if f["waste"]:
            waste_usd = ""
            if s["requests_priced"]:
                waste_usd = f" ({_usd(s['avoidable_cost_usd'])}{', priced requests only' if partial else ''})"
            lines.append(f"Avoidable: {_n(s['avoidable_tokens'])} tokens{waste_usd}, "
                         f"{(s['avoidable_share'] or 0):.1%} of prompt tokens. Fix these, biggest first:")
            for i, w in enumerate(f["waste"], 1):
                cost = f", {_usd(w['avoidable_cost_usd'])}" if w["avoidable_cost_usd"] is not None else ""
                lines.append(f"  {i}. {w['what'].capitalize()} ({_req(w['requests'])}, "
                             f"{_n(w['avoidable_tokens'])} tokens{cost}).")
                for d in w.get("details") or []:
                    lines.append(f"     {d['requests']} of these: {d['what']}.")
                if w.get("fix"):
                    lines.append(f"     Fix: {w['fix']}")
                if w.get("proxy_fix"):
                    lines.append(f"     Prefixwise can do it for you: prefixwise {command} --fix {w['proxy_fix']}"
                                 + (f" ({w['fix_status']})" if w.get("fix_status") else ""))
            ceiling = ceiling_line(s)
            if ceiling:
                lines.append(ceiling)
        elif f["unexplained"] and sum(u["requests"] for u in f["unexplained"]) > (f.get("cache_not_reported") or 0):
            lines.append("No avoidable waste found among the misses whose cause is known.")
        elif f["unexplained"]:
            lines.append("No avoidable waste found in the prompts.")
        else:
            lines.append("No avoidable waste found: every miss had a legitimate cause.")
    blind = f.get("cache_not_reported") or 0
    if blind:
        lines.append(f"This server does not report cached tokens ({_req(blind)} with an unchanged prompt): Ollama "
                     "and many llama.cpp builds leave the count out, so the share served from cache above is not "
                     "known, not zero. Prefixwise still names every change to the prompt. With vLLM, start it with "
                     "--enable-prompt-tokens-details.")
    if f["unexplained"] and sum(u["requests"] for u in f["unexplained"]) > blind:
        n = sum(u["requests"] for u in f["unexplained"]) - blind
        lines.append(f"Cause not determined for {n} request{'s' if n != 1 else ''} "
                     "(`prefixwise report --detail` shows the evidence).")
    if f["expected"]:
        parts = [f"{e['requests']} x {e['what']}" for e in f["expected"][:4]]
        lines.append("Expected misses (not waste): " + "; ".join(parts) + ".")
        for e in f["expected"]:
            for d in e.get("details") or []:
                lines.append(f"  {d['requests']} x {e['what'].split(':')[0]}: {d['what']}.")
                if d["detail"] in DETAIL_FIX:
                    lines.append(f"     Fix: {DETAIL_FIX[d['detail']]}")
    unread = f.get("unread_writes") or []
    if unread:
        tok = sum(u["tokens"] for u in unread)
        n = sum(u["requests"] for u in unread)
        priced = [u for u in unread if u["priced"] == u["requests"] and u["surcharge_usd"] is not None]
        line = (f"Cache writes never read: {_n(tok)} tokens were written to the cache, and the next turn of the "
                f"same conversation replaced the prefix before reading them ({_req(n)})")
        if priced:
            line += (f"; the write surcharge on them cost {_usd(sum(u['surcharge_usd'] for u in priced))} above "
                     "sending them uncached" + ("" if len(priced) == len(unread) else ", priced models only"))
        lines.append(line + ". Fixing the misses above saves this too; it is not counted twice.")
    for t in f.get("ttl") or []:
        n = t["misses"]
        head = (f"{n} {t['model']} request{'s' if n != 1 else ''} missed because the 5-minute cache had expired, "
                "within the hour: a 1-hour cache (cache_control ttl \"1h\") would have kept "
                f"{'them' if n != 1 else 'it'}. ")
        if t["pays"]:
            lines.append(head + f"It would save at most {_usd(t['saved_at_most_usd'])} and cost "
                         f"{_usd(t['extra_write_cost_usd'])} more in cache writes: worth trying, and "
                         "worth measuring (part of each saving is new content that is written anyway).")
        else:
            lines.append(head + f"But it would cost {_usd(t['extra_write_cost_usd'])} more in cache writes and "
                         f"save at most {_usd(t['saved_at_most_usd'])}: it cannot pay, keep the 5-minute cache.")
    if f["failures"]:
        parts = [f"{x['requests']} x {x['error']}" for x in f["failures"]]
        lines.append("Failed requests (no verdict): " + ", ".join(parts) + ".")
    if f.get("rejected_credentials"):
        lines.append("The provider rejected the credentials (401/403). Prefixwise passes your agent's own API key "
                     "through; an agent signed in with a subscription instead of an API key (for example Codex "
                     "with a ChatGPT login) cannot be routed through a proxy.")
    if s["tasks_successful"]:
        lines.append(f"Tokens per successful task: {_n(s['tokens_per_successful_task'])}.")
    return "\n".join(lines)


def render_share(f: dict, since_label: str | None = None) -> str:
    """A short block for pasting into a chat or a thread: the headline numbers, a pricing note when
    the dollars are partial or approximate, the top cause and its fix, the unread writes. Counts and
    dollars only; no prompt text exists to leak. At most ten lines."""
    s = f["summary"]
    if not s["requests"]:
        return "prefixwise: no requests recorded yet."
    lines = ["prefixwise" + (f" ({since_label})" if since_label else "")]
    hit = f"{s['cache_hit_rate']:.0%} from cache" if s["cache_hit_rate"] is not None else "cache hit rate unknown"
    head = f"{_req(s['requests'])}, {_n(s['prompt_tokens'])} prompt tokens, {hit}"
    if s["requests_priced"]:
        head += f", {_usd(s['cost_usd'])}"
    lines.append(head)
    split = bill_line(f.get("bill"), short=True)
    if split:
        lines.append(split)
    notes = []
    if s["requests_priced"] and s["requests_priced"] < s["requests"]:
        notes.append(f"{s['requests_priced']} of {s['requests']} requests priced")
    if f.get("approximate_prices"):
        notes.append("priced at a related model's rate: " + ", ".join(f"{m} as {k}" for m, k in f["approximate_prices"]))
    if f.get("unpriced_models"):
        notes.append("not priced: " + ", ".join(f["unpriced_models"]))
    if notes:
        lines.append("pricing: " + "; ".join(notes))
    if f["waste"]:
        w = f["waste"][0]
        cost = f" ({_usd(s['avoidable_cost_usd'])})" if s["requests_priced"] else ""
        lines.append(f"avoidable: {_n(s['avoidable_tokens'])} tokens{cost}, {(s['avoidable_share'] or 0):.0%} of the prompt"
                     + (", {} to {} from cache if fixed".format(*pair)
                        if (pair := from_to(s.get("cache_hit_rate"), s.get("cache_hit_rate_if_fixed"))) else ""))
        lines.append(f"top cause: {w['what']} ({_req(w['requests'])})")
        if w.get("fix"):
            lines.append(f"fix: {w['fix']}")
    elif f["unexplained"] and sum(u["requests"] for u in f["unexplained"]) > (f.get("cache_not_reported") or 0):
        lines.append("no avoidable waste among the misses whose cause is known")
    elif f["unexplained"]:
        lines.append("no avoidable waste in the prompts")
    else:
        lines.append("no avoidable waste: every miss had a legitimate cause")
    if f.get("cache_not_reported"):
        lines.append("this server does not report cached tokens: the share served from cache is not known")
    unread = f.get("unread_writes") or []
    if unread:
        tok = sum(u["tokens"] or 0 for u in unread)
        usd = sum(u["surcharge_usd"] or 0 for u in unread)
        lines.append(f"cache writes never read back: {_n(tok)} tokens" + (f" ({_usd(usd)} write surcharge)" if usd else ""))
    lines.append("pip install prefixwise  ·  github.com/tejachavali/prefixwise")
    return "\n".join(lines)
