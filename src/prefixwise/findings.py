"""What the ledger means, in plain English: the findings behind `prefixwise report`,
`prefixwise run` and GET /api/v1/findings.

`findings()` returns data (every number from prefixwise.metrics); `render()` turns it
into a few sentences a person can act on: how much was served from cache, what the
waste cost, the fix for each cause, and what was expected.
"""

from __future__ import annotations

from . import metrics
from .fixes import FIX_ADVICE, FIX_STATUS
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
}


# Details under an expected cause that still have something to fix: printed with their fix.
DETAIL_FIX = {
    "volatile_prefix_at_start": "Move the value that changes per session (a timestamp, an id) out of the system "
                                "prompt to the end of the prompt; a timestamp can also be set once per day. Then a "
                                "restarted or parallel session shares the cached tools and system prompt.",
}


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
    errors = metrics.query(ledger, ["requests_failed"], by="error", **window)
    statuses = metrics.query(ledger, ["requests_failed"], by="status", **window)
    return {"summary": {k: s[k] for k in ("requests", "requests_failed", "prompt_tokens", "cached_tokens",
                                          "cache_hit_rate", "avoidable_tokens", "avoidable_share",
                                          "cost_usd", "avoidable_cost_usd", "requests_priced",
                                          "tasks_successful", "tokens_per_successful_task")},
            "waste": waste, "expected": expected, "unexplained": unexplained,
            "rejected_credentials": sum(x["requests_failed"] or 0 for x in statuses if x["status"] in (401, 403)),
            "failures": [{"error": e["error"], "requests": e["requests_failed"]}
                         for e in errors if e["error"] and e["requests_failed"]],
            "ttl": ttl_advice(ledger, prices=prices, **window),
            "unread_writes": unread_writes(ledger, **window)}


def _usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}" if abs(x) >= 1 or x == 0 else f"${x:.4f}"  # cents need precision


def _req(n: int) -> str:
    return f"{n:,} request{'s' if n != 1 else ''}"


def _n(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


def render(f: dict, command: str = "serve") -> str:
    """Plain sentences from findings(). `command` is how the user runs the proxy
    (serve or run), so the advice names a command they can type."""
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
        lines.append(head + ".")
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
        elif f["unexplained"]:
            lines.append("No avoidable waste found among the misses whose cause is known.")
        else:
            lines.append("No avoidable waste found: every miss had a legitimate cause.")
    if f["unexplained"]:
        n = sum(u["requests"] for u in f["unexplained"])
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
