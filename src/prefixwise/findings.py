"""What the ledger means, in plain English: the findings behind `prefixwise report`,
`prefixwise run` and GET /api/v1/findings.

`findings()` returns data (every number from prefixwise.metrics); `render()` turns it
into a few sentences a person can act on: how much was served from cache, what the
waste cost, the fix for each cause, and what was expected.
"""

from __future__ import annotations

from . import metrics
from .fixes import FIX_ADVICE, FIX_STATUS

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
    "breakpoint_misplaced": "the cache breakpoint was placed where nothing can be reused",
    "ttl_expired": "too long since the previous turn: the cache entry expired",
    "evicted": "the engine evicted the cached prefix (memory full)",
    "routing_miss": "nothing changed, but the request reached a machine without the cache",
    "retry_replay": "an identical request was retried and missed",
    "unknown": "the cause could not be determined",
}


def findings(ledger, since: float | None = None, until: float | None = None) -> dict:
    """The facts a person needs, for the whole ledger or a time window."""
    window = {"since": since, "until": until}
    s = metrics.summary(ledger, **window)
    by_cause = metrics.query(ledger, ["requests", "prompt_tokens", "cached_tokens", "avoidable_tokens",
                                      "avoidable_cost_usd", "requests_wasteful"], by="cause", **window)
    waste, expected = [], []
    for r in by_cause:
        cause = r["cause"]
        if cause is None or not r["requests"] or cause == "none":
            continue
        item = {"cause": cause, "what": CAUSE_PLAIN.get(cause, cause), "requests": r["requests"],
                "avoidable_tokens": r["avoidable_tokens"], "avoidable_cost_usd": r["avoidable_cost_usd"]}
        if r["avoidable_tokens"]:
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
    return {"summary": {k: s[k] for k in ("requests", "requests_failed", "prompt_tokens", "cached_tokens",
                                          "cache_hit_rate", "avoidable_tokens", "avoidable_share",
                                          "cost_usd", "avoidable_cost_usd", "requests_priced",
                                          "tasks_successful", "tokens_per_successful_task")},
            "waste": waste, "expected": expected,
            "failures": [{"error": e["error"], "requests": e["requests_failed"]}
                         for e in errors if e["error"] and e["requests_failed"]]}


def _usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}" if abs(x) >= 0.01 or x == 0 else f"${x:.4f}"


def _n(x) -> str:
    return "-" if x is None else f"{x:,.0f}"


def render(f: dict) -> str:
    s = f["summary"]
    if not s["requests"] and not s["requests_failed"]:
        return ("No requests recorded yet. Point your agent's base URL at the proxy, or run it with "
                "`prefixwise run -- <your agent>`.")
    lines = []
    hit = s["cache_hit_rate"]
    head = (f"{_n(s['requests'])} requests, {_n(s['prompt_tokens'])} prompt tokens, "
            f"{hit:.0%} served from cache" if hit is not None else f"{_n(s['requests'])} requests")
    if s["requests_priced"]:
        head += f". Cost {_usd(s['cost_usd'])}"
        if s["requests_priced"] < s["requests"]:
            head += f" ({s['requests_priced']} of {s['requests']} requests priced)"
    lines.append(head + ".")
    if f["waste"]:
        waste_usd = f" ({_usd(s['avoidable_cost_usd'])})" if s["requests_priced"] else ""
        lines.append(f"Avoidable: {_n(s['avoidable_tokens'])} tokens{waste_usd}, "
                     f"{(s['avoidable_share'] or 0):.1%} of prompt tokens. Fix these, biggest first:")
        for i, w in enumerate(f["waste"], 1):
            cost = f", {_usd(w['avoidable_cost_usd'])}" if w["avoidable_cost_usd"] is not None else ""
            lines.append(f"  {i}. {w['what'].capitalize()} ({w['requests']} requests, "
                         f"{_n(w['avoidable_tokens'])} tokens{cost}).")
            if w.get("fix"):
                lines.append(f"     Fix: {w['fix']}")
            if w.get("proxy_fix"):
                lines.append(f"     Prefixwise can do it for you: prefixwise serve --fix {w['proxy_fix']}"
                             + (f" ({w['fix_status']})" if w.get("fix_status") else ""))
    elif s["requests"]:
        lines.append("No avoidable waste found: your prompt cache is being reused as well as it can be.")
    if f["expected"]:
        parts = [f"{e['requests']} x {e['what']}" for e in f["expected"][:4]]
        lines.append("Expected misses (not waste): " + "; ".join(parts) + ".")
    if f["failures"]:
        parts = [f"{x['requests']} x {x['error']}" for x in f["failures"]]
        lines.append("Failed requests (no verdict): " + ", ".join(parts) + ".")
    if s["tasks_successful"]:
        lines.append(f"Tokens per successful task: {_n(s['tokens_per_successful_task'])}.")
    return "\n".join(lines)
