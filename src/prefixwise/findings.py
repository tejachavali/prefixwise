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
    waste, expected, unexplained = [], [], []
    for r in by_cause:
        cause = r["cause"]
        if cause is None or not r["requests"] or cause == "none":
            continue
        item = {"cause": cause, "what": CAUSE_PLAIN.get(cause, cause), "requests": r["requests"],
                "avoidable_tokens": r["avoidable_tokens"], "avoidable_cost_usd": r["avoidable_cost_usd"]}
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
                         for e in errors if e["error"] and e["requests_failed"]]}


def _usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}" if abs(x) >= 1 or x == 0 else f"${x:.4f}"  # cents need precision


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
        head = f"{_n(s['requests'])} requests, {_n(s['prompt_tokens'])} prompt tokens"
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
                lines.append(f"  {i}. {w['what'].capitalize()} ({w['requests']} requests, "
                             f"{_n(w['avoidable_tokens'])} tokens{cost}).")
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
