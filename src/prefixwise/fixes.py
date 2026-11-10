"""Fix toggles and fix advice.

A toggle is a rewrite the proxy can apply to a request before forwarding
it, opt-in and reversible (turn it off and requests go through unchanged).
Only rewrites that cannot change what the prompt means are toggles:

- canonical_tools: tools sorted by name, each tool's keys sorted at every
  depth, and the schema arrays that are sets (`required`, a list `type`)
  sorted; enum, anyOf and other arrays keep their order. The
  provider caches by exact prefix, and tools come first in every
  provider's prefix, so a tool list whose order or key order drifts
  between turns (plugin load order, dict/set iteration, a framework
  rebuilding definitions) misses the whole cache. Sorting is stable:
  same tool set, same bytes, every turn.

- auto_breakpoints (Anthropic): cache_control markers for a request that
  carries none, placed where agents reuse: the end of tools + system and
  the latest message, with a third on the previous user turn when more
  than Anthropic's 20-block lookback lies between. Not lossless for the
  bill (a write costs 1.25x input, 2x for 1 hour, and pays off only when
  read), so the proxy places them only when this edge saw the same
  tools + system within the cache TTL (the session's own previous turn,
  or another session's), the prefix is above the model's minimum, and the
  client set no marker of its own (a client's markers are never moved).
  Proven on Anthropic (E4 live 2026-11-09), so --optimize turns it on.

`--optimize` turns on PROVEN_FIXES, the set with an E4 run behind it, and
any --fix the user names on top.

Everything else is advice: changes only the application can make safely
(moving a timestamp out of the system prompt changes the text the model
reads). FIX_ADVICE maps each wasteful cause to its fix, and says whether
the proxy can apply it.

A fix ships as "recommended" on predicted savings (the avoidable tokens
the ledger attributes to its cause) and is "proven" only after a replay
(experiments/e4) shows realised savings of at least half the prediction
with task success flat.
"""

from __future__ import annotations

import json

from .fingerprint import schema_sets

AVAILABLE_FIXES = ("canonical_tools", "auto_breakpoints")
PROVEN_FIXES = ("canonical_tools", "auto_breakpoints")  # what --optimize turns on; a fix joins after an E4 run (FIX_STATUS)
OPTIMIZE = "optimize"  # the name --optimize stands for in a fix list

# Evidence per fix. Change only with a committed E4 run behind it.
FIX_STATUS = {
    "canonical_tools": {
        "status": "proven",
        "evidence": "E4 live 2026-11-06: 99.9% of predicted savings realised on Anthropic (Haiku 5.5), "
                    "97.5% on vLLM (Qwen3-4B); task success flat (0/60 and 1/60 turns changed). "
                    "experiments/e4/runs/2026-11-06-live.md",
    },
    "auto_breakpoints": {
        "status": "proven",
        "evidence": "E4 live 2026-11-09 on Anthropic (Haiku 5.5), a client with no cache_control: 106.8% of "
                    "predicted savings realised, prompt-side cost down 85.6% with the write surcharge included, "
                    "task success flat (59/60 off, 60/60 on), no marker rejected. Multi-turn sessions; one model. "
                    "experiments/e4/runs/2026-11-09-auto-breakpoints-live.md",
    },
}

MARKER = {"type": "ephemeral"}
_THINKING = ("thinking", "redacted_thinking")  # Anthropic rejects cache_control on these
LOOKBACK_BLOCKS = 20  # Anthropic checks this many blocks before each marker for an earlier entry
_CHARS_PER_TOKEN = 4  # a generous estimate: an over-count places a marker below the minimum, which Anthropic ignores at no cost


def _with_marker(block) -> dict:
    return {**block, "cache_control": MARKER} if isinstance(block, dict) else {"type": "text", "text": str(block),
                                                                                 "cache_control": MARKER}


def _mark_last(content):
    """A message's content with a marker on its last block (a string becomes one text block); None
    when the last block is a thinking block, which cannot carry one."""
    if isinstance(content, list) and content:
        if isinstance(content[-1], dict) and content[-1].get("type") in _THINKING:
            return None
        return content[:-1] + [_with_marker(content[-1])]
    if isinstance(content, str) and content:
        return [_with_marker(content)]
    return None


def _n_blocks(content) -> int:
    return len(content) if isinstance(content, list) else (1 if content not in (None, "") else 0)


def auto_breakpoints(body: dict, min_prefix_tokens: int = 1024) -> dict | None:
    """An Anthropic Messages body with cache_control markers placed, or None when none belongs:
    the client set a marker anywhere (never moved), there is no tools + system prefix, or the
    prefix is estimated below `min_prefix_tokens` (Anthropic would ignore the marker). Marker 1
    ends tools + system (on the last system block, which covers the tools before it; on the last
    tool when there is no system). Marker 2 is on the last block of the last message. Marker 3 is
    on the last block of the previous user turn when more than LOOKBACK_BLOCKS blocks follow it,
    where the previous request's marker 2 sat. Never mutates body."""
    if "cache_control" in json.dumps(body, default=str):
        return None
    tools, system = body.get("tools"), body.get("system")
    if not tools and not system:
        return None
    if len(json.dumps(tools or [], default=str)) + len(json.dumps(system or "", default=str)) \
            < min_prefix_tokens * _CHARS_PER_TOKEN:
        return None
    out = dict(body)
    if isinstance(system, list) and system:
        out["system"] = system[:-1] + [_with_marker(system[-1])]
    elif isinstance(system, str) and system:
        out["system"] = [_with_marker(system)]
    elif isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        out["tools"] = tools[:-1] + [_with_marker(tools[-1])]
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        marked = _mark_last(msgs[-1].get("content"))
        if marked is not None:
            msgs = msgs[:-1] + [{**msgs[-1], "content": marked}]
            # the previous user turn: the previous request ended there, so its marker 2 sat there
            after = _n_blocks(msgs[-1].get("content"))
            for i in range(len(msgs) - 2, -1, -1):
                m = msgs[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    if after > LOOKBACK_BLOCKS:
                        prev = _mark_last(m.get("content"))
                        if prev is not None:
                            msgs = msgs[:i] + [{**m, "content": prev}] + msgs[i + 1:]
                    break
                after += _n_blocks(m.get("content")) if isinstance(m, dict) else 0
            out["messages"] = msgs
    return out


def _tool_name(tool: object) -> str | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])  # OpenAI chat completions
    return str(tool["name"]) if tool.get("name") else None  # Anthropic, Responses API


def _sorted_keys(value: object) -> object:
    # json round trip with sort_keys: nested dicts come back in sorted key order.
    return json.loads(json.dumps(value, sort_keys=True))


def canonical_tools(tools: list) -> list:
    """Tools sorted by name (unnamed tools keep their order, at the end),
    with every tool's keys sorted and its schema's set arrays (`required`, a
    list `type`) sorted. Same set in, same bytes out."""
    named = sorted((t for t in tools if _tool_name(t) is not None), key=_tool_name)
    unnamed = [t for t in tools if _tool_name(t) is None]
    return [_sorted_keys(schema_sets(t)) for t in named] + [_sorted_keys(schema_sets(t)) for t in unnamed]


def apply_fixes(body: dict, fixes: tuple[str, ...] | list[str]) -> tuple[dict, list[str]]:
    """Return (body to forward, fixes that changed it). Never mutates body."""
    applied: list[str] = []
    out = body
    if "canonical_tools" in fixes and isinstance(body.get("tools"), list) and body["tools"]:
        tools = canonical_tools(body["tools"])
        if tools != body["tools"] or json.dumps(tools) != json.dumps(body["tools"]):
            out = {**out, "tools": tools}
            applied.append("canonical_tools")
    return out, applied


def parse_fixes(spec: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """'canonical_tools,other' or a list -> validated tuple. 'optimize' stands for PROVEN_FIXES.
    Unknown names raise."""
    if not spec:
        return ()
    names = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if s and s.strip()]
    if OPTIMIZE in names:
        names = [n for n in names if n != OPTIMIZE] + list(PROVEN_FIXES)
    unknown = [n for n in names if n not in AVAILABLE_FIXES]
    if unknown:
        raise ValueError(f"unknown fix {unknown}; available: {', '.join(AVAILABLE_FIXES)}")
    return tuple(dict.fromkeys(names))


# Cause -> how to fix it. "toggle" names a proxy fix that applies it for you.
FIX_ADVICE: dict[str, dict] = {
    "tool_reserialize": {"toggle": "canonical_tools",
                         "fix": "Send tools in a fixed order with deterministic serialization "
                                "(or let the proxy do it: prefixwise serve --fix canonical_tools)."},
    "volatile_system": {"toggle": None,
                        "fix": "Move dates, times and ids out of the system prompt, to the end of the prompt or "
                               "the latest user message, in one fixed format (a working directory there keeps "
                               "other sessions from sharing the prefix)."},
    "tool_result_mutation": {"toggle": None,
                             "fix": "Store each tool result once, exactly as first sent; never re-render it."},
    "history_edit": {"toggle": None,
                     "fix": "Append to the conversation; do not rewrite earlier messages. If you must, "
                            "send X-Prefixwise-Compaction-Event: 1 and compact at task boundaries."},
    "retry_replay": {"toggle": None,
                     "fix": "Check retry logic: identical requests within seconds are re-billed in full."},
    "breakpoint_misplaced": {"toggle": "auto_breakpoints",
                             "fix": "Anthropic: add a cache_control breakpoint after the stable prefix "
                                    "(tools and system), and one on the last message so the growing "
                                    "history is cached too."},
}


# Detail -> its fix, for details under a cause that is not waste (printed by findings.render).
DETAIL_ADVICE: dict[str, str] = {
    "session_header_dropped": (
        "Send the conversation's id as a body session_id as well as the x-session-id header: SGLang reads the "
        "session from the body only, and a body field passes through sgl-router, which per sglang#47763 forwards "
        "only an allowlist of headers (should_forward_request_header), without x-session-id. SGLang PR #45620 would "
        "read the header when the body has no session_id; it is not merged."),
}


# Anthropic's multipliers on the input rate: a 5-minute cache write 1.25x, a read 0.1x (docs.anthropic.com
# prompt caching pricing, Nov 2026). auto_breakpoints places 5-minute markers only. Against a placed marker
# counts the surcharge above sending the tokens uncached, 0.25 per token written; a read saves 0.9.
_WRITE_SURCHARGE = 0.25
_READ_SAVING = 0.9


def breakpoints_paid(ledger) -> dict | None:
    """Did the markers auto_breakpoints placed pay for themselves? Over the rows it rewrote: the cache
    writes those markers caused, the reads they enabled (Anthropic serves no read without a marker; a
    few may be of entries another client's own markers wrote, whose write is not counted here), and
    the reads' saving against the writes' surcharge, in input-token equivalents. Reads of these writes
    by a client with its own markers are not credited. None when it rewrote nothing. A model from
    the multipliers above, not a bill."""
    with ledger._lock:
        rows = ledger.conn.execute(
            "SELECT cache_write_tokens, cached_tokens FROM entries "
            "WHERE fixes IS NOT NULL AND (',' || fixes || ',') LIKE '%,auto_breakpoints,%' AND error IS NULL").fetchall()
    if not rows:
        return None
    written = sum(w or 0 for w, _ in rows)
    read = sum(r or 0 for _, r in rows)
    surcharge = written * _WRITE_SURCHARGE
    saving = read * _READ_SAVING
    return {"requests": len(rows), "written_tokens": written, "read_tokens": read,
            "write_surcharge_token_equivalents": round(surcharge), "read_saving_token_equivalents": round(saving),
            "net_token_equivalents": round(saving - surcharge),
            "paid": (saving > surcharge) if (written or read) else None}


def recommend(ledger) -> dict:
    """Fixes ranked by the avoidable tokens this ledger attributes to their cause
    (the data behind `prefixwise fixes` and GET /api/v1/fixes)."""
    from . import metrics

    total = metrics.query(ledger, ["prompt_tokens"])
    prompt = total[0]["prompt_tokens"] if total else 0
    rows = {r["cause"]: r for r in metrics.query(
        ledger, ["requests", "avoidable_tokens", "avoidable_cost_usd", "requests_priced"], by="cause")}
    with ledger._lock:
        fixed = dict(ledger.conn.execute(
            "SELECT fixes, COUNT(*) FROM entries WHERE fixes IS NOT NULL GROUP BY fixes").fetchall())
    recs = []
    for cause, advice in FIX_ADVICE.items():
        r = rows.get(cause)
        if not r or not r["avoidable_tokens"]:
            continue
        status = FIX_STATUS.get(advice["toggle"] or "", {})
        recs.append({"cause": cause, "requests": r["requests"], "predicted_savings_tokens": r["avoidable_tokens"],
                     "predicted_savings_usd": r["avoidable_cost_usd"],
                     "unpriced_requests": r["requests"] - (r["requests_priced"] or 0),
                     "share_of_prompt_tokens": (r["avoidable_tokens"] / prompt) if prompt else None,
                     "fix": advice["fix"], "proxy_toggle": advice["toggle"],
                     "status": status.get("status") or "recommended (predicted; prove it with experiments/e4)",
                     "evidence": status.get("evidence")})
    recs.sort(key=lambda x: -x["predicted_savings_tokens"])
    return {"recommendations": recs, "requests_rewritten_by_fixes": fixed, "auto_breakpoints": breakpoints_paid(ledger)}
