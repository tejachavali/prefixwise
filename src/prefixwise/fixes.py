"""Fix toggles and fix advice.

A toggle is a rewrite the proxy can apply to a request before forwarding
it, opt-in and reversible (turn it off and requests go through unchanged).
Only rewrites that cannot change what the prompt means are toggles:

- canonical_tools: tools sorted by name, each tool's keys sorted. The
  provider caches by exact prefix, and tools come first in every
  provider's prefix, so a tool list whose order or key order drifts
  between turns (plugin load order, dict/set iteration, a framework
  rebuilding definitions) misses the whole cache. Sorting is stable:
  same tool set, same bytes, every turn.

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

AVAILABLE_FIXES = ("canonical_tools",)


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
    with every tool's keys sorted. Same set in, same bytes out."""
    named = sorted((t for t in tools if _tool_name(t) is not None), key=_tool_name)
    unnamed = [t for t in tools if _tool_name(t) is None]
    return [_sorted_keys(t) for t in named] + [_sorted_keys(t) for t in unnamed]


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
    """'canonical_tools,other' or a list -> validated tuple. Unknown names raise."""
    if not spec:
        return ()
    names = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if s and s.strip()]
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
                        "fix": "Move dates, times and ids out of the system prompt, to the end of the "
                               "prompt or the latest user message."},
    "tool_result_mutation": {"toggle": None,
                             "fix": "Store each tool result once, exactly as first sent; never re-render it."},
    "history_edit": {"toggle": None,
                     "fix": "Append to the conversation; do not rewrite earlier messages. If you must, "
                            "send X-Prefixwise-Compaction-Event: 1 and compact at task boundaries."},
    "retry_replay": {"toggle": None,
                     "fix": "Check retry logic: identical requests within seconds are re-billed in full."},
    "breakpoint_misplaced": {"toggle": None,
                             "fix": "Anthropic: add a cache_control breakpoint after the stable prefix "
                                    "(tools and system)."},
}
