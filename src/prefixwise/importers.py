"""Offline import of agent session logs: Prefixwise for people a proxy cannot reach.

Claude Code subscribers (Pro/Max) sign in rather than use an API key, so their traffic
cannot go through a proxy. Claude Code writes every session to
~/.claude/projects/<project>/<session>.jsonl (subagents in <session>/subagents/), and
each API response there carries its usage: cache reads, and cache writes split into
5-minute and 1-hour buckets. Between responses the log records what changed: tool,
MCP-instruction, skill and agent-list deltas, compactions, model and date changes.

`import_claude_code` reads those logs (read-only), turns each API request into a
ledger row with a cause inferred from the log, and never stores message text. The
system prompt and tool bodies are not in the log, so a cause is only claimed from an
event the log shows or from timing; a miss with neither is `unknown`, and says so.
Every verdict's evidence starts with "from the log", to tell it from a proxied verdict.

The log format is Claude Code's own, undocumented, and changes between versions:
unknown entries are skipped, never fatal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .ledger import Ledger, LedgerEntry, avoidable_for_turn
from .pricing import PriceTable, breakdown

# Log events that change the prompt before the next request, and the cause each implies.
# First match in prefix order wins (tools come before the system prompt).
_EVENT_CAUSE = (
    ("deferred_tools_delta", "tool_catalog_change", True, "the tool list changed (deferred tools loaded or dropped)"),
    ("mcp_instructions_delta", "system_edit", None, "MCP server instructions in the system prompt changed"),
    ("skill_listing", "system_edit", None, "the skill listing in the prompt changed"),
    ("agent_listing_delta", "system_edit", None, "the agent listing in the prompt changed"),
    ("date", "volatile_system", False, "the date in the prompt changed"),
)
HIT_SHARE = 0.5  # as the proxy: a turn that read at least half of the reusable prefix is a hit


@dataclass
class _Request:
    request_id: str
    sent: float | None  # time of the last log entry before the response (the request's inputs)
    done: float  # response time
    model: str
    usage: dict
    ttl_requested: str | None
    events: list[str] = field(default_factory=list)  # attachment types / "compact_boundary" since the last request


def _ts(raw) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _requests(path: Path) -> list[_Request]:
    """API requests in one log file, in order, deduplicated by request id."""
    out: list[_Request] = []
    by_id: dict[str, _Request] = {}
    pending: list[str] = []
    last_input_ts: float | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            kind, ts = e.get("type"), _ts(e.get("timestamp"))
            if kind == "assistant":
                msg = e.get("message") if isinstance(e.get("message"), dict) else {}
                usage = msg.get("usage")
                # One response is logged as several entries (one per content block), and not every
                # entry carries both ids: the message id (msg_...) and the API request id (req_...)
                # are aliases of one response.
                ids = [str(i) for i in (msg.get("id"), e.get("requestId")) if i]
                if not isinstance(usage, dict) or not ids or ts is None:
                    continue
                r = next((by_id[i] for i in ids if i in by_id), None)
                if r is not None:
                    if (usage.get("output_tokens") or 0) >= (r.usage.get("output_tokens") or 0):
                        r.usage = usage
                    r.done = max(r.done, ts)
                    for i in ids:
                        by_id[i] = r
                    continue
                r = _Request(ids[0], last_input_ts, ts, str(msg.get("model") or e.get("requestedModel") or "unknown"),
                             usage, e.get("promptCacheRequested"), pending)
                for i in ids:
                    by_id[i] = r
                out.append(r)
                pending = []
                continue
            if kind == "attachment":
                att = e.get("attachment") if isinstance(e.get("attachment"), dict) else {}
                if att.get("type"):
                    pending.append(str(att["type"]))
            elif kind == "system" and e.get("subtype") == "compact_boundary":
                pending.append("compact_boundary")
            if ts is not None and kind in ("user", "attachment", "system"):
                last_input_ts = ts
    return out


def _agent_label(log: Path) -> str:
    """'main' for a session, the subagent type (e.g. Explore) for a subagent."""
    if log.parent.name != "subagents":
        return "main"
    meta = log.with_name(log.name.removesuffix(".jsonl") + ".meta.json")
    try:
        return "subagent:" + str(json.loads(meta.read_text()).get("agentType") or "unknown")
    except (OSError, ValueError):
        return "subagent:unknown"


def _granted_ttl(usage: dict) -> float | None:
    created = usage.get("cache_creation") or {}
    if int(created.get("ephemeral_1h_input_tokens") or 0) > 0:
        return 3600.0
    if int(created.get("ephemeral_5m_input_tokens") or 0) > 0:
        return 300.0
    return None


def _classify(prev: _Request | None, prev_ttl: float | None, r: _Request, prompt: int, read: int,
              prev_prompt: int | None, first_cause: str) -> tuple[str, bool | None, str, str | None]:
    """(cause, legit, evidence, detail) from what the log shows."""
    if prev is None:
        return first_cause, True, "first request in this log", None
    if prev.model != r.model:
        return "model_switch", True, f"model {prev.model} -> {r.model}", None
    reusable = min(prev_prompt, prompt) if prev_prompt else prompt
    if prompt > 0 and read >= HIT_SHARE * reusable:
        return "none", None, "read at least half of the previous prompt from the cache", None
    if "compact_boundary" in r.events:
        return "compaction_rewrite", True, "the conversation was compacted before this request", None
    gap = (r.sent or r.done) - prev.done
    if prev_ttl is not None and gap > prev_ttl:
        detail, why = None, f"{gap:.0f} s since the previous response; its cache write was kept {prev_ttl:.0f} s"
        if prev.ttl_requested == "1h" and prev_ttl == 300.0:  # what the previous request asked for
            detail = "ttl_shortened"
            why += " although that request asked for a 1-hour cache"
        return "ttl_expired", True, why, detail
    for event, cause, legit, why in _EVENT_CAUSE:
        if event in r.events:
            return cause, legit, why, None
    return "unknown", None, (f"nothing in the log explains it ({gap:.0f} s gap, within the cache lifetime): "
                             "a prompt change Claude Code does not log, or a server-side miss"), None


def import_claude_code(root: str | os.PathLike | None = None, ledger: Ledger | None = None,
                       prices: PriceTable | None = None, since: float | None = None) -> dict:
    """Import Claude Code session logs under root (default ~/.claude/projects) into ledger.

    Returns counts. Requests already in the ledger (by request id) are skipped, so a
    re-import adds only what is new."""
    root = Path(root or Path.home() / ".claude" / "projects").expanduser()
    ledger = ledger or Ledger()
    prices = prices or PriceTable.load(None)
    logs = sorted(root.glob("*/*.jsonl")) + sorted(root.glob("*/*/subagents/*.jsonl"))
    known = ledger.request_ids()
    seen: set[str] = set()  # imported in this run: the same session can be logged twice (two project dirs)
    stats = {"logs": 0, "requests": 0, "imported": 0, "skipped_known": 0, "duplicates": 0,
             "first": None, "last": None}
    for log in logs:
        reqs = _requests(log)
        if not reqs:
            continue
        stats["logs"] += 1
        agent = _agent_label(log)
        is_sub = agent != "main"
        session = log.parent.parent.name if is_sub else log.stem
        if is_sub:
            session = f"{session}/{log.stem}"
        prev = prev_ttl = prev_prompt = None
        for r in reqs:
            stats["requests"] += 1
            u = r.usage
            read = int(u.get("cache_read_input_tokens") or 0)
            write = int(u.get("cache_creation_input_tokens") or 0)
            prompt = int(u.get("input_tokens") or 0) + read + write
            cause, legit, why, detail = _classify(prev, prev_ttl, r, prompt, read, prev_prompt,
                                                  "first_write")
            if cause == "first_write" and is_sub and read < 0.1 * prompt:
                why = "first request of a subagent: it does not share the parent's prefix"
            gap = (r.sent or r.done) - prev.done if prev is not None else None
            avoidable = avoidable_for_turn(legit, prompt, read, prev_prompt)
            prev, prev_prompt = r, prompt
            prev_ttl = _granted_ttl(u) or prev_ttl
            if since is not None and r.done < since:
                continue
            if r.request_id in seen:
                stats["duplicates"] += 1
                continue
            if r.request_id in known:
                stats["skipped_known"] += 1
                continue
            parts = breakdown("anthropic", u)
            entry = LedgerEntry(
                session_id=session, agent_id=agent, provider="anthropic", model=r.model, cause=cause,
                prompt_tokens=prompt, cached_tokens=read, avoidable_tokens=avoidable, avoidable_cost=None,
                legit=legit, request_id=r.request_id, completion_tokens=int(u.get("output_tokens") or 0),
                cost=prices.request_cost(r.model, "anthropic", u), api="messages",
                uncached_tokens=parts.uncached_input,
                cache_write_tokens=parts.cache_write_5m + parts.cache_write_1h + parts.cache_write,
                gap_seconds=gap, evidence=f"from the log: {why}", detail=detail,
                classifier_version="log-1")
            entry.avoidable_cost = prices.avoidable_cost(r.model, avoidable, prompt)
            if entry.cost is not None:
                entry.price_as_of = prices.as_of
            ledger.record(entry, at=r.done)
            seen.add(r.request_id)
            stats["imported"] += 1
            stats["first"] = r.done if stats["first"] is None else min(stats["first"], r.done)
            stats["last"] = r.done if stats["last"] is None else max(stats["last"], r.done)
    return stats
