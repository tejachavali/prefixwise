"""Offline import of agent session logs: Prefixwise for people a proxy cannot reach.

Claude Code subscribers (Pro/Max) sign in rather than use an API key, so their traffic
cannot go through a proxy. Claude Code writes every session to
~/.claude/projects/<project>/<session>.jsonl (subagents in <session>/subagents/), and
each API response there carries its usage: cache reads, and cache writes split into
5-minute and 1-hour buckets.

`import_claude_code` reads those logs (read-only), turns each API response into a
ledger row, and never stores message text. The prompt itself is not in the log, so a
cause is claimed only from what the log shows changed in the cached prefix:

  none               read at least half of what could be reused (also for a first request
                     that reused an entry written elsewhere: a parent, a fork, a resume)
  first_write        the first request in a log, reading little
  model_switch       the served model changed (caches are per model)
  compaction_rewrite a compaction boundary since the previous request
  system_edit        Claude Code was upgraded (the version changed), or its logged system
                     prompt changed (compared by hash, never stored)
  param_rendered     the effort setting changed, on models where that breaks the cache
  ttl_expired        more time passed since the previous request than its write was kept
                     (Anthropic reports 5-minute and 1-hour writes); `ttl_shortened` when
                     that request asked for 1 hour and was kept 5 minutes
  (Anthropic's own)  a cache_miss_reason in the response, when the client asked for one
  unknown            nothing in the log explains it, and it says so

What Claude Code appends mid-conversation (deferred tool loads, MCP instruction updates,
skill and agent listings, the date, file-change notices) comes after the cached prefix
and does not break it (Claude Code prompt-caching docs), so it is never a cause here;
on an unexplained miss it is mentioned as context. Every verdict's evidence starts with
"from the log". The format is Claude Code's own, undocumented, and changes between
versions: unknown entries and unreadable files are skipped, never fatal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .diagnostics import NO_VERDICT, map_native_reason
from .ledger import Ledger, LedgerEntry, avoidable_for_turn, unread_write
from .pricing import PriceTable, breakdown

HIT_SHARE = 0.5  # as the proxy: a turn that read at least half of the reusable prefix is a hit
# Appended after the cached prefix: context for an unexplained miss, never a cause.
_APPENDED = {"deferred_tools_delta": "deferred tools loaded", "mcp_instructions_delta": "MCP instructions updated",
             "skill_listing": "skill listing", "agent_listing_delta": "agent listing updated", "date": "date notice"}
# Models on which changing effort does not break the cache (Claude Code prompt-caching docs).
_EFFORT_SAFE = re.compile(r"opus-5-5|sonnet-5-5|haiku-5-5|fable-5-1")


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


_USAGE_INTS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _clean_usage(u: dict) -> dict:
    """The usage fields Prefixwise reads, as integers: a log written by another version may differ."""
    clean = {k: _int(u.get(k)) for k in _USAGE_INTS}
    created = u.get("cache_creation")
    if isinstance(created, dict):
        clean["cache_creation"] = {k: _int(created.get(k))
                                   for k in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")}
    if isinstance(u.get("iterations"), list):
        clean["iterations"] = u["iterations"]
    return clean


@dataclass
class _Request:
    request_id: str
    sent: float | None  # time of the last log entry before the response (the request's inputs)
    done: float  # response time
    model: str
    usage: dict
    ttl_requested: str | None
    version: str | None
    effort: str | None
    system_hash: str | None  # of the latest logged system prompt snapshot
    native: str | None  # Anthropic's cache_miss_reason, when the client asked for diagnostics
    events: list[str] = field(default_factory=list)  # attachment types / "compact_boundary" since the last request


def _ts(raw) -> float | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _stream_key(e: dict) -> str:
    """Subagent entries written into an older main log form their own stream."""
    if e.get("agentId"):
        return f"agent:{e['agentId']}"
    return "sidechain" if e.get("isSidechain") else "main"


def _requests(path: Path) -> dict[str, list[_Request]]:
    """API requests in one log file by stream, in order, deduplicated by response id."""
    streams: dict[str, list[_Request]] = {}
    by_id: dict[str, _Request] = {}
    pending: dict[str, list[str]] = {}
    last_input: dict[str, float] = {}
    system_hash: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            kind, ts, key = e.get("type"), _ts(e.get("timestamp")), _stream_key(e)
            if kind == "assistant":
                msg = e.get("message") if isinstance(e.get("message"), dict) else {}
                usage = msg.get("usage")
                model = str(msg.get("model") or e.get("requestedModel") or "unknown")
                if model == "<synthetic>" or e.get("isApiErrorMessage"):  # errors and interrupts, no API call
                    continue
                # One response is logged as several entries (one per content block), and not every
                # entry carries both ids: the message id and the request id are aliases.
                ids = [str(i) for i in (msg.get("id"), e.get("requestId")) if i]
                if not isinstance(usage, dict) or not ids or ts is None:
                    continue
                usage = _clean_usage(usage)
                r = next((by_id[i] for i in ids if i in by_id), None)
                if r is not None:
                    if _int(usage.get("output_tokens")) >= _int(r.usage.get("output_tokens")):
                        r.usage = usage
                    r.done = max(r.done, ts)
                    for i in ids:
                        by_id[i] = r
                    continue
                diag = msg.get("diagnostics") if isinstance(msg.get("diagnostics"), dict) else {}
                miss = diag.get("cache_miss_reason")
                native = (miss.get("type") if isinstance(miss, dict) else miss) or None
                r = _Request(ids[0], last_input.get(key), ts, model, usage, e.get("promptCacheRequested"),
                             e.get("version"), e.get("perTurnEffort") or e.get("effort"), system_hash.get(key),
                             str(native) if native else None, pending.pop(key, []))
                for i in ids:
                    by_id[i] = r
                streams.setdefault(key, []).append(r)
                continue
            if kind == "attachment":
                att = e.get("attachment") if isinstance(e.get("attachment"), dict) else {}
                if att.get("type") == "prompt_snapshot" and att.get("systemPrompt") is not None:
                    system_hash[key] = hashlib.sha256(
                        json.dumps(att["systemPrompt"], sort_keys=True).encode()).hexdigest()[:16]
                elif att.get("type"):
                    pending.setdefault(key, []).append(str(att["type"]))
            elif kind == "system" and e.get("subtype") == "compact_boundary":
                pending.setdefault(key, []).append("compact_boundary")
            if ts is not None and kind in ("user", "attachment", "system"):
                last_input[key] = ts
    return streams


def _agent_label(log: Path, stream: str) -> str:
    """'main' for a session, the subagent type (e.g. Explore) for a subagent."""
    if log.parent.name != "subagents":
        return "main" if stream == "main" else "subagent:unknown"
    meta = log.with_name(log.name.removesuffix(".jsonl") + ".meta.json")
    try:
        return "subagent:" + str(json.loads(meta.read_text()).get("agentType") or "unknown")
    except (OSError, ValueError, AttributeError):
        return "subagent:unknown"


def _granted_ttl(usage: dict) -> float | None:
    created = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    if _int(created.get("ephemeral_1h_input_tokens")) > 0:
        return 3600.0
    if _int(created.get("ephemeral_5m_input_tokens")) > 0:
        return 300.0
    return None


def _classify(prev: _Request | None, prev_ttl: float | None, r: _Request, prompt: int, read: int,
              prev_prompt: int | None, subagent: bool) -> tuple[str, bool | None, str, str | None]:
    """(cause, legit, evidence, detail) from what the log shows."""
    reusable = min(prev_prompt, prompt) if prev_prompt else prompt
    if prompt > 0 and read >= HIT_SHARE * reusable:
        return "none", None, ("read at least half of the reusable prompt from the cache" if prev is not None else
                              "first request in this log, yet it read an existing cache entry "
                              "(written by a parent session, a fork or a resume)"), None
    iterations = r.usage.get("iterations")
    if isinstance(iterations, list) and len(iterations) > 1:
        return "unknown", None, "the response combined several model calls; its usage is their sum", None
    if prev is None:
        return "first_write", True, ("first request of a subagent: it does not share the parent's prefix"
                                     if subagent else "first request in this log"), None
    if prev.model != r.model:
        return "model_switch", True, f"model {prev.model} -> {r.model}", None
    if "compact_boundary" in r.events:
        return "compaction_rewrite", True, "the conversation was compacted before this request", None
    if prev.version and r.version and prev.version != r.version:
        return "system_edit", True, f"Claude Code was upgraded ({prev.version} -> {r.version})", None
    if prev.system_hash and r.system_hash and prev.system_hash != r.system_hash:
        return "system_edit", None, "the system prompt Claude Code logged changed", None
    if prev.effort != r.effort and prev.effort and r.effort and not _EFFORT_SAFE.search(r.model):
        return "param_rendered", None, f"effort changed ({prev.effort} -> {r.effort})", None
    since_use = (r.sent or r.done) - (prev.sent or prev.done)  # the cache was last used when prev was sent
    if prev_ttl is not None and since_use > prev_ttl:
        why = f"{since_use:.0f} s since the previous request; its cache write was kept {prev_ttl:.0f} s"
        detail = None
        if prev.ttl_requested == "1h" and prev_ttl == 300.0:
            detail, why = "ttl_shortened", why + " although that request asked for a 1-hour cache"
        return "ttl_expired", True, why, detail
    if r.native and r.native not in NO_VERDICT and map_native_reason("anthropic", r.native) is not None:
        cause = map_native_reason("anthropic", r.native).value
        return cause, None, f"Anthropic's own diagnosis: {r.native}", None
    seen = sorted({_APPENDED[e] for e in r.events if e in _APPENDED})
    context = (f"; the log shows {', '.join(seen)} before it, which Claude Code appends after the cached prefix"
               if seen else "")
    return "unknown", None, (f"nothing in the log explains it ({since_use:.0f} s since the previous request, within "
                             f"the cache lifetime): a change Claude Code does not log, or a server-side miss"
                             f"{context}"), None


def import_claude_code(root: str | os.PathLike | None = None, ledger: Ledger | None = None,
                       prices: PriceTable | None = None, since: float | None = None) -> dict:
    """Import Claude Code session logs under root (default ~/.claude/projects) into ledger.

    Returns counts. Requests already in the ledger (by response id) are skipped, so a
    re-import adds only what is new."""
    root = Path(root or Path.home() / ".claude" / "projects").expanduser()
    ledger = ledger or Ledger()
    prices = prices or PriceTable.load(None)
    logs = sorted(root.glob("*/*.jsonl")) + sorted(root.glob("*/*/subagents/*.jsonl"))
    known = ledger.request_ids()
    seen: set[str] = set()  # imported in this run: the same session can be logged twice (two project dirs)
    stats = {"logs": 0, "requests": 0, "in_window": 0, "imported": 0, "skipped_known": 0, "duplicates": 0,
             "unreadable": 0, "first": None, "last": None}
    for log in logs:
        try:
            if not log.is_file():
                continue
            streams = _requests(log)
        except (OSError, UnicodeError) as exc:  # unreadable, vanished, a directory named *.jsonl
            stats["unreadable"] += 1
            stats.setdefault("unreadable_files", []).append(f"{log}: {type(exc).__name__}")
            continue
        if not streams:
            continue
        stats["logs"] += 1
        session_base = log.parent.parent.name if log.parent.name == "subagents" else log.stem
        for stream, reqs in streams.items():
            agent = _agent_label(log, stream)
            subagent = agent != "main"
            session = f"{session_base}/{log.stem}" if log.parent.name == "subagents" else (
                session_base if stream == "main" else f"{session_base}/{stream}")
            prev = prev_ttl = prev_prompt = None
            for r in reqs:
                stats["requests"] += 1
                u = r.usage
                read = _int(u.get("cache_read_input_tokens"))
                write = _int(u.get("cache_creation_input_tokens"))
                prompt = _int(u.get("input_tokens")) + read + write
                cause, legit, why, detail = _classify(prev, prev_ttl, r, prompt, read, prev_prompt, subagent)
                # This request's own cache lifetime: what its write was granted, else what it asked for.
                own_ttl = _granted_ttl(u) or {"1h": 3600.0, "5m": 300.0}.get(r.ttl_requested or "")
                gap = (r.sent or r.done) - prev.done if prev is not None else None
                avoidable = avoidable_for_turn(legit, prompt, read, prev_prompt)
                prev_u = prev.usage if prev is not None else None
                prev, prev_prompt_before, prev_prompt = r, prev_prompt, prompt
                prev_ttl = _granted_ttl(u) or prev_ttl
                if since is not None and r.done < since:
                    continue
                stats["in_window"] += 1
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
                    legit=legit, request_id=r.request_id, completion_tokens=_int(u.get("output_tokens")),
                    cost=prices.request_cost(r.model, "anthropic", u), api="messages",
                    uncached_tokens=parts.uncached_input,
                    cache_write_tokens=parts.cache_write_5m + parts.cache_write_1h + parts.cache_write,
                    gap_seconds=gap, evidence=f"from the log: {why}", detail=detail, native_reason=r.native,
                    cache_ttl_seconds=own_ttl,
                    classifier_version="log-2")
                entry.avoidable_cost = prices.avoidable_cost(r.model, avoidable, prompt, "anthropic", u)
                if prev_u is not None and prompt > 0:
                    pw = breakdown("anthropic", prev_u)
                    w5, w1, _ = unread_write(cause, read, pw.cache_write_5m, pw.cache_write_1h, 0)
                    if w5 or w1:
                        entry.unread_write_tokens = w5 + w1
                        entry.unread_write_cost = prices.write_surcharge(r.model, prev_prompt_before or 0, w5, w1)
                if entry.cost is not None:
                    entry.price_as_of = prices.as_of
                ledger.record(entry, at=r.done)
                seen.add(r.request_id)
                stats["imported"] += 1
                stats["first"] = r.done if stats["first"] is None else min(stats["first"], r.done)
                stats["last"] = r.done if stats["last"] is None else max(stats["last"], r.done)
    return stats


# -- Codex ---------------------------------------------------------------------------
#
# Codex writes each thread to ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl (older ones
# move to ~/.codex/archived_sessions): one JSON object per line, {"timestamp", "type",
# "payload"} (openai/codex, codex-rs/history/src/rollout_payload.rs, Nov 2026). Per
# response it writes a `token_usage_record` with the response id and usage (input
# tokens including cached ones, cached input, cache writes, output); older versions
# write only `event_msg` / `token_count` with `last_token_usage`. `turn_context` carries
# the model, `compacted` marks a compaction, `dynamic_tools_updated` the tool catalog,
# `session_meta` the thread id, its parent (subagents) and what it was forked from.
#
# Causes, from what changes OpenAI's cached prefix: model_switch, compaction_rewrite,
# tool_catalog_change (tools come first in the prefix). OpenAI's in-memory cache lifetime
# is not a fixed number, so expiry is never claimed: an unexplained miss is unknown.


@dataclass
class _CodexRequest:
    request_id: str
    sent: float | None
    done: float
    model: str
    usage: dict  # Responses API shape, for pricing
    prompt: int
    cached: int
    tools_hash: str | None
    compacted: bool


def _codex_usage(u: dict) -> tuple[dict, int, int]:
    prompt, cached = _int(u.get("input_tokens")), _int(u.get("cached_input_tokens"))
    shaped = {"input_tokens": prompt, "output_tokens": _int(u.get("output_tokens")),
              "input_tokens_details": {"cached_tokens": cached,
                                       "cache_write_tokens": _int(u.get("cache_write_input_tokens"))}}
    return shaped, prompt, cached


def _codex_requests(path: Path) -> tuple[dict, list[_CodexRequest]]:
    meta: dict = {}
    records: list[_CodexRequest] = []
    counts: list[_CodexRequest] = []  # older logs: token_count events, used only when no records exist
    model, tools_hash, compacted, last_input = "unknown", None, False, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            kind, ts = e.get("type"), _ts(e.get("timestamp"))
            p = e.get("payload") if isinstance(e.get("payload"), dict) else {}
            if kind == "session_meta":
                meta = p.get("meta") if isinstance(p.get("meta"), dict) else p
            elif kind == "turn_context":
                model = str(p.get("model") or model)
            elif kind == "compacted":
                compacted = True
            elif kind == "dynamic_tools_updated":
                tools_hash = hashlib.sha256(json.dumps(p.get("tools"), sort_keys=True).encode()).hexdigest()[:16]
            elif kind == "token_usage_record" and isinstance(p.get("usage"), dict) and ts is not None:
                shaped, prompt, cached = _codex_usage(p["usage"])
                rid = str(p.get("response_id") or f"{path.stem}:{n}")
                records.append(_CodexRequest(rid, last_input, ts, model, shaped, prompt, cached, tools_hash, compacted))
                compacted = False
                continue
            elif kind == "event_msg" and p.get("type") == "token_count" and ts is not None:
                info = p.get("info") if isinstance(p.get("info"), dict) else {}
                last = info.get("last_token_usage")
                if isinstance(last, dict):
                    shaped, prompt, cached = _codex_usage(last)
                    counts.append(_CodexRequest(f"{path.stem}:{n}", last_input, ts, model, shaped, prompt, cached,
                                                tools_hash, compacted))
                    if not records:
                        compacted = False
                continue
            if ts is not None:
                last_input = ts
    return meta, (records or counts)


def _classify_codex(prev: _CodexRequest | None, r: _CodexRequest, forked: bool, subagent: bool
                    ) -> tuple[str, bool | None, str]:
    reusable = min(prev.prompt, r.prompt) if prev is not None and prev.prompt else r.prompt
    if r.prompt > 0 and r.cached >= HIT_SHARE * reusable:
        return "none", None, ("read at least half of the reusable prompt from the cache" if prev is not None else
                              "first request in this log, yet it read an existing cache entry")
    if prev is None:
        if forked:
            return "fork_boundary", True, "first request of a forked thread, reading little of the parent's cache"
        return "first_write", True, ("first request of a subagent thread" if subagent else
                                     "first request in this log")
    if prev.model != r.model:
        return "model_switch", True, f"model {prev.model} -> {r.model}"
    if r.compacted:
        return "compaction_rewrite", True, "the conversation was compacted before this request"
    if prev.tools_hash != r.tools_hash and r.tools_hash is not None:
        return "tool_catalog_change", True, "the tool catalog changed (dynamic tools updated)"
    gap = (r.sent or r.done) - (prev.sent or prev.done)
    return "unknown", None, (f"nothing in the log explains it ({gap:.0f} s since the previous request; OpenAI's "
                             "cache lifetime is not fixed, so expiry, routing and an unlogged prompt change look alike)")


def import_codex(root: str | os.PathLike | None = None, ledger: Ledger | None = None,
                 prices: PriceTable | None = None, since: float | None = None) -> dict:
    """Import Codex rollout logs under root (default ~/.codex) into ledger. Re-imports add only new rows."""
    root = Path(root or Path.home() / ".codex").expanduser()
    ledger = ledger or Ledger()
    prices = prices or PriceTable.load(None)
    logs = sorted(root.rglob("rollout-*.jsonl"))
    known = ledger.request_ids()
    seen: set[str] = set()
    stats = {"logs": 0, "requests": 0, "in_window": 0, "imported": 0, "skipped_known": 0, "duplicates": 0,
             "unreadable": 0, "first": None, "last": None}
    for log in logs:
        try:
            if not log.is_file():
                continue
            meta, reqs = _codex_requests(log)
        except (OSError, UnicodeError) as exc:
            stats["unreadable"] += 1
            stats.setdefault("unreadable_files", []).append(f"{log}: {type(exc).__name__}")
            continue
        if not reqs:
            continue
        stats["logs"] += 1
        session = str(meta.get("id") or log.stem)
        subagent = bool(meta.get("parent_thread_id"))
        agent = "subagent" if subagent else "main"
        prev = None
        for r in reqs:
            stats["requests"] += 1
            cause, legit, why = _classify_codex(prev, r, bool(meta.get("forked_from_id")), subagent)
            gap = (r.sent or r.done) - prev.done if prev is not None else None
            prev = r
            if since is not None and r.done < since:
                continue
            stats["in_window"] += 1
            if r.request_id in seen:
                stats["duplicates"] += 1
                continue
            if r.request_id in known:
                stats["skipped_known"] += 1
                continue
            parts = breakdown("openai", r.usage)
            entry = LedgerEntry(
                session_id=session, agent_id=agent, provider="openai", model=r.model, cause=cause,
                prompt_tokens=r.prompt, cached_tokens=r.cached, avoidable_tokens=0, avoidable_cost=None,
                legit=legit, request_id=r.request_id, completion_tokens=_int(r.usage.get("output_tokens")),
                cost=prices.request_cost(r.model, "openai", r.usage), api="responses",
                uncached_tokens=parts.uncached_input, cache_write_tokens=parts.cache_write,
                gap_seconds=gap, evidence=f"from the log: {why}", classifier_version="codex-log-1")
            if entry.cost is not None:
                entry.price_as_of = prices.as_of
            ledger.record(entry, at=r.done)
            seen.add(r.request_id)
            stats["imported"] += 1
            stats["first"] = r.done if stats["first"] is None else min(stats["first"], r.done)
            stats["last"] = r.done if stats["last"] is None else max(stats["last"], r.done)
    return stats
