"""Offline import of agent session logs: Prefixwise for people a proxy cannot reach.

Claude Code subscribers (Pro/Max) sign in rather than use an API key, so their traffic
cannot go through a proxy. Claude Code writes every session to
~/.claude/projects/<project>/<session>.jsonl (subagents in <session>/subagents/), and
each API response there carries its usage: cache reads, and cache writes split into
5-minute and 1-hour buckets.

`import_claude_code` reads those logs (read-only), turns each API response into a
ledger row, and never stores message text. The prompt itself is not in the log, so a
cause is claimed only from what the log shows changed in the cached prefix:

  none               read what the previous request on this model left in the cache (also for
                     a first request that reused an entry written elsewhere: a parent, a fork,
                     a resume, a sibling subagent); or read at least half of it when nothing in
                     the log explains the rest
  first_write        the first request in a log, reading little
  model_switch       the first request on a model in this log (unless it read its whole
                     prompt); or a switch to a model whose own cache had expired while the
                     conversation went on elsewhere (caches are per model)
  compaction_rewrite a compaction boundary since the previous request on this model
  ttl_expired        more time passed since the previous request on this model was sent than its
                     write was kept (Anthropic reports 5-minute and 1-hour writes; with neither,
                     what the request asked for, or one hour); `ttl_shortened` when that request
                     asked for 1 hour and was kept 5 minutes
  (Anthropic's own)  a cache_miss_reason in the response, when the client asked for one
  system_edit        under half was read and the logged system prompt changed (compared by hash,
                     never stored), or Claude Code was upgraded
  param_rendered     under half was read and the effort setting changed, on models where that
                     breaks the cache
  unknown            nothing in the log explains it, and it says so

Each request is compared with the previous one on the same model that used the cache, since
caches are per model: a small side call on another model in between is not taken for the
conversation, and a call that read and wrote nothing does not count as a use. A request that read less than was left,
after a compaction, an expiry or a model switch, gets that cause even when it read more than half:
a short or just-compacted conversation can lose all of its own history and still read most of its
prompt, because the system prompt and tools at the front can be in the cache independently of it
(measured: about 60,000 tokens read on the first call of a session and after 21 hours idle). A
changed system prompt, effort or version is at the front of the prompt, so it is named only when
under half was read: a read that goes past the front was not broken there.

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
from .fingerprint import schema_sets
from .ledger import Ledger, LedgerEntry, avoidable_for_turn, missed_for_turn, unread_write
from .pricing import PriceTable, breakdown

CLASSIFIER_VERSION = "log-3"  # of the Claude Code rules below; stored on each row
HIT_SHARE = 0.5  # as the proxy: a turn that read at least half of the reusable prefix is a hit
# A turn that read this share of what the previous one left cached reused all of it: the log has
# exact token counts, so a full reuse reads at least 100% (measured on 10,245 real hits).
FULL_SHARE = 0.95
# Attachments Claude Code writes when the response arrives, not when the request is sent (measured:
# stamped within 50 ms of the response on 2,905 of 2,905). They say nothing about the send time.
_AT_RESPONSE = {"deferred_tools_record"}
# The same for a record type not known here: an attachment or system entry stamped this close to the
# response was written with it. Those written with the request are stamped earlier by the time the
# model takes to start answering (measured: 0.5 s at least). A user entry always counts.
RESPONSE_SLACK = 0.1
ANY_TTL = 3600.0  # the longest Anthropic keeps a cache entry: past it, expired under any lifetime
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
    sent: float | None  # when the request was sent: its last logged input, or the previous response's end if later
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
    inputs: dict[str, list[tuple[float, bool]]] = {}  # (stamp, is a user entry) since the stream's previous response
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
                # Sent after its last input and after the previous response ended. An input can be
                # stamped earlier than that (a message queued while the previous response streamed).
                before = streams.get(key)
                stamps = [t for t, user in inputs.pop(key, ()) if t <= ts and (user or t < ts - RESPONSE_SLACK)]
                if before and before[-1].done <= ts:
                    stamps.append(before[-1].done)
                sent = max(stamps, default=None)
                r = _Request(ids[0], sent, ts, model, usage, e.get("promptCacheRequested"),
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
                if att.get("type") in _AT_RESPONSE:
                    continue
            elif kind == "system" and e.get("subtype") == "compact_boundary":
                pending.setdefault(key, []).append("compact_boundary")
            if ts is not None and kind in ("user", "attachment", "system"):
                inputs.setdefault(key, []).append((ts, kind == "user"))
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


@dataclass
class _Lane:
    """The last request on one model in a stream. Caches are per model, so a request is compared
    with this, not with whatever came just before it on another model."""
    req: _Request
    prompt: int
    held: int           # what it left in the cache (read + written)
    ttl: float | None   # how long its write is kept: granted, else carried from an earlier write, else asked for
    granted: bool       # the log showed that lifetime on a write (not only what the request asked for)
    usage: dict
    compactions: int    # compaction boundaries the stream had seen by then
    summed: bool        # its usage is the sum of several model calls: no basis for an exact comparison


def _classify(prev: _Request | None, prev_lane: _Lane | None, same: _Lane | None, r: _Request, prompt: int,
              read: int, subagent: bool, compacted: bool = False) -> tuple[str, bool | None, str, str | None]:
    """(cause, legit, evidence, detail) from what the log shows. prev: the stream's previous request,
    prev_lane: its model's lane; same: the previous request on this request's model that used the
    cache; compacted: a compaction boundary was logged since `same`."""
    several = isinstance(r.usage.get("iterations"), list) and len(r.usage["iterations"]) > 1
    summed = ("unknown", None, "the response combined several model calls; its usage is their sum", None)
    if prev is None:
        if prompt > 0 and read >= HIT_SHARE * prompt:
            return "none", None, ("first request in this log, yet it read an existing cache entry (written by a "
                                  "parent session, a fork, a resume, or another subagent with the same prompt)"), None
        if several:
            return summed
        return "first_write", True, ("first request of a subagent: it does not share the parent's prefix"
                                     if subagent else "first request in this log"), None
    switch = f"model {prev.model} -> {r.model}"
    if same is None:
        # Nothing of this conversation was cached on this model, as far as the log shows: whatever was
        # read (the front of the prompt can be in the cache independently) the rest is the switch's.
        if prompt > 0 and read >= FULL_SHARE * prompt:
            return "none", None, f"first request on {r.model} in this log, yet it read its whole prompt", None
        return summed if several else ("model_switch", True, switch, None)
    switched = prev.model != r.model
    here = " on this model" if switched else ""
    # No more than the previous request left could be read, and no more than this prompt holds.
    exact = same.held > 0 and not same.summed
    reusable = min(same.held, prompt) if exact else min(same.prompt, prompt)
    if exact and prompt > 0 and read >= FULL_SHARE * reusable:
        return "none", None, f"read what the previous request{here} left in the cache", None
    if not exact and prompt > 0 and read >= FULL_SHARE * prompt:
        # Nothing is known to have been left on this model, yet the whole prompt was read.
        return "none", None, "read its whole prompt from the cache", None
    partly = prompt > 0 and read >= HIT_SHARE * reusable
    kept_half = ("none", None, f"read at least half of what the previous request{here} left in the cache, and "
                               "the log shows nothing that explains the rest", None)
    if several or (same.summed and partly):
        # A sum of several calls is no basis for "less was read than was left": only a read under
        # half of the prompt is taken as a miss.
        return kept_half if partly else summed
    was = same.req
    since_use = (r.sent or r.done) - (was.sent or was.done)  # the cache was last used when that request was sent
    expired = since_use > (same.ttl if same.ttl is not None else ANY_TTL)
    # Less was read than was left. A compaction, an expiry or a model switch explains it even above
    # half: the front of the prompt can be in the cache independently of this conversation.
    if switched and expired and prev_lane is not None:
        there = prev_lane.req  # the other model's last request that used its cache
        since_prev = (r.sent or r.done) - (there.sent or there.done)
        # The conversation was on the other model if that request carried a prompt of this one's
        # size. A small side call (a title, a classification) is not the conversation.
        carried = min(prev_lane.prompt, prompt) >= HIT_SHARE * max(prev_lane.prompt, prompt)
        if carried and since_prev <= (prev_lane.ttl if prev_lane.ttl is not None else ANY_TTL):
            # It was live there: staying would have hit.
            return "model_switch", True, (f"{switch}; {r.model}'s own cache, last used {since_use:.0f} s ago, "
                                          "had expired"), None
    if compacted and (not switched or (exact and prompt < same.held)):
        # (After a spell on another model, only a prompt smaller than what this model held was the
        # one compacted: a side call on this model in between was not.)
        return "compaction_rewrite", True, "the conversation was compacted before this request", None
    upgraded = (f"Claude Code was upgraded ({was.version} -> {r.version})"
                if was.version and r.version and was.version != r.version else None)
    if expired:
        # An expiry explains the miss by itself. A change in the same gap is context, not the cause:
        # the cache was gone either way.
        why = f"{since_use:.0f} s since the previous request{here}; " + (
            "longer than any cache lifetime (1 hour)" if same.ttl is None else
            f"its cache write was kept {same.ttl:.0f} s" if same.granted else
            f"it asked for a cache kept {same.ttl:.0f} s")
        detail = None
        if was.ttl_requested == "1h" and same.ttl == 300.0:
            detail, why = "ttl_shortened", why + " although that request asked for a 1-hour cache"
        also = [x for x in (switch if switched else None, upgraded) if x]
        return "ttl_expired", True, why + (f"; also in between: {'; '.join(also)}" if also else ""), detail
    if r.native and r.native not in NO_VERDICT and map_native_reason("anthropic", r.native) is not None:
        cause = map_native_reason("anthropic", r.native).value
        return cause, None, f"Anthropic's own diagnosis: {r.native}", None
    if partly:
        # The system prompt, the effort and the version are at the front of the prompt. A read that
        # went past the front was not broken there, whatever changed.
        return kept_half
    if was.system_hash and r.system_hash and was.system_hash != r.system_hash:
        return "system_edit", None, "the system prompt Claude Code logged changed", None
    if was.effort != r.effort and was.effort and r.effort and not _EFFORT_SAFE.search(r.model):
        return "param_rendered", None, f"effort changed ({was.effort} -> {r.effort})", None
    if upgraded:
        return "system_edit", True, upgraded + ", which can change its system prompt and tools", None
    seen = sorted({_APPENDED[e] for e in r.events if e in _APPENDED})
    context = (f"; the log shows {', '.join(seen)} before it, which Claude Code appends after the cached prefix"
               if seen else "")
    return "unknown", None, (f"nothing in the log explains it ({since_use:.0f} s since the previous request{here}, "
                             f"within the cache lifetime): a change Claude Code does not log, or a server-side miss"
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
    # Rows an earlier version of these rules classified. They are kept as they are (the logs they
    # came from may be gone), and counted, so the caller can say how to get them classified again.
    with ledger.reader() as conn:
        older = {row[0] for row in conn.execute(
            "SELECT request_id FROM entries WHERE classifier_version LIKE 'log-%' AND classifier_version != ?",
            (CLASSIFIER_VERSION,))}
    seen: set[str] = set()  # imported in this run: the same session can be logged twice (two project dirs)
    stats = {"logs": 0, "requests": 0, "in_window": 0, "imported": 0, "skipped_known": 0, "duplicates": 0,
             "unreadable": 0, "older_verdicts": 0, "first": None, "last": None}
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
            prev = None
            lanes: dict[str, _Lane] = {}  # per model: caches are per model
            compactions = 0
            for r in reqs:
                stats["requests"] += 1
                u = r.usage
                read = _int(u.get("cache_read_input_tokens"))
                write = _int(u.get("cache_creation_input_tokens"))
                prompt = _int(u.get("input_tokens")) + read + write
                compactions += r.events.count("compact_boundary")
                same = lanes.get(r.model)
                prev_lane = lanes.get(prev.model) if prev is not None else None
                cause, legit, why, detail = _classify(prev, prev_lane, same, r, prompt, read, subagent,
                                                      compacted=same is not None and compactions > same.compactions)
                # This request's own cache lifetime: what its write was granted, else what it asked for.
                asked = {"1h": 3600.0, "5m": 300.0}.get(r.ttl_requested or "")
                own_ttl = _granted_ttl(u) or asked
                # Since the previous turn on this model (the ledger's definition), else the stream's.
                before = same.req if same is not None else prev
                gap = max(0.0, (r.sent or r.done) - before.done) if before is not None else None
                prev_u = same.usage if same is not None else None
                prev_prompt_before = same.prompt if same is not None else None
                held = same.held if same is not None and same.held > 0 else None
                avoidable = avoidable_for_turn(legit, prompt, read, prev_prompt_before, held)
                # An expiry gets a figure only when nothing else changed in the gap, as far as the log
                # shows: the same model throughout, the same Claude Code version, the same system
                # prompt snapshot. And only when what the previous request left is exact (not a sum of
                # several calls). (`same.held`, not `held`: zero is "left nothing", not "unknown".)
                was = same.req if same is not None else None
                only_expired = (was is not None and not same.summed and prev is not None and prev.model == r.model
                                and not (was.version and r.version and was.version != r.version)
                                and not (was.system_hash and r.system_hash and was.system_hash != r.system_hash))
                missed = missed_for_turn(cause, legit, prompt, read, prev_prompt_before,
                                         same.held if same is not None else None, unchanged=only_expired)
                several = isinstance(u.get("iterations"), list) and len(u["iterations"]) > 1
                if read + write > 0 or same is None or same.held == 0:
                    # A call that read and wrote nothing did not touch this model's cache: what the
                    # lane holds, since when and for how long stay as they were.
                    granted = _granted_ttl(u)
                    lanes[r.model] = _Lane(req=r, prompt=prompt, held=read + write,
                                           ttl=granted or (same.ttl if same else None) or asked,
                                           granted=bool(granted or (same and same.granted)), usage=u,
                                           compactions=compactions, summed=several)
                prev = r
                if since is not None and r.done < since:
                    continue
                stats["in_window"] += 1
                if r.request_id in seen:
                    stats["duplicates"] += 1
                    continue
                if r.request_id in known:
                    stats["skipped_known"] += 1
                    stats["older_verdicts"] += r.request_id in older
                    continue
                parts = breakdown("anthropic", u)
                entry = LedgerEntry(
                    session_id=session, agent_id=agent, provider="anthropic", model=r.model, cause=cause,
                    prompt_tokens=prompt, cached_tokens=read, avoidable_tokens=avoidable, avoidable_cost=None,
                    legit=legit, request_id=r.request_id, completion_tokens=_int(u.get("output_tokens")),
                    **prices.cost_fields(r.model, "anthropic", u), api="messages",
                    uncached_tokens=parts.uncached_input,
                    cache_write_tokens=parts.cache_write_5m + parts.cache_write_1h + parts.cache_write,
                    gap_seconds=gap, evidence=f"from the log: {why}", detail=detail, native_reason=r.native,
                    cache_ttl_seconds=own_ttl,
                    classifier_version=CLASSIFIER_VERSION)
                entry.avoidable_cost = prices.avoidable_cost(r.model, avoidable, prompt, "anthropic", u)
                entry.missed_tokens = missed
                if missed is not None:
                    entry.missed_cost = prices.missed_cost(r.model, missed, prompt, "anthropic", u)
                if entry.cost is None:  # no rates, or no usage: no dollar figure of any kind
                    entry.avoidable_cost = entry.missed_cost = None
                # A compaction's summary call reads the previous write, and Claude Code does not log
                # that call: the write was not left unread. A request that used no cache at all leaves
                # its model's lane as it was, so the write is counted at the next request that does.
                if prev_u is not None and prompt > 0 and cause != "compaction_rewrite" and read + write > 0:
                    pw = breakdown("anthropic", prev_u)
                    w5, w1, _ = unread_write(cause, read, pw.cache_write_5m, pw.cache_write_1h, 0,
                                             _int(prev_u.get("cache_read_input_tokens")))
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
# tool_catalog_change and tool_reserialize (the same tools in another order; tools come first in
# the prefix). OpenAI's in-memory cache lifetime
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
    tools_hash: str | None  # the tool list as sent, in order
    compacted: bool
    tools_set_hash: str | None = None  # the same tools in any order: equal with tools_hash changed is a reorder


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
    model, tools_hash, tools_set_hash, compacted, last_input = "unknown", None, None, False, None
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
                tools = p.get("tools")
                tools_hash = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:16]
                # Order-independent: the multiset of tools. Reported on Reddit (kantorcodes1): Codex rebuilds
                # its dynamic tool list and sends the same tools in another order, which is waste
                # (tool_reserialize), not a catalog change.
                tools_set_hash = (hashlib.sha256(json.dumps(sorted(json.dumps(schema_sets(t), sort_keys=True) for t in tools))
                                                 .encode()).hexdigest()[:16] if isinstance(tools, list) else None)
            elif kind == "token_usage_record" and isinstance(p.get("usage"), dict) and ts is not None:
                shaped, prompt, cached = _codex_usage(p["usage"])
                rid = str(p.get("response_id") or f"{path.stem}:{n}")
                records.append(_CodexRequest(rid, last_input, ts, model, shaped, prompt, cached, tools_hash, compacted,
                                             tools_set_hash))
                compacted = False
                continue
            elif kind == "event_msg" and p.get("type") == "token_count" and ts is not None:
                info = p.get("info") if isinstance(p.get("info"), dict) else {}
                last = info.get("last_token_usage")
                if isinstance(last, dict):
                    shaped, prompt, cached = _codex_usage(last)
                    counts.append(_CodexRequest(f"{path.stem}:{n}", last_input, ts, model, shaped, prompt, cached,
                                                tools_hash, compacted, tools_set_hash))
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
        if prev.tools_set_hash is not None and prev.tools_set_hash == r.tools_set_hash:
            return "tool_reserialize", False, "the same tools were sent in a different order (dynamic tools updated)"
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
        prev, prev_prompt = None, None
        for r in reqs:
            stats["requests"] += 1
            cause, legit, why = _classify_codex(prev, r, bool(meta.get("forked_from_id")), subagent)
            gap = (r.sent or r.done) - prev.done if prev is not None else None
            # Waste (legit False, e.g. a reordered tool list) is priced as the proxy and the Claude Code
            # importer price it: the previous prompt that could have been read but was not.
            held = None  # what the previous request left in the cache, when the log says (OpenAI rarely does)
            if prev is not None:
                write = _int((prev.usage.get("input_tokens_details") or {}).get("cache_write_tokens"))
                held = prev.cached + write if write else None
            avoidable = avoidable_for_turn(legit, r.prompt, r.cached, prev_prompt, held)
            prev, prev_prompt = r, r.prompt
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
                prompt_tokens=r.prompt, cached_tokens=r.cached, avoidable_tokens=avoidable, avoidable_cost=None,
                legit=legit, request_id=r.request_id, completion_tokens=_int(r.usage.get("output_tokens")),
                **prices.cost_fields(r.model, "openai", r.usage), api="responses",
                uncached_tokens=parts.uncached_input, cache_write_tokens=parts.cache_write,
                gap_seconds=gap, evidence=f"from the log: {why}", classifier_version="codex-log-2")
            entry.avoidable_cost = prices.avoidable_cost(r.model, avoidable, r.prompt, "openai", r.usage)
            if entry.cost is None:  # no rates, or no usage: no dollar figure of any kind
                entry.avoidable_cost = None
            if entry.cost is not None:
                entry.price_as_of = prices.as_of
            ledger.record(entry, at=r.done)
            seen.add(r.request_id)
            stats["imported"] += 1
            stats["first"] = r.done if stats["first"] is None else min(stats["first"], r.done)
            stats["last"] = r.done if stats["last"] is None else max(stats["last"], r.done)
    return stats
