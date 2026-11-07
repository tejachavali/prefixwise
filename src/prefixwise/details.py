"""Sub-reasons: the why behind a cause, for the misses agents cause in 2026.

The cause says what broke reuse first; the detail says why, where the evidence
allows. Each detail is claimed only from facts in hand (the request bodies,
the provider's usage, what this proxy saw in flight), never guessed:

  ttl_shortened          ttl_expired: the request asked for a 1-hour cache, but the
                         previous write was cached for 5 minutes (Anthropic reports
                         writes by TTL). Seen in Claude Code issues (#46829 and the
                         October 2026 cluster).
  lookback_exceeded      the prefix was unchanged but out of reach: the previous
                         write point is more than 21 content blocks before every
                         breakpoint of this request (Anthropic checks at most 20
                         positions per breakpoint; measured: 21 added blocks hit,
                         22 missed). Cause becomes breakpoint_misplaced: an earlier
                         breakpoint fixes it.
  markers_ignored        cache_control was sent and the prompt was long enough, yet
                         the provider wrote nothing and read nothing: a gateway may
                         have stripped the markers, or the endpoint does not cache.
  sent_before_cache_written
                         another request with the same tools and system prompt was
                         still in flight when this one was sent (parallel fan-out,
                         subagents started together), so there was nothing to read yet.
  shared_prefix_not_reused
                         first request of a session whose tools and system prompt
                         another session had just cached (within 5 minutes), yet it
                         read nothing: worth checking cache keys, TTLs, tool order.
  thinking_stripped      history_edit where the only change to earlier messages is
                         removed thinking/reasoning blocks (a client or gateway
                         dropping them).
  cache_key_changed      OpenAI prompt_cache_key differs from the previous turn's.

Standard library only; pure functions over plain facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

LOOKBACK_CLAIM = 21  # positions behind a breakpoint beyond which a hit is never found (see above)
SHARED_PREFIX_WINDOW = 300.0  # seconds: no provider keeps a cache entry for less


@dataclass
class Facts:
    """What the proxy knows about one request, beyond its block fingerprints."""
    provider: str = ""
    ttl_requested: float | None = None  # seconds the request asked for (Anthropic)
    write_5m: int = 0  # tokens the provider wrote with each TTL (Anthropic usage)
    write_1h: int = 0
    cache_write: int = 0  # all cache writes, any provider
    has_breakpoint: bool = False
    breakpoints: list[int] = field(default_factory=list)  # content-block positions carrying cache_control
    n_blocks: int = 0
    nothink: list[str] = field(default_factory=list)  # per-message hashes without thinking/reasoning blocks
    had_thinking: bool = False
    cache_key: str | None = None
    prefix_key: str | None = None  # hash of tools + system as sent
    inflight_same_prefix: bool = False  # another request with this prefix_key was in flight when this was sent
    shared_prefix_age: float | None = None  # seconds since another session finished a request with this prefix


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=False, default=str).encode()).hexdigest()[:16]


_THINKING = ("thinking", "redacted_thinking", "reasoning")


def body_facts(provider: str, body: dict) -> Facts:
    """The request-side facts (usage-side ones are filled in at the response)."""
    f = Facts(provider=provider)
    tools, system = body.get("tools") or [], body.get("system", body.get("instructions"))
    f.prefix_key = _h([tools, system]) if (tools or system) else None
    f.cache_key = body.get("prompt_cache_key")
    pos = 0

    def blocks_of(value):
        if isinstance(value, list):
            return value
        return [value] if value not in (None, "", []) else []

    for t in tools:
        if isinstance(t, dict) and "cache_control" in t:
            f.breakpoints.append(pos)
        pos += 1
    for b in blocks_of(system) if provider == "anthropic" else []:
        if isinstance(b, dict) and "cache_control" in b:
            f.breakpoints.append(pos)
        pos += 1
    items = body.get("messages")
    if items is None:
        items = body.get("input") if isinstance(body.get("input"), list) else []
    for m in items or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        kept = []
        if m.get("type") in _THINKING:  # Responses API reasoning item
            f.had_thinking = True
            continue
        for b in blocks_of(content):
            if isinstance(b, dict) and b.get("type") in _THINKING:
                f.had_thinking = True
            else:
                kept.append({k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b)
            if isinstance(b, dict) and "cache_control" in b:
                f.breakpoints.append(pos)
            pos += 1
        f.nothink.append(_h([m.get("role"), m.get("type"), kept]))
    f.n_blocks = pos
    if body.get("cache_control") and pos:  # automatic caching: the breakpoint is the last block
        f.breakpoints.append(pos - 1)
    f.has_breakpoint = bool(f.breakpoints)
    if provider == "anthropic" and '"ttl": "1h"' in json.dumps(body):
        f.ttl_requested = 3600.0
    elif provider == "anthropic":
        f.ttl_requested = 300.0
    return f


def usage_facts(f: Facts, usage: dict, cache_write: int) -> Facts:
    created = (usage or {}).get("cache_creation") or {}
    f.write_5m = int(created.get("ephemeral_5m_input_tokens") or 0)
    f.write_1h = int(created.get("ephemeral_1h_input_tokens") or 0)
    f.cache_write = int(cache_write or 0)
    return f


def detail_for(cause: str, prev: Facts | None, curr: Facts, *, prompt_tokens: int, cached_tokens: int,
               gap_seconds: float | None, min_prefix: int | None
               ) -> tuple[tuple[str, bool | None] | None, str | None, str]:
    """((new cause, its legitimacy) or None to keep the cause, detail or None, evidence to append)."""
    cold = prompt_tokens > 0 and cached_tokens < 0.1 * prompt_tokens

    # Asked for 1 hour, granted 5 minutes: after more than 5 minutes the entry is gone. The
    # classifier timed the gap against the hour that was asked for, so it saw no expiry.
    if prev is not None and cold and curr.ttl_requested == 3600.0 and prev.write_5m > 0 and prev.write_1h == 0 \
            and gap_seconds is not None and gap_seconds > 300 and cause in ("ttl_expired", "routing_miss", "unknown"):
        why = (f"the request asked for a 1-hour cache, but the previous write ({prev.write_5m} tokens) was cached "
               f"for 5 minutes, and {gap_seconds:.0f} s passed")
        return (None if cause == "ttl_expired" else ("ttl_expired", True)), "ttl_shortened", why

    if prev is not None and cold and curr.provider == "anthropic" and prev.breakpoints and curr.breakpoints \
            and cause in ("ttl_expired", "routing_miss", "unknown") and curr.n_blocks >= prev.n_blocks:
        written = max(prev.breakpoints)  # the furthest prefix the previous request wrote
        if all(b - written > LOOKBACK_CLAIM or b < written for b in curr.breakpoints):
            reach = min((b - written for b in curr.breakpoints if b >= written), default=None)
            return ("breakpoint_misplaced", False), "lookback_exceeded", (
                f"prefix unchanged but out of reach: the previous write ends at block {written} and the nearest "
                f"breakpoint now is {reach} blocks later; Anthropic looks back at most 20 positions per "
                "breakpoint. Add a breakpoint at the start of each turn")

    # Only on a real miss: Anthropic filters earlier turns' thinking blocks itself, so
    # dropping them does not always cost anything.
    if cause == "history_edit" and cold and prev is not None and prev.had_thinking and prev.nothink \
            and curr.nothink[:len(prev.nothink)] == prev.nothink:
        return None, "thinking_stripped", ("the only change to earlier messages is removed thinking/reasoning "
                                           "blocks (a client or gateway dropping them)")

    # Nothing written and nothing read despite markers: the root cause of every miss that
    # follows, so it outranks the timing and sharing explanations below.
    # The minimum length applies to the prefix up to a breakpoint, which this proxy cannot
    # count in tokens: claim only when a breakpoint covers (nearly) the whole prompt.
    if curr.provider == "anthropic" and curr.has_breakpoint and cached_tokens == 0 and curr.cache_write == 0 \
            and max(curr.breakpoints) >= curr.n_blocks - 2 \
            and prompt_tokens > 0 and (min_prefix is None or prompt_tokens >= min_prefix) \
            and cause not in ("none", "below_min_prefix"):
        return None, "markers_ignored", (
            "cache_control was sent and the prompt was long enough, but the provider wrote and read nothing: "
            "a gateway may have stripped the markers, or the endpoint does not cache")

    if cold and curr.inflight_same_prefix and cause in ("first_write", "fork_boundary", "routing_miss", "unknown"):
        return None, "sent_before_cache_written", (
            "another request with the same tools and system prompt was still in flight when this one was sent, "
            "so nothing was cached yet; stagger parallel requests, or warm the prefix first")

    if cause == "first_write" and cold and curr.shared_prefix_age is not None \
            and curr.shared_prefix_age <= SHARED_PREFIX_WINDOW:
        return None, "shared_prefix_not_reused", (
            f"another session finished a request with the same tools and system prompt {curr.shared_prefix_age:.0f} s "
            "earlier, yet nothing was read from the cache: check cache keys, TTLs and tool order")

    if cause in ("routing_miss", "unknown", "ttl_expired") and prev is not None \
            and prev.cache_key != curr.cache_key and (prev.cache_key or curr.cache_key):
        return None, "cache_key_changed", "prompt_cache_key differs from the previous turn's"

    return None, None, ""
