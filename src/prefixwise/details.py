"""Sub-reasons: the why behind a cause, for the ways agents break the cache in practice.

The cause says what broke reuse first; the detail says why, where the evidence allows.
A detail is claimed only when every precondition below holds; "no detail" is always
an acceptable answer, a wrong one is not. Sources: Anthropic's prompt-caching and
extended-thinking docs (Nov 2026), Claude Code / Codex / opencode / LiteLLM issues.

  ttl_shortened      The previous request asked for a 1-hour cache on every breakpoint
                     (and used no server tools, which add a 5-minute breakpoint), yet
                     Anthropic reports its write in the 5-minute bucket only, and more
                     than 5 minutes passed before this request was sent. Promotes a
                     routing_miss/unknown to ttl_expired.
  lookback_exceeded  Anthropic checks at most 20 positions back from each breakpoint
                     (a run of tool_use blocks, or of tool_result blocks, is one
                     position). Claimed when tools, system and earlier messages are
                     provably unchanged, the previous request wrote at its breakpoint,
                     the gap is within 5 minutes, and every breakpoint now is at least
                     22 positions past that write (20 per the docs, plus a margin for
                     how blocks are counted). Promotes routing_miss/unknown to
                     breakpoint_misplaced: an earlier breakpoint fixes it.
  thinking_stripped  history_edit on a model that keeps earlier turns' thinking (Opus
                     4.5+, Sonnet 4.6+, Haiku 5.5, Fable, Mythos), where the only
                     change to earlier messages is removed thinking blocks. On other
                     models the API strips them itself, so it is no cause.
  markers_ignored    cache_control on the last block, a prompt of at least 8,192 tokens
                     (twice the largest minimum of any model), and nothing written or
                     read: the markers did not reach a caching endpoint.
  sent_before_cache_written
                     The first request of a session, sent within 1 second of another
                     request with the same model, credentials, tools and system prompt
                     that was still waiting for its response (a fan-out): the entry did
                     not exist yet.
  shared_prefix_not_reused
                     Anthropic: the first request of a session, within 5 minutes of
                     another session's request (same model, credentials, tools and
                     system prompt) that wrote an entry ending inside tools/system,
                     with a breakpoint here that could reach it, yet it read little.
  cache_key_changed  OpenAI prompt_cache_key differs from the previous turn's.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

LOOKBACK_POSITIONS = 20  # Anthropic: positions checked per breakpoint, the breakpoint itself first
LOOKBACK_CLAIM = LOOKBACK_POSITIONS + 2  # claim only this far out: block counting may differ slightly
MARKERS_MIN_PROMPT = 8192  # twice the largest minimum cacheable prefix of any Claude model
FANOUT_SECONDS = 1.0  # a request sent this soon after a same-prefix one: its prefill cannot be done
SHARED_WINDOW = 300.0  # seconds: the default Anthropic cache lifetime
_THINKING = ("thinking", "redacted_thinking")
_COLLAPSE = ("tool_use", "tool_result")  # a run of these is one lookback position
# Anthropic models that keep earlier turns' thinking in context (thinking docs, "preservation by model").
_KEEPS_THINKING = re.compile(r"opus-4-[5-9]|opus-[5-9]|sonnet-4-[6-9]|sonnet-[5-9]|haiku-5-5|haiku-[6-9]|fable|mythos")


def keeps_thinking(model: str) -> bool:
    return bool(_KEEPS_THINKING.search((model or "").lower()))


@dataclass
class Facts:
    """What the proxy knows about one request beyond its block fingerprints."""
    provider: str = ""
    model: str = ""
    prefix_key: str | None = None  # model, credentials, tools and system prompt as sent
    sys_end: int = 0  # lookback positions taken by tools and system
    breakpoints: list[int] = field(default_factory=list)  # lookback positions carrying cache_control
    n_pos: int = 0
    msgs: list[str] = field(default_factory=list)  # per-message hashes (cache_control removed)
    nothink: list[str] = field(default_factory=list)  # the same, thinking blocks removed
    had_thinking: bool = False
    all_1h: bool = False  # every cache_control asks for ttl 1h
    server_tools: bool = False  # server tools add an automatic 5-minute breakpoint
    cache_key: str | None = None
    sent_at: float | None = None
    done_at: float | None = None
    write_5m: int = 0
    write_1h: int = 0
    cache_write: int = 0
    fanout_age: float | None = None  # seconds since a same-prefix request still in flight was sent
    shared_age: float | None = None  # seconds since another session wrote an entry inside tools/system
    shared_write_pos: int | None = None  # where that entry ends


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=False, default=str).encode()).hexdigest()[:16]


def _strip_markers(value):
    if isinstance(value, dict):
        return {k: _strip_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_markers(v) for v in value]
    return value


def _cache_controls(value, out: list) -> None:
    if isinstance(value, dict):
        cc = value.get("cache_control")
        if isinstance(cc, dict):
            out.append(cc)
        for k, v in value.items():
            if k != "cache_control":
                _cache_controls(v, out)
    elif isinstance(value, list):
        for v in value:
            _cache_controls(v, out)


def body_facts(provider: str, body: dict, model: str = "", credential: str = "") -> Facts:
    """The request-side facts (the response fills in the rest)."""
    f = Facts(provider=provider, model=model or str(body.get("model") or ""))
    tools = body.get("tools") or []
    f.server_tools = any(isinstance(t, dict) and t.get("type") and t.get("type") != "custom"
                         and "input_schema" not in t for t in tools)
    items = body.get("messages")
    if items is None:
        items = body.get("input") if isinstance(body.get("input"), list) else []
    system = body.get("system", body.get("instructions"))
    if system is None:  # OpenAI chat: the system prompt is the first system/developer message
        system = next((m.get("content") for m in items or []
                       if isinstance(m, dict) and m.get("role") in ("system", "developer")), None)
    if tools or system:
        f.prefix_key = _h([provider, f.model, credential, _strip_markers(tools), _strip_markers(system)])
    f.cache_key = body.get("prompt_cache_key")
    controls: list = []
    _cache_controls(body, controls)
    f.all_1h = bool(controls) and all(c.get("ttl") == "1h" for c in controls)

    pos = 0
    for t in tools:
        if isinstance(t, dict) and "cache_control" in t:
            f.breakpoints.append(pos)
        pos += 1
    if provider == "anthropic":
        for b in system if isinstance(system, list) else ([system] if system else []):
            if isinstance(b, dict) and "cache_control" in b:
                f.breakpoints.append(pos)
            pos += 1
    f.sys_end = pos
    last_type = None
    for m in items or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        blocks = content if isinstance(content, list) else ([content] if content not in (None, "") else [])
        kept = []
        for b in blocks:
            btype = b.get("type") if isinstance(b, dict) else "text"
            if btype in _THINKING:
                f.had_thinking = True
            else:
                kept.append(_strip_markers(b))
            if not (btype in _COLLAPSE and btype == last_type):
                pos += 1
            last_type = btype
            if isinstance(b, dict) and "cache_control" in b:
                f.breakpoints.append(pos - 1)
        f.msgs.append(_h(_strip_markers(m)))
        f.nothink.append(_h([m.get("role"), m.get("type"), kept]))
    f.n_pos = pos
    if body.get("cache_control") and pos:  # automatic caching: the breakpoint is the last block
        f.breakpoints.append(pos - 1)
    return f


def usage_facts(f: Facts, usage: dict, cache_write: int) -> Facts:
    created = (usage or {}).get("cache_creation") or {}
    f.write_5m = int(created.get("ephemeral_5m_input_tokens") or 0)
    f.write_1h = int(created.get("ephemeral_1h_input_tokens") or 0)
    f.cache_write = int(cache_write or 0)
    return f


def detail_for(cause: str, prev: Facts | None, curr: Facts, *, prompt_tokens: int, cached_tokens: int
               ) -> tuple[tuple[str, bool | None] | None, str | None, str]:
    """((new cause, its legitimacy) or None to keep the cause, detail or None, evidence to append)."""
    if prompt_tokens <= 0:
        return None, None, ""
    cold = cached_tokens < 0.1 * prompt_tokens
    read = f"read {cached_tokens:,} of {prompt_tokens:,} prompt tokens from the cache"
    gap = (curr.sent_at - prev.done_at) if prev is not None and curr.sent_at and prev.done_at else None
    anthropic = curr.provider == "anthropic"

    if anthropic and prev is not None and cold and gap is not None and gap > 300 \
            and cause in ("ttl_expired", "routing_miss", "unknown") \
            and prev.all_1h and not prev.server_tools and prev.write_5m > 0 and prev.write_1h == 0:
        why = (f"the previous request asked for a 1-hour cache on every breakpoint, but its write "
               f"({prev.write_5m:,} tokens) was kept for 5 minutes, and {gap:.0f} s passed; this request {read}")
        return (None if cause == "ttl_expired" else ("ttl_expired", True)), "ttl_shortened", why

    if anthropic and prev is not None and cold and cause in ("routing_miss", "unknown") \
            and gap is not None and gap <= 300 and prev.cache_write > 0 and prev.breakpoints and curr.breakpoints \
            and prev.prefix_key == curr.prefix_key and prev.model == curr.model \
            and curr.msgs[:len(prev.msgs)] == prev.msgs:
        written = max(prev.breakpoints)
        ahead = [b - written for b in curr.breakpoints if b >= written]
        if ahead and min(ahead) >= LOOKBACK_CLAIM:
            return ("breakpoint_misplaced", False), "lookback_exceeded", (
                f"prompt unchanged but out of reach: the previous write ends at position {written} and the "
                f"nearest breakpoint now is {min(ahead)} positions later; Anthropic looks back at most "
                f"{LOOKBACK_POSITIONS}. Add a breakpoint at the start of each turn")

    if anthropic and prev is not None and cold and cause == "history_edit" and keeps_thinking(curr.model) \
            and prev.had_thinking and prev.nothink and curr.nothink[:len(prev.nothink)] == prev.nothink:
        return None, "thinking_stripped", (
            "the only change to earlier messages is removed thinking blocks; this model keeps them in context, "
            "so removing them changes the prompt")

    if anthropic and curr.breakpoints and max(curr.breakpoints) == curr.n_pos - 1 and cached_tokens == 0 \
            and curr.cache_write == 0 and prompt_tokens >= MARKERS_MIN_PROMPT \
            and cause not in ("none", "below_min_prefix"):
        return None, "markers_ignored", (
            "cache_control was on the last block and the prompt far above any minimum, yet nothing was written "
            "or read: the markers did not reach a caching endpoint (a gateway stripping them?)")

    if cold and cause in ("first_write", "fork_boundary") and curr.fanout_age is not None \
            and curr.fanout_age <= FANOUT_SECONDS:
        return None, "sent_before_cache_written", (
            f"sent {curr.fanout_age:.2f} s after another request with the same model, tools and system prompt "
            f"that was still waiting for its response, so its cache entry did not exist yet; this request {read}. "
            "Stagger parallel requests, or warm the prefix first")

    if anthropic and cold and cause == "first_write" and curr.shared_age is not None \
            and curr.shared_age <= SHARED_WINDOW and curr.shared_write_pos is not None \
            and any(0 <= b - curr.shared_write_pos < LOOKBACK_POSITIONS for b in curr.breakpoints):
        return None, "shared_prefix_not_reused", (
            f"another session wrote a cache entry for the same model, tools and system prompt "
            f"{curr.shared_age:.0f} s earlier, within reach of a breakpoint here, yet this request {read}")

    if cause in ("routing_miss", "unknown", "ttl_expired") and prev is not None \
            and prev.cache_key != curr.cache_key and (prev.cache_key or curr.cache_key):
        return None, "cache_key_changed", "prompt_cache_key differs from the previous turn's"

    return None, None, ""
