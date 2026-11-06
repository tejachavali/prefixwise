"""Miss-cause classifier (v2 taxonomy, 19 causes).

Exactly one cause per request: the first divergence in prefix order.
``UNKNOWN`` when evidence is absent, never a guess.

Legitimacy is a separate column from cause: a legitimate change still
costs tokens, but it is not waste. A policy can later mark a catalog
change avoidable (masking instead of removing) without renaming the cause.

Provider order (verified, not assumed):
- Anthropic: tools, system, messages.
- OpenAI: tools are injected before developer instructions, so
  tools, developer/system, messages. (OpenAI Prompt Caching 201.)

Rules run in v2 order; the first that fires wins, the rest are recorded
as secondary. Provider native reasons (Anthropic 6, OpenAI 9) map onto
this taxonomy in diagnostics.py; they are a partial agreement signal,
not ground truth: providers report the earliest divergence only, in a
coarse taxonomy, and neither names TTL expiry, routing, or eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .fingerprint import BlockFingerprint


class MissCause(str, Enum):
    FIRST_WRITE = "first_write"
    MODEL_SWITCH = "model_switch"
    VOLATILE_SYSTEM = "volatile_system"
    SYSTEM_EDIT = "system_edit"
    TOOL_RESERIALIZE = "tool_reserialize"
    TOOL_CATALOG_CHANGE = "tool_catalog_change"
    COMPACTION_REWRITE = "compaction_rewrite"
    HISTORY_EDIT = "history_edit"
    TOOL_RESULT_MUTATION = "tool_result_mutation"
    FORK_BOUNDARY = "fork_boundary"
    PARAM_RENDERED = "param_rendered"
    BELOW_MIN_PREFIX = "below_min_prefix"
    BREAKPOINT_MISPLACED = "breakpoint_misplaced"
    TTL_EXPIRED = "ttl_expired"
    EVICTED = "evicted"
    ROUTING_MISS = "routing_miss"
    RETRY_REPLAY = "retry_replay"
    NONE = "none"
    UNKNOWN = "unknown"


# Provider minimum cacheable prefix, tokens. Anthropic is model-dependent:
# 512-4,096 (512 for Claude Fable/Mythos/Opus/Sonnet/Haiku 5.x, 2,048 for
# Mythos Preview / Opus 4.7, 4,096 for Opus 4.6/4.5 and Haiku 4.5,
# 1,024 for Opus 4.8 / Sonnet 5 / Sonnet 4.6). Use the model-specific
# value when known; the dict below is the per-model table, not a range.
ANTHROPIC_MIN_PREFIX = {
    "512": 512, "2048": 2048, "4096": 4096, "1024": 1024,
}
OPENAI_MIN_PREFIX = 1024
# No provider keeps a cache entry for less than ~5 minutes (Anthropic default
# 5 min; OpenAI in-memory 5-10 min). A cold cache within this gap of the
# last turn is not explained by expiry.
MIN_PROVIDER_TTL_SECONDS = 300
GEMINI_MIN_PREFIX = 4096  # Gemini 3.x; 2,048 on 2.5. ai.google.dev, Nov 2026.


@dataclass
class RequestSnapshot:
    model: str
    provider: str = "unknown"
    blocks: list[BlockFingerprint] = field(default_factory=list)
    # Prompt-affecting params that change the rendered request without
    # changing any block: tool_choice, thinking/reasoning, output_format,
    # context_management, beta headers, temperature is NOT one of these.
    params: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    # Seconds since the previous turn in this session; None = unknown.
    gap_seconds: float | None = None
    ttl_seconds: float | None = None
    compaction_event: bool = False
    fork_parent_session_id: str | None = None
    session_id: str | None = None
    # vLLM KV events for this session between the two turns.
    evicted_block_hashes: list[str] = field(default_factory=list)
    # Anthropic cache_control present anywhere in the request.
    has_breakpoint: bool = False
    breakpoint_after_volatile: bool = False
    min_prefix_tokens: int | None = None


@dataclass
class Classification:
    cause: MissCause
    legit: bool | None  # True = legitimate, False = waste, None = n/a
    evidence: str
    secondary: list[MissCause] = field(default_factory=list)
    native_reason: str | None = None
    classifier_version: str = "2.0"


def _blocks_by_role(blocks: list[BlockFingerprint]) -> dict[str, list[BlockFingerprint]]:
    out: dict[str, list[BlockFingerprint]] = {}
    for b in blocks:
        out.setdefault(b.role, []).append(b)
    return out


def _tool_multiset(blocks: list[BlockFingerprint]) -> list[str]:
    return sorted(b.canonical for b in blocks if b.role == "tools")


def _same(a: BlockFingerprint, b: BlockFingerprint) -> bool:
    return a.raw == b.raw and a.canonical == b.canonical and a.masked == b.masked


# Roles that grow by appending turn over turn. For these, the previous
# turn's blocks must be an unchanged prefix of the current turn's; new
# blocks after that prefix are ordinary conversation growth, not a
# divergence. System and tool blocks must match exactly.
_APPEND_ROLES = ("messages", "tool_results", "user")


def _role_unchanged(role: str, p: list[BlockFingerprint], c: list[BlockFingerprint]) -> bool:
    if role in _APPEND_ROLES:
        return len(p) <= len(c) and all(_same(a, b) for a, b in zip(p, c))
    return len(p) == len(c) and all(_same(a, b) for a, b in zip(p, c))


def classify(prev: RequestSnapshot | None, curr: RequestSnapshot,
             *, native_reason: str | None = None) -> Classification:
    """Classify one turn against the previous turn in the same session."""
    if prev is None:
        if curr.fork_parent_session_id:
            return Classification(MissCause.FORK_BOUNDARY, True,
                                  f"first turn of a session forked from {curr.fork_parent_session_id}",
                                  native_reason=native_reason)
        return Classification(MissCause.FIRST_WRITE, True, "no prior request in session",
                              native_reason=native_reason)
    if prev.model != curr.model:
        return Classification(MissCause.MODEL_SWITCH, True,
                              f"model {prev.model} -> {curr.model}", native_reason=native_reason)

    prev_by_role = _blocks_by_role(prev.blocks)
    curr_by_role = _blocks_by_role(curr.blocks)
    secondary: list[MissCause] = []

    # First divergence in prefix order: system, tools, messages, tool_results, user.
    for role in ("system", "system_dynamic", "tools", "messages", "tool_results", "user"):
        p, c = prev_by_role.get(role, []), curr_by_role.get(role, [])
        if _role_unchanged(role, p, c):
            continue
        if role in ("system", "system_dynamic"):
            if [b.masked for b in p] == [b.masked for b in c]:
                return Classification(MissCause.VOLATILE_SYSTEM, False,
                                      "system raw hash changed, masked hash equal (dates/ids/numbers only)",
                                      secondary, native_reason)
            return Classification(MissCause.SYSTEM_EDIT, None,
                                  "system masked hash changed", secondary, native_reason)
        if role == "tools":
            if _tool_multiset(prev.blocks) == _tool_multiset(curr.blocks):
                return Classification(MissCause.TOOL_RESERIALIZE, False,
                                      "tool multiset of canonical hashes equal; sequence or raw bytes differ",
                                      secondary, native_reason)
            return Classification(MissCause.TOOL_CATALOG_CHANGE, True,
                                  "tool multiset differs", secondary, native_reason)
        if role == "messages":
            cause = MissCause.COMPACTION_REWRITE if curr.compaction_event else MissCause.HISTORY_EDIT
            legit = True if curr.compaction_event else False
            ev = "previous message blocks rewritten (not a prefix of this turn) with a compaction event" \
                if curr.compaction_event else \
                "previous message blocks rewritten (not a prefix of this turn) without a compaction event"
            return Classification(cause, legit, ev, secondary, native_reason)
        if role == "tool_results":
            if curr.compaction_event:
                # Context editing that clears old tool results is a compaction.
                return Classification(MissCause.COMPACTION_REWRITE, True,
                                      "earlier tool results cleared or rewritten with a compaction event",
                                      secondary, native_reason)
            return Classification(MissCause.TOOL_RESULT_MUTATION, False,
                                  "an earlier tool result changed or was removed without a compaction event",
                                  secondary, native_reason)
        # user role divergence with no other cause: fall through to param/TTL checks.
        secondary.append(MissCause.HISTORY_EDIT)

    # Blocks equal (or only the user turn changed): check params, prefix, TTL, eviction.
    if prev.params != curr.params:
        changed = sorted(set(prev.params) ^ set(curr.params) |
                         {k for k in prev.params.keys() & curr.params.keys() if prev.params[k] != curr.params[k]})
        return Classification(MissCause.PARAM_RENDERED, None,
                              f"blocks equal; prompt-affecting params changed: {changed}",
                              secondary, native_reason)

    min_prefix = curr.min_prefix_tokens
    if min_prefix is None:
        min_prefix = {"openai": OPENAI_MIN_PREFIX, "gemini": GEMINI_MIN_PREFIX}.get(curr.provider, None)
    # 0 prompt tokens means usage was absent, not a short prompt: fall
    # through to UNKNOWN rather than claim BELOW_MIN_PREFIX.
    if min_prefix is not None and 0 < curr.prompt_tokens < min_prefix:
        return Classification(MissCause.BELOW_MIN_PREFIX, None,
                              f"cacheable prefix {curr.prompt_tokens} below provider minimum {min_prefix}",
                              secondary, native_reason)

    if curr.provider == "anthropic" and not curr.has_breakpoint:
        return Classification(MissCause.BREAKPOINT_MISPLACED, False,
                              "Anthropic: no cache_control anywhere, or breakpoint after a volatile block",
                              secondary, native_reason)
    if curr.breakpoint_after_volatile:
        return Classification(MissCause.BREAKPOINT_MISPLACED, False,
                              "breakpoint sits after a volatile block", secondary, native_reason)

    # Every rule below reads usage. Usage absent (0 prompt tokens, seen live
    # on OpenAI 2026-11-06) is no evidence of a hit, a retry or a routing
    # miss: say UNKNOWN rather than guess from a zero.
    if curr.prompt_tokens <= 0:
        return Classification(MissCause.UNKNOWN, None,
                              "blocks unchanged but the provider returned no usage; cannot tell hit from miss",
                              secondary, native_reason)

    # Append-only turn: compare cache_read to the reusable prior prefix,
    # not to this turn's prompt, which includes the newly appended tail.
    # A real hit wins over every timing rule below: if the cache served
    # the prefix, it did not expire and was not evicted.
    reusable = min(prev.prompt_tokens, curr.prompt_tokens) if prev.prompt_tokens else curr.prompt_tokens
    if curr.prompt_tokens > 0 and curr.cached_tokens >= reusable * 0.5:
        return Classification(MissCause.NONE, None,
                              "append-only turn; cache_read at least half of the prior prefix",
                              secondary, native_reason)

    # From here the prefix was reusable but cache_read was low.
    # Engine eviction events are direct evidence; they outrank inference.
    if curr.evicted_block_hashes:
        return Classification(MissCause.EVICTED, None,
                              "BlockRemoved events for this session's blocks between the two turns",
                              secondary, native_reason)

    # Identical replay within a short window is a retry, not a cache miss.
    if prev.blocks and curr.blocks and \
       [b.raw for b in prev.blocks] == [b.raw for b in curr.blocks] and \
       curr.gap_seconds is not None and curr.gap_seconds < 60 and curr.cached_tokens == 0:
        return Classification(MissCause.RETRY_REPLAY, False,
                              "identical request within the retry window", secondary, native_reason)

    gap, ttl = curr.gap_seconds, curr.ttl_seconds
    if ttl is not None and gap is not None and gap > ttl:
        return Classification(MissCause.TTL_EXPIRED, True,
                              f"gap {gap:.0f}s since last turn exceeds the cache TTL {ttl:.0f}s",
                              secondary, native_reason)

    if curr.cached_tokens == 0 and curr.prompt_tokens > 0 and gap is not None and \
       (gap <= (ttl if ttl is not None else MIN_PROVIDER_TTL_SECONDS)):
        window = f"TTL {ttl:.0f}s" if ttl is not None else f"shortest provider TTL {MIN_PROVIDER_TTL_SECONDS}s"
        return Classification(MissCause.ROUTING_MISS, None,
                              f"blocks and params equal, gap {gap:.0f}s within {window}, "
                              "no eviction evidence, nothing read from cache",
                              secondary, native_reason)
    if curr.prompt_tokens > 0 and gap is not None and ttl is None:
        return Classification(MissCause.UNKNOWN, None,
                              f"prefix reusable but cache_read low after a {gap:.0f}s gap with unknown TTL:"
                              " expiry or routing, not distinguishable",
                              secondary, native_reason)
    return Classification(MissCause.UNKNOWN, None,
                          "prefix reusable but cache_read low; no timing or usage evidence to name a cause",
                          secondary, native_reason)


def classify_first_divergence(changed_role: str | None, *, model_changed: bool = False,
                              compaction_event: bool = False) -> MissCause:
    """Backwards-compatible helper from v1 (13-cause API)."""
    if model_changed:
        return MissCause.MODEL_SWITCH
    if changed_role is None:
        return MissCause.FIRST_WRITE
    if changed_role in ("system", "system_dynamic"):
        return MissCause.VOLATILE_SYSTEM
    if changed_role == "tools":
        return MissCause.TOOL_RESERIALIZE
    if changed_role == "messages":
        return MissCause.COMPACTION_REWRITE if compaction_event else MissCause.HISTORY_EDIT
    if changed_role == "tool_results":
        return MissCause.TOOL_RESULT_MUTATION
    return MissCause.UNKNOWN
