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

import re
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


# Provider minimum cacheable prefix, tokens. Anthropic's is per model (prompt-caching page, read
# 2026-11-26): 512 for the 5.x models except Sonnet 5; 1,024 for Opus 4.8, Sonnet 5, Sonnet 4.6,
# Sonnet 4.5, Opus 4.1, Opus 4 and Sonnet 4; 2,048 for Mythos Preview and Opus 4.7; 4,096 for
# Opus 4.6, Opus 4.5 and Haiku 4.5. The Claude 3 models are from Anthropic's earlier table
# (2,048 for the Haiku models, 1,024 for the others). First match wins.
_MINOR = r"(?!\d)"  # the minor is this digit alone (a date may follow)
_NO_MINOR = r"(?![-._]?\d{1,2}(?!\d))"  # the major alone: no minor after it (a date, @date or [1m] may follow)
_ANTHROPIC_MIN_PREFIX = [(re.compile(pattern), tokens) for pattern, tokens in (
    (rf"opus-4[-.][56]{_MINOR}|haiku-4[-.]5{_MINOR}", 4096),
    (rf"opus-4[-.]7{_MINOR}|mythos-preview", 2048),
    (rf"opus-4[-.][018]{_MINOR}|sonnet-4[-.][056]{_MINOR}", 1024),   # 4-0 is how some ids spell Opus 4 / Sonnet 4
    (rf"(opus|sonnet)-4{_NO_MINOR}", 1024),                           # Opus 4 and Sonnet 4
    (rf"sonnet-5{_NO_MINOR}", 1024),                                  # Sonnet 5 itself; Sonnet 5.5 is 512
    (r"haiku-3|3-5-haiku|3-haiku", 2048),
    (r"claude-3-|sonnet-3|opus-3", 1024),
)]
OPENAI_MIN_PREFIX = 1024
MAX_ANTHROPIC_TTL_SECONDS = 3600  # the longest lifetime a cache_control can ask for
# Twice the largest minimum cacheable prefix of any Claude model: a prompt this long whose last
# block carries cache_control, yet wrote and read nothing, had its markers ignored (the
# `markers_ignored` detail in prefixwise.details).
MARKERS_MIN_PROMPT = 8192
# No provider keeps a cache entry for less than ~5 minutes (Anthropic default
# 5 min; OpenAI in-memory 5-10 min). A cold cache within this gap of the
# last turn is not explained by expiry.
MIN_PROVIDER_TTL_SECONDS = 300
GEMINI_MIN_PREFIX = 4096  # Gemini 3.x; 2,048 on 2.5. ai.google.dev, Nov 2026.


# Optional route prefixes (openai/, openrouter/openai/) and a fine-tune prefix (ft:); gpt-oss is an
# open-weight model other hosts serve under their own caching rules.
_OPENAI_MODEL = re.compile(r"^(?:[\w.-]+/)*(?:ft:)?(?:gpt-(?!oss)|chatgpt-|o\d|codex-|computer-use-)")


def openai_model(model: str) -> bool:
    """An OpenAI model id (gpt-*, o1/o3/o4*, chatgpt-*, codex-*): OpenAI's own caching rules apply.
    Anything else sent in OpenAI's request shape is another provider's model."""
    return bool(_OPENAI_MODEL.match((model or "").lower()))


def anthropic_min_prefix(model: str) -> int:
    """The model's minimum cacheable prefix (table above). A model not in it gets
    the lowest known minimum, so "below the minimum" is claimed only when no
    Anthropic model could have cached the prompt."""
    m = (model or "").lower()
    for pattern, tokens in _ANTHROPIC_MIN_PREFIX:
        if pattern.search(m):
            return tokens
    return 512  # Claude 5.x (Fable, Mythos, Opus, Sonnet, Haiku), and the floor for unknown models


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
    # Tokens this request wrote to the cache (0 when the provider does not report writes).
    # What the cache holds after a turn is what it read plus what it wrote; the next turn can
    # read no more than that, however long the uncached tail after the last breakpoint was.
    cache_written: int = 0
    # Seconds since the previous turn in this session ended (idle time); None = unknown.
    gap_seconds: float | None = None
    # Seconds since the previous turn was SENT: the age of what it left in the cache. A provider
    # counts a cache entry's lifetime from the request that wrote or read it, so the previous turn's
    # generation time is part of the age (measured on Anthropic: two 5-minute entries read nothing
    # 318 s after the request that wrote them, 153 s and 292 s after its response ended). None =
    # unknown: the rules then fall back to gap_seconds.
    cache_age_seconds: float | None = None
    sent_at: float | None = None  # unix time this request was sent (the next turn's age counts from it)
    ttl_seconds: float | None = None
    compaction_event: bool = False
    fork_parent_session_id: str | None = None
    session_id: str | None = None
    # Self-hosted engine (vLLM, SGLang) behind the proxy, and how many KV caches it has
    # (data-parallel ranks, or replicas behind a router), as far as Prefixwise knows.
    engine: bool = False
    engine_caches: int = 1
    block_size: int | None = None  # tokens per KV block, when the engine's events have shown it
    # From the engine's KV events (prefixwise.engines; None = nothing to judge):
    # evicted_blocks > 0 when the block where the cache read stopped was removed (on a hit: blocks
    # the read covered that had been removed, so they came back from a host tier: host_restored);
    # evicted_blocks == 0 and resident_blocks > 0 when that block is still on the GPU.
    evicted_blocks: int | None = None
    resident_blocks: int | None = None
    # Several caches: (cache that served the previous request, cache that served this one,
    # blocks of the previous request still resident on the first), when the first still held
    # the block where this request's cache read stopped.
    routed: tuple | None = None
    # The proxy found no earlier turn of this session (on this model) whose conversation this
    # one continues: its first message matches none of them. A new thread, e.g. a subagent.
    new_thread: bool = False
    # The session names no single conversation (no conversation id came with the request): it
    # holds every conversation that named none, so a new opening message is a new conversation.
    shared_session: bool = False
    # The previous turn marked a block inside (or at the end of) tools+system, and this request
    # marks the same block: what that turn cached up to there, this request could read.
    shared_breakpoint: bool = False
    # Anthropic: the usage carried a cache-write count (cache_creation_input_tokens), so a 0 is a
    # count, not a missing field.
    write_reported: bool = False
    # cache_control on the request's last block: the marked prefix is the whole prompt.
    last_block_marked: bool = False
    # (at least, at most): an estimate of the tokens this request's cache_control covered, that is
    # everything up to and including the block with its last marker, by that part's share of the
    # request's bytes times its prompt tokens (prefixwise.details.marked_tokens). None: no estimate
    # (no marker, or bytes that are not text's).
    marked_tokens: tuple[int, int] | None = None
    # False when the usage is a placeholder (offline `explain` without usage): rules that read
    # the cache's response must not fire.
    usage_known: bool = True
    cache_reported: bool = True  # False: the server's usage has no cached-token count at all
    # The compared turn is not the session's latest on this model: another conversation came
    # between (interleaved subagents), so this request was matched to its own conversation.
    interleaved: bool = False
    # Anthropic cache_control present anywhere in the request.
    has_breakpoint: bool = False
    # What this ledger has seen of the cache's lifetime on this provider and model
    # (prefixwise.lifetime.Observed), used only when no TTL is known for the request.
    observed_lifetime: object | None = None
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
    # False when the verdict overrides the pair the provider compared (an expired entry, a change
    # that cost nothing): its native reason then describes another question, not a disagreement.
    score: bool = True
    # True when the prefix was unchanged and less than half of it was read, whatever cause that was
    # then given: the evidence prefixwise.lifetime learns cache lifetimes from.
    cold_unchanged: bool = False
    # True when no comparison was made (the turn at hand is not this request's predecessor) and the
    # verdict rests on this request's own read: no detail, no lifetime evidence, no "first changed"
    # block belongs on such a row.
    own_read: bool = False


def _blocks_by_role(blocks: list[BlockFingerprint]) -> dict[str, list[BlockFingerprint]]:
    out: dict[str, list[BlockFingerprint]] = {}
    for b in blocks:
        out.setdefault(b.role, []).append(b)
    return out


def _tool_multiset(blocks: list[BlockFingerprint]) -> list[str]:
    return sorted(b.canonical for b in blocks if b.role == "tools")


_WALKED = ("messages", "tool_results")  # the conversation, walked in the order it was sent


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


CACHE_NOT_REPORTED = ("the prompt's reusable part did not change, but the server reports no cached-token count "
                      "(no prompt_tokens_details in its usage), so a hit cannot be told from a miss")
# A Responses turn continuing a response this proxy never saw: nothing to compare, hit or miss.
UNSEEN_PARENT = ("previous_response_id points to a response this proxy never saw;"
                 " the cached prefix cannot be reconstructed")


def _min_prefix(curr: RequestSnapshot) -> int | None:
    """The provider's minimum cacheable prefix for this request; None when there is
    none to apply (self-hosted engines)."""
    if curr.min_prefix_tokens is not None:
        return curr.min_prefix_tokens
    if not curr.cache_reported or curr.engine:  # Ollama, llama.cpp; vLLM and SGLang cache from one block
        return None
    if curr.provider == "anthropic":
        return anthropic_min_prefix(curr.model)
    if curr.provider == "openai":
        # OpenAI's own minimum: another API behind the same request shape (DeepSeek, Tencent, OpenRouter,
        # Together) caches by its own rules, so no minimum is claimed there.
        return OPENAI_MIN_PREFIX if openai_model(curr.model) else None
    return {"gemini": GEMINI_MIN_PREFIX}.get(curr.provider)


def _age(curr: RequestSnapshot) -> float | None:
    """Seconds the cache entry had to survive: since the previous turn was sent, or, when that is
    not known, since it ended."""
    return curr.cache_age_seconds if curr.cache_age_seconds is not None else curr.gap_seconds


def _since(curr: RequestSnapshot, short: bool = False) -> str:
    """The age in words: 'gap 340s since last turn' ('gap 340s' when `short`), or, when the previous
    turn took long to answer, the age with the idle time beside it so the two numbers do not read as
    a contradiction. An age is never called a gap on a row whose idle gap is another number."""
    age, gap = _age(curr), curr.gap_seconds
    if curr.cache_age_seconds is None or gap is None or abs(age - gap) < 1:
        return f"gap {age:.0f}s" if short else f"gap {age:.0f}s since last turn"
    return f"{age:.0f}s since the previous turn was sent (its response ended {gap:.0f}s ago)"


def _markers_ignored(snap: RequestSnapshot) -> bool:
    """The conditions of the `markers_ignored` detail (prefixwise.details), on a snapshot."""
    return (snap.provider == "anthropic" and snap.last_block_marked and snap.usage_known
            and snap.cached_tokens == 0 and snap.cache_written == 0 and snap.prompt_tokens >= MARKERS_MIN_PROMPT)


def _wrote_nothing(prev: RequestSnapshot) -> bool:
    """The previous turn read 0 and wrote 0 on Anthropic: it added nothing to the cache."""
    return (prev.provider == "anthropic" and prev.usage_known and prev.prompt_tokens > 0
            and prev.cached_tokens + prev.cache_written == 0)


def _left_nothing(prev: RequestSnapshot, curr: RequestSnapshot) -> bool:
    """The previous turn carried cache_control and is known to have left nothing in the cache: what
    its markers covered was below the model's minimum cacheable size, or they were not honoured
    (`_after_nothing_left` says which, as far as that is known). Claimed only where all of these
    hold:

    - Anthropic's own API, not a self-hosted engine, and the turn's usage carried a cache-write
      count (a server that reports no writes says nothing by its 0);
    - it read 0 and wrote 0, with a prompt;
    - its last block was not marked: a marker there covers the whole prompt, and a whole prompt
      below the minimum is the rule before any comparison (2.1), while one at or above it that
      wrote nothing had its markers ignored;
    - this request does not meet the `markers_ignored` detail's conditions, which then stands."""
    return (_wrote_nothing(prev) and prev.write_reported and prev.has_breakpoint
            and not prev.engine and not curr.engine
            and not prev.last_block_marked and not _markers_ignored(curr))


def _after_nothing_left(prev: RequestSnapshot, curr: RequestSnapshot, secondary: list, native_reason: str | None,
                        changed: MissCause | None = None) -> Classification:
    """The verdict for a request that read nothing after a turn that left nothing (`_left_nothing`):
    never waste, since that turn left nothing in the cache to lose, to expire or to be missed. Why
    it left nothing is named only as far as it is known, from the estimate of what its markers
    covered (`prev.marked_tokens`, at least and at most):

    - at least twice the model's minimum cacheable size: the markers were not honoured, `unknown`;
    - at most under twice the minimum: too small to cache, or too close to the minimum for an
      estimate to say otherwise, `below_min_prefix`;
    - no estimate, or bounds on either side of that line: `below_min_prefix` too, in words that
      claim neither.

    `changed`: a change this request carries, which cost nothing and is kept as a secondary cause."""
    floor = _min_prefix(curr) or anthropic_min_prefix(curr.model)
    low, high = prev.marked_tokens if prev.marked_tokens is not None else (None, None)
    cost = f"; the {changed.value} after it cost nothing" if changed is not None else ""
    also = secondary + ([changed] if changed is not None else [])
    about = lambda n, bound: f"about {n:,}" if low == high else f"{bound} about {n:,}"  # noqa: E731
    if low is not None and low >= 2 * floor:
        return Classification(MissCause.UNKNOWN, None,
                              f"the previous turn carried cache_control on {about(low, 'at least')} tokens "
                              f"(estimated from the size of its blocks), above the model's minimum cacheable size "
                              f"({floor} tokens), yet the provider reported nothing read and nothing written, so "
                              "that turn left nothing in the cache for this request to read. Its markers were not "
                              f"honoured (a gateway that strips them, or a server that fills the write count with 0)"
                              f"{cost}", also, native_reason, score=False)
    if high is not None and high < 2 * floor:
        size = (f"{about(high, 'at most')} tokens, estimated from the size of its blocks, was below the model's "
                f"minimum cacheable size ({floor} tokens), or not clearly above it")
    else:
        size = f"was below the model's minimum cacheable size ({floor} tokens), or was not cached"
        size = "whose size is not known, " + size
    return Classification(MissCause.BELOW_MIN_PREFIX, None,
                          "the previous turn left nothing in the cache (it carried cache_control, yet read and wrote "
                          f"0 tokens): the part its markers covered, {size}, so there was nothing to read{cost}",
                          also, native_reason, score=changed is None)


def own_read(curr: RequestSnapshot, why: str, native_reason: str | None = None,
             served_all_left: bool = False) -> Classification:
    """The verdict for a request that has no turn of its own to be compared with: no change is
    claimed. `none` when at least half of its own prompt came from the cache, else `unknown` (a
    server that reports no cached-token count reads as 0 here, so it is never a hit). Never scored
    against the provider's reason. `served_all_left`: the compared turn is a later turn of its own
    conversation and the cache served all that turn left of what this request repeats; that is a
    hit too, whatever share of the prompt it is."""
    half = curr.usage_known and curr.prompt_tokens > 0 and curr.cached_tokens >= 0.5 * curr.prompt_tokens
    also = ("" if half or not served_all_left else
            ": under half of its prompt, but all that the compared turn left in the cache of what it repeats")
    return Classification(MissCause.NONE if half or served_all_left else MissCause.UNKNOWN, None,
                          f"{why}; no change is claimed against a turn that is not its previous turn, and it is "
                          f"judged by its own read: {curr.cached_tokens:,} of {curr.prompt_tokens:,} prompt tokens "
                          f"came from the cache{also}", native_reason=native_reason, score=False, own_read=True)


def _reusable(prev: RequestSnapshot, curr: RequestSnapshot) -> int:
    """What this turn could have read: the previous turn's prompt (or this one's, if shorter);
    on Anthropic, no more than what that turn left in the cache (what it read plus what it
    wrote, which ends at its last breakpoint), so the uncached tail after the breakpoint is
    never expected back. Other providers report no writes (or round the reads), so there the
    whole prompt stands."""
    reusable = min(prev.prompt_tokens, curr.prompt_tokens) if prev.prompt_tokens else curr.prompt_tokens
    held = prev.cached_tokens + prev.cache_written
    if held > 0 and (prev.provider == "anthropic" or prev.cache_written > 0):
        return min(reusable, held)
    return reusable  # nothing known to be held (or no write counts): the whole prompt stands


# Cache reads are reported in whole blocks on every provider but Anthropic: OpenAI in increments
# of 128 tokens, vLLM and SGLang in their block (page) size, 1 to 128. A full read of a prefix is
# reported rounded down, so a count less than one block short of the prefix is a full read; one
# that is a whole block or more short is not (a 2,048-token prefix read in full says 2,048, not
# 1,920). Behind an engine the previous prompt also ends in its generation prompt, and the part of
# it the rendered history never repeats (Qwen3's 4-token `<think>\n\n</think>\n\n` stub when
# thinking is off) puts the last matching block a few tokens earlier still. When the engine's
# events have shown its block size, the slack is that block plus GENERATION_TAIL; otherwise it is
# the largest block any provider uses. Either way it is capped at an eighth of the prefix, so a
# short prefix read only in part is not rounded up to a hit. Live (experiments/e_template, vLLM,
# 16-token blocks): reads 3 to 25 tokens short of the previous prompt on turns that were hits, 25
# being one block plus 9 tokens; a divergence further back than block + tail is caught.
READ_GRANULARITY = 128
GENERATION_TAIL = 16


def _read_slack(curr: RequestSnapshot, target: int) -> int:
    if curr.provider == "anthropic":  # exact counts
        return 0
    slack = curr.block_size + GENERATION_TAIL if curr.block_size else READ_GRANULARITY
    return min(slack, target // 8)


def _served_all(prev: RequestSnapshot, curr: RequestSnapshot) -> bool:
    """The cache served everything the previous turn left in it that this turn could reuse, to
    within the provider's reporting granularity (`_read_slack`)."""
    if not curr.usage_known or curr.prompt_tokens <= 0:
        return False
    target = _reusable(prev, curr)
    if target <= 0 or curr.cached_tokens <= 0:
        return False
    slack = _read_slack(curr, target)
    return curr.cached_tokens >= target or curr.cached_tokens + slack > target


def served_all(prev: RequestSnapshot, curr: RequestSnapshot) -> bool:
    """Public form of `_served_all`, for the details that need the same hit check."""
    return _served_all(prev, curr)


def _served(prev: RequestSnapshot, curr: RequestSnapshot) -> str:
    """'N of M tokens', with the rounding named when the count is short of the prefix."""
    target = _reusable(prev, curr)
    note = "" if curr.cached_tokens >= target else "; reads are reported in whole cache blocks"
    return f"{curr.cached_tokens} of {target} tokens{note}"


def _stable_uncached_tail(prev: RequestSnapshot, curr: RequestSnapshot, secondary: list) -> tuple[int, int, int]:
    """Anthropic, the previous turn's blocks a prefix of this one's: (held, shared, tail). `held` is
    what the previous turn left in the cache (its reads plus its write, which ends at its last
    breakpoint), `shared` the previous prompt, all of which this turn repeats, and `tail` the part
    of it after the breakpoint: stable, yet sent uncached on every turn. The tail counts only when
    the cache served all it held (so nothing else explains the miss) and the tail is at least the
    model's minimum cacheable size: a noise floor, since a breakpoint placed after the tail would
    cache it whatever its size. A breakpoint that moves forward each turn leaves no tail. Not
    applied when a message block was replaced (a one-shot side call's, or a rewritten last turn:
    HISTORY_EDIT in `secondary`) or the prompt shrank, since then the previous prompt is not all
    repeated. (0, 0, 0) when the rule does not apply."""
    if (prev.provider != "anthropic" or not curr.usage_known or not curr.has_breakpoint
            or MissCause.HISTORY_EDIT in secondary or curr.prompt_tokens < prev.prompt_tokens):
        return 0, 0, 0
    held = prev.cached_tokens + prev.cache_written
    if held <= 0 or prev.prompt_tokens <= 0 or curr.cached_tokens < held:
        return 0, 0, 0
    shared = prev.prompt_tokens
    tail = shared - held
    if tail < max(_min_prefix(curr) or 0, 1):
        return 0, 0, 0
    return held, shared, tail


_VOLATILE_NAMES = {"date": "a date or timestamp", "time": "a time", "id": "an id"}


def _volatile_kinds(prev_blocks: list, curr_blocks: list) -> str:
    """What the changed system blocks carry (dates, times, ids), from the fingerprints' kinds: the
    candidates for what changed, since the values themselves are never kept. Numbers are left out
    (nearly every prompt has one) and a time is implied by a date. A path is named apart: it is not
    masked, so it did not change here, but it blocks reuse across sessions and machines. '' on older
    records or malformed kinds."""
    kinds: list[str] = []
    path = False
    for b in list(prev_blocks) + list(curr_blocks):
        found = getattr(b, "volatile", None)
        if not isinstance(found, (list, tuple)):
            continue
        for k in found:
            if k == "path":
                path = True
            elif isinstance(k, str) and k in _VOLATILE_NAMES and k not in kinds:
                kinds.append(k)
    if "date" in kinds and "time" in kinds:
        kinds.remove("time")
    if not kinds:
        return ""
    names = [_VOLATILE_NAMES[k] for k in kinds]
    joined = names[0] if len(names) == 1 else " and ".join(names)
    note = (f" (the system prompt carries {joined}: move it to the end of the prompt, or drop it"
            + ("; it also carries a file path, which other sessions and machines cannot share" if path else "") + ")")
    return note


def _first_change(prev: RequestSnapshot, curr: RequestSnapshot, secondary: list,
                  native_reason: str | None, notes: dict | None = None,
                  conversation: bool = True) -> Classification | None:
    """The cause named by the first part of the prompt that changed, or None if none did
    (a changed user turn only adds to `secondary`). notes["new_conversation"] is set when the
    opening message differs because this is another conversation, not an edit of this one. That
    is decided only after tools and system compared equal: those are the lane's, and a change in
    them stands against any of its turns.
    `conversation=False` stops after the tools and system (an earlier point of the compared turn's
    conversation changed nothing in the conversation)."""
    prev_by_role = _blocks_by_role(prev.blocks)
    curr_by_role = _blocks_by_role(curr.blocks)
    # First divergence in prefix order. Anthropic and OpenAI put tools before the system
    # prompt (module docstring); a self-hosted engine's chat template decides, so there the
    # system prompt, which templates render first, is checked first.
    order = ("system", "system_dynamic", "tools") if curr.engine else ("tools", "system", "system_dynamic")
    for role in order:
        p, c = prev_by_role.get(role, []), curr_by_role.get(role, [])
        if _role_unchanged(role, p, c):
            continue
        if role in ("system", "system_dynamic"):
            if [b.masked for b in p] == [b.masked for b in c]:
                return Classification(MissCause.VOLATILE_SYSTEM, False,
                                      "system raw hash changed, masked hash equal: only volatile values differ"
                                      + _volatile_kinds(p, c), secondary, native_reason)
            return Classification(MissCause.SYSTEM_EDIT, None,
                                  "system masked hash changed", secondary, native_reason)
        if _tool_multiset(prev.blocks) == _tool_multiset(curr.blocks):
            return Classification(MissCause.TOOL_RESERIALIZE, False,
                                  "tool multiset of canonical hashes equal; sequence or raw bytes differ",
                                  secondary, native_reason)
        return Classification(MissCause.TOOL_CATALOG_CHANGE, True,
                              "tool multiset differs", secondary, native_reason)

    # The conversation grows by appending: the previous turn's blocks must be an unchanged prefix of
    # this turn's. The first block that is not, in the order they were sent (messages and tool
    # results interleaved), is where the cache stopped matching, and its kind names the cause.
    if not conversation:
        return None
    pc = [b for b in prev.blocks if b.role in _WALKED]  # ("user" blocks, wire records only, are judged apart, below)
    cc = [b for b in curr.blocks if b.role in _WALKED]
    i = next((k for k, (a, b) in enumerate(zip(pc, cc)) if a.role != b.role or not _same(a, b)), None)
    if i is None and len(pc) > len(cc):
        i = len(cc)  # earlier blocks removed (truncated history)
    if i is None:
        if not _role_unchanged("user", prev_by_role.get("user", []), curr_by_role.get("user", [])):
            secondary.append(MissCause.HISTORY_EDIT)  # a changed user block alone: no cause of its own
        return None
    role = pc[i].role
    if role == "messages":
        p, c = prev_by_role.get("messages", []), curr_by_role.get("messages", [])
        one_shot = len(p) == 1
        if (not curr.compaction_event and p and c and p[0].raw != c[0].raw
                and (one_shot or (len(c) == 1 and curr.shared_session))):
            # The opening message differs, and either the previous request was a one-shot (a single
            # message: an agent's "title this" side call, a classification), or this request is a
            # single message in a session that names no single conversation (traffic with no
            # session id all lands in one). Either way it is another conversation on the same tools
            # and system prompt, not an edit of this one's history: judge it by whether the cache
            # served the shared prefix, and never as an unchanged prompt that missed (classify()).
            # Any other request with another opening (several messages, or a single one in a
            # session that names a conversation) is a rewritten history: an edit, or a compaction
            # with its event.
            if notes is not None:
                notes["new_conversation"] = True
            if not one_shot:
                return None  # the previous conversation's tool results are not this one's to keep
            secondary.append(MissCause.HISTORY_EDIT)
            pt, ct = prev_by_role.get("tool_results", []), curr_by_role.get("tool_results", [])
            if not _role_unchanged("tool_results", pt, ct):
                role = "tool_results"
            else:
                return None
        else:
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
    return None


def classify(prev: RequestSnapshot | None, curr: RequestSnapshot,
             *, native_reason: str | None = None, earlier_point: str | None = None) -> Classification:
    """Classify one turn against the previous turn in the same session. `earlier_point`: this
    request is an earlier point of the compared turn's own conversation (the collector's finding,
    in words): the rules that need no comparison apply, then a change in the tools or the system
    prompt is named like any other, and with none it is judged by its own read."""
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
    if curr.new_thread:
        if curr.usage_known and curr.prompt_tokens > 0 and curr.cached_tokens >= curr.prompt_tokens * 0.5:
            return Classification(MissCause.NONE, None,
                                  "a new conversation in this session, its shared prefix served from the cache",
                                  native_reason=native_reason)
        return Classification(MissCause.FIRST_WRITE, True,
                              "a new conversation in this session: its opening matches none of the session's "
                              "conversations on this model and its system prompt differs (a subagent of "
                              "another type)", native_reason=native_reason)

    # The previous request was too short for the provider to cache at all, so
    # nothing could have been reused, whatever changed since: not waste.
    floor = _min_prefix(curr)
    if floor is not None and 0 < prev.prompt_tokens < floor and curr.cached_tokens == 0:
        return Classification(MissCause.BELOW_MIN_PREFIX, None,
                              f"previous request was {prev.prompt_tokens} tokens, below the provider's minimum "
                              f"cacheable prefix ({floor}): nothing was cached to reuse",
                              native_reason=native_reason)

    secondary: list[MissCause] = []
    notes: dict = {}
    changed = _first_change(prev, curr, secondary, native_reason, notes, conversation=not earlier_point)
    if changed is None and earlier_point:
        # Tools and system are as they were, and nothing in the conversation changed: its
        # predecessor is the turn before that point, not the one at hand, so no change is claimed.
        # The turn at hand is still its own conversation's, so reading all that turn left of what
        # this request repeats is a hit.
        return own_read(curr, earlier_point, native_reason, served_all_left=_served_all(prev, curr))
    if (changed is not None and floor is not None and curr.usage_known
            and 0 < curr.prompt_tokens < floor and curr.cached_tokens == 0):
        # This request is itself too short to cache: no version of it could have been read
        # from the cache, so the change it carries cost nothing (it stays in secondary).
        return Classification(MissCause.BELOW_MIN_PREFIX, None,
                              f"cacheable prefix {curr.prompt_tokens} below provider minimum {floor}; the "
                              f"{changed.cause.value} in it cost nothing",
                              secondary + [changed.cause], native_reason, score=False)
    if changed is not None:
        if curr.usage_known and curr.prompt_tokens > 0 and curr.cached_tokens == 0 and _left_nothing(prev, curr):
            # Nothing was in the cache to lose (and so nothing to expire: this comes before the TTL).
            return _after_nothing_left(prev, curr, secondary, native_reason, changed.cause)
        age, ttl = _age(curr), curr.ttl_seconds
        if (curr.usage_known and ttl is not None and age is not None and age > ttl
                and curr.prompt_tokens > 0 and curr.cached_tokens == 0):
            # The entry had expired before this request: nothing was reusable, whatever changed.
            return Classification(MissCause.TTL_EXPIRED, True,
                                  f"{_since(curr)} exceeds the cache TTL {ttl:.0f}s and nothing "
                                  f"was read; the {changed.cause.value} after it cost nothing more",
                                  secondary + [changed.cause], native_reason, score=False)
        if (curr.provider == "anthropic" and not curr.engine and not curr.has_breakpoint and curr.usage_known
                and curr.cached_tokens == 0 and _wrote_nothing(prev)
                and MissCause.BREAKPOINT_MISPLACED not in changed.secondary):
            # The request carries no cache_control either, and the previous turn wrote nothing: the
            # change is still the first thing to fix (a marker on a prefix that changes every turn
            # buys writes nobody reads), but nothing is cached until the markers are there too.
            # (Not behind a self-hosted engine, which caches without markers.)
            changed.secondary = [*changed.secondary, MissCause.BREAKPOINT_MISPLACED]
            changed.evidence += ("; the request also carries no cache_control, so nothing is cached until both "
                                 "are fixed")
        if _served_all(prev, curr):
            # The provider served all of the prefix the previous turn could share: the change
            # was after it (or did not reach the rendered prompt) and cost nothing.
            return Classification(MissCause.NONE, None,
                                  f"{changed.cause.value} seen, but the cache served all of the reusable "
                                  f"prefix ({_served(prev, curr)}): it cost nothing",
                                  secondary + [changed.cause], native_reason, score=False)
        return changed

    # Blocks equal (or only the user turn changed). A full hit wins over every inference below:
    # if the cache served all of the reusable prefix, a changed param did not reach the rendered
    # prompt, the markers were not needed (or were not misplaced) and the minimum did not apply.
    # One exception: a breakpoint that stays early while the stable prompt grows past it. The
    # cache serves all it holds, turn after turn, and the stable part after the breakpoint is paid
    # uncached every time: that is a misplaced breakpoint, not a hit.
    new_conversation = bool(notes.get("new_conversation"))
    if _served_all(prev, curr):
        # (A new conversation repeats only the front of the previous prompt: no tail to speak of.)
        held, shared, tail = (0, 0, 0) if new_conversation else _stable_uncached_tail(prev, curr, secondary)
        if tail:
            return Classification(MissCause.BREAKPOINT_MISPLACED, False,
                                  f"breakpoint too early: the cache held {held:,} tokens and served them all, but "
                                  f"both turns share a {shared:,}-token prefix; the {tail:,} stable tokens after "
                                  "the last breakpoint are sent uncached every turn (a breakpoint after them, or "
                                  "on the last message, would cache them)", secondary, native_reason)
        if prev.params != curr.params:
            secondary.append(MissCause.PARAM_RENDERED)
        return Classification(MissCause.NONE, None,
                              f"the cache served all of the reusable prefix ({_served(prev, curr)})",
                              secondary, native_reason)
    if prev.params != curr.params:
        changed = sorted(set(prev.params) ^ set(curr.params) |
                         {k for k in prev.params.keys() & curr.params.keys() if prev.params[k] != curr.params[k]})
        same = "a new conversation on the same tools and system prompt" if new_conversation else "blocks equal"
        return Classification(MissCause.PARAM_RENDERED, None,
                              f"{same}; prompt-affecting params changed: {changed}",
                              secondary, native_reason)

    min_prefix = _min_prefix(curr)
    # 0 prompt tokens means usage was absent, not a short prompt: fall
    # through to UNKNOWN rather than claim BELOW_MIN_PREFIX.
    if min_prefix is not None and 0 < curr.prompt_tokens < min_prefix:
        return Classification(MissCause.BELOW_MIN_PREFIX, None,
                              f"cacheable prefix {curr.prompt_tokens} below provider minimum {min_prefix}",
                              secondary, native_reason)

    age, ttl = _age(curr), curr.ttl_seconds
    # A request with no marker states no lifetime (the edge's ttl_seconds is then the 5-minute default
    # a marker would have had): the longest a marker can ask for is the lifetime to beat. A marker
    # that is there, after a volatile block, states its own.
    unmarked = curr.provider == "anthropic" and not curr.has_breakpoint
    longest = MAX_ANTHROPIC_TTL_SECONDS if unmarked else ttl
    if ((unmarked or curr.breakpoint_after_volatile)
            and curr.usage_known and longest is not None and age is not None and age > longest
            and curr.prompt_tokens > 0 and curr.cached_tokens == 0):
        # With markers in the right place the entry would have expired all the same: nothing here
        # was avoidable. (Every changed-block cause is weighed this way above.)
        return Classification(MissCause.TTL_EXPIRED, True,
                              f"{_since(curr)} exceeds the {'longest cache TTL' if unmarked else 'cache TTL'} "
                              f"{longest:.0f}s and nothing was read; the missing or misplaced cache_control cost "
                              "nothing more", secondary + [MissCause.BREAKPOINT_MISPLACED], native_reason, score=False)
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
                              ("a new conversation in this session" if new_conversation else "blocks unchanged")
                              + " but the provider returned no usage; cannot tell hit from miss",
                              secondary, native_reason)
    # A server that reports no cached-token count (Ollama, many llama.cpp builds) may have reused
    # its cache: its 0 is "not reported", so no timing or routing rule can read it.
    # KV events that saw the blocks removed, or the request served elsewhere, are evidence on their own.
    if not curr.cache_reported and not curr.evicted_blocks and not curr.routed:
        return Classification(MissCause.UNKNOWN, None, CACHE_NOT_REPORTED, secondary, native_reason)

    # Append-only turn: compare cache_read to the reusable prior prefix,
    # not to this turn's prompt, which includes the newly appended tail.
    # A real hit wins over every timing rule below: if the cache served
    # the prefix, it did not expire and was not evicted.
    reusable = min(prev.prompt_tokens, curr.prompt_tokens) if prev.prompt_tokens else curr.prompt_tokens
    if curr.prompt_tokens > 0 and curr.cached_tokens >= reusable * 0.5:
        return Classification(MissCause.NONE, None,
                              "a new conversation in this session, its shared prefix served from the cache"
                              if new_conversation else
                              "append-only turn; cache_read at least half of the prior prefix",
                              secondary, native_reason)

    # Anthropic writes only at a breakpoint: a previous turn without one left nothing to read,
    # so a zero read on the first turn that carries one is the first write, not a miss.
    # A part read now (another conversation's write of the same system prompt) does not change that.
    if curr.provider == "anthropic" and curr.has_breakpoint and not prev.has_breakpoint:
        return Classification(MissCause.FIRST_WRITE, True,
                              "the previous turn carried no cache_control, so nothing was written for this turn "
                              "to read; this turn's write is the first", secondary, native_reason)

    # A new conversation shares only the tools and system prompt before its opening message with the
    # compared turn. Was that part in the cache, where this request could read it? Only one case is
    # known: on Anthropic's own API, the previous turn marked a block of tools+system, read or wrote
    # something, and this request marks the same block. Then, with nothing read, that prefix is
    # judged by the expiry and routing rules below like any prefix that did not change.
    shared_missed = (new_conversation and curr.shared_breakpoint and curr.provider == "anthropic"
                     and not curr.engine and curr.cached_tokens == 0
                     and prev.cached_tokens + prev.cache_written > 0)
    if new_conversation and not shared_missed:
        # Otherwise how large the shared part is, or whether it was cached at all, is not known:
        # no routing miss or expiry is inferred, and the turn teaches nothing about how long the
        # cache lasts. It is the conversation's first write; whether a shared part was missed is
        # left open (legit None).
        return Classification(MissCause.FIRST_WRITE, None,
                              "a new conversation in this session (its opening message is not the previous "
                              "request's): only the tools and system prompt before it could be reused; "
                              f"{curr.cached_tokens:,} of its {curr.prompt_tokens:,} prompt tokens were read, and "
                              "the size of that shared part is not known, so no miss of it is claimed",
                              secondary, native_reason)

    if curr.usage_known and curr.cached_tokens == 0 and _left_nothing(prev, curr):
        # Nothing was in the cache: nothing for a router to miss, nothing to expire and nothing a
        # retry could have read.
        return _after_nothing_left(prev, curr, secondary, native_reason)

    # From here the prefix was reusable but cache_read was low.
    # Engine eviction events are direct evidence; they outrank inference.
    if curr.evicted_blocks:
        return Classification(MissCause.EVICTED, None,
                              "confirmed: the block where the cache read stopped was removed from GPU memory "
                              f"({curr.evicted_blocks} of the previous request's KV blocks removed; KV events)",
                              secondary, native_reason)

    if curr.routed:
        before, now, held = curr.routed
        return Classification(MissCause.ROUTING_MISS, None,
                              f"confirmed: served by {now}, while {before}, which served this session's previous "
                              f"request, still held the prefix ({held} of its blocks resident; KV events). "
                              "Route each session to the same cache",
                              secondary, native_reason)

    if (curr.provider == "openai" and not curr.engine and curr.min_prefix_tokens is None
            and not openai_model(curr.model) and 0 < curr.prompt_tokens < OPENAI_MIN_PREFIX):
        # Another API in OpenAI's shape (DeepSeek, Tencent, a gateway) caches by its own minimum, unknown here:
        # a short prompt that missed may simply be below it, so no retry, routing miss or minimum is claimed (live: DeepSeek cached nothing of a 239-token prompt
        # repeated three times).
        return Classification(MissCause.UNKNOWN, None,
                              f"a {curr.prompt_tokens}-token prompt on an OpenAI-compatible API whose minimum "
                              "cacheable length is not known: it may be below that minimum",
                              secondary, native_reason)
    # Identical replay within a short window is a retry, not a cache miss.
    if prev.blocks and curr.blocks and \
       [b.raw for b in prev.blocks] == [b.raw for b in curr.blocks] and \
       curr.gap_seconds is not None and curr.gap_seconds < 60:
        return Classification(MissCause.RETRY_REPLAY, False,
                              "identical request within the retry window", secondary, native_reason)

    # One self-hosted engine: there is no routing, so an unchanged prefix that was
    # not reused is usually an eviction. Events can rule that out (docs/ENGINES.md).
    if curr.engine:
        if curr.evicted_blocks == 0 and curr.resident_blocks:
            where = ("the cache that served it" if curr.engine_caches > 1 else "the engine")
            also = ("" if curr.engine_caches > 1 else
                    ", or the request went to another data-parallel rank")
            return Classification(MissCause.UNKNOWN, None,
                                  f"not an eviction: {where} still holds the block where the cache read stopped "
                                  f"({curr.resident_blocks} of the previous request's blocks resident; KV events). "
                                  "The token prefix changed there although the request did not (chat template "
                                  f"rendering){also}",
                                  secondary, native_reason)
        if curr.engine_caches > 1:
            return Classification(MissCause.UNKNOWN, None,
                                  f"{curr.engine_caches} KV caches behind this upstream (data-parallel ranks or "
                                  "replicas): an eviction and a request sent to another cache look the same, and "
                                  "the KV events could not tell them apart for this request",
                                  secondary, native_reason)
        return Classification(MissCause.EVICTED, None,
                              "inferred: one self-hosted engine, prefix unchanged but not reused. Usually the "
                              "engine evicted it (GPU memory full); chat template re-rendering or data-parallel "
                              "routing look the same. Confirm with --kv-events",
                              secondary, native_reason)

    gap, ttl, age = curr.gap_seconds, curr.ttl_seconds, _age(curr)
    # How much of the unchanged prefix came back, named in every verdict below: a partial read is
    # a miss of its own size, not a hit.
    read = (f"read {curr.cached_tokens:,} of the {reusable:,}-token unchanged prefix" if curr.cached_tokens
            else "nothing read from cache")
    observed = curr.observed_lifetime if ttl is None else None
    # What is judged from here: the unchanged prompt, or, for a new conversation, the tools and
    # system prompt it shares with the previous request (which is not evidence of how long an
    # unchanged prompt lasts).
    unchanged = not shared_missed  # (cold_unchanged: an unchanged prefix that came back short)
    what = ("a new conversation on the tools and system prompt the previous request cached at a marker this "
            "one repeats; " if shared_missed else "")
    if ttl is not None and age is not None and age > ttl:
        return Classification(MissCause.TTL_EXPIRED, True,
                              f"{what}{_since(curr)} exceeds the cache TTL {ttl:.0f}s; {read}",
                              secondary, native_reason, cold_unchanged=unchanged)

    # Within the TTL (or, when it is unknown, within the shortest TTL any provider keeps), an
    # unchanged prefix that came back short did not expire. With no eviction evidence, the request
    # most likely reached a cache that did not hold it (another server). A partial read is the same
    # miss as a zero read, only smaller: implicit caches (OpenAI, DeepSeek) do both.
    # On Anthropic a part read after more than 5 minutes is what a mixed lifetime looks like: a 1-hour
    # breakpoint early in the prompt still read, a 5-minute one after it expired. The request's TTL is
    # the longest any breakpoint asked for, so a part read is judged by the shortest (5 minutes).
    # A stated lifetime is compared with the entry's age (since the previous turn was sent); an
    # observed one with the idle gap, which is what it was learned from.
    limit = ttl if ttl is not None else MIN_PROVIDER_TTL_SECONDS
    alive = getattr(observed, "alive_up_to", None)
    by_observation = alive is not None and alive > limit
    if by_observation:
        limit = alive  # the cache has been seen to hold an unchanged prefix this long here
    if curr.cached_tokens and curr.provider == "anthropic":
        limit, by_observation = min(limit, MIN_PROVIDER_TTL_SECONDS), False
    # Which lifetime decides: the stated one (or the shortest any provider keeps), against the
    # entry's age; only when the age is past that does an observed lifetime decide, against the
    # idle gap. The evidence names the one that decided.
    stated = ttl if ttl is not None else MIN_PROVIDER_TTL_SECONDS
    if curr.cached_tokens and curr.provider == "anthropic":
        stated = min(stated, MIN_PROVIDER_TTL_SECONDS)
    by_observation = by_observation and not (age is not None and age <= stated)
    measure = gap if by_observation else age
    if not by_observation:
        limit = stated
    if curr.prompt_tokens > 0 and measure is not None and measure <= limit:
        window = (f"the observed lifetime on this model: {observed.describe()}" if by_observation
                  else f"TTL {ttl:.0f}s" if ttl is not None and limit == ttl
                  else f"{MIN_PROVIDER_TTL_SECONDS // 60} minutes (the shortest TTL a breakpoint can have)"
                  if curr.provider == "anthropic" and curr.cached_tokens
                  else f"shortest provider TTL {MIN_PROVIDER_TTL_SECONDS}s")
        # (An observed lifetime is measured in idle gaps; a stated one against the entry's age.)
        span = f"gap {gap:.0f}s" if by_observation else _since(curr, short=True)
        return Classification(MissCause.ROUTING_MISS, None,
                              f"inferred: {what or 'blocks and params equal, '}{span} within {window}, "
                              f"no eviction evidence, {read}",
                              secondary, native_reason, cold_unchanged=unchanged)
    if curr.prompt_tokens > 0 and age is not None and curr.cached_tokens and curr.provider == "anthropic":
        return Classification(MissCause.UNKNOWN, None,
                              f"a part read ({read}), {_since(curr)}: a 5-minute breakpoint may "
                              "have expired while an earlier 1-hour one held, or the request reached another "
                              "server: expiry or routing, not distinguishable",
                              secondary, native_reason, cold_unchanged=True)
    cold = getattr(observed, "cold_from", None)
    if curr.prompt_tokens > 0 and gap is not None and cold is not None and gap >= cold:
        return Classification(MissCause.TTL_EXPIRED, True,
                              f"inferred: gap {gap:.0f}s is past the observed lifetime on this model "
                              f"({observed.describe()}); {read}",
                              secondary, native_reason, cold_unchanged=True)
    if curr.prompt_tokens > 0 and gap is not None and ttl is None:
        seen = f"; observed on this model: {observed.describe()}" if observed is not None else ""
        return Classification(MissCause.UNKNOWN, None,
                              f"prefix reusable but cache_read low ({read}), {_since(curr)}, with unknown"
                              f" TTL: expiry or routing, not distinguishable{seen}",
                              secondary, native_reason, cold_unchanged=True)
    return Classification(MissCause.UNKNOWN, None,
                          f"{what}prefix reusable but cache_read low ({read}); no timing or usage evidence to name "
                          "a cause", secondary, native_reason, cold_unchanged=unchanged)


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


def first_divergence(prev: list, curr: list, engine: bool = False) -> dict | None:
    """Locate the first block, in prefix order, where two requests differ. Behind a self-hosted
    engine the chat template renders the system prompt before the tools, as `_first_change` reads
    them; the index stays the block's position in the request."""
    if engine:
        rank = {"system": 0, "system_dynamic": 0, "tools": 1}
        prev = sorted(prev, key=lambda b: rank.get(b.role, 2))
        curr = sorted(curr, key=lambda b: rank.get(b.role, 2))
    for i, (a, b) in enumerate(zip(prev, curr)):
        i = getattr(a, "index", i)
        if a.role != b.role:
            return {"index": i, "role": f"{a.role} -> {b.role}", "kind": "block order changed"}
        if a.raw == b.raw:
            continue
        if a.canonical == b.canonical:
            kind = "same content, different bytes (key order / whitespace / serialization)"
        elif a.masked == b.masked:
            kind = "only volatile values changed (dates, times, ids, numbers)"
        else:
            kind = "content changed"
        return {"index": i, "role": a.role, "kind": kind}
    if len(curr) > len(prev):
        return {"index": len(prev), "role": curr[len(prev)].role,
                "kind": "appended after the previous request (normal growth, not a divergence)"}
    if len(curr) < len(prev):
        return {"index": len(curr), "role": prev[len(curr)].role,
                "kind": "previous blocks removed (history truncated or compacted)"}
    return None
