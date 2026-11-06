"""Offline miss explanation: why would (or did) the cache miss between two requests?

No proxy, no API key. Give it request bodies as the client sent them,
optionally with the usage the provider returned, and it runs the same
fingerprinting and classifier as the proxy.

Without usage, the explanation is about the prompt itself: it assumes the
cache would have served the whole shared prefix, so the answer is "what in
the request breaks reuse", not "what the provider actually did". TTL
expiry, eviction, and routing need real usage or engine events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import RequestSnapshot, classify
from .fingerprint import BlockFingerprint, Fingerprinter
from .proxy import completion_tokens_of, extract_blocks, has_cache_breakpoint, normalize_usage

PARAM_KEYS = ("tool_choice", "thinking", "reasoning", "output_format", "text_format", "verbosity")


def detect_provider(body: dict) -> str:
    """Anthropic Messages bodies carry system at the top level; OpenAI ones do not."""
    return "anthropic" if "system" in body else "openai"


@dataclass
class Turn:
    body: dict
    usage: dict | None = None
    compaction_event: bool = False
    gap_seconds: float | None = None
    session_id: str = "default"


@dataclass
class Explanation:
    cause: str
    legit: bool | None
    evidence: str
    divergence: dict | None
    provider: str
    offline: bool
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    secondary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def first_divergence(prev: list[BlockFingerprint], curr: list[BlockFingerprint]) -> dict | None:
    """Locate the first block, in prefix order, where two requests differ."""
    for i, (a, b) in enumerate(zip(prev, curr)):
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


def _snapshot(turn: Turn, provider: str, fp: Fingerprinter) -> RequestSnapshot:
    blocks = fp.fingerprint_blocks(extract_blocks(turn.body, provider))
    if turn.usage is not None:
        prompt, cached = normalize_usage(provider, turn.usage)
        min_prefix = None
    else:
        # Offline: pretend the shared prefix was fully served, so only a
        # divergence in the request itself can explain a miss.
        prompt, cached, min_prefix = 1, 1, 0
    return RequestSnapshot(model=str(turn.body.get("model", "unknown")), provider=provider,
                           blocks=blocks, prompt_tokens=prompt, cached_tokens=cached,
                           gap_seconds=turn.gap_seconds, compaction_event=turn.compaction_event,
                           params={k: turn.body.get(k) for k in PARAM_KEYS if turn.body.get(k) is not None},
                           has_breakpoint=has_cache_breakpoint(turn.body),
                           min_prefix_tokens=min_prefix)


def explain_pair(prev: Turn | None, curr: Turn, provider: str | None = None,
                 fingerprinter: Fingerprinter | None = None) -> Explanation:
    provider = provider or detect_provider(curr.body)
    fp = fingerprinter or Fingerprinter()
    curr_snap = _snapshot(curr, provider, fp)
    prev_snap = _snapshot(prev, provider, fp) if prev is not None else None
    result = classify(prev_snap, curr_snap)
    div = first_divergence(prev_snap.blocks, curr_snap.blocks) if prev_snap else None
    if div and result.cause.value == "tool_reserialize" and div["kind"] == "content changed":
        div["kind"] = "tools reordered (same tool set, different order)"
    return Explanation(cause=result.cause.value, legit=result.legit, evidence=result.evidence,
                       divergence=div, provider=provider, offline=curr.usage is None,
                       prompt_tokens=curr_snap.prompt_tokens if curr.usage is not None else None,
                       cached_tokens=curr_snap.cached_tokens if curr.usage is not None else None,
                       secondary=[c.value for c in result.secondary])


def _turn_from_record(rec: dict) -> Turn:
    """A record is a bare request body, or {"request"|"body": ..., "usage": ..., ...}."""
    if not isinstance(rec, dict):
        raise ValueError("each record must be a JSON object")
    body = rec.get("request") or rec.get("body")
    if body is None:
        return Turn(body=rec, session_id=str(rec.get("session_id") or "default"))
    usage = rec.get("usage") or (rec.get("response") or {}).get("usage")
    return Turn(body=body, usage=usage,
                compaction_event=bool(rec.get("compaction_event", False)),
                gap_seconds=rec.get("gap_seconds"),
                session_id=str(rec.get("session_id") or body.get("session_id") or "default"))


def load_turns(path: str | Path) -> list[Turn]:
    """Load one JSON record, a JSON array of records, or JSONL (one record per line)."""
    text = Path(path).read_text()
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = data if isinstance(data, list) else [data]
    return [_turn_from_record(r) for r in records]


def explain_session(turns: list[Turn], provider: str | None = None) -> list[Explanation]:
    """Explain every turn against the previous turn in the same session."""
    fp = Fingerprinter()
    last: dict[str, Turn] = {}
    out = []
    for t in turns:
        out.append(explain_pair(last.get(t.session_id), t, provider, fp))
        last[t.session_id] = t
    return out


__all__ = ["Turn", "Explanation", "explain_pair", "explain_session", "load_turns",
           "first_divergence", "detect_provider", "completion_tokens_of"]
