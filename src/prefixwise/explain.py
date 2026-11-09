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

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import RequestSnapshot, classify, first_divergence
from .fingerprint import BlockFingerprint, Fingerprinter
from .pricing import breakdown
from .records import PARAM_KEYS as _PARAM_KEYS, param_fingerprint
from .proxy import (cache_ttl_seconds, completion_tokens_of, extract_blocks, has_cache_breakpoint,
                    normalize_usage)

PARAM_KEYS = _PARAM_KEYS  # the same prompt-affecting parameters the proxy compares


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
    diff: str | None = None  # --diff: the first differing block, before and after, from the files given

    def to_dict(self) -> dict:
        return dict(self.__dict__)


DIFF_LINES = 40
DIFF_SIDE_LINES = 600  # lines of each side kept around the first difference before diffing


def _block_text(value) -> str:
    """A block for reading: a string as is; a dict or list as indented JSON with the newlines inside
    its strings made real, so a long system prompt or message diffs line by line instead of as one
    multi-kilobyte JSON string literal. For eyes, not for parsing back."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=1, ensure_ascii=False).replace("\\n", "\n")


def _lines(text: str) -> list[str]:
    return text.split("\n")  # not splitlines(): a trailing newline or a CR is a difference too


def block_diff(prev_body: dict, curr_body: dict, provider: str, index: int, limit: int = DIFF_LINES) -> str | None:
    """A unified diff of block `index` (prefix order, as fingerprinted: cache markers removed) between
    two request bodies the caller holds. Hashes say where a request changed; this says what, from the
    user's own files, on their machine. The proxy and the ledger never see it. The common head and
    tail of the two sides are dropped and each side is cut to DIFF_SIDE_LINES before diffing, so a
    large block with scattered edits does not cost seconds; the output is cut at `limit` lines."""
    prev_blocks, curr_blocks = extract_blocks(prev_body, provider), extract_blocks(curr_body, provider)
    before_text = _block_text(prev_blocks[index][1]) if index < len(prev_blocks) else ""
    after_text = _block_text(curr_blocks[index][1]) if index < len(curr_blocks) else ""
    if before_text == after_text:
        return None
    before, after = _lines(before_text), _lines(after_text)
    head = 0
    while head < len(before) and head < len(after) and before[head] == after[head]:
        head += 1
    tail = 0
    while (tail < len(before) - head and tail < len(after) - head
           and before[-1 - tail] == after[-1 - tail]):
        tail += 1
    start = max(0, head - 1)  # one line of context before the change
    b = before[start:len(before) - tail + 1][:DIFF_SIDE_LINES]
    a = after[start:len(after) - tail + 1][:DIFF_SIDE_LINES]
    label = f" (from line {start + 1})" if start else ""
    lines = list(difflib.unified_diff(b, a, "previous request" + label, "this request" + label, lineterm="", n=1))
    if not lines:  # the same lines, different bytes: line endings or a trailing newline
        return "(differs only in line endings or trailing whitespace: the text is the same line by line)"
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines"]
    return "\n".join(lines)


def _snapshot(turn: Turn, provider: str, fp: Fingerprinter) -> RequestSnapshot:
    blocks = fp.fingerprint_blocks(extract_blocks(turn.body, provider))
    written = 0
    if turn.usage is not None:
        prompt, cached = normalize_usage(provider, turn.usage)
        parts = breakdown(provider, turn.usage)
        written = parts.cache_write_5m + parts.cache_write_1h + parts.cache_write
        min_prefix = None
    else:
        # Offline: pretend the shared prefix was fully served, so only a
        # divergence in the request itself can explain a miss.
        prompt, cached, min_prefix = 1, 1, 0
    return RequestSnapshot(model=str(turn.body.get("model", "unknown")), provider=provider,
                           blocks=blocks, prompt_tokens=prompt, cached_tokens=cached, cache_written=written,
                           gap_seconds=turn.gap_seconds, compaction_event=turn.compaction_event,
                           params={k: param_fingerprint(turn.body.get(k)) for k in PARAM_KEYS
                                   if turn.body.get(k) is not None},
                           has_breakpoint=has_cache_breakpoint(turn.body),
                           ttl_seconds=cache_ttl_seconds(provider, turn.body),
                           min_prefix_tokens=min_prefix, usage_known=turn.usage is not None)


def explain_pair(prev: Turn | None, curr: Turn, provider: str | None = None,
                 fingerprinter: Fingerprinter | None = None, diff: bool = False) -> Explanation:
    provider = provider or detect_provider(curr.body)
    fp = fingerprinter or Fingerprinter()
    curr_snap = _snapshot(curr, provider, fp)
    prev_snap = _snapshot(prev, provider, fp) if prev is not None else None
    result = classify(prev_snap, curr_snap)
    div = first_divergence(prev_snap.blocks, curr_snap.blocks) if prev_snap else None
    if div and result.cause.value == "tool_reserialize" and div["kind"] == "content changed":
        div["kind"] = "tools reordered (same tool set, different order)"
    text = None
    if diff and div and prev is not None and "appended after" not in div["kind"]:
        text = block_diff(prev.body, curr.body, provider, div["index"])
    return Explanation(cause=result.cause.value, legit=result.legit, evidence=result.evidence,
                       divergence=div, provider=provider, offline=curr.usage is None,
                       prompt_tokens=curr_snap.prompt_tokens if curr.usage is not None else None,
                       cached_tokens=curr_snap.cached_tokens if curr.usage is not None else None,
                       secondary=[c.value for c in result.secondary], diff=text)


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


def explain_session(turns: list[Turn], provider: str | None = None, diff: bool = False) -> list[Explanation]:
    """Explain every turn against the previous turn in the same session."""
    fp = Fingerprinter()
    last: dict[str, Turn] = {}
    lanes: dict[tuple, Turn] = {}  # (session, model): the same per-model lanes as the proxy
    out = []
    for t in turns:
        lane = (t.session_id, str(t.body.get("model", "unknown")))
        prev = lanes.get(lane) or last.get(t.session_id)
        out.append(explain_pair(prev, t, provider, fp, diff=diff))
        last[t.session_id] = t
        lanes[lane] = t
    return out


__all__ = ["Turn", "Explanation", "explain_pair", "explain_session", "load_turns",
           "first_divergence", "detect_provider", "completion_tokens_of"]
