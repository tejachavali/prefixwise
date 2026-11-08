"""The turn record: what the edge hands the collector, and all the collector needs.

Prefixwise has two halves. The **edge** sits where the request is (the proxy, `prefixwise.wrap`,
the LiteLLM callback): it fingerprints the request, reads the response's usage, and emits one
record per turn. The **collector** (`prefixwise.collector`) compares each record with the
previous turn of its conversation, names the cause, prices it and writes the ledger. In one
process (`prefixwise serve`, `prefixwise.wrap`) the edge calls the collector directly; across
machines the same record travels over the wire (`prefixwise.transport`).

A record never holds prompt text: blocks are hashes (`prefixwise.fingerprint`), the facts are
hashes, positions and counts (`prefixwise.details`), the usage is the provider's token counts.
`TurnRecord.to_dict()` / `from_dict()` are its wire form; `RECORD_VERSION` changes when a
field's meaning does, and a collector refuses a version it does not know rather than guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .details import Facts
from .fingerprint import BlockFingerprint

RECORD_VERSION = 1

# Prompt-affecting request parameters the classifier compares (RequestSnapshot.params). A change
# in one of these re-renders the prompt without changing any block.
PARAM_KEYS = ("tool_choice", "thinking", "reasoning", "output_format",
              # SGLang leaves extra_key out of its KV-event hashes, so a change must be seen here
              # (cache_salt is in them)
              "extra_key",
              "text_format", "text", "verbosity")


class RecordError(ValueError):
    """A record the collector cannot accept (unknown version, missing or malformed fields)."""


@dataclass
class TurnRecord:
    # -- identity --------------------------------------------------------------------------
    provider: str  # anthropic | openai (any OpenAI-compatible server)
    api: str  # chat | responses
    model: str  # as requested
    session_id: str
    agent_id: str
    request_id: str
    task_id: str | None = None
    session_is_user: bool = False  # the session id is an end-user id, not a conversation id
    fork_of: str | None = None  # X-Prefixwise-Fork-Of
    compaction_event: bool = False
    stream: bool = False
    previous_response_id: str | None = None  # OpenAI Responses: a stitched turn
    # -- the request, as the provider received it (no text) ---------------------------------
    blocks: list[BlockFingerprint] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    has_breakpoint: bool = False  # Anthropic cache_control anywhere
    ttl_seconds: float | None = None  # the cache lifetime that applies to this request
    facts: Facts = field(default_factory=Facts)  # positions, per-message hashes, write counts
    fixes_applied: list[str] = field(default_factory=list)
    sent_at: float = 0.0  # unix time the request was sent (the gap is measured to it)
    # -- the engine join, when the edge subscribes to KV events ------------------------------
    engine: str | None = None  # vllm | sglang | None
    engine_caches: int = 1
    engine_verdict: list | None = None  # [evicted_blocks, resident_blocks, routed] or None
    # -- the response ---------------------------------------------------------------------
    status: int = 200
    error: str | None = None  # set on a failure record (no verdict)
    usage: dict = field(default_factory=dict)
    response_model: str | None = None  # the model the provider says answered (priced by it)
    response_id: str | None = None
    native_reason: str | None = None  # the provider's own cache-miss reason, if it gave one
    latency_ms: float | None = None  # upstream time
    overhead_ms: float | None = None  # the edge's own time
    done_at: float = 0.0  # unix time the response completed
    version: int = RECORD_VERSION

    @property
    def failed(self) -> bool:
        return self.status >= 400 or self.error is not None

    # -- wire form -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocks"] = [asdict(b) for b in self.blocks]
        d["facts"] = asdict(self.facts)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TurnRecord":
        if not isinstance(d, dict):
            raise RecordError("a record must be an object")
        version = d.get("version")
        if version != RECORD_VERSION:
            raise RecordError(f"record version {version!r}; this collector reads version {RECORD_VERSION}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise RecordError(f"unknown record fields: {unknown}")
        for name in ("provider", "api", "model", "session_id", "agent_id", "request_id"):
            if not isinstance(d.get(name), str) or not d[name]:
                raise RecordError(f"record field {name!r} must be a non-empty string")
        try:
            blocks = [BlockFingerprint(**b) for b in d.get("blocks") or []]
            facts = Facts(**(d.get("facts") or {}))
        except TypeError as exc:
            raise RecordError(f"malformed record: {exc}") from None
        for b in blocks:
            if not all(isinstance(getattr(b, h), str) and len(getattr(b, h)) == 64 for h in ("raw", "canonical", "masked")):
                raise RecordError("a block hash is not a 64-character digest")
        rest = {k: v for k, v in d.items() if k not in ("blocks", "facts")}
        if not isinstance(rest.get("usage", {}), dict) or not isinstance(rest.get("params", {}), dict):
            raise RecordError("usage and params must be objects")
        rec = cls(blocks=blocks, facts=facts, **rest)
        if rec.engine_verdict is not None:
            rec.engine_verdict = list(rec.engine_verdict)
        return rec
