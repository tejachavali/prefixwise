"""Canonical block fingerprinting (v2).

Split a rendered request into blocks in prefix order and store three
hashes per block:

- ``raw``: SHA-256 (or HMAC) over the exact bytes as sent. This is the
  only hash that can see TOOL_RESERIALIZE, which is defined as
  canonical-equal, raw-different.
- ``canonical``: SHA-256 (or HMAC) over canonicalised bytes (sorted keys,
  fixed separators, NFC unicode, no trailing-whitespace variance).
- ``masked``: hash over canonical bytes with volatile values masked out
  (dates, times, numbers, UUIDs, ids). A system block whose raw hash
  changed but whose masked hash is equal is VOLATILE_SYSTEM, not a
  SYSTEM_EDIT.

Hashes are stored, never raw prompts, by default.

Privacy: a bare hash of a short or predictable block (a four-word user
message, a boilerplate system line) is dictionary-guessable. When an
HMAC key is supplied, every hash of a short/predictable block is an
HMAC with the per-install key, replacing bare SHA-256 entirely for
that block (raw, canonical, and masked). The short-block threshold
is a policy choice (default 64 tokens, approximate, ~256 characters),
not a verified fact; a 60-token boilerplate line can still be
predictable, and a long block can still be guessable.

Soft tokens / summaries are fingerprinted as distinct block types:
their content is never hashed as text. A soft-token block hashes
only its type and token count.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field

BLOCK_ROLES = (
    "system",
    "system_dynamic",
    "tools",
    "messages",
    "tool_results",
    "user",
    "summary",
    "soft_tokens",
)

# Policy choice, not a verified fact (see module docstring).
DEFAULT_SHORT_BLOCK_CHARS = 256

# Masked hashes are memoised by canonical hash: an agent resends the same
# system prompt, tools and history every turn, and masking (six regex passes)
# was ~80% of the proxy's per-request CPU on long prompts. Bounded LRU.
MASK_CACHE_ENTRIES = 32768

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b", re.IGNORECASE)
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_WITH_DIGIT_RE = re.compile(r"\b\w*\d\w*\b")


def mask_volatiles(text: str) -> str:
    """Replace volatile values (dates, times, ids, numbers) with placeholders."""
    text = _DATE_RE.sub("<DATE>", text)
    text = _TIME_RE.sub("<TIME>", text)
    text = _UUID_RE.sub("<ID>", text)
    text = _HEX_ID_RE.sub("<ID>", text)
    text = _NUMBER_RE.sub("<N>", text)
    # Tokens like abc123 / session-42 are ids, not stable words.
    text = _TOKEN_WITH_DIGIT_RE.sub("<ID>", text)
    return text


def canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8")


def raw_bytes(value: object) -> bytes:
    """Exact bytes as sent: JSON without key sorting, as the caller rendered it."""
    if isinstance(value, (str, bytes)):
        raw = value if isinstance(value, bytes) else value.encode("utf-8")
    else:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return raw


def _hash(data: bytes, hmac_key: bytes | None) -> str:
    if hmac_key:
        return hmac_mod.new(hmac_key, data, hashlib.sha256).hexdigest()
    return hashlib.sha256(data).hexdigest()


@dataclass
class BlockFingerprint:
    role: str
    index: int
    raw: str
    canonical: str
    masked: str
    byte_length: int
    hmac: bool = False
    # First-divergence position in tokens is filled by the caller when a
    # tokenizer is available; None means "not measured", never a guess.
    token_estimate: int | None = None


@dataclass
class Fingerprinter:
    """Per-install fingerprinter. Supply hmac_key to protect short blocks."""

    hmac_key: bytes | None = None
    short_block_chars: int = DEFAULT_SHORT_BLOCK_CHARS
    _masked_cache: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _masked_hash(self, canon: bytes, canon_hash: str, key: bytes | None) -> str:
        # canon_hash already depends on key, so it is a safe cache key for either mode.
        with self._cache_lock:
            hit = self._masked_cache.get(canon_hash)
            if hit is not None:
                self._masked_cache.move_to_end(canon_hash)
                return hit
        masked = _hash(mask_volatiles(canon.decode("utf-8", "replace")).encode("utf-8"), key)
        with self._cache_lock:
            self._masked_cache[canon_hash] = masked
            if len(self._masked_cache) > MASK_CACHE_ENTRIES:
                self._masked_cache.popitem(last=False)
        return masked

    def _is_short(self, raw: bytes) -> bool:
        return len(raw) <= self.short_block_chars

    def fingerprint_block(self, role: str, index: int, value: object) -> BlockFingerprint:
        if role == "soft_tokens":
            # Never hash soft-token content as text: hash type + count only.
            count = value if isinstance(value, int) else len(value)  # type: ignore[arg-type]
            payload = canonical_bytes({"type": "soft_tokens", "count": count})
            digest = _hash(payload, self.hmac_key)
            return BlockFingerprint(role=role, index=index, raw=digest, canonical=digest,
                                    masked=digest, byte_length=len(payload), hmac=bool(self.hmac_key))
        raw = raw_bytes(value)
        canon = canonical_bytes(value)
        # HMAC replaces bare hashing for short/predictable blocks entirely.
        key = self.hmac_key if self._is_short(raw) else None
        canon_hash = _hash(canon, key)
        return BlockFingerprint(
            role=role, index=index,
            raw=_hash(raw, key), canonical=canon_hash, masked=self._masked_hash(canon, canon_hash, key),
            byte_length=len(raw), hmac=bool(key),
        )

    def fingerprint_blocks(self, blocks: list[tuple[str, object]]) -> list[BlockFingerprint]:
        return [self.fingerprint_block(role, i, value) for i, (role, value) in enumerate(blocks)]

    def prefix_hash(self, fingerprints: list[BlockFingerprint]) -> str:
        joined = "".join(f.canonical for f in fingerprints).encode("utf-8")
        return _hash(joined, self.hmac_key)


def block_hash(value: object) -> str:
    """Backwards-compatible helper: canonical SHA-256 of a value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fingerprint_blocks(blocks: list[tuple[str, object]], hmac_key: bytes | None = None) -> list[BlockFingerprint]:
    return Fingerprinter(hmac_key=hmac_key).fingerprint_blocks(blocks)


def prefix_hash(fingerprints: list[BlockFingerprint]) -> str:
    joined = "".join(f.canonical for f in fingerprints).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()
