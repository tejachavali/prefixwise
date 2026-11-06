"""Native provider diagnostics chaining (v2).

The proxy auto-chains the provider's own diagnostic identifier so the
provider's miss reason is recorded beside Prefixwise's cause on every
turn. The provider reason is a *partial agreement signal*, not ground
truth: providers report the earliest divergence only, in a coarse
taxonomy, and neither names TTL expiry, routing, or eviction.

Anthropic (Claude API only, not Bedrock / Google-managed):
- GA Sep 23, 2026. The beta header ``cache-diagnosis-2026-04-07`` is
  NO LONGER REQUIRED. Send ``diagnostics: {"previous_message_id": id}``
  (first turn: null to opt in). Chaining does not change caching
  behaviour or load prior content. ZDR-eligible.
- Reasons (6): model_changed, system_changed, tools_changed,
  messages_changed, previous_message_not_found, unavailable.
  ``unavailable`` fires when model/system/tools match but tool_choice,
  thinking, context_management, output_config, output_format, or beta
  headers differ: that is PARAM_RENDERED.

OpenAI (Responses API only, GPT-5.6+):
- Send ``prompt_cache_options.comparison_response_id`` with a baseline
  response id. Does not load the conversation or change caching.
  Streaming: read ``prompt_cache_diagnostics`` from ``event.response``
  in the ``response.completed`` event.
- Types (4): cache_hit, cache_miss, comparison_response_not_found,
  unavailable. Reasons (9): model_changed, prompt_cache_key_changed,
  service_tier_changed, tools_changed, text_format_changed,
  reasoning_effort_changed, verbosity_changed, context_compacted,
  input_changed. No TTL-expiry or routing reason exists.

Gemini: no miss diagnostics (verified Nov 2026). Only a hit count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import MissCause

ANTHROPIC_REASONS = {
    "model_changed": MissCause.MODEL_SWITCH,
    "system_changed": MissCause.SYSTEM_EDIT,
    "tools_changed": MissCause.TOOL_CATALOG_CHANGE,  # or TOOL_RESERIALIZE; ambiguous, see below
    "messages_changed": MissCause.HISTORY_EDIT,  # or COMPACTION_REWRITE; hook decides
    "previous_message_not_found": MissCause.UNKNOWN,
    "unavailable": MissCause.PARAM_RENDERED,
}

OPENAI_REASONS = {
    "model_changed": MissCause.MODEL_SWITCH,
    "prompt_cache_key_changed": MissCause.UNKNOWN,
    "service_tier_changed": MissCause.ROUTING_MISS,
    "tools_changed": MissCause.TOOL_CATALOG_CHANGE,
    "text_format_changed": MissCause.PARAM_RENDERED,
    "reasoning_effort_changed": MissCause.PARAM_RENDERED,
    "verbosity_changed": MissCause.PARAM_RENDERED,
    "context_compacted": MissCause.COMPACTION_REWRITE,
    "input_changed": MissCause.HISTORY_EDIT,
}


def anthropic_diagnostics(previous_message_id: str | None) -> dict:
    """Build the Anthropic diagnostics object. No beta header needed (GA Sep 23 2026)."""
    return {"diagnostics": {"previous_message_id": previous_message_id}}


def openai_comparison_options(comparison_response_id: str | None) -> dict:
    """Build OpenAI prompt_cache_options for diagnostics chaining."""
    if comparison_response_id is None:
        return {}
    return {"prompt_cache_options": {"comparison_response_id": comparison_response_id}}


def map_native_reason(provider: str, reason: str | None) -> MissCause | None:
    if reason is None:
        return None
    table = {"anthropic": ANTHROPIC_REASONS, "openai": OPENAI_REASONS}.get(provider, {})
    return table.get(reason)


@dataclass
class AgreementScorer:
    """Score the classifier against native reasons where both exist.

    Two metrics ship with the classifier:
    - agreement: share of turns where Prefixwise's cause maps onto the
      native reason when both exist (E1 target >= 95%).
    - coverage: share of misses not UNKNOWN (E1 target: UNKNOWN <= 10%).
    A miss providers lump into ``unavailable`` that Prefixwise names
    PARAM_RENDERED counts as agreement when the parameter diff is
    present in the request.
    """

    total_with_native: int = 0
    agreed: int = 0
    total_misses: int = 0
    unknown_misses: int = 0

    def record(self, cause: MissCause, native_reason: str | None, provider: str) -> bool | None:
        """Record one turn. Returns True/False for agreement, None if no native reason."""
        is_miss = cause not in (MissCause.NONE, MissCause.FIRST_WRITE)
        if is_miss:
            self.total_misses += 1
            if cause is MissCause.UNKNOWN:
                self.unknown_misses += 1
        if native_reason is None:
            return None
        mapped = map_native_reason(provider, native_reason)
        if mapped is None:
            return None
        self.total_with_native += 1
        # tools_changed / messages_changed are ambiguous: either reserialize
        # or catalog change, either history edit or compaction, both agree.
        ambiguous = {
            (MissCause.TOOL_RESERIALIZE, MissCause.TOOL_CATALOG_CHANGE),
            (MissCause.HISTORY_EDIT, MissCause.COMPACTION_REWRITE),
        }
        agreed = cause == mapped or (cause, mapped) in ambiguous or (mapped, cause) in ambiguous
        if agreed:
            self.agreed += 1
        return agreed

    @property
    def agreement_rate(self) -> float | None:
        if not self.total_with_native:
            return None
        return self.agreed / self.total_with_native

    @property
    def unknown_rate(self) -> float | None:
        if not self.total_misses:
            return None
        return self.unknown_misses / self.total_misses
