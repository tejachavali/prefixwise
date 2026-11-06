from prefixwise.classifier import MissCause
from prefixwise.diagnostics import AgreementScorer, anthropic_diagnostics, map_native_reason, openai_comparison_options


def test_anthropic_no_beta_header():
    # GA Sep 23 2026: diagnostics object only, no beta header anywhere.
    assert anthropic_diagnostics("msg-1") == {"diagnostics": {"previous_message_id": "msg-1"}}
    assert anthropic_diagnostics(None) == {"diagnostics": {"previous_message_id": None}}


def test_openai_comparison_options():
    assert openai_comparison_options("resp-1") == {"prompt_cache_options": {"comparison_response_id": "resp-1"}}
    assert openai_comparison_options(None) == {}


def test_native_reason_mapping():
    assert map_native_reason("anthropic", "model_changed") is MissCause.MODEL_SWITCH
    assert map_native_reason("anthropic", "unavailable") is MissCause.PARAM_RENDERED
    assert map_native_reason("openai", "context_compacted") is MissCause.COMPACTION_REWRITE
    assert map_native_reason("openai", "tools_changed") is MissCause.TOOL_CATALOG_CHANGE
    assert map_native_reason("gemini", "anything") is None


def test_agreement_scorer():
    s = AgreementScorer()
    assert s.record(MissCause.MODEL_SWITCH, "model_changed", "anthropic") is True
    assert s.record(MissCause.TOOL_RESERIALIZE, "tools_changed", "anthropic") is True  # ambiguous, agrees
    assert s.record(MissCause.VOLATILE_SYSTEM, "tools_changed", "anthropic") is False
    assert s.agreement_rate == 2 / 3
    assert s.unknown_rate == 0.0
    s.record(MissCause.UNKNOWN, None, "openai")
    assert s.unknown_rate == 1 / 4


def test_native_agreement_accepts_refinements_not_wrong_places():
    from prefixwise.classifier import MissCause as C
    from prefixwise.diagnostics import AgreementScorer

    s = AgreementScorer()
    # Refinements of the provider's coarse reason agree.
    assert s.record(C.VOLATILE_SYSTEM, "system_changed", "anthropic") is True
    assert s.record(C.TOOL_RESULT_MUTATION, "messages_changed", "anthropic") is True
    assert s.record(C.SYSTEM_EDIT, "input_changed", "openai") is True
    # A different block, or a different kind of change, does not.
    assert s.record(C.SYSTEM_EDIT, "tools_changed", "anthropic") is False
    assert s.record(C.TOOL_RESERIALIZE, "input_changed", "openai") is False
    assert s.record(C.HISTORY_EDIT, "unavailable", "anthropic") is False
    assert s.record(C.ROUTING_MISS, "model_changed", "openai") is False
    assert s.record(C.NONE, "not_a_real_reason", "openai") is None  # unmapped: not scored


def test_reads_anthropic_live_shape():
    """Exact objects from the 2026-11-06 live probe (claude-haiku-5-5)."""
    from prefixwise.proxy import native_reason_from, native_type_from

    miss = {"id": "msg_x", "diagnostics": {"cache_miss_reason": {"type": "tools_changed",
                                                                 "cache_missed_input_tokens": 9986}}}
    hit = {"id": "msg_y", "diagnostics": None}
    assert native_reason_from(miss) == "tools_changed" and native_type_from(miss) == "cache_miss"
    assert native_reason_from(hit) is None and native_type_from(hit) is None


def test_reads_openai_shapes_and_unavailable():
    from prefixwise.proxy import native_reason_from, native_type_from

    unavailable = {"prompt_cache_diagnostics": {"type": "unavailable"}}  # observed live 2026-11-06
    assert native_reason_from(unavailable) is None and native_type_from(unavailable) == "unavailable"
    miss = {"prompt_cache_diagnostics": {"type": "cache_miss", "reason": "tools_changed"}}  # documented
    assert native_reason_from(miss) == "tools_changed"
