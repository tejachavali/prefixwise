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
