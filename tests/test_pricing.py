"""Dollar costs: arithmetic pinned to hand-computed values; unknown rates stay None."""

import json

import pytest

from prefixwise.cli import main
from prefixwise.pricing import PriceTable, breakdown
from prefixwise.proxy import PrefixwiseProxy

T = PriceTable.load()


def test_builtin_table_is_dated_and_sourced():
    assert T.as_of == "2026-11-07"
    for name, r in T.models.items():
        assert "platform.claude.com" in r["source"], name  # official page only
        assert all(isinstance(r[k], (int, float)) for k in ("input", "output", "cache_read")), name


def test_anthropic_request_cost_by_hand():
    usage = {"input_tokens": 50, "cache_read_input_tokens": 1900, "cache_creation_input_tokens": 50,
             "cache_creation": {"ephemeral_5m_input_tokens": 50, "ephemeral_1h_input_tokens": 0},
             "output_tokens": 16}
    # Haiku 5.5: 50 x 0.10 + 1900 x 0.01 + 50 x 0.125 + 16 x 0.50 = 38.25 per 1M
    assert T.request_cost("claude-haiku-5-5", "anthropic", usage) == pytest.approx(38.25e-6)


def test_one_hour_writes_and_long_prompt_tier():
    usage = {"input_tokens": 10_000, "cache_read_input_tokens": 110_000, "cache_creation_input_tokens": 1_000,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1_000},
             "output_tokens": 100}
    # 121,000-token prompt > 100,000: whole request at the long-prompt rates.
    expected = (10_000 * 0.50 + 110_000 * 0.05 + 1_000 * 1.00 + 100 * 2.50) / 1e6
    assert T.request_cost("claude-haiku-5-5", "anthropic", usage) == pytest.approx(expected)


def test_model_ids_match_snapshots_and_cloud_prefixes_but_not_other_models():
    assert T.rates_for("claude-haiku-5-5-20261007") is T.models["claude-haiku-5-5"]
    assert T.rates_for("us.anthropic.claude-sonnet-5-5-v1:0") is T.models["claude-sonnet-5-5"]
    assert T.rates_for("claude-haiku-5") is None and T.rates_for("gpt-5.6-luna") is None


def test_unknown_is_none_never_zero(tmp_path):
    assert T.request_cost("gpt-5.6-luna", "openai", {"input_tokens": 100, "output_tokens": 1}) is None
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"models": {"my-model": {"input": 1.0, "cache_read": 0.1, "output": 2.0}}}))
    t = PriceTable.load(str(path))
    usage = {"prompt_tokens": 1000, "completion_tokens": 10,
             "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 400}}
    assert t.request_cost("my-model", "openai", usage) is None  # writes present, no write rate
    path.write_text(json.dumps({"models": {"my-model": {"input": 1.0, "cache_read": 0.1, "output": 2.0,
                                                        "cache_write": "input"}}}))
    t = PriceTable.load(str(path))
    assert t.request_cost("my-model", "openai", usage) == pytest.approx((400 * 1.0 + 600 * 0.1 + 10 * 2.0) / 1e6)


def test_user_file_overrides_and_validates(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"models": {"claude-haiku-5-5": {"input": 0.05, "cache_read": 0.005, "output": 0.25}}}))
    assert PriceTable.load(str(path)).rates_for("claude-haiku-5-5")["input"] == 0.05
    path.write_text(json.dumps({"models": {"x": {"input": -1}}}))
    with pytest.raises(ValueError):
        PriceTable.load(str(path))


def test_avoidable_cost_is_the_cache_discount_forgone():
    # Sonnet 5.5: 10,000 avoidable tokens x (2.00 - 0.10) = $0.019
    assert T.avoidable_cost("claude-sonnet-5-5", 10_000, 12_000) == pytest.approx(0.019)
    assert T.avoidable_cost("unknown-model", 10_000, 12_000) is None


def test_openai_breakdown_reads_responses_and_chat_shapes():
    b = breakdown("openai", {"input_tokens": 5881, "input_tokens_details": {"cache_write_tokens": 24,
                                                                           "cached_tokens": 5854},
                             "output_tokens": 16})  # live gpt-5.6-luna usage, 2026-11-06
    assert (b.uncached_input, b.cache_read, b.cache_write, b.output) == (3, 5854, 24, 16)


def test_proxy_records_cost_and_report_shows_dollars(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    proxy = PrefixwiseProxy(ledger_path=path, mock=True)
    body = {"model": "claude-haiku-5-5", "session_id": "s", "messages": [{"role": "user", "content": "hi"}],
            "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}
    proxy.handle("anthropic", dict(body))
    _, entry = proxy.handle("anthropic", dict(body))
    assert entry.cost is not None and entry.cost > 0 and entry.price_as_of == "2026-11-07"
    proxy.handle("openai", {"model": "gpt-5.6-luna", "session_id": "o", "messages": []})
    main(["report", "--ledger", path])
    out = capsys.readouterr().out
    assert "Cost" in out and "2 of 3 requests priced" in out
    main(["report", "--ledger", path, "--json"])
    s = json.loads(capsys.readouterr().out)["summary"]
    assert s["priced_requests"] == 2 and s["cost_per_successful_task"] is None


def test_prices_command(capsys):
    assert main(["prices"]) == 0
    assert "claude-haiku-5-5" in capsys.readouterr().out
