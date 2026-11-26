"""Dollar costs: arithmetic pinned to hand-computed values; unknown rates stay None."""

import json

import pytest

from prefixwise.cli import main
from prefixwise.findings import findings, render
from prefixwise.pricing import PriceTable, breakdown
from prefixwise.proxy import PrefixwiseProxy

T = PriceTable.load()


def test_builtin_table_is_dated_and_sourced():
    assert T.as_of == "2026-11-26"
    official = {"anthropic": "platform.claude.com", "openai": "developers.openai.com"}
    for name, r in T.models.items():
        assert official[r["provider"]] in r["source"], name  # official pages only
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
    # a Claude id the table does not list is priced at the nearest version of its family, and said so
    assert T.resolve("claude-haiku-5")[0] == "claude-haiku-4-5" and T.resolve("claude-haiku-5")[2] is True
    assert T.rates_for("gpt-5.6") is None  # no family rule for OpenAI ids


def test_unknown_is_none_never_zero(tmp_path):
    assert T.request_cost("gpt-5.6", "openai", {"input_tokens": 100, "output_tokens": 1}) is None
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
    assert entry.cost is not None and entry.cost > 0 and entry.price_as_of == "2026-11-26"
    proxy.handle("openai", {"model": "gpt-5.6", "session_id": "o", "messages": []})  # unpriced
    main(["report", "--ledger", path])
    out = capsys.readouterr().out
    assert "Cost" in out and "2 of 3 requests priced" in out
    main(["report", "--ledger", path, "--json"])
    s = json.loads(capsys.readouterr().out)["summary"]
    assert s["priced_requests"] == 2 and s["cost_per_successful_task"] is None


def test_prices_command(capsys):
    assert main(["prices"]) == 0
    assert "claude-haiku-5-5" in capsys.readouterr().out


def test_openai_prices_from_official_page():
    # Live gpt-5.6-luna usage (2026-11-06): 5881 input, 5854 cached, 24 written, 16 output.
    usage = {"input_tokens": 5881, "input_tokens_details": {"cache_write_tokens": 24, "cached_tokens": 5854},
             "output_tokens": 16}
    expected = (3 * 0.20 + 5854 * 0.02 + 24 * 0.25 + 16 * 1.20) / 1e6
    assert T.request_cost("gpt-5.6-luna", "openai", usage) == pytest.approx(expected)
    assert T.rates_for("gpt-5.6") is None  # no plain gpt-5.6 row on the page


def test_promotional_price_expires():
    from datetime import date

    assert T.rates_for("gpt-5.6-sol", today=date(2026, 11, 30)) is not None
    assert T.rates_for("gpt-5.6-sol", today=date(2026, 12, 1)) is None  # unknown after the promotion


def test_matching_does_not_borrow_another_models_price():
    assert T.rates_for("gpt-5.6-sol-pro") is None
    assert T.rates_for("claude-sonnet-5-5-fast") is None
    assert T.rates_for("claude-sonnet-5-5-20261001") is T.models["claude-sonnet-5-5"]
    assert T.rates_for("anthropic.claude-sonnet-4-6-v1:0") is T.models["claude-sonnet-4-6"]


def test_dated_snapshots_are_priced_as_their_model():
    """Anthropic's dated id is the model itself. OpenAI answers an alias under a dated snapshot id; snapshots
    of one alias have had different prices, so it is priced at the alias's rate and said to be approximate."""
    assert T.resolve("claude-haiku-4-5-20251001") == ("claude-haiku-4-5", T.models["claude-haiku-4-5"], False)
    assert T.resolve("gpt-5.6-terra-2026-10-15") == ("gpt-5.6-terra", T.models["gpt-5.6-terra"], True)
    assert T.resolve("openai/gpt-5.6-terra-2026-10-15")[0] == "gpt-5.6-terra"
    for other in ("gpt-5.6-terra-pro-2026-10-15",   # another model, still not borrowed
                  "gpt-5.6-terra-2026-10", "gpt-5.6-terra-0000-00-00", "gpt-5.6-terra-2026-13-01",  # not dates
                  "gpt-5.6-terra-latest"):
        assert T.rates_for(other) is None, other


def test_a_response_with_no_usage_has_no_cost():
    for provider, model, usage in (("openai", "gpt-5.6-terra", {}), ("anthropic", "claude-opus-5-5", {}),
                                   ("openai", "gpt-5.6-terra", {"prompt_tokens": 0, "completion_tokens": 0}),
                                   # no request has an empty prompt: output alone is not a usage report
                                   ("openai", "gpt-5.6-terra", {"prompt_tokens": 0, "completion_tokens": 7})):
        assert T.request_cost(model, provider, usage) is None, usage
    assert T.cost_fields("claude-opus-5-5", "anthropic", {})["cost"] is None
    # a prompt with no output (max_tokens 0, to warm a cache) is a real request
    assert T.request_cost("claude-opus-5-5", "anthropic", {"input_tokens": 1000, "output_tokens": 0}) == pytest.approx(0.004)


def _proxy_answering(model: str, usage: dict, path: str = ":memory:"):
    from prefixwise.proxy import PrefixwiseProxy

    proxy = PrefixwiseProxy(ledger_path=path, upstream_url="http://unused.invalid", api_key="k")
    reply = {"id": "chatcmpl-1", "usage": usage, **({"model": model} if model else {})}
    proxy._forward = lambda *a, **k: (200, dict(reply))
    return proxy


USAGE = {"prompt_tokens": 3000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 0}}
ASK = {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "hi"}]}
SESSION = {"x-prefixwise-session-id": "s"}


def test_the_model_that_answered_is_priced_never_the_one_asked_for():
    """Behind a gateway's fallback routing another model can answer, at another price."""
    for answered in ("gpt-5.6-sol-pro", "llama-3.3-70b"):
        entry = _proxy_answering(answered, USAGE).handle("openai", dict(ASK), SESSION)[1]
        assert entry.cost is None and entry.avoidable_cost is None, answered
    entry = _proxy_answering("gpt-5.6-terra", USAGE).handle("openai", dict(ASK), SESSION)[1]
    assert entry.cost == pytest.approx(T.request_cost("gpt-5.6-terra", "openai", USAGE))   # Terra's rate, not Luna's


def test_no_usage_is_reported_as_no_usage_not_as_a_missing_price(tmp_path, capsys):
    path = str(tmp_path / "l.db")
    proxy = _proxy_answering("gpt-5.6-terra", {}, path)
    for _ in range(3):
        entry = proxy.handle("openai", {**ASK, "model": "gpt-5.6-terra"}, SESSION)[1]
    assert entry.cost is None and entry.avoidable_cost is None
    found = findings(proxy.ledger)
    assert found["unpriced_models"] == [] and found["no_usage_requests"] == 3
    text = render(found)
    assert "No cost for 3 requests whose responses reported no usage." in text and "Not priced" not in text
    assert main(["report", "--ledger", path, "--by", "model"]) == 0
    out = capsys.readouterr().out
    assert "no response reported usage" in out and "price table" not in out


def test_rows_priced_before_a_model_had_its_own_rates_are_named(tmp_path):
    """0.12.7 priced Fable 5.2 as Fable 5.1, 25% high, and said so. Those rows keep their stored cost;
    with 5.2's own row in the table the report must still say how they were priced."""
    from prefixwise.ledger import Ledger, LedgerEntry

    led = Ledger(str(tmp_path / "l.db"))
    row = dict(session_id="s", agent_id="a", provider="anthropic", model="claude-fable-5-2", cause="none",
               prompt_tokens=1_000_000, cached_tokens=0, avoidable_tokens=0, avoidable_cost=0.0)
    led.record(LedgerEntry(**row, cost=15.0, price_as_of="2026-11-09", request_id="old"))
    led.record(LedgerEntry(**row, cost=12.0, price_as_of="2026-11-26", request_id="new"))
    found = findings(led)
    assert found["priced_at_an_earlier_rate"] == [("claude-fable-5-2", 1, "2026-11-26")]
    assert "1 requests on claude-fable-5-2 (recorded before the table of 2026-11-26)" in render(found)


def test_fable_and_mythos_5_2_have_their_own_rows():
    """Read from Anthropic's pricing page on 2026-11-26: $8 in, $40 out, reads at 0.025x input."""
    for model in ("claude-fable-5-2", "claude-mythos-5-2"):
        key, rates, approximate = T.resolve(model)
        assert key == model and approximate is False
        assert (rates["input"], rates["cache_write_5m"], rates["cache_write_1h"], rates["cache_read"],
                rates["output"]) == (8.0, 10.0, 16.0, 0.2, 40.0)
    usage = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 100_000,
             "cache_creation_input_tokens": 2000, "cache_creation": {"ephemeral_1h_input_tokens": 2000}}
    assert T.request_cost("claude-fable-5-2", "anthropic", usage) == pytest.approx(
        (1000 * 8 + 100_000 * 0.2 + 2000 * 16 + 100 * 40) / 1e6)
