from prefixwise.ledger import Ledger, LedgerEntry, avoidable_tokens, reuse_adjusted_reads


def entry(**kw):
    d = dict(session_id="s1", agent_id="a1", provider="openai", model="gpt-5.6",
             cause="volatile_system", prompt_tokens=1000, cached_tokens=200,
             avoidable_tokens=800, avoidable_cost=0.01, task_success=True)
    d.update(kw)
    return LedgerEntry(**d)


def test_avoidable_tokens():
    assert avoidable_tokens(1000, 200) == 800
    assert avoidable_tokens(1000, 200, first_write_tokens=800) == 0


def test_tokens_per_successful_task():
    led = Ledger()
    led.record(entry(prompt_tokens=1000))
    led.record(entry(prompt_tokens=500))
    led.record(entry(prompt_tokens=999, task_success=False))
    assert led.tokens_per_successful_task() == 750.0
    assert led.tokens_per_successful_task("nope") is None


def test_rollup_by_cause():
    led = Ledger()
    led.record(entry(cause="volatile_system", avoidable_tokens=800))
    led.record(entry(cause="tool_reserialize", avoidable_tokens=100))
    by_cause = {r["cause"]: r for r in led.rollup_by_cause()}
    assert by_cause["volatile_system"]["avoidable_tokens"] == 800


def test_reuse_adjusted_reads_is_a_model():
    # 1.25 / 0.9 ~= 1.4, 2.5 / 0.9 ~= 2.8 (model, not a verified fact).
    assert abs(reuse_adjusted_reads(1.25, 0.1) - 1.388) < 0.01
    assert abs(reuse_adjusted_reads(2.5, 0.1) - 2.777) < 0.01
