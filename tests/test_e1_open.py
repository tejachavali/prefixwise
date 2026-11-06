"""Unit tests for the E1-open harness scoring logic.

No GPU, no server: scoring is tested on mock turn results, and the
mock-mode run is tested end to end through the real proxy, classifier,
and ledger (only the upstream is simulated).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "e1_open"))

from e1_open import (  # noqa: E402
    build_body,
    is_pass,
    retry_upstream_ids,
    score_results,
    simulate_cached_tokens,
)


def _results(pairs):
    return [{"injected": inj, "predicted": pred, "cached_tokens": 0,
             "prompt_tokens": 2000} for inj, pred in pairs]


def test_score_results_perfect():
    pairs = [("volatile_system", "volatile_system"), ("none", "none"),
             ("first_write", "first_write")]
    report = score_results(_results(pairs))
    assert report["total_turns"] == 3
    assert report["agreement"] == 1.0
    assert report["unknown_rate"] == 0.0
    assert report["per_cause"]["volatile_system"]["precision"] == 1.0
    assert is_pass(report)


def test_score_results_unknown_rate_counts_misses_only():
    # UNKNOWN on a NONE turn is not a miss; UNKNOWN on a miss counts.
    pairs = [("none", "unknown"), ("volatile_system", "unknown"),
             ("volatile_system", "volatile_system")]
    report = score_results(_results(pairs))
    assert report["misses"] == 2
    assert report["unknown_rate"] == 0.5
    assert not is_pass(report)


def test_score_results_pass_thresholds():
    # 95% agreement and 10% UNKNOWN is the boundary: 19/20 agree, 1/19 unknown misses.
    pairs = [("volatile_system", "volatile_system")] * 19 + [("volatile_system", "unknown")]
    report = score_results(_results(pairs))
    assert report["agreement"] == 0.95
    assert report["unknown_rate"] == 1 / 20
    assert is_pass(report)
    pairs = [("volatile_system", "volatile_system")] * 18 + [("volatile_system", "unknown")] * 2
    assert not is_pass(score_results(_results(pairs)))


def test_simulate_cached_tokens():
    assert simulate_cached_tokens("first_write") == 0
    assert simulate_cached_tokens("retry_replay") == 0
    assert simulate_cached_tokens("none") > 1000  # cache hit: most of the prefix
    assert simulate_cached_tokens("volatile_system") < simulate_cached_tokens("none")


def test_build_body_mutates_only_the_injected_block():
    from datetime import date

    base = build_body(session_id="s", model="m", injected="none", turn=2, base_date=date(2026, 11, 5))
    volatile = build_body(session_id="s", model="m", injected="volatile_system", turn=3, base_date=date(2026, 11, 5))
    # Same tools, same messages, only the system date differs.
    assert base["tools"] == volatile["tools"]
    assert base["messages"][1] == volatile["messages"][1]
    assert base["messages"][0]["content"] != volatile["messages"][0]["content"]


def test_retry_upstream_ids_are_fresh_per_turn():
    # A fresh upstream session + fresh cache_salt per retry turn, so
    # the vLLM prefix is cold. Different turns get different ids;
    # the same turn is deterministic.
    s1, salt1 = retry_upstream_ids("e1-open-s5", 6)
    s2, salt2 = retry_upstream_ids("e1-open-s5", 7)
    assert s1 != s2 and salt1 != salt2
    assert retry_upstream_ids("e1-open-s5", 6) == (s1, salt1)
    assert "e1-open-s5" in s1  # still traceable to the proxy session


def test_retry_turns_carry_fresh_upstream_session(tmp_path):
    from e1_open import run

    results, report = run(vllm_url="http://127.0.0.1:8000", model="Qwen/Qwen3-4B",
                          model_alias="Qwen/Qwen3-4B-alias", mock=True,
                          output_dir=tmp_path)
    retries = [r for r in results if r["injected"] == "retry_replay"]
    assert len(retries) == 10
    # Every retry turn got its own fresh upstream session id, distinct
    # from the proxy session id and from every other retry turn.
    upstream = [r["upstream_session_id"] for r in retries]
    assert all(u is not None and u.startswith("e1-open-") for u in upstream)
    assert len(set(upstream)) == 10
    assert all(u != r["session_id"] for u, r in zip(upstream, retries))
    # And they still score as retry_replay in mock mode.
    assert all(r["predicted"] == "retry_replay" for r in retries)
    assert is_pass(report)


def test_proxy_pops_upstream_session_override():
    # The harness-only flag must never reach the upstream: the proxy
    # pops it and uses it only as the X-Session-ID for forwarding,
    # while its own snapshot chain keeps the original session.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from prefixwise.proxy import PrefixwiseProxy

    seen: dict = {}

    def _capture(provider, body, request_id, session_id):
        seen["body"] = dict(body)
        seen["session_id"] = session_id
        return {"prompt_tokens": 2000, "cached_tokens": 0}, "resp-1"

    proxy = PrefixwiseProxy()
    proxy._forward = _capture  # type: ignore[method-assign]
    body = {"model": "m", "session_id": "s1",
            "messages": [{"role": "user", "content": "hi"}],
            "prefixwise_upstream_session_id": "s1-retry-6",
            "cache_salt": "s1-retry-6"}
    proxy.handle("vllm", dict(body))
    assert "prefixwise_upstream_session_id" not in seen["body"]
    assert seen["session_id"] == "s1-retry-6"  # upstream sees the fresh session
    assert "s1" in proxy.last_snapshot  # proxy chain keeps the original session


def test_mock_run_passes_100_turns(tmp_path):
    from e1_open import run

    results, report = run(vllm_url="http://127.0.0.1:8000", model="Qwen/Qwen3-8B",
                          model_alias="Qwen/Qwen3-8B-alias", mock=True,
                          output_dir=tmp_path)
    assert len(results) == 100
    assert report["total_turns"] == 100
    assert report["agreement"] == 1.0
    assert report["unknown_rate"] == 0.0
    assert is_pass(report)
    # Every required injected cause is covered.
    causes = {r["injected"] for r in results}
    assert {"first_write", "none", "volatile_system", "tool_reserialize",
            "tool_catalog_change", "compaction_rewrite", "model_switch",
            "retry_replay"} <= causes
    # Results are JSON-serialisable for the report writer.
    json.dumps(results)
