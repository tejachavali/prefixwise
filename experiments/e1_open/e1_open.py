"""E1-open: validate the Prefixwise classifier without API credits.

Runs 100 turns against a vLLM OpenAI-compatible server serving an open
Hugging Face model (default Qwen/Qwen3-8B). Each turn has a KNOWN
injected cause, so the injected cause is the ground truth: the
classifier is scored against what the harness deliberately changed,
not against a provider's native reason (open models have none).

Turns are chained in sessions of 10, so each turn is scored against
the previous turn in the same session, exactly as the classifier is
defined (first divergence in prefix order):

Session A (x5, coding questions):
  1 FIRST_WRITE, 2 NONE, 3 VOLATILE_SYSTEM, 4 VOLATILE_SYSTEM,
  5 TOOL_RESERIALIZE, 6 TOOL_RESERIALIZE, 7 TOOL_CATALOG_CHANGE,
  8 TOOL_CATALOG_CHANGE, 9 SYSTEM_EDIT, 10 SYSTEM_EDIT

Session B (x5, coding questions):
  1 FIRST_WRITE, 2 NONE, 3 COMPACTION_REWRITE, 4 COMPACTION_REWRITE,
  5 NONE, 6 RETRY_REPLAY, 7 RETRY_REPLAY, 8 MODEL_SWITCH,
  9 MODEL_SWITCH, 10 NONE

That is 100 turns: FIRST_WRITE 10, NONE 20, VOLATILE_SYSTEM 10,
TOOL_RESERIALIZE 10, TOOL_CATALOG_CHANGE 10, SYSTEM_EDIT 10,
COMPACTION_REWRITE 10, RETRY_REPLAY 10, MODEL_SWITCH 10.

How each injection is made (all other blocks identical to the
previous turn, so the first divergence is the injected one):
- VOLATILE_SYSTEM: system date changes only (masked hash equal).
- TOOL_RESERIALIZE: tool list reversed, same tool set.
- TOOL_CATALOG_CHANGE: one tool removed, then restored.
- COMPACTION_REWRITE: history replaced by a summary, with
  prefixwise_compaction_event=True (PreCompact snapshot).
- MODEL_SWITCH: model field switches to the served-model alias
  (vllm_server.sh starts vLLM with --served-model-name MODEL
  MODEL-alias). A single-model vLLM server rejects an unknown model
  id, so if the upstream rejects the alias the harness falls back to
  an in-process classification of the same request (marked
  upstream_error=True in the report) instead of dropping the turn.
- RETRY_REPLAY: identical request, sent as a fresh upstream
  session with a fresh cache_salt, so the vLLM prefix is cold
  (cached_tokens == 0) and the classifier can fire RETRY_REPLAY
  (identical blocks, gap < 60 s, cached 0). The proxy session
  stays the original session, so the turn is still compared
  against the previous turn in the same session. Why both:
  POST /reset_prefix_cache returns HTTP 404 on vLLM 0.33.0
  (live run Nov 5-6, 2026, retry_replay 0/10, scored as NONE),
  so it is no longer used. A fresh session_id alone would NOT
  be enough: session_id is event metadata on KV events, not a
  cache partition. cache_salt is the documented mechanism
  (vLLM Automatic Prefix Caching docs): it is mixed into the
  first block's hash, so only requests with the same salt can
  reuse cached blocks. A fresh salt per retry turn therefore
  guarantees a cold prefix. The salt is not part of the app
  request blocks, so the fingerprint/classifier see an identical
  request, exactly as a real retry looks to the app layer.
- NONE: identical request, expect a cache hit.

Pass criteria (same as E1-hosted, adapted): agreement >= 95%,
UNKNOWN <= 10% of misses (misses = not NONE / FIRST_WRITE).

Modes:
- Live (default): --vllm-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B
  The Prefixwise proxy forwards to vLLM with X-Request-Id and
  X-Session-ID set, so the engine join is exercised.
- Mock (--mock): no GPU, no server. The proxy's _forward is replaced
  with a simulator that returns cached_tokens per injected cause,
  so the scoring path (proxy -> classifier -> ledger) runs locally.
  This is a smoke test of the harness, not evidence about vLLM.

Usage:
  python3 experiments/e1_open/e1_open.py --mock
  python3 experiments/e1_open/e1_open.py --vllm-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prefixwise.classifier import MissCause, RequestSnapshot, classify  # noqa: E402
from prefixwise.fingerprint import Fingerprinter  # noqa: E402
from prefixwise.proxy import PrefixwiseProxy, extract_blocks  # noqa: E402

PROMPT_TOKENS_MOCK = 2000

# Long, stable system prompt: ~1,800 tokens of real text, so the
# prefix is worth caching and a volatile date is a small part of it.
_STABLE_PARAGRAPH = (
    "You are Prefixwise, a helpful coding assistant for Python and "
    "inference engineering. Answer concisely, with concrete examples, "
    "and show the exact code or command when one exists. Do not invent "
    "library APIs, model names, or version numbers; if you are unsure, "
    "say so and name what would settle it. Prefer the standard library "
    "over a new dependency, and prefer a small, tested change over a "
    "rewrite. When a question is about caching, name the cache layer, "
    "the key, and the eviction rule before giving an answer. "
)
STABLE_SYSTEM = "You are a helpful coding assistant.\n" + _STABLE_PARAGRAPH * 14

TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "search the web for documentation", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "calc", "description": "calculate an arithmetic expression", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "fetch", "description": "fetch a url and return its text", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}},
]

QUESTIONS = [
    "Explain how prefix caching works in vLLM, in three sentences.",
    "When does a KV cache block get evicted, and what decides it?",
    "What is RadixAttention in SGLang, and how is it different from paged attention?",
    "Why does reordering tool definitions hurt prompt cache hit rate?",
    "How does LMCache share KV blocks between requests?",
]

# Session scripts: (injected cause, mutation) per turn, turn 1 is the baseline.
SESSION_A = ["first_write", "none", "volatile_system", "volatile_system",
             "tool_reserialize", "tool_reserialize", "tool_catalog_change",
             "tool_catalog_change", "system_edit", "system_edit"]
SESSION_B = ["first_write", "none", "compaction_rewrite", "compaction_rewrite",
             "none", "retry_replay", "retry_replay", "model_switch",
             "model_switch", "none"]


def simulate_cached_tokens(injected: str, prompt_tokens: int = PROMPT_TOKENS_MOCK) -> int:
    """Mock vLLM cached_tokens for an injected cause (mock mode only).

    Values are chosen from the mechanism, not measured: a cache hit
    returns most of the prefix, a change early in the prefix (tools)
    returns almost nothing, a volatile system change returns the
    tools block that precedes it, a compaction returns the new
    summary only. The classifier's block-divergence rules fire
    before the cached-token checks for every cause except NONE and
    RETRY_REPLAY, so these values only decide those two.
    """
    fractions = {
        "first_write": 0.0,
        "none": 0.9,
        "volatile_system": 0.2,
        "tool_reserialize": 0.0,
        "tool_catalog_change": 0.0,
        "compaction_rewrite": 0.1,
        "model_switch": 0.0,
        "retry_replay": 0.0,
        "system_edit": 0.2,
    }
    return int(prompt_tokens * fractions.get(injected, 0.0))


def build_body(*, session_id: str, model: str, injected: str, turn: int,
               base_date: date, question: str = QUESTIONS[0]) -> dict:
    """Build the request body for one turn, mutating only the injected block.

    The user question is fixed for a whole session: if it changed every
    turn, every turn would differ in messages and NONE / RETRY_REPLAY
    turns could never be identical to the previous turn.
    """
    # Date: only VOLATILE_SYSTEM turns change the date. Turn 3 uses
    # base_date + 1 day, turn 4 returns to base_date; both are volatile
    # changes only (masked hash stays equal). Every other turn keeps
    # base_date, so a SYSTEM_EDIT turn differs in wording, not date.
    volatile_date = base_date + timedelta(days=1 if (injected == "volatile_system" and turn == 3) else 0)
    system = STABLE_SYSTEM + f"\nToday is {volatile_date.isoformat()}."
    if injected == "system_edit" and turn == 9:
        # Turn 9: real wording change (masked hash differs), not just a date.
        # Turn 10 restores the original wording, which is itself a system edit.
        system = system.replace("You are a helpful coding assistant.",
                                "You are a concise, helpful coding assistant.")
    tools = list(TOOLS)
    if injected == "tool_reserialize" and turn == 5:
        tools = list(reversed(tools))  # turn 6 restores the original order
    if injected == "tool_catalog_change" and turn == 7:
        tools = tools[:2]  # fetch removed; turn 8 restores it
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    if injected == "compaction_rewrite":
        # Each compaction produces a different summary, so turn 4's
        # history differs from turn 3's and is still a rewrite.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"[Summary of prior turns, compaction {turn}] Prefix caching was discussed; the user asked about vLLM eviction."},
            {"role": "user", "content": question},
        ]
    body: dict = {
        "model": model,
        "session_id": session_id,
        "messages": messages,
        "tools": tools,
        "max_tokens": 16,
        "temperature": 0,
    }
    if injected in ("none", "retry_replay") and turn in (5, 6, 7):
        # Session B, turns 5-7: these follow a compaction, so the
        # identical request to repeat is the compacted history from
        # turn 4, not the pre-compaction baseline. Without this the
        # "identical" turn would differ in messages and be classified
        # HISTORY_EDIT instead of NONE / RETRY_REPLAY.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "[Summary of prior turns, compaction 4] Prefix caching was discussed; the user asked about vLLM eviction."},
            {"role": "user", "content": question},
        ]
        # The user question must also match turn 4's, not turn 5/6/7's index.
        messages[-1]["content"] = question
    if injected == "compaction_rewrite":
        body["prefixwise_compaction_event"] = True
    body["messages"] = messages
    return body


def score_results(results: list[dict]) -> dict:
    """Score a list of turn results against their injected causes."""
    total = len(results)
    agreed = sum(1 for r in results if r["predicted"] == r["injected"])
    misses = [r for r in results if r["injected"] not in ("none", "first_write")]
    unknown = sum(1 for r in misses if r["predicted"] == "unknown")
    per_cause: dict[str, dict] = {}
    for r in results:
        c = per_cause.setdefault(r["injected"], {"n": 0, "correct": 0})
        c["n"] += 1
        if r["predicted"] == r["injected"]:
            c["correct"] += 1
    for c in per_cause.values():
        c["precision"] = c["correct"] / c["n"] if c["n"] else None
    agreement = agreed / total if total else None
    unknown_rate = unknown / len(misses) if misses else None
    return {
        "total_turns": total,
        "agreement": agreement,
        "unknown_rate": unknown_rate,
        "misses": len(misses),
        "unknown_misses": unknown,
        "per_cause": per_cause,
        "pass": bool(agreement is not None and agreement >= 0.95
                     and unknown_rate is not None and unknown_rate <= 0.10),
    }


def is_pass(report: dict) -> bool:
    return bool(report["pass"])


def write_report(results: list[dict], report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"e1_open_{stamp}.json"
    md_path = output_dir / f"e1_open_{stamp}.md"
    json_path.write_text(json.dumps({"report": report, "results": results}, indent=2))
    lines = ["# E1-open report", "",
             f"Turns: {report['total_turns']}  |  Agreement: {report['agreement']:.1%}  |  "
             f"UNKNOWN rate (misses): {report['unknown_rate']:.1%}  |  "
             f"Pass: {'YES' if report['pass'] else 'NO'}", "",
             "## Per-cause precision", "",
             "| Injected cause | n | Correct | Precision |",
             "|---|---:|---:|---:|"]
    for cause, c in sorted(report["per_cause"].items()):
        lines.append(f"| {cause} | {c['n']} | {c['correct']} | {c['precision']:.1%} |")
    lines += ["", "## Misses (predicted != injected)", ""]
    for r in results:
        if r["predicted"] != r["injected"]:
            lines.append(f"- turn {r['turn']} session {r['session_id']}: "
                         f"injected {r['injected']}, predicted {r['predicted']}, "
                         f"cached {r['cached_tokens']}/{r['prompt_tokens']}")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def _classify_direct(proxy: PrefixwiseProxy, session_id: str, body: dict,
                     cached_tokens: int = 0, prompt_tokens: int = PROMPT_TOKENS_MOCK) -> str:
    """In-process classification of one request, used only when the
    upstream rejects it (MODEL_SWITCH alias not served). Same
    classifier, same fingerprinting, no duplicate logic."""
    prev = proxy.last_snapshot.get(session_id)
    fps = Fingerprinter().fingerprint_blocks(extract_blocks(body, "vllm"))
    curr = RequestSnapshot(model=str(body.get("model", "unknown")), provider="vllm",
                           blocks=fps, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                           compaction_event=bool(body.get("prefixwise_compaction_event", False)))
    result = classify(prev, curr)
    proxy.last_snapshot[session_id] = curr
    return result.cause.value


def retry_upstream_ids(session_id: str, turn_no: int) -> tuple[str, str]:
    """Fresh upstream identity for a RETRY_REPLAY turn.

    Returns (upstream_session_id, cache_salt): a new session id and
    a new salt per retry turn, so the vLLM prefix is cold. The
    proxy session is unchanged; only the identity sent upstream
    changes. See the RETRY_REPLAY note in the module docstring.
    """
    fresh = f"{session_id}-retry-{turn_no}"
    return fresh, fresh


def run(*, vllm_url: str, model: str, model_alias: str, mock: bool,
        output_dir: Path, max_turns: int | None = None) -> tuple[list[dict], dict]:
    proxy = PrefixwiseProxy(upstream_url=vllm_url, api_key="sk-dummy") if not mock else PrefixwiseProxy()
    # Mock mode: replace the proxy's forwarder with a simulator that
    # returns cached_tokens per injected cause. Real proxy, real
    # classifier, real ledger; only the upstream is simulated.
    state = {"injected": "first_write"}

    if mock:
        def _mock_forward(provider, body, request_id, session_id):
            return ({"prompt_tokens": PROMPT_TOKENS_MOCK,
                     "cached_tokens": simulate_cached_tokens(state["injected"])},
                    f"resp-{request_id[:8]}")
        proxy._forward = _mock_forward  # type: ignore[method-assign]

    results: list[dict] = []
    turn_no = 0
    base_date = date(2026, 11, 5)
    for session_idx, script in enumerate([SESSION_A] * 5 + [SESSION_B] * 5):
        session_id = f"e1-open-s{session_idx}"
        for turn_idx, injected in enumerate(script, start=1):
            turn_no += 1
            if max_turns is not None and turn_no > max_turns:
                break
            model_for_turn = model_alias if injected == "model_switch" and turn_idx in (8,) else model
            # Turn 9 in session B switches back to the original model.
            if injected == "model_switch" and turn_idx == 9:
                model_for_turn = model
            state["injected"] = injected
            body = build_body(session_id=session_id, model=model_for_turn,
                              injected=injected, turn=turn_idx, base_date=base_date,
                              question=QUESTIONS[session_idx % len(QUESTIONS)])
            # RETRY_REPLAY: fresh upstream session + fresh cache_salt,
            # proxy session unchanged (see retry_upstream_ids).
            # No /reset_prefix_cache call: it 404s on vLLM 0.33.0.
            retry_upstream_session_id = None
            if injected == "retry_replay":
                retry_upstream_session_id, retry_salt = retry_upstream_ids(session_id, turn_no)
                body["prefixwise_upstream_session_id"] = retry_upstream_session_id
                body["cache_salt"] = retry_salt
            upstream_error = False
            try:
                result, _entry = proxy.handle("vllm", dict(body))
                predicted = result["cause"]
                usage = result["usage"]
                cached = int(usage.get("cached_tokens", 0) or 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            except Exception as exc:  # upstream rejected the model alias, etc.
                upstream_error = True
                predicted = _classify_direct(proxy, session_id, body)
                cached, prompt_tokens = 0, PROMPT_TOKENS_MOCK
                print(f"turn {turn_no}: upstream error ({exc}), classified in-process as {predicted}")
            results.append({
                "turn": turn_no, "session_id": session_id, "injected": injected,
                "predicted": predicted, "cached_tokens": cached,
                "prompt_tokens": prompt_tokens,
                "agreement": predicted == injected,
                "unknown": predicted == "unknown",
                "upstream_error": upstream_error,
                "upstream_session_id": retry_upstream_session_id,
            })
    report = score_results(results)
    return results, report


def main() -> int:
    ap = argparse.ArgumentParser(description="E1-open: classifier validation on open models via vLLM.")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-alias", default="Qwen/Qwen3-8B-alias")
    ap.add_argument("--mock", action="store_true", help="simulate vLLM, no GPU/server needed")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--max-turns", type=int, default=None)
    args = ap.parse_args()

    if args.mock:
        print("E1-open in MOCK mode: simulated vLLM cached_tokens, no GPU needed.")
    else:
        print(f"E1-open live: vLLM at {args.vllm_url}, model {args.model} (alias {args.model_alias})")
    results, report = run(vllm_url=args.vllm_url, model=args.model,
                          model_alias=args.model_alias, mock=args.mock,
                          output_dir=Path(args.output_dir), max_turns=args.max_turns)
    json_path, md_path = write_report(results, report, Path(args.output_dir))
    print(f"Turns: {report['total_turns']}  Agreement: {report['agreement']:.1%}  "
          f"UNKNOWN rate: {report['unknown_rate']:.1%}  Pass: {'YES' if report['pass'] else 'NO'}")
    for cause, c in sorted(report["per_cause"].items()):
        print(f"  {cause}: {c['correct']}/{c['n']} = {c['precision']:.1%}")
    print(f"Report: {json_path}\n        {md_path}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
