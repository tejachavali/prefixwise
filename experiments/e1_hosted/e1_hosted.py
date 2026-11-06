"""E1-hosted: Prefixwise against the providers' own cache diagnostics.

E1-open proved the classifier names injected causes on live vLLM. E1-hosted
asks the provider-facing question: on OpenAI (Responses API, with
prompt_cache_options.comparison_response_id chained) and Anthropic
(Messages API, with diagnostics.previous_message_id chained), does
Prefixwise's cause agree with the provider's own miss reason?

Each turn has a KNOWN injected cause, so it is scored twice:
- vs injected ground truth (as in E1-open), and
- vs the provider's native reason, via diagnostics.AgreementScorer.
Pass (E1-hosted target): native agreement >= 95% where a native reason
exists, and UNKNOWN <= 10% of misses.

Sessions of 10 turns, state carried forward (a mutation stays in place
for the following turns, as a real harness would leave it):

  A: first_write, none, volatile_system, none, tool_reserialize, none,
     tool_catalog_change, none, system_edit, none
  B: first_write, none, param_rendered, none, compaction_rewrite, none,
     model_switch, model_switch (back), none, none

"none" turns append one exchange. RETRY_REPLAY and TTL_EXPIRED are not
injected: a hosted cache cannot be made cold on demand, and a TTL test
means waiting 5+ minutes per turn.

Modes (cheapest first):
  --mock    no network, no keys: simulated usage and native reasons in the
            shapes Prefixwise assumes. Harness smoke test only.
  --probe   2 real turns per provider; prints where usage and diagnostics
            actually appear in the response. Run this before a full run:
            if the providers put diagnostics somewhere Prefixwise does not
            read, the full run would measure nothing.
  (default) the full run: --repeats sessions-pairs per provider
            (default 5 -> 100 turns per provider).

Keys: OPENAI_API_KEY / ANTHROPIC_API_KEY, or none at all when a network
proxy injects them per host (this repo's cloud sessions do). Every
turn's raw usage and raw diagnostics objects are saved, so the result can
be audited without re-spending.

Usage:
  python experiments/e1_hosted/e1_hosted.py --mock
  python experiments/e1_hosted/e1_hosted.py --probe --providers anthropic
  python experiments/e1_hosted/e1_hosted.py --providers anthropic,openai --estimate
  python experiments/e1_hosted/e1_hosted.py --providers anthropic --repeats 5 --max-input-tokens 800000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prefixwise.diagnostics import AgreementScorer, map_native_reason  # noqa: E402
from prefixwise.classifier import MissCause  # noqa: E402
from prefixwise.proxy import PrefixwiseProxy, native_reason_from, normalize_usage  # noqa: E402

UPSTREAMS = {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"}
# Cheapest models that should carry native diagnostics. OpenAI's GPT-5.6
# is a family (Sol / Terra / Luna); the exact API ids are not verified
# here: run --probe first, which stops at the first 404 for a wrong id.
DEFAULT_MODELS = {"anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5"),
                  "openai": ("gpt-5.6-luna", None)}
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

SESSION_A = ["first_write", "none", "volatile_system", "none", "tool_reserialize", "none",
             "tool_catalog_change", "none", "system_edit", "none"]
SESSION_B = ["first_write", "none", "param_rendered", "none", "compaction_rewrite", "none",
             "model_switch", "model_switch", "none", "none"]

# ~4,700 tokens of stable instructions: above every provider's minimum
# cacheable prefix (Anthropic is up to 4,096 on some models), so a miss
# is never BELOW_MIN_PREFIX by accident.
_PARAGRAPH = (
    "You are a careful coding agent working in a user's repository. Read before you edit, "
    "keep changes small and tested, explain the reason for each change in one sentence, "
    "never invent APIs or versions, prefer the standard library, and stop to ask when a "
    "request is ambiguous. When you run tools, quote the exact command and summarise its "
    "output; when a test fails, find the root cause before changing code. "
)
STABLE_INSTRUCTIONS = _PARAGRAPH * 70
EST_CHARS_PER_TOKEN = 4

_TOOL_SPECS = [
    ("read_file", "Read a file from the repository", {"path": {"type": "string"}}),
    ("run_tests", "Run the test suite and return the summary", {"pattern": {"type": "string"}}),
    ("search_code", "Search the repository for a string", {"query": {"type": "string"}}),
]


def tool_defs(provider: str) -> list[dict]:
    out = []
    for name, desc, props in _TOOL_SPECS:
        schema = {"type": "object", "properties": props}
        if provider == "anthropic":
            out.append({"name": name, "description": desc, "input_schema": schema})
        else:  # Responses API: flat function tools
            out.append({"type": "function", "name": name, "description": desc, "parameters": schema})
    return out


class SessionState:
    """The request a harness would send, mutated in place turn by turn."""

    def __init__(self, provider: str, model: str, session_id: str, base_date: date):
        self.provider, self.model, self.session_id = provider, model, session_id
        self.date = base_date
        self.wording = "You are a coding agent."
        self.tools = tool_defs(provider)
        self.history: list[dict] = []
        self.tool_choice_any = False
        self.verbosity = "low"
        self.n = 0

    def system_text(self) -> str:
        return f"{self.wording}\n{STABLE_INSTRUCTIONS}\nToday is {self.date.isoformat()}."

    def apply(self, injected: str, turn: int, alt_model: str | None, base_model: str) -> dict:
        """Mutate state for this turn's injected cause; return extra request headers."""
        headers: dict[str, str] = {}
        if injected == "volatile_system":
            self.date += timedelta(days=1)
        elif injected == "tool_reserialize":
            self.tools = list(reversed(self.tools))
        elif injected == "tool_catalog_change":
            self.tools = self.tools[:-1]
        elif injected == "system_edit":
            self.wording = "You are a concise, careful coding agent."
        elif injected == "param_rendered":
            if self.provider == "anthropic":
                self.tool_choice_any = True
            else:
                self.verbosity = "medium"
        elif injected == "compaction_rewrite":
            self.history = [{"role": "user", "content": f"[Summary of turns 1-{turn - 1}: the user asked "
                                                         "for a failing test to be fixed; tests were run.]"},
                            {"role": "assistant", "content": "Understood, continuing from the summary."}]
            headers["X-Prefixwise-Compaction-Event"] = "1"
        elif injected == "model_switch":
            self.model = alt_model if self.model == base_model else base_model
        return headers

    def next_user_turn(self) -> None:
        self.n += 1
        if self.history:
            self.history.append({"role": "assistant", "content": f"Step {self.n - 1} done."})
        self.history.append({"role": "user", "content": f"Continue with step {self.n} of the fix."})

    def body(self) -> dict:
        if self.provider == "anthropic":
            cc = {"type": "ephemeral"}
            msgs = copy.deepcopy(self.history)
            last = msgs[-1]
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": cc}]
            body = {"model": self.model, "max_tokens": 16,
                    "system": [{"type": "text", "text": self.system_text(), "cache_control": cc}],
                    "tools": self.tools, "messages": msgs}
            body["tool_choice"] = {"type": "any"} if self.tool_choice_any else {"type": "auto"}
            return body
        return {"model": self.model, "instructions": self.system_text(), "tools": self.tools,
                "input": copy.deepcopy(self.history), "max_output_tokens": 16,
                "text": {"verbosity": self.verbosity},
                "prompt_cache_key": f"prefixwise-e1-{self.session_id}"}

    def est_input_tokens(self) -> int:
        return len(json.dumps(self.body())) // EST_CHARS_PER_TOKEN


def plan(provider: str, repeats: int, alt_model: str | None) -> list[tuple[int, list[str]]]:
    scripts = []
    for r in range(repeats):
        scripts.append((2 * r, SESSION_A))
        b = SESSION_B if alt_model else [("none" if c == "model_switch" else c) for c in SESSION_B]
        scripts.append((2 * r + 1, b))
    return scripts


# -- mock upstream: the shapes Prefixwise assumes, for a free smoke test ----------------

_MOCK_NATIVE = {
    "anthropic": {"volatile_system": "system_changed", "system_edit": "system_changed",
                  "tool_reserialize": "tools_changed", "tool_catalog_change": "tools_changed",
                  "param_rendered": "unavailable", "compaction_rewrite": "messages_changed",
                  "model_switch": "model_changed"},
    "openai": {"volatile_system": "input_changed", "system_edit": "input_changed",
               "tool_reserialize": "tools_changed", "tool_catalog_change": "tools_changed",
               "param_rendered": "verbosity_changed", "compaction_rewrite": "input_changed",
               "model_switch": "model_changed"},
}


def mock_forward_factory(provider: str, state: dict):
    def forward(prov, body, request_id, session_id, client_headers, api="chat"):
        injected, tokens = state["injected"], state["tokens"]
        hit = injected == "none"
        cached = int(tokens * 0.9) if hit else 0
        reason = None if hit or injected == "first_write" else _MOCK_NATIVE[provider][injected]
        diag = None if injected == "first_write" else {"type": "cache_hit" if hit else "cache_miss",
                                                      "reason": reason}
        if provider == "anthropic":
            payload = {"id": f"msg_{request_id[:8]}",
                       "usage": {"input_tokens": tokens - cached, "cache_read_input_tokens": cached,
                                 "cache_creation_input_tokens": 0, "output_tokens": 1}}
            if diag:
                payload["diagnostics"] = diag
        else:
            payload = {"id": f"resp_{request_id[:8]}",
                       "usage": {"input_tokens": tokens, "input_tokens_details": {"cached_tokens": cached},
                                 "output_tokens": 1}}
            if diag:
                payload["prompt_cache_diagnostics"] = diag
        return 200, payload
    return forward


# -- runner ----------------------------------------------------------------------------

def make_proxy(provider: str, mock: bool) -> PrefixwiseProxy:
    if mock:
        return PrefixwiseProxy(mock=True)
    return PrefixwiseProxy(upstream_url=UPSTREAMS[provider],
                           api_key=os.environ.get(KEY_ENV[provider]) or None, mock=False)


def run_provider(provider: str, *, model: str, alt_model: str | None, repeats: int, mock: bool,
                 max_input_tokens: int | None, probe: bool = False) -> tuple[list[dict], dict]:
    proxy = make_proxy(provider, mock)
    state = {"injected": "first_write", "tokens": 0}
    if mock:
        proxy._forward = mock_forward_factory(provider, state)  # type: ignore[method-assign]
    api = "chat" if provider == "anthropic" else "responses"
    scorer = AgreementScorer()
    results, spent = [], 0
    scripts = [(0, SESSION_A[:2])] if probe else plan(provider, repeats, alt_model)
    base_date = date(2026, 11, 6)
    stamp = time.strftime("%H%M%S")
    for idx, script in scripts:
        sess = SessionState(provider, model, f"{provider}-{stamp}-s{idx}", base_date)
        for turn, injected in enumerate(script, start=1):
            extra = sess.apply(injected, turn, alt_model, model)
            sess.next_user_turn()
            body = sess.body()
            est = sess.est_input_tokens()
            if max_input_tokens is not None and spent + est > max_input_tokens:
                print(f"budget: stopping before {provider} session {idx} turn {turn} "
                      f"({spent:,} + ~{est:,} > {max_input_tokens:,} input tokens)")
                return results, score(results, scorer)
            state.update(injected=injected, tokens=est)
            headers = {"X-Prefixwise-Session-Id": sess.session_id, **extra}
            result, _ = proxy.handle(provider, body, headers, api=api)
            payload = result["response"]
            if result["status"] >= 400:
                print(f"{provider} s{idx} t{turn}: HTTP {result['status']}: {json.dumps(payload)[:300]}")
                results.append({"provider": provider, "session": idx, "turn": turn, "injected": injected,
                                "error": payload, "status": result["status"]})
                if result["status"] in (401, 403, 404):
                    return results, score(results, scorer)  # key, access or model problem: stop spending
                continue
            prompt, cached = normalize_usage(provider, payload.get("usage") or {})
            spent += prompt
            native = native_reason_from(payload)
            agree_native = scorer.record(MissCause(result["cause"]), native, provider)
            row = {"provider": provider, "session": idx, "turn": turn, "model": sess.model,
                   "injected": injected, "predicted": result["cause"], "native_reason": native,
                   "native_mapped": (map_native_reason(provider, native).value
                                     if map_native_reason(provider, native) else None),
                   "agree_injected": result["cause"] == injected, "agree_native": agree_native,
                   "prompt_tokens": prompt, "cached_tokens": cached,
                   "raw_usage": payload.get("usage"),
                   "raw_diagnostics": {k: payload.get(k) for k in ("diagnostics", "prompt_cache_diagnostics")
                                       if payload.get(k) is not None},
                   "response_keys": sorted(payload.keys())}
            results.append(row)
            if probe:
                print(json.dumps({k: row[k] for k in ("turn", "injected", "predicted", "native_reason",
                                                      "prompt_tokens", "cached_tokens", "raw_usage",
                                                      "raw_diagnostics", "response_keys")}, indent=2))
    return results, score(results, scorer)


def score(results: list[dict], scorer: AgreementScorer) -> dict:
    ok = [r for r in results if "error" not in r]
    misses = [r for r in ok if r["injected"] not in ("none", "first_write")]
    per: dict[str, dict] = {}
    for r in ok:
        c = per.setdefault(r["injected"], {"n": 0, "agree_injected": 0, "with_native": 0, "agree_native": 0})
        c["n"] += 1
        c["agree_injected"] += r["agree_injected"]
        if r["agree_native"] is not None:
            c["with_native"] += 1
            c["agree_native"] += bool(r["agree_native"])
    unknown = sum(1 for r in misses if r["predicted"] == "unknown")
    native_rate = scorer.agreement_rate
    unknown_rate = unknown / len(misses) if misses else None
    return {"turns": len(ok), "errors": len(results) - len(ok),
            "agreement_injected": (sum(r["agree_injected"] for r in ok) / len(ok)) if ok else None,
            "native_turns": scorer.total_with_native, "agreement_native": native_rate,
            "unknown_rate": unknown_rate,
            "input_tokens": sum(r["prompt_tokens"] for r in ok),
            "cached_tokens": sum(r["cached_tokens"] for r in ok),
            "per_cause": per,
            "pass": bool(native_rate is not None and native_rate >= 0.95
                         and unknown_rate is not None and unknown_rate <= 0.10)}


def estimate(providers: list[str], repeats: int, models: dict) -> dict:
    """Token estimate for a full run, from the exact request bodies the run would send."""
    out = {}
    for p in providers:
        model, alt = models[p]
        total = writes = 0
        for idx, script in plan(p, repeats, alt):
            sess = SessionState(p, model, f"est-{idx}", date(2026, 11, 6))
            for turn, injected in enumerate(script, start=1):
                sess.apply(injected, turn, alt, model)
                sess.next_user_turn()
                est = sess.est_input_tokens()
                total += est
                if injected != "none":
                    writes += est  # a miss re-processes (and on Anthropic re-writes) the prefix
        out[p] = {"turns": sum(len(s) for _, s in plan(p, repeats, alt)), "input_tokens": total,
                  "uncached_tokens_if_misses_miss": writes, "cached_tokens_if_hits_hit": total - writes,
                  "output_tokens_max": 16 * sum(len(s) for _, s in plan(p, repeats, alt))}
    return out


def write_report(provider_results: dict, out_dir: Path, mode: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"e1_hosted_{mode}_{stamp}.json"
    path.write_text(json.dumps({"mode": mode, **provider_results}, indent=2))
    md = [f"# E1-hosted ({mode})", ""]
    for p, (rows, rep) in provider_results.items():
        fmt = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
        md += [f"## {p}", "",
               f"Turns {rep['turns']} (errors {rep['errors']})  |  native agreement {fmt(rep['agreement_native'])}"
               f" over {rep['native_turns']} turns with a native reason  |  injected agreement "
               f"{fmt(rep['agreement_injected'])}  |  UNKNOWN {fmt(rep['unknown_rate'])}  |  "
               f"Pass: {'YES' if rep['pass'] else 'NO'}",
               f"Input tokens {rep['input_tokens']:,} (cached {rep['cached_tokens']:,})", "",
               "| Injected | n | vs injected | with native | vs native |", "|---|---:|---:|---:|---:|"]
        for c, v in sorted(rep["per_cause"].items()):
            md.append(f"| {c} | {v['n']} | {v['agree_injected']} | {v['with_native']} | {v['agree_native']} |")
        md += ["", "Disagreements:", ""]
        for r in rows:
            if "error" in r:
                md.append(f"- s{r['session']} t{r['turn']} {r['injected']}: HTTP {r['status']}")
            elif not r["agree_injected"] or r["agree_native"] is False:
                md.append(f"- s{r['session']} t{r['turn']} injected {r['injected']}, predicted {r['predicted']}, "
                          f"native {r['native_reason']}, cached {r['cached_tokens']}/{r['prompt_tokens']}")
        md.append("")
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1-hosted: Prefixwise vs provider-native cache diagnostics.")
    ap.add_argument("--providers", default="anthropic,openai")
    ap.add_argument("--repeats", type=int, default=5, help="session pairs per provider (10 turns each)")
    ap.add_argument("--mock", action="store_true", help="no network: harness smoke test")
    ap.add_argument("--probe", action="store_true", help="2 real turns per provider; print response shapes")
    ap.add_argument("--estimate", action="store_true", help="print token estimate for a full run and exit")
    ap.add_argument("--max-input-tokens", type=int, default=None, help="stop before exceeding this budget")
    for p in ("anthropic", "openai"):
        ap.add_argument(f"--{p}-model", default=DEFAULT_MODELS[p][0])
        ap.add_argument(f"--{p}-alt-model", default=DEFAULT_MODELS[p][1],
                        help="second model for MODEL_SWITCH turns (omit to skip them)")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args(argv)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    models = {p: (getattr(args, f"{p}_model"), getattr(args, f"{p}_alt_model")) for p in providers}
    if args.estimate:
        print(json.dumps(estimate(providers, args.repeats, models), indent=2))
        return 0
    mode = "mock" if args.mock else "probe" if args.probe else "live"
    all_results = {}
    for p in providers:
        model, alt = models[p]
        print(f"== {p} ({mode}): model {model}" + (f", alt {alt}" if alt else ", no MODEL_SWITCH"))
        rows, rep = run_provider(p, model=model, alt_model=alt, repeats=args.repeats, mock=args.mock,
                                 max_input_tokens=args.max_input_tokens, probe=args.probe)
        all_results[p] = (rows, rep)
        fmt = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
        print(f"   turns {rep['turns']} errors {rep['errors']} | native agreement {fmt(rep['agreement_native'])}"
              f" ({rep['native_turns']} turns) | injected agreement {fmt(rep['agreement_injected'])}"
              f" | UNKNOWN {fmt(rep['unknown_rate'])} | input tokens {rep['input_tokens']:,}"
              f" | Pass {'YES' if rep['pass'] else 'NO'}")
    path = write_report(all_results, Path(args.output_dir), mode)
    print(f"Report: {path}\n        {path.with_suffix('.md')}")
    if args.probe:
        return 0
    return 0 if all(rep["pass"] for _, rep in all_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
