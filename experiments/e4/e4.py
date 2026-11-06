"""E4: does applying a fix save tokens without hurting task success?

Replays recorded agent sessions twice through the Prefixwise proxy:
  arm OFF: requests as recorded (the bug in place)
  arm ON:  the same requests with a proxy fix applied (canonical_tools)

The recorded sessions have a common real-world bug: the agent rebuilds
its tool list between turns (plugin reload, dict rebuild), so tool order
and key order drift. Tools come first in every provider's prefix, so each
drift misses the whole cache. Each turn asks for a task that needs one
specific tool; task success = the model called that tool.

Metrics (README build order, E4):
  predicted  = avoidable tokens the OFF arm's ledger attributes to
               tool_reserialize (what `prefixwise fixes` would promise)
  realised   = cached tokens ON - cached tokens OFF, same turns
  Pass: >= 50 requests per arm, realised >= 50% of predicted, and task
  success flat (ON success >= OFF success - 5 points).

Arms are isolated so one cannot warm the other's cache: vLLM by a
per-arm cache_salt; Anthropic by waiting out the 5-minute TTL between
arms (--arm-gap).

Modes:
  --record   write a fresh recorded-sessions file (deterministic, --seed)
  --mock     no network: a prefix-cache simulator (cached = longest
             shared prefix with an earlier request in the same arm),
             always-correct tool calls. Harness smoke test only.
  live       --provider vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
             --provider anthropic --model claude-haiku-5-5

Usage:
  python experiments/e4/e4.py --mock
  python experiments/e4/e4.py --provider vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B
  python experiments/e4/e4.py --provider anthropic --model claude-haiku-5-5 --max-input-tokens 900000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prefixwise.proxy import PrefixwiseProxy  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SESSIONS = HERE / "sessions" / "tool-order-drift.jsonl"
UPSTREAMS = {"anthropic": "https://api.anthropic.com"}

TOOLS = {
    "read_file": ("Read a file from the repository and return its contents.",
                  {"path": {"type": "string", "description": "Repository-relative file path"}}),
    "run_tests": ("Run the project's test suite and return the summary.",
                  {"pattern": {"type": "string", "description": "Optional test name filter"}}),
    "search_code": ("Search the repository source code for a symbol or string.",
                    {"query": {"type": "string", "description": "Symbol or text to find"}}),
    "calculator": ("Evaluate an arithmetic expression exactly.",
                   {"expression": {"type": "string", "description": "e.g. 1234 * 5678"}}),
    "git_log": ("Show recent git commits.",
                {"count": {"type": "integer", "description": "How many commits"}}),
    "web_search": ("Search the web for up-to-date information.",
                   {"query": {"type": "string", "description": "Search query"}}),
}
# The harness's "natural" order: deliberately not alphabetical, so the fix
# changes something even on turns without drift.
BASE_ORDER = ["search_code", "run_tests", "read_file", "calculator", "git_log", "web_search"]
TASKS = [
    ("Show me the contents of src/app/config.py.", "read_file"),
    ("Run the test suite and tell me what fails.", "run_tests"),
    ("Where is the function parse_settings defined in this codebase?", "search_code"),
    ("What is 48213 * 9127? I need the exact number.", "calculator"),
    ("Show me the last 5 commits.", "git_log"),
    ("What is the latest stable release of the Rust compiler? Look it up.", "web_search"),
]

_PARAGRAPH = (
    "You are a coding agent working in the user's repository. Use the provided tools whenever a "
    "request needs information you do not have: read files instead of guessing their contents, run "
    "tests instead of predicting results, search the code instead of assuming where things are, use "
    "the calculator for arithmetic, check git history with git_log, and search the web for anything "
    "that changes over time. Call exactly one tool when one is needed. "
)
SYSTEM = _PARAGRAPH * 24


# -- recorded sessions -----------------------------------------------------------------

def record(path: Path, sessions: int, turns: int, p_reload: float, seed: int) -> list[dict]:
    """Deterministic recording: per turn, whether the agent 'reloaded' its
    tools (new order and new key order) and which task it asked."""
    rng = random.Random(seed)
    specs = []
    for s in range(sessions):
        order, key_seed = list(BASE_ORDER), None
        for t in range(1, turns + 1):
            reloaded = t > 1 and rng.random() < p_reload
            if reloaded:
                order = rng.sample(BASE_ORDER, len(BASE_ORDER))
                key_seed = rng.randrange(1_000_000)
            task_idx = (s + t - 1) % len(TASKS)
            specs.append({"session": s, "turn": t, "tool_order": list(order), "key_seed": key_seed,
                          "reloaded": reloaded, "task": task_idx, "expected_tool": TASKS[task_idx][1]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in specs))
    return specs


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _keyed(d: dict, rng: random.Random | None) -> dict:
    """The same dict with its keys in a (seeded) shuffled order: a rebuilt
    definition serializes differently even when nothing changed."""
    if rng is None:
        return d
    keys = list(d)
    rng.shuffle(keys)
    return {k: d[k] for k in keys}


def render_tools(provider: str, order: list[str], key_seed: int | None) -> list[dict]:
    rng = random.Random(key_seed) if key_seed is not None else None
    out = []
    for name in order:
        desc, props = TOOLS[name]
        schema = _keyed({"type": "object", "properties": props}, rng)
        if provider == "anthropic":
            out.append(_keyed({"name": name, "description": desc, "input_schema": schema}, rng))
        else:
            fn = _keyed({"name": name, "description": desc, "parameters": schema}, rng)
            out.append({"type": "function", "function": fn})
    return out


def render(provider: str, model: str, spec: dict, history: list[dict]) -> dict:
    question = TASKS[spec["task"]][0]
    tools = render_tools(provider, spec["tool_order"], spec["key_seed"])
    if provider == "anthropic":
        cc = {"type": "ephemeral"}
        msgs = copy.deepcopy(history) + [{"role": "user", "content": [{"type": "text", "text": question,
                                                                       "cache_control": cc}]}]
        return {"model": model, "max_tokens": 512, "tools": tools, "tool_choice": {"type": "auto"},
                "system": [{"type": "text", "text": SYSTEM, "cache_control": cc}], "messages": msgs}
    msgs = [{"role": "system", "content": SYSTEM}] + copy.deepcopy(history) + [{"role": "user", "content": question}]
    body = {"model": model, "max_tokens": 256, "temperature": 0, "tools": tools, "tool_choice": "auto",
            "messages": msgs}
    if provider == "vllm":
        body["chat_template_kwargs"] = {"enable_thinking": False}  # Qwen3: answer, don't think
    return body


def tool_called(provider: str, payload: dict) -> str | None:
    if provider == "anthropic":
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block.get("name")
        return None
    try:
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        return calls[0]["function"]["name"] if calls else None
    except (KeyError, IndexError, TypeError):
        return None


# -- mock upstream: a prefix-cache simulator ---------------------------------------------

def _prompt_text(body: dict) -> str:
    """The prompt as the provider would tokenize it, in prefix order."""
    parts = [json.dumps(t) for t in body.get("tools") or []]
    parts.append(json.dumps(body.get("system")))
    parts += [json.dumps(m) for m in body.get("messages") or []]
    return "\n".join(parts)


class MockCache:
    BLOCK = 16  # tokens per cache block, as vLLM

    def __init__(self):
        self.seen: list[str] = []

    def lookup(self, text: str) -> tuple[int, int]:
        best = 0
        for prev in self.seen:
            n = 0
            for a, b in zip(prev, text):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        self.seen.append(text)
        prompt = len(text) // 4
        cached = min((best // 4) // self.BLOCK * self.BLOCK, prompt)
        return prompt, cached


def mock_forward(provider: str, cache: MockCache, state: dict):
    def forward(prov, body, request_id, session_id, client_headers, api="chat"):
        prompt, cached = cache.lookup(_prompt_text(body))
        tool = state["expected"]
        if provider == "anthropic":
            return 200, {"id": f"msg_{request_id[:8]}", "content": [{"type": "tool_use", "name": tool, "input": {}}],
                         "usage": {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                                   "cache_creation_input_tokens": 0, "output_tokens": 20}}
        return 200, {"id": f"chatcmpl-{request_id[:8]}",
                     "choices": [{"message": {"role": "assistant", "content": None,
                                              "tool_calls": [{"type": "function",
                                                              "function": {"name": tool, "arguments": "{}"}}]}}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": 20,
                               "prompt_tokens_details": {"cached_tokens": cached}}}
    return forward


# -- runner ------------------------------------------------------------------------------

def run_arm(arm: str, specs: list[dict], *, provider: str, model: str, base_url: str | None,
            mock: bool, salt: str, budget: dict) -> list[dict]:
    fixes = ("canonical_tools",) if arm == "on" else ()
    if mock:
        proxy = PrefixwiseProxy(mock=True, fixes=fixes)
        state = {"expected": None}
        proxy._forward = mock_forward(provider, MockCache(), state)  # type: ignore[method-assign]
    else:
        url = base_url if provider == "vllm" else UPSTREAMS[provider]
        key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else None
        proxy = PrefixwiseProxy(upstream_url=url, api_key=key, mock=False, fixes=fixes)
        state = {"expected": None}
    rows, histories = [], {}
    for spec in specs:
        history = histories.setdefault(spec["session"], [])
        body = render(provider, model, spec, history)
        if provider == "vllm":
            body["cache_salt"] = salt  # isolates this arm's cache from the other arm's
        est = len(_prompt_text(body)) // 4
        if budget["max"] is not None and budget["spent"] + est > budget["max"]:
            print(f"budget: stopping {arm} before session {spec['session']} turn {spec['turn']}")
            break
        state["expected"] = spec["expected_tool"]
        headers = {"X-Prefixwise-Session-Id": f"e4-{arm}-{salt}-s{spec['session']}"}
        result, _ = proxy.handle(provider, body, headers)
        if result["status"] >= 400:
            print(f"{arm} s{spec['session']} t{spec['turn']}: HTTP {result['status']}: "
                  f"{json.dumps(result['response'])[:300]}")
            rows.append({"arm": arm, **spec, "error": result["response"], "status": result["status"]})
            if result["status"] in (401, 403, 404):
                break
            continue
        budget["spent"] += result["prompt_tokens"]
        called = tool_called(provider, result["response"])
        rows.append({"arm": arm, **spec, "cause": result["cause"], "prompt_tokens": result["prompt_tokens"],
                     "cached_tokens": result["cached_tokens"], "avoidable_tokens": result["avoidable_tokens"],
                     "fixes_applied": result.get("fixes_applied") or [], "tool_called": called,
                     "success": called == spec["expected_tool"]})
        # Grow the conversation the way the agent did: question, then its answer.
        q = TASKS[spec["task"]][0]
        history += ([{"role": "user", "content": [{"type": "text", "text": q}]}] if provider == "anthropic"
                    else [{"role": "user", "content": q}])
        history.append({"role": "assistant", "content": f"Called {called or 'no tool'}; result noted."})
    return rows


def score(off: list[dict], on: list[dict]) -> dict:
    ok_off = [r for r in off if "error" not in r]
    ok_on = [r for r in on if "error" not in r]
    keys_on = {(r["session"], r["turn"]): r for r in ok_on}
    paired = [(r, keys_on[(r["session"], r["turn"])]) for r in ok_off if (r["session"], r["turn"]) in keys_on]
    predicted = sum(r["avoidable_tokens"] for r, _ in paired if r["cause"] == "tool_reserialize")
    realised = sum(b["cached_tokens"] for _, b in paired) - sum(a["cached_tokens"] for a, _ in paired)
    succ = lambda rows: (sum(r["success"] for r in rows) / len(rows)) if rows else None  # noqa: E731
    s_off, s_on = succ([a for a, _ in paired]), succ([b for _, b in paired])
    causes = lambda rows: {c: sum(1 for r in rows if r["cause"] == c) for c in sorted({r["cause"] for r in rows})}  # noqa: E731
    ratio = (realised / predicted) if predicted else None
    flat = s_off is not None and s_on is not None and s_on >= s_off - 0.05
    return {"paired_requests": len(paired), "errors": {"off": len(off) - len(ok_off), "on": len(on) - len(ok_on)},
            "predicted_savings_tokens": predicted, "realised_savings_tokens": realised,
            "realised_over_predicted": ratio,
            "prompt_tokens": {"off": sum(a["prompt_tokens"] for a, _ in paired),
                              "on": sum(b["prompt_tokens"] for _, b in paired)},
            "cached_tokens": {"off": sum(a["cached_tokens"] for a, _ in paired),
                              "on": sum(b["cached_tokens"] for _, b in paired)},
            "task_success": {"off": s_off, "on": s_on}, "success_flat": flat,
            "causes": {"off": causes([a for a, _ in paired]), "on": causes([b for _, b in paired])},
            "reloaded_turns": sum(1 for a, _ in paired if a["reloaded"]),
            "pass": bool(len(paired) >= 50 and predicted > 0 and ratio is not None and ratio >= 0.5 and flat)}


def write_report(rep: dict, off: list[dict], on: list[dict], out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"e4_{label}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"report": rep, "off": off, "on": on}, indent=2))
    pct = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
    md = [f"# E4 ({label}): canonical_tools", "",
          f"Paired requests {rep['paired_requests']} (errors off {rep['errors']['off']}, on {rep['errors']['on']}), "
          f"{rep['reloaded_turns']} with tool-order drift  |  Pass: {'YES' if rep['pass'] else 'NO'}", "",
          "| | OFF (as recorded) | ON (canonical_tools) |", "|---|---:|---:|",
          f"| Prompt tokens | {rep['prompt_tokens']['off']:,} | {rep['prompt_tokens']['on']:,} |",
          f"| Cached tokens | {rep['cached_tokens']['off']:,} | {rep['cached_tokens']['on']:,} |",
          f"| Task success | {pct(rep['task_success']['off'])} | {pct(rep['task_success']['on'])} |",
          f"| Causes | {rep['causes']['off']} | {rep['causes']['on']} |", "",
          f"Predicted savings (OFF tool_reserialize avoidable): {rep['predicted_savings_tokens']:,} tokens",
          f"Realised savings (cached ON - cached OFF): {rep['realised_savings_tokens']:,} tokens "
          f"({pct(rep['realised_over_predicted'])} of predicted; pass needs >= 50%)",
          f"Task success flat (ON >= OFF - 5 points): {'yes' if rep['success_flat'] else 'NO'}", ""]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E4: canonical_tools replay, predicted vs realised savings.")
    ap.add_argument("--provider", default="vllm", choices=["vllm", "anthropic"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server")
    ap.add_argument("--sessions-file", default=str(DEFAULT_SESSIONS))
    ap.add_argument("--record", action="store_true", help="write a fresh recording and exit")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--p-reload", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--arm-gap", type=float, default=None,
                    help="seconds between arms (default: 330 for anthropic, past the 5-min cache TTL; 0 otherwise)")
    ap.add_argument("--max-input-tokens", type=int, default=None)
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    path = Path(args.sessions_file)
    if args.record:
        specs = record(path, args.sessions, args.turns, args.p_reload, args.seed)
        print(f"recorded {len(specs)} turns ({sum(s['reloaded'] for s in specs)} with tool reloads) -> {path}")
        return 0
    specs = load(path)
    stamp = time.strftime("%H%M%S")
    budget = {"spent": 0, "max": args.max_input_tokens}
    gap = args.arm_gap if args.arm_gap is not None else (330.0 if args.provider == "anthropic" and not args.mock else 0.0)
    kw = dict(provider=args.provider, model=args.model, base_url=args.base_url, mock=args.mock, budget=budget)
    print(f"E4 {'mock' if args.mock else 'live'}: {args.provider} {args.model}, {len(specs)} recorded turns per arm")
    off = run_arm("off", specs, salt=f"e4-off-{stamp}", **kw)
    if gap:
        print(f"waiting {gap:.0f}s between arms so the OFF arm's cache expires")
        time.sleep(gap)
    on = run_arm("on", specs, salt=f"e4-on-{stamp}", **kw)
    rep = score(off, on)
    label = "mock" if args.mock else f"{args.provider}"
    path = write_report(rep, off, on, Path(args.output_dir), label)
    print(path.with_suffix(".md").read_text())
    print(f"Report: {path}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
