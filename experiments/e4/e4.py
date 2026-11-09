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

A second fix, `--fix auto_breakpoints` (Anthropic only), replays sessions from a client that
sets no cache_control at all (sessions/no-markers.jsonl: the same tasks, no tool drift):
  arm OFF: no marker anywhere, so Anthropic caches nothing
  arm ON:  the proxy places the markers
  predicted = avoidable tokens the OFF ledger attributes to breakpoint_misplaced
  realised  = cached tokens ON - cached tokens OFF
  It changes the bill (a write costs 1.25x), so Pass also needs the ON arm to cost less in
  dollars than the OFF arm, write surcharge included, and no request rejected for its markers.
  The OFF arm writes nothing, so no gap between arms is needed; each run's system prompt
  carries the run's own tag so an earlier run's cache cannot serve this one.

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
NO_MARKER_SESSIONS = HERE / "sessions" / "no-markers.jsonl"  # for --fix auto_breakpoints: p_reload 0
# fix -> the cause whose avoidable tokens are the prediction
FIX_CAUSE = {"canonical_tools": "tool_reserialize", "auto_breakpoints": "breakpoint_misplaced"}
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


def render(provider: str, model: str, spec: dict, history: list[dict], markers: bool = True,
           tag: str = "") -> dict:
    """`markers=False`: a client that sets no cache_control (the auto_breakpoints replay). `tag`:
    text appended to the system prompt, the same in both arms of one run."""
    question = TASKS[spec["task"]][0]
    tools = render_tools(provider, spec["tool_order"], spec["key_seed"])
    if provider == "anthropic":
        cc = {"cache_control": {"type": "ephemeral"}} if markers else {}
        msgs = copy.deepcopy(history) + [{"role": "user", "content": [{"type": "text", "text": question, **cc}]}]
        return {"model": model, "max_tokens": 512, "tools": tools, "tool_choice": {"type": "auto"},
                "system": [{"type": "text", "text": SYSTEM + tag, **cc}], "messages": msgs}
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

    def lookup_marked(self, text: str, marked: bool) -> tuple[int, int, int]:
        """As Anthropic: (prompt, read, written). Without a marker nothing is read or written;
        with one, the longest stored prefix is read and the rest of the prompt written."""
        if not marked:
            return len(text) // 4, 0, 0
        prompt, cached = self.lookup(text)
        return prompt, cached, prompt - cached

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


def _without_markers(value):
    if isinstance(value, dict):
        return {k: _without_markers(v) for k, v in value.items() if k != "cache_control"}
    return [_without_markers(v) for v in value] if isinstance(value, list) else value


def mock_forward(provider: str, cache: MockCache, state: dict, by_marker: bool = False):
    """`by_marker`: bill as Anthropic does, reading and writing only when the request carries a
    cache_control marker (the auto_breakpoints replay); the cache compares prompts without them."""
    def forward(prov, body, request_id, session_id, client_headers, api="chat"):
        written = 0
        if by_marker:
            prompt, cached, written = cache.lookup_marked(_prompt_text(_without_markers(body)),
                                                          "cache_control" in json.dumps(body))
        else:
            prompt, cached = cache.lookup(_prompt_text(body))
        tool = state["expected"]
        if provider == "anthropic":
            return 200, {"id": f"msg_{request_id[:8]}", "content": [{"type": "tool_use", "name": tool, "input": {}}],
                         "usage": {"input_tokens": prompt - cached - written, "cache_read_input_tokens": cached,
                                   "cache_creation_input_tokens": written, "output_tokens": 20}}
        return 200, {"id": f"chatcmpl-{request_id[:8]}",
                     "choices": [{"message": {"role": "assistant", "content": None,
                                              "tool_calls": [{"type": "function",
                                                              "function": {"name": tool, "arguments": "{}"}}]}}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": 20,
                               "prompt_tokens_details": {"cached_tokens": cached}}}
    return forward


# -- runner ------------------------------------------------------------------------------

def run_arm(arm: str, specs: list[dict], *, provider: str, model: str, base_url: str | None,
            mock: bool, salt: str, budget: dict, fix: str = "canonical_tools", tag: str = "") -> list[dict]:
    fixes = (fix,) if arm == "on" else ()
    markers = fix != "auto_breakpoints"  # that replay is a client that sets none
    if mock:
        proxy = PrefixwiseProxy(mock=True, fixes=fixes)
        state = {"expected": None}
        proxy._forward = mock_forward(provider, MockCache(), state, by_marker=not markers)  # type: ignore[method-assign]
    else:
        url = base_url if provider == "vllm" else UPSTREAMS[provider]
        key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else None
        proxy = PrefixwiseProxy(upstream_url=url, api_key=key, mock=False, fixes=fixes)
        state = {"expected": None}
    rows, histories = [], {}
    for spec in specs:
        history = histories.setdefault(spec["session"], [])
        body = render(provider, model, spec, history, markers=markers, tag=tag)
        if provider == "vllm":
            body["cache_salt"] = salt  # isolates this arm's cache from the other arm's
        est = len(_prompt_text(body)) // 4
        if budget["max"] is not None and budget["spent"] + est > budget["max"]:
            print(f"budget: stopping {arm} before session {spec['session']} turn {spec['turn']}")
            break
        state["expected"] = spec["expected_tool"]
        headers = {"X-Prefixwise-Session-Id": f"e4-{arm}-{salt}-s{spec['session']}"}
        result, entry = proxy.handle(provider, body, headers)
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
                     "success": called == spec["expected_tool"],
                     "cost": entry.cost if entry is not None else None,
                     "cache_write_tokens": (entry.cache_write_tokens or 0) if entry is not None else 0,
                     # the provider refused the markers and the client's request went as it was
                     "markers_rejected": (provider, "auto_breakpoints") in proxy.rejected_fields})
        # Grow the conversation the way the agent did: question, then its answer.
        q = TASKS[spec["task"]][0]
        history += ([{"role": "user", "content": [{"type": "text", "text": q}]}] if provider == "anthropic"
                    else [{"role": "user", "content": q}])
        history.append({"role": "assistant", "content": f"Called {called or 'no tool'}; result noted."})
    return rows


def score(off: list[dict], on: list[dict], fix: str = "canonical_tools") -> dict:
    ok_off = [r for r in off if "error" not in r]
    ok_on = [r for r in on if "error" not in r]
    keys_on = {(r["session"], r["turn"]): r for r in ok_on}
    paired = [(r, keys_on[(r["session"], r["turn"])]) for r in ok_off if (r["session"], r["turn"]) in keys_on]
    predicted = sum(r["avoidable_tokens"] for r, _ in paired if r["cause"] == FIX_CAUSE[fix])
    realised = sum(b["cached_tokens"] for _, b in paired) - sum(a["cached_tokens"] for a, _ in paired)
    succ = lambda rows: (sum(r["success"] for r in rows) / len(rows)) if rows else None  # noqa: E731
    s_off, s_on = succ([a for a, _ in paired]), succ([b for _, b in paired])
    causes = lambda rows: {c: sum(1 for r in rows if r["cause"] == c) for c in sorted({r["cause"] for r in rows})}  # noqa: E731
    ratio = (realised / predicted) if predicted else None
    flat = s_off is not None and s_on is not None and s_on >= s_off - 0.05
    # Dollars, for a fix that changes the bill: every paired request priced, or no verdict on cost.
    costs = [(a.get("cost"), b.get("cost")) for a, b in paired]
    priced = bool(costs) and all(x is not None and y is not None for x, y in costs)
    cost = {"off": sum(x for x, _ in costs), "on": sum(y for _, y in costs)} if priced else None
    bill = {"cost_usd": cost, "saved_usd": (cost["off"] - cost["on"]) if cost else None,
            "cache_write_tokens": {"off": sum(a.get("cache_write_tokens") or 0 for a, _ in paired),
                                   "on": sum(b.get("cache_write_tokens") or 0 for _, b in paired)},
            "requests_rewritten": sum(1 for _, b in paired if fix in (b.get("fixes_applied") or [])),
            "markers_rejected": any(b.get("markers_rejected") for _, b in paired)}
    pays = fix != "auto_breakpoints" or bool(cost and cost["on"] < cost["off"] and not bill["markers_rejected"])
    return {"fix": fix, **bill, "paired_requests": len(paired), "errors": {"off": len(off) - len(ok_off), "on": len(on) - len(ok_on)},
            "predicted_savings_tokens": predicted, "realised_savings_tokens": realised,
            "realised_over_predicted": ratio,
            "prompt_tokens": {"off": sum(a["prompt_tokens"] for a, _ in paired),
                              "on": sum(b["prompt_tokens"] for _, b in paired)},
            "cached_tokens": {"off": sum(a["cached_tokens"] for a, _ in paired),
                              "on": sum(b["cached_tokens"] for _, b in paired)},
            "task_success": {"off": s_off, "on": s_on}, "success_flat": flat,
            "causes": {"off": causes([a for a, _ in paired]), "on": causes([b for _, b in paired])},
            "reloaded_turns": sum(1 for a, _ in paired if a["reloaded"]),
            "pass": bool(len(paired) >= 50 and predicted > 0 and ratio is not None and ratio >= 0.5 and flat
                         and pays)}


def write_report(rep: dict, off: list[dict], on: list[dict], out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"e4_{label}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"report": rep, "off": off, "on": on}, indent=2))
    pct = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
    fix = rep.get("fix", "canonical_tools")
    md = [f"# E4 ({label}): {fix}", "",
          f"Paired requests {rep['paired_requests']} (errors off {rep['errors']['off']}, on {rep['errors']['on']}), "
          f"{rep['reloaded_turns']} with tool-order drift  |  Pass: {'YES' if rep['pass'] else 'NO'}", "",
          f"| | OFF (as recorded) | ON ({fix}) |", "|---|---:|---:|",
          f"| Prompt tokens | {rep['prompt_tokens']['off']:,} | {rep['prompt_tokens']['on']:,} |",
          f"| Cached tokens | {rep['cached_tokens']['off']:,} | {rep['cached_tokens']['on']:,} |",
          f"| Task success | {pct(rep['task_success']['off'])} | {pct(rep['task_success']['on'])} |",
          f"| Causes | {rep['causes']['off']} | {rep['causes']['on']} |", "",
          f"Predicted savings (OFF {FIX_CAUSE[fix]} avoidable): {rep['predicted_savings_tokens']:,} tokens",
          f"Realised savings (cached ON - cached OFF): {rep['realised_savings_tokens']:,} tokens "
          f"({pct(rep['realised_over_predicted'])} of predicted; pass needs >= 50%)",
          f"Task success flat (ON >= OFF - 5 points): {'yes' if rep['success_flat'] else 'NO'}", ""]
    if fix == "auto_breakpoints":
        c = rep.get("cost_usd")
        md += [f"Requests rewritten (markers placed): {rep['requests_rewritten']} of {rep['paired_requests']}",
               f"Cache writes: OFF {rep['cache_write_tokens']['off']:,} tokens, ON {rep['cache_write_tokens']['on']:,} "
               "(OFF must be 0: no marker, nothing written)",
               (f"Cost: OFF ${c['off']:.4f}, ON ${c['on']:.4f}, saved ${rep['saved_usd']:.4f} "
                f"({rep['saved_usd'] / c['off']:.1%}), write surcharge included; pass needs ON below OFF"
                if c else "Cost: not every request was priced, so no verdict on dollars (pass needs one)"),
               f"Markers rejected by the provider: {'YES' if rep['markers_rejected'] else 'no'}", ""]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E4: canonical_tools replay, predicted vs realised savings.")
    ap.add_argument("--provider", default="vllm", choices=["vllm", "anthropic"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server")
    ap.add_argument("--fix", default="canonical_tools", choices=sorted(FIX_CAUSE),
                    help="the proxy fix under test (auto_breakpoints: Anthropic only, a client without markers)")
    ap.add_argument("--sessions-file", default=None,
                    help="recorded sessions (default: tool-order-drift.jsonl; no-markers.jsonl for auto_breakpoints)")
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
    if args.fix == "auto_breakpoints":
        if not args.mock and args.provider != "anthropic":
            ap.error("--fix auto_breakpoints is for --provider anthropic")
        args.provider = "anthropic"
        if not args.mock and args.model == "Qwen/Qwen3-4B":
            ap.error("--fix auto_breakpoints needs --model (e.g. claude-haiku-5-5)")
        if args.mock and args.model == "Qwen/Qwen3-4B":
            args.model = "claude-haiku-5-5"  # a priced model, so the mock run has dollars to compare
    path = Path(args.sessions_file or (NO_MARKER_SESSIONS if args.fix == "auto_breakpoints" else DEFAULT_SESSIONS))
    if args.record:
        specs = record(path, args.sessions, args.turns, args.p_reload, args.seed)
        print(f"recorded {len(specs)} turns ({sum(s['reloaded'] for s in specs)} with tool reloads) -> {path}")
        return 0
    specs = load(path)
    stamp = time.strftime("%H%M%S")
    budget = {"spent": 0, "max": args.max_input_tokens}
    # auto_breakpoints: the OFF arm carries no marker and so writes nothing; no cache to wait out
    gap = args.arm_gap if args.arm_gap is not None else (
        330.0 if args.provider == "anthropic" and not args.mock and args.fix != "auto_breakpoints" else 0.0)
    tag = f" Run tag {time.strftime('%Y%m%d')}-{stamp}." if args.fix == "auto_breakpoints" else ""
    kw = dict(provider=args.provider, model=args.model, base_url=args.base_url, mock=args.mock, budget=budget,
              fix=args.fix, tag=tag)
    print(f"E4 {'mock' if args.mock else 'live'}: {args.fix}, {args.provider} {args.model}, "
          f"{len(specs)} recorded turns per arm")
    off = run_arm("off", specs, salt=f"e4-off-{stamp}", **kw)
    if gap:
        print(f"waiting {gap:.0f}s between arms so the OFF arm's cache expires")
        time.sleep(gap)
    on = run_arm("on", specs, salt=f"e4-on-{stamp}", **kw)
    rep = score(off, on, args.fix)
    label = ("mock" if args.mock else f"{args.provider}") + ("" if args.fix == "canonical_tools" else f"_{args.fix}")
    path = write_report(rep, off, on, Path(args.output_dir), label)
    print(path.with_suffix(".md").read_text())
    print(f"Report: {path}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
