"""E5: Prefixwise's accuracy at scale, on real agent conversations with injected changes.

E1 scored the classifier on 120 scripted turns. This replays many real conversations from a public
trace dataset (or generated ones) through the real Prefixwise proxy into a real engine, turn by
turn, and at random turns injects one known change that then stays, as an agent's would: tools
reordered, a tool added or removed, a tool's description edited, a timestamp in the system prompt
changed, the system prompt reworded, an earlier message edited, an earlier tool result edited.
Every other turn is plain growth (the next user message appended). Each turn is scored against
what was done:

- a change turn is correct when the verdict is the change's cause;
- a growth turn is a false alarm when the verdict is any change cause, or waste;
- `unknown` is counted on its own (no claim is not a wrong claim);
- a change seen (kept as a secondary cause) on a turn whose read fell short of the prefix by less
  than one reporting unit is counted apart: usage cannot tell that loss from block rounding;
- engine causes on growth turns (`evicted`, `routing_miss`) are counted apart: under memory
  pressure they can be true.

Usage:
  python3 experiments/e5_scale/e5_scale.py --mock                     # smoke test, no GPU
  python3 experiments/e5_scale/e5_scale.py --upstream http://127.0.0.1:8000 --model Qwen/Qwen3-4B \\
      --source hf --dataset nvidia/Open-SWE-Traces --limit 200

The mock is an independent prefix cache (it caches the literal prefix of the rendered prompt, in
16-token blocks), not Prefixwise's own logic, so a mock score is a real check of the classifier
against a cache that behaves like one, though not evidence about a particular engine.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prefixwise.proxy import PrefixwiseProxy  # noqa: E402

CHANGES = ("tool_reserialize", "tool_catalog_change", "volatile_system", "system_edit", "history_edit",
           "tool_result_mutation")
WASTE = {"tool_reserialize", "volatile_system", "history_edit", "tool_result_mutation", "retry_replay",
         "breakpoint_misplaced"}
CHANGE_CAUSES = set(CHANGES) | {"compaction_rewrite"}
WORDS = ("search", "read", "write", "list", "grep", "fetch", "edit", "run", "plan", "test", "diff", "open")


# -- conversations ------------------------------------------------------------------------------

def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(c.get("text") if isinstance(c, dict) else c) for c in content)
    if isinstance(content, dict):
        return _text(content.get("text") or content.get("content") or "")
    return "" if content is None else str(content)


ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant",
            "system": "system", "developer": "system", "tool": "tool", "function": "tool", "observation": "tool",
            "ipython": "tool"}


def normalize(row: dict, max_chars: int) -> list[dict] | None:
    """A dataset row as a list of {role, text} (user, assistant, tool, system), or None."""
    for key in ("messages", "conversations", "conversation", "trajectory", "turns", "dialog"):
        msgs = row.get(key)
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                msgs = None
        if isinstance(msgs, list) and msgs:
            out = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = ROLE_MAP.get(str(m.get("role") or m.get("from") or "").lower())
                text = _text(m.get("content", m.get("value")))[:max_chars]
                if role and text.strip():
                    out.append({"role": role, "text": text})
            if sum(m["role"] == "user" for m in out) >= 2:
                return out
    return None


def generated(rng: random.Random, n_turns: int) -> list[dict]:
    out = [{"role": "system", "text": "You are a careful coding agent. " * rng.randint(20, 50)}]
    for t in range(n_turns):
        out.append({"role": "user", "text": f"Step {t}: " + " ".join(rng.choices(WORDS, k=20))})
        if rng.random() < 0.6:
            out.append({"role": "tool", "text": "output: " + " ".join(rng.choices(WORDS, k=60))})
        out.append({"role": "assistant", "text": "Done: " + " ".join(rng.choices(WORDS, k=15))})
    return out


def load(args, rng) -> list[list[dict]]:
    if args.source == "generated":
        return [generated(rng, args.turns) for _ in range(args.limit)]
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    convs = []
    for row in ds:
        conv = normalize(row, args.max_chars)
        if conv:
            convs.append(conv)
        if len(convs) >= args.limit:
            break
    return convs


def _tools(rng) -> list[dict]:
    names = rng.sample(WORDS, 4)
    return [{"type": "function", "function": {
        "name": n, "description": f"{n.capitalize()} files in the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                       "required": ["path", "limit"]}}} for n in names]


class State:
    """What the agent sends: tools, a system prompt with a timestamp, and the conversation so far,
    rendered as OpenAI chat messages with well-formed tool calls."""

    def __init__(self, conv: list[dict], rng: random.Random):
        sys_text = next((m["text"] for m in conv if m["role"] == "system"), "You are a coding agent.")
        self.system = sys_text + "\nCurrent time: 2026-11-12T09:00:00Z."
        self.tools = _tools(rng)
        self.turns: list[list[dict]] = []  # one list of messages per user turn of the trace
        cur: list[dict] = []
        for m in conv:
            if m["role"] == "system":
                continue
            if m["role"] == "user" and cur:
                self.turns.append(cur)
                cur = []
            cur.append(m)
        if cur:
            self.turns.append(cur)
        self.k = 0  # turns sent so far
        self.msgs: list[dict] = []
        self.calls = 0

    def _render(self, m: dict) -> list[dict]:
        if m["role"] == "tool":
            self.calls += 1
            cid = f"call_{self.calls}"
            return [{"role": "assistant", "content": None, "tool_calls": [
                        {"id": cid, "type": "function", "function": {"name": self.tools[0]["function"]["name"],
                                                                     "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": cid, "content": m["text"]}]
        return [{"role": m["role"], "content": m["text"]}]

    def next_body(self, model: str) -> dict | None:
        """The next request: everything so far plus the next user turn (its user message only; the
        rest of the turn, tool results and the reply, is appended after the response)."""
        if self.k >= len(self.turns):
            return None
        if self.k > 0:
            for m in self.turns[self.k - 1][1:]:
                self.msgs.extend(self._render(m))
        self.msgs.extend(self._render(self.turns[self.k][0]))
        self.k += 1
        return {"model": model, "max_tokens": 8, "temperature": 0, "tools": copy.deepcopy(self.tools),
                "messages": [{"role": "system", "content": self.system}] + copy.deepcopy(self.msgs)}


def mutate(state: State, change: str, rng: random.Random) -> bool:
    """Apply one change to the agent's state (it stays for later turns). False if not applicable."""
    if change == "tool_reserialize":
        order = list(range(len(state.tools)))
        while order == sorted(order):
            rng.shuffle(order)
        state.tools = [state.tools[i] for i in order]
    elif change == "tool_catalog_change":
        if rng.random() < 0.5 and len(state.tools) > 2:
            state.tools.pop(rng.randrange(len(state.tools)))
        else:
            free = [w for w in WORDS if w not in {t["function"]["name"] for t in state.tools}]
            state.tools.insert(rng.randrange(len(state.tools) + 1), _tools(rng)[0] | {})
            state.tools[-1]["function"]["name"] = rng.choice(free) + "_x"
    elif change == "volatile_system":
        head = state.system.rsplit("Current time: ", 1)[0]
        state.system = head + "Current time: 2026-11-12T%02d:%02d:%02dZ." % (
            rng.randrange(10, 24), rng.randrange(60), rng.randrange(60))
    elif change == "system_edit":
        state.system = "Be concise. " + state.system
    elif change == "history_edit":
        users = [i for i, m in enumerate(state.msgs[:-1]) if m["role"] == "user"]
        if not users:
            return False
        state.msgs[rng.choice(users)]["content"] += " (edited)"
    elif change == "tool_result_mutation":
        tools = [i for i, m in enumerate(state.msgs) if m["role"] == "tool"]
        if not tools:
            return False
        state.msgs[rng.choice(tools)]["content"] += " [re-rendered]"
    return True


# -- an independent prefix cache, for the mock ----------------------------------------------------

class PrefixCache:
    """Caches the literal rendered prompt per session, in 16-token blocks (4 chars a token). The
    rendering puts the system prompt first, then tools, then messages, as engine chat templates do."""

    def __init__(self):
        self.last: dict[str, str] = {}

    @staticmethod
    def render(body: dict) -> str:
        msgs = body["messages"]
        system = "".join(json.dumps(m, sort_keys=True) for m in msgs if m["role"] == "system")
        rest = "".join(json.dumps(m, sort_keys=True) for m in msgs if m["role"] != "system")
        return system + json.dumps(body.get("tools") or []) + rest

    def serve(self, session: str, body: dict) -> dict:
        text = self.render(body)
        prev = self.last.get(session, "")
        common = 0
        for a, b in zip(prev, text):
            if a != b:
                break
            common += 1
        self.last[session] = text
        prompt = max(1, len(text) // 4)
        cached = min(prompt, (common // 4) // 16 * 16)
        return {"id": "chatcmpl-e5", "object": "chat.completion", "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": cached}}}


# -- the run ----------------------------------------------------------------------------------

def run(args) -> dict:
    rng = random.Random(args.seed)
    convs = load(args, rng)
    ledger = str(Path(args.out) / f"e5_{int(time.time())}.db")
    if args.mock:
        cache = PrefixCache()
        proxy = PrefixwiseProxy(ledger_path=ledger, mock=False, upstream_url="http://e5-mock.invalid",
                                engine="vllm", kv_events="", fixes=(), otel="", collector_url="")
        proxy._forward = lambda provider, body, rid, sid, headers, api="chat": (200, cache.serve(sid, body))
    else:
        proxy = PrefixwiseProxy(ledger_path=ledger, mock=False, upstream_url=args.upstream,
                                engine=args.engine, kv_events=args.kv_events or "", fixes=(), otel="",
                                collector_url="", full_attention=args.full_attention)
    turns = []
    for ci, conv in enumerate(convs):
        state = State(conv, rng)
        sid = f"e5-{args.seed}-{ci}"
        for k in range(args.turns):
            label = "first_write" if k == 0 else "none"
            if k > 0 and rng.random() < args.change_rate:
                change = rng.choice(CHANGES)
                if mutate(state, change, rng):
                    label = change
            body = state.next_body(args.model)
            if body is None:
                break
            result, entry = proxy.handle("openai", body, {"x-prefixwise-session-id": sid})
            if entry is None:
                turns.append({"conv": ci, "turn": k, "label": label, "cause": None, "error": result.get("status")})
                continue
            turns.append({"conv": ci, "turn": k, "label": label, "cause": entry.cause, "detail": entry.detail,
                          "prompt": entry.prompt_tokens, "cached": entry.cached_tokens,
                          "secondary": entry.secondary_causes, "evidence": entry.evidence})
    proxy.close()
    return score(turns) | {"turns": turns, "conversations": len(convs), "ledger": ledger}


def score(turns: list[dict]) -> dict:
    ok = [t for t in turns if t["cause"] is not None]
    changes = [t for t in ok if t["label"] in CHANGES]
    growth = [t for t in ok if t["label"] == "none"]
    per = {}
    for c in CHANGES:
        lab = [t for t in changes if t["label"] == c]
        pred = [t for t in ok if t["cause"] == c]
        per[c] = {"injected": len(lab), "named": sum(t["cause"] == c for t in lab),
                  "unknown": sum(t["cause"] == "unknown" for t in lab),
                  "predicted": len(pred), "predicted_correct": sum(t["label"] == c for t in pred)}
    false_alarms = [t for t in growth if t["cause"] in CHANGE_CAUSES or t["cause"] in WASTE]
    engine_on_growth = [t for t in growth if t["cause"] in ("evicted", "routing_miss")]
    confusion = collections.Counter((t["label"], t["cause"]) for t in ok)
    return {
        "requests": len(turns), "errors": len(turns) - len(ok),
        "change_turns": len(changes), "changes_named": sum(t["cause"] == t["label"] for t in changes),
        "changes_seen_within_rounding": sum(t["cause"] == "none" and t["label"] in (t.get("secondary") or "")
                                            for t in changes),
        "changes_wrong": sum(t["cause"] not in (t["label"], "unknown", "none") for t in changes),
        "changes_unknown": sum(t["cause"] == "unknown" for t in changes),
        "growth_turns": len(growth), "false_alarms": len(false_alarms),
        "engine_causes_on_growth": len(engine_on_growth),
        "unknown_on_growth": sum(t["cause"] == "unknown" for t in growth),
        "per_cause": per,
        "confusion": {f"{a} -> {b}": n for (a, b), n in sorted(confusion.items())},
    }


def report(res: dict) -> str:
    lines = [f"# E5 accuracy at scale", "",
             f"{res['conversations']} conversations, {res['requests']} requests ({res['errors']} errors).", "",
             f"- Change turns named exactly: **{res['changes_named']} of {res['change_turns']}**"
             f" ({res['changes_unknown']} unknown, {res['changes_seen_within_rounding']} seen within one reporting"
             f" unit, {res['changes_wrong']} named as another cause)",
             f"- False alarms on growth turns: **{res['false_alarms']} of {res['growth_turns']}**",
             f"- Engine causes on growth turns (evicted / routing): {res['engine_causes_on_growth']}",
             f"- Unknown on growth turns: {res['unknown_on_growth']}", "",
             "| change | injected | named | unknown | precision |", "|---|---:|---:|---:|---:|"]
    for c, p in res["per_cause"].items():
        prec = f"{p['predicted_correct']}/{p['predicted']}" if p["predicted"] else "-"
        lines.append(f"| {c} | {p['injected']} | {p['named']} | {p['unknown']} | {prec} |")
    lines += ["", "Confusion (label -> verdict):", ""]
    lines += [f"- {k}: {v}" for k, v in res["confusion"].items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mock", action="store_true", help="an independent prefix cache, no engine")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--kv-events", default="", help="KV event endpoint, or auto")
    ap.add_argument("--full-attention", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--source", choices=["generated", "hf"], default="generated")
    ap.add_argument("--dataset", default="nvidia/Open-SWE-Traces")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=60, help="conversations")
    ap.add_argument("--turns", type=int, default=10, help="turns per conversation, at most")
    ap.add_argument("--max-chars", type=int, default=1500, help="per message, so prompts fit the context")
    ap.add_argument("--change-rate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    res = run(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (Path(args.out) / f"e5_{stamp}.json").write_text(json.dumps(res, indent=2))
    text = report(res)
    (Path(args.out) / f"e5_{stamp}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
