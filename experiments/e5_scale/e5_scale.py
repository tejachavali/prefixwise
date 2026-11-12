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


MESSAGE_KEYS = ("messages", "conversations", "conversation", "trajectory", "turns", "dialog", "history",
                "events", "steps", "transcript", "trace")


def _message(m) -> dict | None:
    """One message as {role, content}: flat ({role, content} or {from, value}) or an agent log line
    with the message nested ({"type": "user", "message": {"role": ..., "content": ...}}, as Claude
    Code writes)."""
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except ValueError:
            return None
    if not isinstance(m, dict):
        return None
    inner = m.get("message") if isinstance(m.get("message"), dict) else m
    role = inner.get("role") or inner.get("from") or m.get("role") or m.get("type") or m.get("speaker")
    content = inner.get("content", inner.get("value", inner.get("text")))
    return {"role": role, "content": content}


def _is_tool_result(content) -> bool:
    return isinstance(content, list) and bool(content) and all(
        isinstance(c, dict) and c.get("type") in ("tool_result", "function_call_output") for c in content)


def _block_text(c) -> str:
    if isinstance(c, dict) and c.get("type") in ("tool_use", "function_call"):
        return json.dumps(c.get("input", c.get("arguments", "")), sort_keys=True)
    return _text(c.get("text") or c.get("content") or "") if isinstance(c, dict) else _text(c)


def split(conv: list[dict]) -> tuple[bool, list[list[dict]]]:
    """(agent trace?, the requests). An agent trace (the model's tool calls are in it) makes a
    request at every model call: everything up to the next assistant message, which is appended
    after the response. A chat makes one at every user message."""
    body = [m for m in conv if m["role"] != "system"]
    steps = any(m.get("tool_calls") for m in body)
    turns: list[list[dict]] = []
    cur: list[dict] = []
    for m in body:
        if steps and cur and m["role"] != "assistant" and cur[-1]["role"] == "assistant":
            turns.append(cur)
            cur = []
        elif not steps and m["role"] == "user" and cur:
            turns.append(cur)
            cur = []
        cur.append(m)
    if cur:
        turns.append(cur)
    if steps and turns and all(m["role"] == "assistant" for m in turns[-1]):
        turns[-2:] = [turns[-2] + turns[-1]] if len(turns) > 1 else []  # a reply alone is no request
    return steps, turns


class Conv(list):
    """A conversation (a list of messages) and the tools its agent sent, when the trace has them."""
    tools = None


def _arguments(value, max_chars: int) -> str:
    """A tool call's arguments as a JSON string, its string values shortened like message text, so
    it stays valid JSON (a template or server may parse it)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return json.dumps({"raw": value[:max_chars]})
    if value is None:
        return "{}"
    if not isinstance(value, dict):  # a template may expect an object
        value = {"value": value}
    return json.dumps({k: (v[:max_chars] if isinstance(v, str) else v) for k, v in value.items()})


def _tool_calls(calls, max_chars: int) -> list[dict] | None:
    """OpenAI-shaped tool calls from a trace message, or None."""
    if not isinstance(calls, list) or not calls:
        return None
    out = []
    for i, c in enumerate(calls):
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append({"id": str(c.get("id") or f"call_t{i}"), "type": "function",
                    "function": {"name": str(fn["name"]), "arguments": _arguments(fn.get("arguments", {}), max_chars)}})
    return out or None


def _trace_tools(row: dict) -> list[dict] | None:
    tools = row.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except ValueError:
            return None
    ok = [t for t in tools if isinstance(t, dict) and isinstance(t.get("function"), dict)
          and t["function"].get("name")] if isinstance(tools, list) else []
    return [{"type": "function", "function": t["function"]} for t in ok] or None


def normalize(row: dict, max_chars: int) -> list[dict] | None:
    """A dataset row as a list of {role, text} (user, assistant, tool, system; an agent's tool calls
    and tool-call ids kept), or None. Usable when it makes two or more requests: two user turns
    (a chat), or two model calls (an agent trace: one task, then tool calls and results)."""
    for key in MESSAGE_KEYS:
        msgs = row.get(key)
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                msgs = [line for line in msgs.splitlines() if line.strip()]  # JSON Lines
        if isinstance(msgs, list) and msgs:
            out = []
            for raw in msgs:
                m = _message(raw)
                if m is None:
                    continue
                role = ROLE_MAP.get(str(m["role"] or "").lower())
                content = m["content"]
                if role == "user" and _is_tool_result(content):
                    role = "tool"  # Anthropic sends tool results as user messages
                text = ("\n".join(_block_text(c) for c in content) if isinstance(content, list)
                        else _text(content))[:max_chars]
                msg = {"role": role, "text": text}
                calls = _tool_calls(raw.get("tool_calls"), max_chars) if role == "assistant" and isinstance(raw, dict) else None
                if calls:
                    msg["tool_calls"] = calls
                if role == "tool" and isinstance(raw, dict) and raw.get("tool_call_id"):
                    msg["tool_call_id"] = str(raw["tool_call_id"])
                if role and (text.strip() or calls or "tool_call_id" in msg):  # an empty result is a result
                    out.append(msg)
            if isinstance(row.get("system_prompt"), str) and row["system_prompt"].strip() and \
                    not any(m["role"] == "system" for m in out):
                out.insert(0, {"role": "system", "text": row["system_prompt"][:max_chars * 4]})
            if any(m["role"] == "user" for m in out) and len(split(out)[1]) >= 2:  # two requests or more
                conv = Conv(out)
                conv.tools = _trace_tools(row)
                return conv
    return None


def shape(value, depth: int = 0) -> object:
    """The structure of a dataset row: field names and types, never its text."""
    if depth > 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: shape(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, list):
        return [shape(value[0], depth + 1), f"... {len(value)} items"] if value else []
    if isinstance(value, str):
        if value[:1] in "[{":
            try:
                return {"json string": shape(json.loads(value), depth + 1)}
            except ValueError:
                pass
        return f"str({len(value)})"
    return type(value).__name__


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
    from datasets import get_dataset_split_names, load_dataset  # pip install datasets
    split = args.split
    if split is None:
        names = get_dataset_split_names(args.dataset, args.config)
        if len(names) != 1:
            raise SystemExit(f"e5: {args.dataset} has splits {names}; pick one with --split")
        split = names[0]
    ds = load_dataset(args.dataset, args.config, split=split, streaming=True)
    convs, scanned, first = [], 0, None
    for row in ds:
        scanned += 1
        first = first if first is not None else row
        conv = normalize(row, args.max_chars)
        if conv:
            convs.append(conv)
        if len(convs) >= args.limit or scanned >= args.limit * 50:
            break
    if not convs:  # measuring nothing must not look like a result
        raise SystemExit(f"e5: no conversation with two or more user turns in the first {scanned} rows of "
                         f"{args.dataset} ({split}); the row shape (field names and types, no text):\n"
                         + json.dumps(shape(first), indent=1) if first is not None else
                         f"e5: {args.dataset} ({split}) yielded no rows")
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
        self.tools = copy.deepcopy(getattr(conv, "tools", None)) or _tools(rng)  # the trace's own tools
        self.steps, self.turns = split(conv)  # per request: what is sent, then what is appended after
        self.k = 0  # turns sent so far
        self.msgs: list[dict] = []
        self.calls = 0

    def _render(self, m: dict) -> list[dict]:
        if m.get("tool_calls"):  # the trace's own call
            return [{"role": "assistant", "content": m["text"] or None, "tool_calls": copy.deepcopy(m["tool_calls"])}]
        if m["role"] == "tool" and m.get("tool_call_id"):  # the trace's own result
            return [{"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["text"]}]
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
        if self.steps:  # send the inputs of this step; its reply follows the response
            if self.k > 0:
                for m in self.turns[self.k - 1]:
                    if m["role"] == "assistant":
                        self.msgs.extend(self._render(m))
            for m in self.turns[self.k]:
                if m["role"] != "assistant":
                    self.msgs.extend(self._render(m))
            self.k += 1
            return {"model": model, "max_tokens": 8, "temperature": 0, "tools": copy.deepcopy(self.tools),
                    "messages": [{"role": "system", "content": self.system}] + copy.deepcopy(self.msgs)}
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
        if len(state.tools) < 2:
            return False
        order = list(range(len(state.tools)))
        while order == sorted(order):
            rng.shuffle(order)
        state.tools = [state.tools[i] for i in order]
    elif change == "tool_catalog_change":
        if rng.random() < 0.5 and len(state.tools) > 2:
            state.tools.pop(rng.randrange(len(state.tools)))
        else:
            free = [w for w in WORDS if w not in {t["function"]["name"] for t in state.tools}]
            at = rng.randrange(len(state.tools) + 1)  # draws in the order the recorded runs made them
            added = _tools(rng)[0]
            state.tools.insert(at, added)
            added["function"]["name"] = rng.choice(free) + "_x"  # the added tool, not whichever is last
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
                turns.append({"conv": ci, "turn": k, "label": label, "cause": None, "error": result.get("status"),
                              "error_text": _error_text(result)})
                if len(turns) == 5 and all(t["cause"] is None for t in turns):  # the server refuses: say why, now
                    proxy.close()
                    raise SystemExit(f"e5: the first 5 requests all failed; the first: {turns[0]['error_text']}")
                continue
            turns.append({"conv": ci, "turn": k, "label": label, "cause": entry.cause, "detail": entry.detail,
                          "prompt": entry.prompt_tokens, "cached": entry.cached_tokens,
                          "secondary": entry.secondary_causes, "evidence": entry.evidence})
    proxy.close()
    first_error = next((t["error_text"] for t in turns if t.get("error_text")), None)
    return score(turns) | {"turns": turns, "conversations": len(convs), "ledger": ledger, "first_error": first_error}


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


def _error_text(result: dict) -> str:
    """HTTP status and the server's own message, short: what a failed run must show."""
    resp = result.get("response")
    err = resp.get("error") if isinstance(resp, dict) else None
    msg = (err.get("message") if isinstance(err, dict) else err) or (resp if resp else "")
    return f"HTTP {result.get('status')}: {str(msg)[:300]}"


def report(res: dict) -> str:
    lines = [f"# E5 accuracy at scale", "",
             f"{res['conversations']} conversations, {res['requests']} requests ({res['errors']} errors).", "",
             *([f"First error: {res['first_error']}", ""] if res.get("first_error") else []),
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
    ap.add_argument("--split", default=None, help="default: the dataset's only split")
    ap.add_argument("--inspect", action="store_true",
                    help="print the splits and the first row's shape (field names and types, no text), then stop")
    ap.add_argument("--limit", type=int, default=60, help="conversations")
    ap.add_argument("--turns", type=int, default=10, help="turns per conversation, at most")
    ap.add_argument("--max-chars", type=int, default=1500, help="per message, so prompts fit the context")
    ap.add_argument("--change-rate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args(argv)
    if args.inspect:
        from datasets import get_dataset_split_names, load_dataset
        names = get_dataset_split_names(args.dataset, args.config)
        print("splits:", names)
        split = args.split or names[0]
        row = next(iter(load_dataset(args.dataset, args.config, split=split, streaming=True)), None)
        if row is None:
            print(f"{split} has no rows")
            return 1
        print(f"first row of {split}:", json.dumps(shape(row), indent=1))
        print("parses as a conversation:", bool(normalize(row, args.max_chars)))
        return 0
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
