"""E-template: what a self-hosted engine's chat template does to a conversation's cached prefix.

A client re-sends the same message blocks every turn; the engine renders them through the model's
chat template. Where the rendering changes between turns, the token prefix changes although the
request did not, and the engine's KV cache cannot serve what the template re-wrote. Measured per
turn, through the real Prefixwise proxy (in-process, like E-evict), with the engine's KV events on:

  cached            what the engine served (usage)
  prev_prompt       the previous turn's prompt: the prefix any cache can serve
  prev_full         previous prompt + the previous output: what an engine that caches decoded
                    blocks (vLLM, SGLang) could serve if the template re-rendered nothing
  reuse             "full" (cached >= prev_full - slack), "prompt" (the previous output was
                    re-rendered: cached ~ prev_prompt), "break" (cached < prev_prompt - slack: a
                    prefix break, the condition behind Prefixwise's "not an eviction"), "cold";
                    slack = one block plus 8 tokens: block rounding, and Qwen3's 4-token think stub
                    (`<think>\n\n</think>\n\n`) that the generation prompt carries only while
                    thinking is off, so a toggle moves the prefix end by 4 tokens, not a re-render
  verdict           Prefixwise's cause / detail / evidence for the turn

Variants (each a session of --turns turns against the same engine):
  kept       the assistant's answer appended as returned, <think> block and all
  stripped   the <think>...</think> block removed client-side before appending
  nothink    chat_template_kwargs: {"enable_thinking": false} on every turn
  toggle     enable_thinking flips each turn (a parameter change: param_rendered)
  tools      a tool-call loop (the model calls read_file, gets a result, answers)

Qwen3's template drops the <think> block of earlier assistant turns and re-renders their content
at a shifted position (measured on CPU, 2026-11-09: the shared prefix with the previous turn's
prompt+output is exactly the previous prompt), so the prediction is reuse = "prompt" on every
variant but the first turn, never "break": the previous output is re-prefilled every turn, a tail
cost invisible in Prefixwise's verdicts (a clean `none`) and in cache-hit rates as a share. For
`stripped`, prev_full still counts the think tokens the client deleted, so its "lost tail" is the
client's choice, not the template's; the CPU mode shows the template alone. Each variant gets its
own system prompt: the engine's prefix cache is global, and `kept` and `stripped` render the same
bytes from turn 2 on, so a shared prompt would let one variant serve the other's blocks. Turns the
model cut off (`finish_reason: length`, an unclosed <think>) are marked `truncated`; max_tokens
is 1,500 so that is rare.

    python experiments/e_template/e_template.py --mock
    python experiments/e_template/e_template.py --engine vllm --url http://127.0.0.1:8000 --model Qwen/Qwen3-4B \\
        --kv-events tcp://127.0.0.1:5557
    python experiments/e_template/e_template.py --engine sglang --url http://127.0.0.1:30000 --model Qwen/Qwen3-4B \\
        --kv-events tcp://127.0.0.1:5557 --full-attention
    python experiments/e_template/e_template.py --cpu-template Qwen/Qwen3-4B      # no server: the template alone

--cpu-template renders turns with the model's own chat template (transformers) and diffs token
ids between turn N's prompt+output and turn N+1's prompt, naming where the divergence starts and
which template clause is behind it; with a Llama-3.x model it also renders two dates to show the
date clause's prefix break. No GPU, no server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prefixwise.proxy import PrefixwiseProxy  # noqa: E402

VARIANTS = ("kept", "stripped", "nothink", "toggle", "tools")
THINK = re.compile(r"<think>.*?</think>\s*", re.S)
QUESTIONS = ["Explain prefix caching in one paragraph.", "Now in one sentence.", "Why does it miss?",
             "Name two causes of a miss.", "Which one is the agent's fault?", "Summarise all of the above."]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}}]


def system_text(words: int) -> str:
    base = ("You are a careful coding assistant. Answer briefly and precisely. The repository uses Python, "
            "pytest and a SQLite ledger. Prefer plain language. ")
    out = base
    while len(out.split()) < words:
        out += base
    return out


def body(model: str, system: str, msgs: list, variant: str, turn: int) -> dict:
    b = {"model": model, "messages": [{"role": "system", "content": system}, *msgs], "max_tokens": 1500, "temperature": 0}
    if variant == "nothink":
        b["chat_template_kwargs"] = {"enable_thinking": False}
    elif variant == "toggle":
        b["chat_template_kwargs"] = {"enable_thinking": bool(turn % 2)}
    elif variant == "tools":
        b["tools"] = TOOLS
    return b


def assistant_message(payload: dict, variant: str) -> dict:
    """The assistant turn as a client appends it: the message the engine returned (content with or
    without the think block, tool calls as structured), per variant."""
    msg = dict(payload["choices"][0]["message"])
    content = msg.get("content") or ""
    if variant == "stripped":
        content = THINK.sub("", content)
    out = {"role": "assistant", "content": content}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


THINK_STUB = 4  # tokens the generation prompt carries only while thinking is off (Qwen3)


def classify_reuse(cached: int, prev_prompt: int | None, prev_full: int | None, block: int) -> str:
    slack = block + THINK_STUB + 4
    if prev_prompt is None:
        return "first"
    if prev_full is not None and cached >= prev_full - slack:
        return "full"
    if cached >= prev_prompt - slack:
        return "prompt"
    if cached > block:
        return "break"
    return "cold"


def run_variant(proxy: PrefixwiseProxy, args, variant: str, block: int) -> list[dict]:
    sid = f"template-{variant}-{int(time.time())}"
    system = f"[{variant}] " + system_text(args.system_words)  # its own prefix: the engine's cache is global
    msgs: list = []
    rows = []
    prev = None  # (prompt, completion)
    for turn in range(args.turns):
        q = QUESTIONS[turn % len(QUESTIONS)]
        if variant == "tools" and turn == 0:
            q = "Use the read_file tool to read a.py, then tell me what it does."
        msgs.append({"role": "user", "content": q})
        b = body(args.model, system, msgs, variant, turn)
        res, entry = proxy.handle("vllm", {**b, "session_id": sid})
        if res["status"] >= 400:
            raise SystemExit(f"{variant} turn {turn}: upstream {res['status']}: {json.dumps(res['response'])[:300]}")
        usage = res["response"].get("usage") or {}
        prompt, cached = res["prompt_tokens"], res["cached_tokens"]
        completion = int(usage.get("completion_tokens") or 0)
        prev_prompt = prev[0] if prev else None
        prev_full = (prev[0] + prev[1]) if prev else None
        reuse = classify_reuse(cached, prev_prompt, prev_full, block)
        choice = res["response"]["choices"][0]
        content = choice["message"].get("content") or ""
        truncated = choice.get("finish_reason") == "length" or ("<think>" in content and "</think>" not in content)
        row = {"turn": turn + 1, "prompt": prompt, "cached": cached, "completion": completion,
               "prev_prompt": prev_prompt, "prev_full": prev_full, "reuse": reuse,
               "lost_tail": max(0, (prev_full or 0) - cached) if reuse == "prompt" else 0,
               "cause": res["cause"], "detail": getattr(entry, "detail", None),
               "evidence": (res.get("evidence") or "")[:200],
               "think_in_content": "<think>" in content, "finish_reason": choice.get("finish_reason"),
               "truncated": truncated}
        rows.append(row)
        print(f"{variant:9s} turn {turn + 1}: prompt {prompt} cached {cached} completion {completion} "
              f"prev_prompt {prev_prompt} prev_full {prev_full} -> {reuse:6s} {res['cause']}"
              + (f" [{row['detail']}]" if row["detail"] else "") + (" TRUNCATED" if truncated else ""), flush=True)
        msg = assistant_message(res["response"], variant)
        msgs.append(msg)
        if msg.get("tool_calls"):  # answer the call; the next user question follows the tool result
            for call in msg["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": call["id"], "content": "def main():\n    print('hi')\n"})
        prev = (prompt, completion)
    return rows


def mock_forward(args):
    """A Qwen3-like engine: the previous output's think block is dropped on re-rendering, so the
    next turn reads the previous prompt and no more; a changed chat_template_kwargs renders anew."""
    state: dict = {}

    def forward(provider, b, request_id, session_id, client_headers, api="chat"):
        prompt = sum(len(str(m.get("content") or "").split()) for m in b["messages"]) * 13 // 10 + 20
        key = session_id
        last = state.get(key)
        cached = 0
        if last is not None and last[2] == json.dumps(b.get("chat_template_kwargs"), sort_keys=True):
            cached = min(last[0], prompt)  # the previous prompt, never the previous output
        completion = 120 if not (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False else 30
        state[key] = (prompt, completion, json.dumps(b.get("chat_template_kwargs"), sort_keys=True))
        content = ("<think>\nthinking about it\n</think>\n\n" if completion == 120 else "") + "An answer."
        msg = {"role": "assistant", "content": content}
        if b.get("tools") and len(b["messages"]) <= 2:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                       "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]}
        return 200, {"id": f"chatcmpl-{request_id[:8]}", "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                               "prompt_tokens_details": {"cached_tokens": cached}}}
    return forward


def cpu_template(model: str) -> dict:
    """Render with the model's own chat template and diff token ids: where does turn N+1's prompt
    leave turn N's prompt+output, and which clause is behind it. No server."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    def ids(msgs, **kw):
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
        if hasattr(out, "keys") and "input_ids" in out:
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    def first_diff(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    sys_ = {"role": "system", "content": system_text(40)}
    u1 = {"role": "user", "content": QUESTIONS[0]}
    answer = "Prefix caching reuses the KV cache of a shared prompt prefix across requests."
    a_think = {"role": "assistant", "content": "<think>\nA short explanation of KV reuse.\n</think>\n\n" + answer}
    a_plain = {"role": "assistant", "content": answer}
    u2 = {"role": "user", "content": QUESTIONS[1]}
    out: dict = {"model": model, "clauses": [l.strip()[:160] for l in (tok.chat_template or "").splitlines()
                                             if re.search(r"reasoning|think|date|strftime|tojson", l)][:12]}
    t1 = ids([sys_, u1])
    for name, a in (("kept", a_think), ("stripped", a_plain)):
        out1 = tok.encode(a["content"] + "<|im_end|>\n" if "<|im_start|>" in tok.decode(t1) else a["content"],
                          add_special_tokens=False)
        t2 = ids([sys_, u1, a, u2])
        k = first_diff(t1 + out1, t2)
        out[name] = {"turn1_prompt": len(t1), "turn1_output": len(out1), "turn2_prompt": len(t2),
                     "shared_with_prompt_plus_output": k, "diverges_at": tok.decode(t2[k:k + 12]),
                     "reuse": "full" if k >= len(t1) + len(out1) else "prompt" if k >= len(t1) else "break"}
    try:
        a = ids([sys_, u1], enable_thinking=True)
        b = ids([sys_, u1], enable_thinking=False)
        k = first_diff(a, b)
        out["enable_thinking"] = {"shared": k, "lens": [len(a), len(b)], "tail": tok.decode(b[k:]),
                                  "where": "tail" if k >= min(len(a), len(b)) - 8 else "prefix"}
    except Exception as exc:  # noqa: BLE001 - a template without the switch
        out["enable_thinking"] = {"error": type(exc).__name__}
    try:  # tool-call arguments as the servers hand them to the template: a dict, re-serialised by tojson
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                    "function": {"name": "read_file", "arguments": {"path": "a.py", "n": 1}}}]}
        res = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        r = tok.apply_chat_template([sys_, u1, call, res, u2], tools=TOOLS, tokenize=False, add_generation_prompt=True)
        i = r.find("<tool_call>")
        out["tool_call_rendered"] = r[i:r.find("</tool_call>") + 12] if i >= 0 else r[-300:]
    except Exception as exc:  # noqa: BLE001
        out["tool_call_rendered"] = f"error: {type(exc).__name__}: {exc}"[:200]
    try:  # a date clause: two renderings of the same conversation on two days
        a = ids([sys_, u1], date_string="08 Nov 2026")
        b = ids([sys_, u1], date_string="09 Nov 2026")
        k = first_diff(a, b)
        out["date"] = {"renders_date": a != b, "shared": k, "len": len(a),
                       "where": ("prefix" if a != b and k < len(a) // 2 else "tail" if a != b else "none")}
    except Exception as exc:  # noqa: BLE001
        out["date"] = {"error": type(exc).__name__}
    return out


def write(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"e_template_{result.get('engine') or 'cpu'}_{time.strftime('%Y%m%d-%H%M%S')}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1))
    lines = [f"# E-template: {result.get('engine') or 'cpu'}{' (mock)' if result.get('mock') else ''}", "",
             f"Model {result['model']}; {result.get('turns', 0)} turns per variant; block {result.get('block')} tokens.", ""]
    if "variants" in result:
        lines += ["| variant | turn | prompt | cached | completion | prev_prompt | prev_full | reuse | lost tail | Prefixwise |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
        for name, rows in result["variants"].items():
            for r in rows:
                tag = f" [{r['detail']}]" if r.get("detail") else ""
                cut = " (truncated)" if r.get("truncated") else ""
                lines.append(f"| {name} | {r['turn']} | {r['prompt']} | {r['cached']} | {r['completion']}{cut} | {r['prev_prompt'] or ''} | "
                             f"{r['prev_full'] or ''} | {r['reuse']} | {r['lost_tail']} | `{r['cause']}`{tag} |")
        s = result["summary"]
        lines += ["", f"- Turns with a previous turn: {s['eligible']}; reuse full {s['full']}, prompt-only {s['prompt']}, "
                      f"break {s['break']}, cold {s['cold']}",
                  f"- Previous output re-prefilled (lost tail), total: {s['lost_tail_total']:,} tokens "
                  f"({s['lost_tail_share']:.0%} of eligible turns' prompt tokens)",
                  f"- Prefixwise verdicts: " + ", ".join(f"`{k}` x{v}" for k, v in s["verdicts"].items()),
                  f"- `unknown` (\"not an eviction\") turns: {s['not_an_eviction']}; truncated answers: {s['truncated']}"]
        eng = result.get("engine_check") or {}
        lines += ["", f"Engine check notes: {'; '.join(eng.get('notes') or []) or 'none'}",
                  f"KV-event stream at the end: `{json.dumps(eng.get('events'))}`"]
    if "cpu" in result:
        lines += ["", "## Template alone (CPU)", "", "```json", json.dumps(result["cpu"], indent=1), "```"]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return stem.with_suffix(".md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--kv-events", help="the engine's KV-event endpoint, e.g. tcp://127.0.0.1:5557")
    ap.add_argument("--full-attention", action="store_true")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--system-words", type=int, default=300, help="system prompt size, in words")
    ap.add_argument("--block", type=int, default=16, help="the engine's block size in tokens (SGLang page size: 1)")
    ap.add_argument("--mock", action="store_true", help="a simulated Qwen3-like engine, no GPU")
    ap.add_argument("--cpu-template", metavar="MODEL", help="no server: render with the model's chat template and diff")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args(argv)
    result: dict = {"model": args.cpu_template or args.model, "turns": args.turns, "block": args.block, "mock": args.mock}
    if args.cpu_template:
        result["cpu"] = cpu_template(args.cpu_template)
        print(json.dumps(result["cpu"], indent=1))
    else:
        result["engine"] = args.engine
        if args.mock:
            proxy = PrefixwiseProxy(engine=args.engine)
            proxy._forward = mock_forward(args)  # type: ignore[method-assign]
        else:
            proxy = PrefixwiseProxy(upstream_url=args.url, api_key="sk-dummy", engine=args.engine,
                                    kv_events=args.kv_events, full_attention=args.full_attention)
        variants = {}
        for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
            variants[v] = run_variant(proxy, args, v, args.block)
        rows = [r for rs in variants.values() for r in rs]
        eligible = [r for r in rows if r["prev_prompt"] is not None]
        verdicts: dict = {}
        for r in rows:
            verdicts[r["cause"]] = verdicts.get(r["cause"], 0) + 1
        result["variants"] = variants
        result["summary"] = {"eligible": len(eligible), **{k: sum(r["reuse"] == k for r in eligible)
                                                           for k in ("full", "prompt", "break", "cold")},
                             "lost_tail_total": sum(r["lost_tail"] for r in eligible),
                             "lost_tail_share": (sum(r["lost_tail"] for r in eligible) / max(1, sum(r["prompt"] for r in eligible))),
                             "verdicts": verdicts,
                             "not_an_eviction": sum(1 for r in rows if r["cause"] == "unknown" and "not an eviction" in r["evidence"]),
                             "truncated": sum(1 for r in rows if r.get("truncated"))}
        result["engine_check"] = {"notes": proxy.engine_notes,
                                  "events": proxy.engine_events.snapshot() if proxy.engine_events is not None else None}
        if proxy.engine_events is not None:
            proxy.engine_events.stop()
    md = write(result, Path(args.output_dir))
    print(f"\n{md.read_text()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
