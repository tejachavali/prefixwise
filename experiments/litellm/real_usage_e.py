"""Test E: Prefixwise's verdicts against a real provider's cache numbers.

A real Claude Opus 4.6 agent run (mini-swe-agent on SWE-bench, from swe-bench/experiments
analysis/test_data), recorded with LiteLLM, cache_control at the end of the prompt (set_cache_control:
default_end), and Anthropic's own usage per call (cached tokens read, cache writes). Each call's request
is rebuilt as LiteLLM sends it to Anthropic (translation captured as in test B, marker on the last
message) and paired with the real usage, then imported with `prefixwise import gateway`.

Usage: python real_usage_e.py OUT.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_b as tb
import litellm

D = tb.D + "/swebench-experiments/analysis/test_data/claude-4-6-opus.json"
t = json.load(open(D))
raw = t["messages"]
msgs = [tb.norm_message(m) for m in raw]
bash = next(iter(tb.load_mini(1)))[2]
out = open(sys.argv[1], "w")
n = 0
prev_end = None
for i, m in enumerate(raw):
    if m.get("role") != "assistant":
        continue
    extra = m.get("extra") or {}
    usage = (extra.get("response") or {}).get("usage") or {}
    d = usage.get("prompt_tokens_details") or {}
    prompt, read, write = usage.get("prompt_tokens") or 0, d.get("cached_tokens") or 0, d.get("cache_creation_tokens") or 0
    tb.captured.clear()
    try:
        litellm.completion(model="anthropic/claude-opus-4-6", messages=msgs[:i], tools=bash, api_key="k",
                           api_base=tb.DEAD, num_retries=0, max_retries=0, timeout=2,
                           cache_control_injection_points=[{"location": "message", "index": -1}])
    except Exception:
        pass
    body = tb.captured[-1]
    body = json.loads(body) if isinstance(body, str) else body
    end = extra.get("timestamp")
    start = max(prev_end or end - 2, end - 30) if prev_end else end - 2
    response = {"id": f"msg_{n}", "type": "message", "model": "claude-opus-4-6",
                "usage": {"input_tokens": prompt - read - write, "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write, "output_tokens": usage.get("completion_tokens") or 0}}
    out.write(json.dumps({"request": body, "response": response, "start_time": end - 1.0, "end_time": end,
                          "provider": "anthropic", "headers": {"X-Prefixwise-Session-Id": "opus-run"}}) + "\n")
    prev_end = end
    n += 1
out.close()
print("calls", n)
