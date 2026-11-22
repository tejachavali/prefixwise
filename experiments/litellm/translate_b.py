"""Test B: is LiteLLM's translation append-only on real agent conversations?

For each real trajectory (OpenAI chat format, as the agent sent it), every model call's request is the
message list before that assistant message. Each request is translated by LiteLLM for each target
provider format (the body LiteLLM builds, captured in log_pre_api_call; the endpoint is a closed port,
so nothing is sent anywhere). Prompt caching needs step N's translated prompt to be an exact prefix of
step N+1's: same tools and system, and step N's messages unchanged at the front of step N+1's. Cache
markers (cache_control) are ignored, since they are expected to move.

Usage: python translate_b.py OUT.json [n_mini] [n_tau]
"""
import ast, glob, json, os, sys, time
import litellm
from litellm.integrations.custom_logger import CustomLogger

D = os.environ.get("PREFIXWISE_DATASETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"))
DEAD = "http://127.0.0.1:9"
os.environ.update(AWS_ACCESS_KEY_ID="AKIAFAKE", AWS_SECRET_ACCESS_KEY="fake", AWS_REGION_NAME="us-east-1")
litellm.suppress_debug_info = True

TARGETS = {  # name: (model, message-list key in the body)
    "openai": ("openai/gpt-4o", "messages"),
    "anthropic": ("anthropic/claude-haiku-5-5", "messages"),
    "bedrock-converse": ("bedrock/converse/anthropic.claude-3-5-haiku-20241022-v1:0", "messages"),
    "bedrock-invoke": ("bedrock/invoke/anthropic.claude-3-5-haiku-20241022-v1:0", "messages"),
    "gemini": ("gemini/gemini-2.5-flash", "contents"),
}

captured = []
class Cap(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        captured.append((kwargs.get("additional_args") or {}).get("complete_input_dict"))
litellm.callbacks = [Cap()]


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def norm_message(m):
    out = {"role": m["role"]}
    if m.get("content") is not None or m["role"] != "assistant":
        out["content"] = m.get("content") if m.get("content") is not None else ""
    calls = m.get("tool_calls")
    if calls:
        out["tool_calls"] = []
        for c in calls:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            out["tool_calls"].append({"id": c.get("id"), "type": "function",
                                      "function": {"name": fn.get("name"),
                                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id")
    return out


def load_mini(n):
    src = open(f"{D}/SWE-agent_mini-swe-agent/src/minisweagent/models/utils/actions_toolcall.py").read()
    tree = ast.parse(src)
    bash = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BASH_TOOL")
    for f in sorted(glob.glob(f"{D}/mini-gemini35/trajs/*.traj.json"))[:n]:
        t = json.load(open(f))
        yield "mini-swe-agent/" + os.path.basename(f), [norm_message(m) for m in t["messages"]], [bash]


def load_tau(n):
    files = sorted(glob.glob(f"{D}/sierra-research_tau-bench/historical_trajectories/*.json"))
    per = max(1, n // len(files))
    for f in files:
        for i, e in enumerate(json.load(open(f))[:per]):
            msgs = [norm_message(m) for m in e["traj"]]
            schema = {}  # tool schemas from the arguments the run used (the repo's own live in Python code)
            for m in msgs:
                for c in m.get("tool_calls") or []:
                    props = schema.setdefault(c["function"]["name"], {})
                    try:
                        for k in json.loads(c["function"]["arguments"] or "{}"):
                            props[k] = {"type": "string"}
                    except ValueError:
                        pass
            tools = [{"type": "function", "function": {"name": k, "description": k,
                                                       "parameters": {"type": "object", "properties": v}}}
                     for k, v in sorted(schema.items())]
            yield f"tau-bench/{os.path.basename(f)}#{i}", msgs, tools


def translate(target, messages, tools):
    model, _ = TARGETS[target]
    kw = dict(model=model, messages=messages, num_retries=0, max_retries=0, timeout=2, api_base=DEAD)
    if tools:
        kw["tools"] = tools
    if target in ("openai", "anthropic", "gemini"):
        kw["api_key"] = "k"
    captured.clear()
    try:
        litellm.completion(**kw)
    except Exception:
        pass
    body = captured[-1] if captured else None
    if isinstance(body, str):
        body = json.loads(body)
    return strip(body) if isinstance(body, dict) else None


def check(name, msgs, tools, results):
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    for target, (_, key) in TARGETS.items():
        prev = None
        r = results.setdefault(target, {"pairs": 0, "ok": 0, "untranslated": 0, "examples": []})
        for i in steps:
            body = translate(target, msgs[:i], tools)
            if body is None:
                r["untranslated"] += 1
                prev = None
                continue
            if prev is not None:
                r["pairs"] += 1
                a, b = prev[key], body[key]
                rest_a = {k: v for k, v in prev.items() if k != key}
                rest_b = {k: v for k, v in body.items() if k != key}
                same_rest = rest_a == rest_b
                prefix = len(a) <= len(b) and b[:len(a)] == a
                if same_rest and prefix:
                    r["ok"] += 1
                elif len(r["examples"]) < 5:
                    where = None
                    if not prefix:
                        where = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                    r["examples"].append({"trajectory": name, "step": i, "other_fields_equal": same_rest,
                                          "changed_keys": sorted(k for k in set(rest_a) | set(rest_b)
                                                                 if rest_a.get(k) != rest_b.get(k)),
                                          "first_changed_message": where,
                                          "before": json.dumps(a[where])[:300] if where is not None and where < len(a) else None,
                                          "after": json.dumps(b[where])[:300] if where is not None and where < len(b) else None})
            prev = body


if __name__ == "__main__":
    out = sys.argv[1]
    n_mini = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_tau = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    results, t0, n = {}, time.time(), 0
    for name, msgs, tools in list(load_mini(n_mini)) + list(load_tau(n_tau)):
        check(name, msgs, tools, results)
        n += 1
    json.dump({"litellm": __import__("importlib.metadata").metadata.version("litellm"), "trajectories": n,
               "seconds": round(time.time() - t0), "results": results}, open(out, "w"), indent=1)
    for t, r in results.items():
        print(f"{t:18s} pairs={r['pairs']:5d} append-only={r['ok']:5d} broken={r['pairs'] - r['ok']:5d} untranslated={r['untranslated']}")
