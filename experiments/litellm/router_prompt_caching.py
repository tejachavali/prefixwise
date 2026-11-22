"""Finding 2: does the Router's prompt_caching check keep a conversation on one deployment?

Two cachemock deployments (ports 18601 and 18602), one mini-swe-agent trajectory with fresh text per
run, 12 turns; prints the deployment (0 or 1) that served each turn.
  anthropic-markers MODEL   cache_control written in the system message by the caller
  anthropic-inject MODEL    markers added through cache_control_injection_points
  openai                    OpenAI format (automatic caching, no markers)
"""
import asyncio, copy, sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litellm
from litellm import Router
import translate_b as tb
print("litellm.callbacks at start:", litellm.callbacks)
name, msgs, tools = list(tb.load_mini(1))[0]
tag = uuid.uuid4().hex[:8]
msgs = copy.deepcopy(msgs)
fmt = sys.argv[1]
if fmt == "anthropic-markers":
    msgs[0] = {"role": "system", "content": [{"type": "text", "text": f"[{tag}] " + msgs[0]["content"], "cache_control": {"type": "ephemeral"}}]}
else:
    msgs[0]["content"] = f"[{tag}] " + msgs[0]["content"]
steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and msgs[i-1]["role"] != "assistant"][:12]
model = "openai/gpt-4o" if fmt == "openai" else ("anthropic/" + (sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-5"))
dep = [{"model_name": "group", "litellm_params": {"model": model, "api_key": "k",
        "api_base": f"http://127.0.0.1:{p}" + ("/v1" if fmt == "openai" else "")}} for p in (18601, 18602)]
async def main():
    r = Router(model_list=dep, routing_strategy="simple-shuffle", optional_pre_call_checks=["prompt_caching"], num_retries=0)
    print("callbacks after Router():", [type(c).__name__ for c in litellm.callbacks])
    ids = {d["model_info"]["id"]: n for n, d in enumerate(r.model_list)}
    seq = []
    for i in steps:
        extra = ({"cache_control_injection_points": [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]}
                 if fmt == "anthropic-inject" else {})
        resp = await r.acompletion(model="group", messages=msgs[:i], tools=tools, **extra)
        seq.append(ids.get((resp._hidden_params or {}).get("model_id")))
        await asyncio.sleep(0.2)
    print(fmt, "deployment per turn:", seq, "tokens(turn1):", litellm.token_counter(model=model, messages=msgs[:steps[0]], tools=tools))
asyncio.run(main())
