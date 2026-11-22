"""Test D: LiteLLM Router across two deployments of one model, each a provider with its own cache
(two cachemock processes: two replicas, or two API keys of one provider). Real trajectories are
replayed with each routing setup. Ground truth: the deployment that served each call (the response's
model_id). A turn moved to the other deployment misses its cache, and Prefixwise should call it
routing_miss (inferred on a hosted API); a turn kept on its deployment should not be.

Usage: python router_d.py OUT.json n_mini n_tau PORT_A PORT_B
"""
import asyncio, copy, json, os, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litellm
from litellm import Router
import translate_b as tb
from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.sdk import Recorder

OUT, N_MINI, N_TAU, PA, PB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
LEDGER = OUT.replace(".json", ".db")
if os.path.exists(LEDGER):
    os.remove(LEDGER)
litellm.callbacks = [PrefixwiseCallback(recorder=Recorder(LEDGER, agent_id="router"))]


def deployments(fmt):
    out = []
    for port in (PA, PB):
        if fmt == "openai":
            out.append({"model_name": "group", "litellm_params": {"model": "openai/gpt-4o", "api_key": "k",
                                                                  "api_base": f"http://127.0.0.1:{port}/v1"}})
        else:
            out.append({"model_name": "group", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "k",
                                                                  "api_base": f"http://127.0.0.1:{port}"}})
    return out


async def replay(router, name, msgs, tools, fmt, setup):
    session = f"{setup}|{fmt}|{name}"
    msgs = copy.deepcopy(msgs)
    msgs[0] = {"role": "system", "content": f"[{session}] " + str(msgs[0]["content"])}
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    served = []
    for i in steps:
        # session_affinity reads metadata.session_id (not litellm_session_id); Prefixwise reads either
        kw = dict(model="group", messages=msgs[:i], litellm_session_id=session, metadata={"session_id": session})
        if tools:
            kw["tools"] = tools
        if fmt == "anthropic":
            kw["cache_control_injection_points"] = [{"location": "message", "role": "system"},
                                                    {"location": "message", "index": -1}]
        r = await router.acompletion(**kw)
        served.append((r._hidden_params or {}).get("model_id"))
        await asyncio.sleep(0.01)
    return session, served


async def main():
    trajs = list(tb.load_mini(N_MINI)) + list(tb.load_tau(N_TAU))
    cases = []
    for setup in ("load-balanced", "prompt_caching", "session_affinity"):
        for fmt in ("openai", "anthropic"):
            checks = [] if setup == "load-balanced" else [setup]
            router = Router(model_list=deployments(fmt), routing_strategy="simple-shuffle",
                            optional_pre_call_checks=checks or None, num_retries=0)
            for name, msgs, tools in trajs:
                cases.append((setup, fmt) + await replay(router, name, msgs, tools, fmt, setup))
    await asyncio.sleep(5)
    return trajs, cases


t0 = time.time()
trajs, cases = asyncio.run(main())
db = sqlite3.connect(LEDGER)
summary = {}
for setup, fmt, session, served in cases:
    rows = list(db.execute("SELECT cause, prompt_tokens, cached_tokens FROM entries WHERE session_id=? ORDER BY id",
                           (session,)))
    s = summary.setdefault(f"{setup}/{fmt}", {"calls": 0, "recorded": 0, "switches": 0, "prompt": 0, "cached": 0,
                                              "routing_miss_on_switch": 0, "other_cause_on_switch": {},
                                              "routing_miss_without_switch": 0})
    s["calls"] += len(served); s["recorded"] += len(rows)
    for n, (cause, prompt, cached) in enumerate(rows):
        s["prompt"] += prompt or 0; s["cached"] += cached or 0
        if n == 0 or n >= len(served):
            continue
        switched = served[n] != served[n - 1]
        if switched:
            s["switches"] += 1
            if cause == "routing_miss":
                s["routing_miss_on_switch"] += 1
            else:
                s["other_cause_on_switch"][cause] = s["other_cause_on_switch"].get(cause, 0) + 1
        elif cause == "routing_miss":
            s["routing_miss_without_switch"] += 1
for v in summary.values():
    v["share_from_cache"] = round(v["cached"] / v["prompt"], 3) if v["prompt"] else None
json.dump({"litellm": __import__("importlib.metadata").metadata.version("litellm"), "trajectories": len(trajs),
           "seconds": round(time.time() - t0), "summary": summary}, open(OUT, "w"), indent=1)
for k, v in summary.items():
    print(k, v)
