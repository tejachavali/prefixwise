"""Test C: real agent conversations replayed through LiteLLM with Prefixwise's callback attached,
against a provider with a content-addressed prompt cache (cachemock.py).

Each trajectory is replayed call by call (the request before each assistant message), in one session.
Either unchanged (every recorded cause must be first_write, none, or a legitimate one: no waste on an
append-only conversation), or with one change injected from a random step k on (that step must be
named as the injected cause; no other step may be named as waste).

Usage: python replay_c.py OUT.json n_mini n_tau seed
"""
import copy, json, os, random, sqlite3, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litellm
import cachemock
import translate_b as tb
from http.server import ThreadingHTTPServer
from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.sdk import Recorder

OUT, N_MINI, N_TAU, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
LEDGER = OUT.replace(".json", ".db")
if os.path.exists(LEDGER):
    os.remove(LEDGER)
srv = ThreadingHTTPServer(("127.0.0.1", 0), cachemock.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"
litellm.callbacks = [PrefixwiseCallback(recorder=Recorder(LEDGER, agent_id="replay"))]
rng = random.Random(SEED)
LEGIT = {"first_write", "none", "below_min_prefix"}
STAMP = " Current time: 2026-11-22T10:00:01Z."


def inject(kind, msgs, tools):
    """The change, applied to a copy of (msgs, tools); None when this trajectory cannot carry it."""
    msgs, tools = copy.deepcopy(msgs), copy.deepcopy(tools)
    if kind == "tool_reserialize":
        if len(tools) < 2:
            return None
        tools = tools[1:] + tools[:1]
    elif kind == "volatile_system":
        msgs[0]["content"] = msgs[0]["content"].replace(STAMP, " Current time: 2026-11-22T10:07:42Z.")
    elif kind == "system_edit":
        msgs[0]["content"] = "You must be concise. " + msgs[0]["content"]
    elif kind == "history_edit":
        msgs[1]["content"] = str(msgs[1]["content"]) + " (edited)"
    elif kind == "tool_result_mutation":
        j = next((i for i, m in enumerate(msgs) if m["role"] == "tool"), None)
        if j is None:
            return None
        msgs[j]["content"] = "[cleared] " + str(msgs[j]["content"])
    return msgs, tools


def replay(name, msgs, tools, provider, kind):
    session = f"{provider}|{kind}|{name}"
    msgs = copy.deepcopy(msgs)
    msgs[0] = {"role": "system", "content": f"[{session}] " + str(msgs[0]["content"]) + STAMP}  # unique per replay
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    if len(steps) < 4:
        return None
    k_index = rng.randrange(2, len(steps)) if kind != "unchanged" else None
    changed = inject(kind, msgs, tools) if kind != "unchanged" else None
    if kind != "unchanged" and changed is None:
        return None
    if kind == "tool_result_mutation":  # the edited tool result must already be in the history at step k
        j = next(i for i, m in enumerate(msgs) if m["role"] == "tool")
        k_index = max(k_index, next(n for n, s in enumerate(steps) if s > j + 1))
        if k_index >= len(steps):
            return None
    for n, i in enumerate(steps):
        m, t = (changed if k_index is not None and n >= k_index else (msgs, tools))
        kw = dict(model="openai/gpt-4o" if provider == "openai" else "anthropic/claude-haiku-5-5",
                  messages=m[:i], api_key="k", litellm_session_id=session, num_retries=0)
        if t:
            kw["tools"] = t
        if provider == "openai":
            kw["api_base"] = BASE + "/v1"
        else:
            kw["api_base"] = BASE
            kw["cache_control_injection_points"] = [{"location": "message", "role": "system"},
                                                    {"location": "message", "index": -1}]
        litellm.completion(**kw)
    return session, len(steps), k_index


cases = []
trajs = list(tb.load_mini(N_MINI)) + list(tb.load_tau(N_TAU))
t0 = time.time()
for name, msgs, tools in trajs:
    for provider in ("openai", "anthropic"):
        for kind in ("unchanged", "tool_reserialize", "volatile_system", "system_edit", "history_edit",
                     "tool_result_mutation"):
            r = replay(name, msgs, tools, provider, kind)
            if r:
                cases.append((provider, kind) + r)
time.sleep(5)
db = sqlite3.connect(LEDGER)
summary, bad = {}, []
for provider, kind, session, n_steps, k in cases:
    rows = list(db.execute("SELECT cause, secondary_causes, prompt_tokens, cached_tokens FROM entries "
                           "WHERE session_id=? ORDER BY id", (session,)))
    s = summary.setdefault(f"{provider}/{kind}", {"replays": 0, "calls": 0, "recorded": 0, "false_alarms": 0,
                                                   "injected_named": 0, "injected_seen_cost_nothing": 0,
                                                   "injected_missed": 0})
    s["replays"] += 1; s["calls"] += n_steps; s["recorded"] += len(rows)
    for n, (cause, secondary, prompt, cached) in enumerate(rows):
        if k is not None and n == k:
            if cause == kind:
                s["injected_named"] += 1
            elif cause == "none" and kind in (secondary or ""):
                s["injected_seen_cost_nothing"] += 1
            else:
                s["injected_missed"] += 1
                if len(bad) < 20:
                    bad.append({"session": session, "step": n, "expected": kind, "got": cause,
                                "secondary": secondary, "prompt": prompt, "cached": cached})
        elif cause not in LEGIT:
            s["false_alarms"] += 1
            if len(bad) < 20:
                bad.append({"session": session, "step": n, "injected_at": k, "got": cause, "secondary": secondary,
                            "prompt": prompt, "cached": cached})
json.dump({"litellm": __import__("importlib.metadata").metadata.version("litellm"), "seed": SEED,
           "trajectories": len(trajs), "seconds": round(time.time() - t0), "summary": summary, "problems": bad},
          open(OUT, "w"), indent=1)
for k, v in summary.items():
    print(f"{k:34s} {v}")
print("problems:", len(bad))
for b in bad[:8]:
    print("  ", b)
