"""agent_eval (LiteLLM 1.112.0.dev2), TOOL_LOOP harness, direct mode, Prefixwise callback attached.
Upstream: Prefixwise's mock provider (tool calls on, simulated prefix cache)."""
import json, sqlite3, sys, threading, tempfile
import litellm
from litellm.harness import Harness
from litellm.harness.evaluation import agent_eval, EvalTask, EvalTarget, PredicateCheck
from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder

proxy = PrefixwiseProxy(mock=True); proxy.mock_tool_calls = True
srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
ledger = sys.argv[1]
cb = PrefixwiseCallback(recorder=Recorder(ledger, agent_id="litellm"))
litellm.callbacks = [cb]

def read_note(name: str) -> str:
    """Read a note by name."""
    return f"note {name}: " + "lorem ipsum " * 50

tasks = [EvalTask(id=f"task{i}", prompt=f"Task {i}: read the note called n{i} and summarise it.",
                  check=PredicateCheck(lambda r: True)) for i in range(3)]
target = EvalTarget(Harness.TOOL_LOOP, model="openai/gpt-5.6", tools=[read_note])
card = agent_eval(tasks, [target], trials=2, api_base=base, api_key="sk-test", timeout=60)
print(card.to_markdown().split("\n\n")[0])
import time; time.sleep(2)
db = sqlite3.connect(ledger)
cols = [r[1] for r in db.execute("PRAGMA table_info(entries)")]
print("rows:")
for r in db.execute("SELECT session_id, task_id, agent_id, cause, prompt_tokens, cached_tokens, evidence FROM entries ORDER BY id"):
    print("  ", r[:6], "|", (r[6] or "")[:90])
