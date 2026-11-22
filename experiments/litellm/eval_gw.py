import sqlite3, sys, time
from litellm.harness import Harness
from litellm.harness.evaluation import agent_eval, EvalTask, EvalTarget, PredicateCheck
def read_note(name: str) -> str:
    """Read a note by name."""
    return f"note {name}: " + "lorem ipsum " * 50
tasks = [EvalTask(id=f"task{i}", prompt=f"Task {i}: read the note called n{i} and summarise it.",
                  check=PredicateCheck(lambda r: True)) for i in range(3)]
target = EvalTarget(Harness.TOOL_LOOP, model=f"litellm_proxy/{sys.argv[2]}", tools=[read_note])
card = agent_eval(tasks, [target], trials=2, api_base="http://127.0.0.1:18532", api_key="sk-anything", timeout=60)
print(card.to_markdown().split("\n\n")[0].splitlines()[-1])
time.sleep(3)
db = sqlite3.connect(sys.argv[1])
for r in db.execute("SELECT session_id, task_id, agent_id, cause, prompt_tokens, cached_tokens FROM entries ORDER BY id"):
    print("  ", r)
