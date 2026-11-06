"""Point an OpenAI SDK agent at Prefixwise and record task outcomes.

    prefixwise serve --mock            # or: --upstream https://api.openai.com
    pip install openai && python examples/openai_agent.py
    prefixwise report --by task_id
"""
import json
import urllib.request

from openai import OpenAI

PROXY = "http://127.0.0.1:8080"


def record_outcome(task_id: str, success: bool) -> None:
    req = urllib.request.Request(f"{PROXY}/tasks", method="POST",
                                 data=json.dumps({"task_id": task_id, "success": success}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).close()


def run_task(task_id: str, question: str) -> bool:
    client = OpenAI(base_url=f"{PROXY}/v1", api_key="sk-...",  # your key, or any string in mock mode
                    default_headers={"X-Prefixwise-Task-Id": task_id,
                                     "X-Prefixwise-Session-Id": task_id,
                                     "X-Prefixwise-Agent-Id": "my-agent-v1"})
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}]
    for _ in range(3):  # a tiny agent loop: the conversation grows each turn
        raw = client.chat.completions.with_raw_response.create(model="gpt-5.6", messages=messages)
        reply = raw.parse().choices[0].message.content
        print(f"{task_id}: cause={raw.headers['x-prefixwise-cause']} "
              f"avoidable={raw.headers['x-prefixwise-avoidable-tokens']}")
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Continue."}]
    return True  # replace with your own success check (tests pass, answer graded, ...)


if __name__ == "__main__":
    for i, q in enumerate(["Summarise RFC 9110 in one line.", "Name three HTTP methods."]):
        task = f"task-{i}"
        record_outcome(task, run_task(task, q))
