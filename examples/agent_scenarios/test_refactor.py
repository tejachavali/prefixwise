"""A scenario test for `prefixwise check`: one small agent loop, run against the gate's mock.

    prefixwise check -- pytest examples/agent_scenarios

`prefixwise check` starts a local mock provider and puts its address in OPENAI_BASE_URL and
ANTHROPIC_BASE_URL, which the SDKs read; no key is used and nothing leaves the machine. This agent
has a bug no functional test catches: its tool list comes back in another order each turn (here a
rotation; in real code, a list built from a set or a dict of plugins). The gate fails on it.
Set SORT_TOOLS=1 to send the tools in one order: the gate passes.
"""

import os

from openai import OpenAI

TOOLS = [
    {"type": "function",
     "function": {"name": name, "description": f"The {name} tool.",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    for name in ("read_file", "run_tests", "edit_file")
]


def run_agent(task: str, turns: int = 3) -> list[dict]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-used-by-the-mock"))
    messages = [{"role": "system", "content": "You are a coding agent. " * 300},
                {"role": "user", "content": task}]
    for turn in range(turns):
        tools = TOOLS if os.environ.get("SORT_TOOLS") else TOOLS[turn % 3:] + TOOLS[:turn % 3]
        reply = client.chat.completions.create(model="gpt-5.6-luna", messages=messages, tools=tools)
        message = reply.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls or []:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
        if not message.tool_calls:
            messages.append({"role": "user", "content": "Go on."})
    return messages


def test_refactor():
    assert len(run_agent("Fix the failing test.")) > 4
