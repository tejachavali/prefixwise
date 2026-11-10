"""JSON Schema arrays that are sets (`required`, a list `type`) in another order are the same tool:
the detector names the miss tool_reserialize (waste), and canonical_tools sends the same bytes.
Every other array keeps its order. From a Reddit question (r/LLMDevs) about nested tool schemas."""

import json

from prefixwise.classifier import classify
from prefixwise.details import body_facts
from prefixwise.fingerprint import Fingerprinter, schema_sets
from prefixwise.fixes import canonical_tools
from prefixwise.classifier import RequestSnapshot
from prefixwise.proxy import extract_blocks


def tool(required=("since", "lang"), filters_required=None, **extra):
    filters = {"type": "object",
               "properties": {"lang": {"type": "string", "enum": ["en", "fr"]}, "since": {"type": ["string", "null"]}},
               "required": list(filters_required or required)}
    params = {"type": "object", "properties": {"q": {"type": "string"}, "filters": filters},
              "required": ["q", "filters"]}
    params["properties"].update(extra)
    return {"type": "function", "function": {"name": "search", "parameters": params}}


def verdict(tools_a, tools_b):
    fp = Fingerprinter()
    msgs = [{"role": "system", "content": "You are a search agent."}, {"role": "user", "content": "find it"}]

    def snap(tools, cached):
        body = {"model": "gpt-5", "tools": tools, "messages": msgs}
        return RequestSnapshot(model="gpt-5", provider="openai", blocks=fp.fingerprint_blocks(extract_blocks(body, "openai")),
                               prompt_tokens=2000, cached_tokens=cached, gap_seconds=5.0)
    return classify(snap(tools_a, 0), snap(tools_b, 0)).cause.value


def test_a_reordered_required_list_is_the_same_tool_and_the_fix_sends_the_same_bytes():
    a, b = [tool(("since", "lang"))], [tool(("lang", "since"))]
    assert verdict(a, b) == "tool_reserialize"  # was tool_catalog_change, a "legitimate" change
    assert json.dumps(canonical_tools(a)) == json.dumps(canonical_tools(b))


def test_a_reordered_type_list_is_the_same_tool():
    a = [tool()]
    b = [tool()]
    b[0]["function"]["parameters"]["properties"]["filters"]["properties"]["since"]["type"] = ["null", "string"]
    assert verdict(a, b) == "tool_reserialize"
    assert json.dumps(canonical_tools(a)) == json.dumps(canonical_tools(b))


def test_a_real_change_to_required_is_still_a_change():
    assert verdict([tool(("since", "lang"))], [tool(("lang",))]) == "tool_catalog_change"


def test_enum_and_anyof_order_are_kept():
    a, b = [tool()], [tool()]
    b[0]["function"]["parameters"]["properties"]["filters"]["properties"]["lang"]["enum"] = ["fr", "en"]
    assert verdict(a, b) == "tool_catalog_change"
    assert canonical_tools(b)[0]["function"]["parameters"]["properties"]["filters"]["properties"]["lang"]["enum"] == ["fr", "en"]
    any_of = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert schema_sets(any_of) == any_of
    assert schema_sets({"anyOf": [{"type": "integer"}, {"type": "string"}]})["anyOf"][0] == {"type": "integer"}


def test_literals_and_property_names_are_untouched():
    schema = {"type": "object",
              "properties": {"required": {"type": "array", "items": {"type": "string"},
                                          "default": ["b", "a"], "examples": [["z", "y"]]},
                             "default": {"type": "object", "properties": {"x": {"type": "string"}},
                                         "required": ["x"]}},
              "required": ["required", "default"],
              "default": {"required": ["b", "a"], "type": ["string", "null"]},
              "const": {"type": ["string", "null"]}}
    out = schema_sets(schema)
    assert out["required"] == ["default", "required"]  # the schema's own set: sorted
    assert out["properties"]["required"]["default"] == ["b", "a"]  # a literal value: kept
    assert out["properties"]["required"]["examples"] == [["z", "y"]]
    assert out["default"] == {"required": ["b", "a"], "type": ["string", "null"]}
    assert out["const"] == {"type": ["string", "null"]}
    assert out["properties"]["default"]["required"] == ["x"]  # a property named "default" is a schema


def test_required_outside_a_schema_object_and_unknown_type_names_are_kept():
    assert schema_sets({"required": ["b", "a"]}) == {"required": ["b", "a"]}  # no properties, not type object
    assert schema_sets({"type": ["string", "Custom"]}) == {"type": ["string", "Custom"]}


def test_the_cross_session_prefix_key_agrees():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    fa = body_facts("openai", {"model": "m", "tools": [tool(("since", "lang"))], "messages": msgs}, model="m")
    fb = body_facts("openai", {"model": "m", "tools": [tool(("lang", "since"))], "messages": msgs}, model="m")
    assert fa.prefix_key_canonical == fb.prefix_key_canonical
    fc = body_facts("openai", {"model": "m", "tools": [tool(("lang",))], "messages": msgs}, model="m")
    assert fc.prefix_key_canonical != fa.prefix_key_canonical


def test_a_type_list_holding_a_schema_is_kept_and_never_an_error():
    """Review: a draft-03 union type (`[{"type": "string"}, "null"]`) made the type-name check raise, and
    body_facts runs on every request, so the proxy answered 500 even with the fix off."""
    union = {"type": [{"type": "string"}, "null"]}
    t = tool(note=union)
    body = {"model": "gpt-5", "tools": [t], "messages": [{"role": "user", "content": "hi"}]}
    body_facts("openai", body)  # no error
    assert canonical_tools([t])[0]["function"]["parameters"]["properties"]["note"] == union
    assert schema_sets({"type": [["string"], "null"]}) == {"type": [["string"], "null"]}


def test_an_anthropic_input_schema_is_the_same_tool_in_another_required_order():
    def ant(required):
        return {"name": "search", "description": "Search.",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}, "k": {"type": "integer"}},
                                 "required": required}}
    a, b = ant(["q", "k"]), ant(["k", "q"])
    assert json.dumps(canonical_tools([a])) == json.dumps(canonical_tools([b]))
    fp = Fingerprinter()
    msgs = [{"role": "user", "content": "find it"}]

    def blocks(t):
        return fp.fingerprint_blocks(extract_blocks({"model": "claude-x", "tools": [t], "messages": msgs}, "anthropic"))
    tools_a = [x for x in blocks(a) if x.role == "tools"]
    tools_b = [x for x in blocks(b) if x.role == "tools"]
    assert [x.canonical for x in tools_a] == [x.canonical for x in tools_b]
    assert [x.raw for x in tools_a] != [x.raw for x in tools_b]


def test_a_schema_too_deep_to_walk_is_left_as_it_is():
    deep = {"type": "object", "properties": {}}
    node = deep
    for _ in range(5000):
        nxt = {"type": "object", "properties": {}}
        node["properties"]["x"] = nxt
        node = nxt
    assert schema_sets(deep) is deep  # returned unchanged, no RecursionError
