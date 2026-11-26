"""Claude on Amazon Bedrock, recorded from boto3. The responses are botocore's Stubber, which checks
every request and response against the service's own published model: no AWS account, no network."""

import io
import json

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.response import StreamingBody  # noqa: E402
from botocore.stub import Stubber  # noqa: E402

import prefixwise  # noqa: E402
from prefixwise.bedrock import anthropic_usage, converse_request  # noqa: E402
from prefixwise.pricing import PriceTable  # noqa: E402
from prefixwise.sdk import Recorder  # noqa: E402

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
SYSTEM = "You are careful. " * 300


def client():
    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="AKIDEXAMPLE",
                        aws_secret_access_key="example")


def rows(rec):
    with rec.ledger.reader() as c:
        return c.execute("SELECT provider, model, session_id, task_id, cause, prompt_tokens, cached_tokens, "
                         "cache_write_tokens, cost, error FROM entries ORDER BY id").fetchall()


def usage(fresh=50, read=None, write=None, out=10, details=None):
    u = {"inputTokens": fresh, "outputTokens": out, "totalTokens": fresh + out + (read or 0) + (write or 0)}
    if read is not None:
        u["cacheReadInputTokens"] = read
    if write is not None:
        u["cacheWriteInputTokens"] = write
    if details is not None:
        u["cacheDetails"] = details
    return u


def reply(u, text="ok", request_id="req-1", stop="end_turn"):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": stop,
            "usage": u, "metrics": {"latencyMs": 120}, "ResponseMetadata": {"RequestId": request_id}}


SYS = [{"text": SYSTEM}, {"cachePoint": {"type": "default"}}]
U1 = {"role": "user", "content": [{"text": "hi"}]}
A1 = {"role": "assistant", "content": [{"text": "ok"}]}
U2 = {"role": "user", "content": [{"text": "more"}]}


def test_converse_is_recorded_and_judged_by_anthropics_rules():
    rec = Recorder(":memory:")
    c = client()
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000), request_id="r1"),
                          {"modelId": MODEL, "system": SYS, "messages": [U1]})
        stub.add_response("converse", reply(usage(fresh=30, read=3000, write=0), request_id="r2"),
                          {"modelId": MODEL, "system": SYS, "messages": [U1, A1, U2]})
        wrapped = prefixwise.wrap(c, recorder=rec)
        assert wrapped is c
        with prefixwise.session("b1", task_id="t1"):
            first = c.converse(modelId=MODEL, system=SYS, messages=[U1])
            second = c.converse(modelId=MODEL, system=SYS, messages=[U1, A1, U2])
        stub.assert_no_pending_responses()
    assert first["output"]["message"]["content"] == [{"text": "ok"}]  # the SDK's own response, unchanged
    assert second["usage"]["cacheReadInputTokens"] == 3000
    out = rows(rec)
    assert [r[:8] for r in out] == [
        ("anthropic", MODEL, "b1", "t1", "first_write", 3020, 0, 3000),
        ("anthropic", MODEL, "b1", "t1", "none", 3030, 3000, 0)]
    assert all(r[8] and r[8] > 0 for r in out)  # priced: the Bedrock id names the model


def test_a_changed_system_prompt_on_bedrock_is_named():
    rec = Recorder(":memory:")
    c = client()
    changed = [{"text": "Today is 2026-11-26. " + SYSTEM}, {"cachePoint": {"type": "default"}}]
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3010)))
        prefixwise.wrap(c, recorder=rec)
        with prefixwise.session("b2"):
            c.converse(modelId=MODEL, system=SYS, messages=[U1])
            c.converse(modelId=MODEL, system=changed, messages=[U1, A1, U2])
    assert rows(rec)[1][4] in ("system_edit", "volatile_system")


def test_usage_is_read_as_bedrock_reports_it():
    # inputTokens is what was neither read nor written, as Anthropic's input_tokens is
    assert anthropic_usage(usage(fresh=7, read=100, write=40, out=3)) == {
        "input_tokens": 7, "output_tokens": 3, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 40}
    # the split by lifetime, when it accounts for the whole write
    split = anthropic_usage(usage(write=40, details=[{"ttl": "1h", "inputTokens": 30}, {"ttl": "5m", "inputTokens": 10}]))
    assert split["cache_creation"] == {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 30}
    assert "cache_creation" not in anthropic_usage(usage(write=40, details=[{"ttl": "1h", "inputTokens": 30}]))
    # no cache counts at all (no cachePoint): the write is not reported, not zero
    plain = anthropic_usage(usage())
    assert "cache_creation_input_tokens" not in plain and "cache_read_input_tokens" not in plain


def test_a_one_hour_write_is_priced_at_the_one_hour_rate():
    rec = Recorder(":memory:")
    c = client()
    one_hour = [{"text": SYSTEM}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=0, read=0, write=3000,
                                                  details=[{"ttl": "1h", "inputTokens": 3000}])))
        prefixwise.wrap(c, recorder=rec)
        c.converse(modelId=MODEL, system=one_hour, messages=[U1])
    rates = PriceTable.load("").rates_for(MODEL)
    cost = rows(rec)[0][8]
    assert cost == pytest.approx((3000 * rates["cache_write_1h"] + 10 * rates["output"]) / 1e6)
    body = converse_request({"modelId": MODEL, "system": one_hour, "messages": [U1]})
    assert body["system"] == [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def test_tools_and_inference_config_are_read():
    body = converse_request({
        "modelId": MODEL, "messages": [U1], "inferenceConfig": {"maxTokens": 64, "temperature": 0.2},
        "toolConfig": {"tools": [{"toolSpec": {"name": "read", "description": "read a file",
                                               "inputSchema": {"json": {"type": "object"}}}},
                                 {"cachePoint": {"type": "default"}}]}})
    assert body["tools"] == [{"name": "read", "description": "read a file", "input_schema": {"type": "object"},
                              "cache_control": {"type": "ephemeral"}}]
    assert body["max_tokens"] == 64 and body["temperature"] == 0.2


def test_invoke_model_is_recorded_and_the_caller_still_reads_the_body():
    rec = Recorder(":memory:")
    c = client()
    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16,
               "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
               "messages": [{"role": "user", "content": "hi"}]}
    answer = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
              "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
              "usage": {"input_tokens": 20, "output_tokens": 5, "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 3000}}
    data = json.dumps(answer).encode()
    with Stubber(c) as stub:
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec)
        with prefixwise.session("i1"):
            result = c.invoke_model(modelId=MODEL, body=json.dumps(request))
    assert json.loads(result["body"].read()) == answer  # the caller's body: the same bytes
    out = rows(rec)
    assert [r[:8] for r in out] == [("anthropic", MODEL, "i1", None, "first_write", 3020, 0, 3000)]


def test_an_application_inference_profile_on_invoke_model_is_judged_by_the_model_it_names():
    rec = Recorder(":memory:")
    c = client()
    arn = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/abc123"
    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    answer = {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
              "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
              "usage": {"input_tokens": 20, "output_tokens": 5}}
    data = json.dumps(answer).encode()
    with Stubber(c) as stub:
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec)
        c.invoke_model(modelId=arn, body=json.dumps(request).encode())
    out = rows(rec)
    assert out[0][1] == "claude-sonnet-4-5-20250929" and out[0][8] and out[0][8] > 0


def test_other_models_streams_and_file_bodies_are_not_recorded_and_pass_through():
    rec = Recorder(":memory:")
    c = client()
    nova = {"text": "x"}
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage()))
        data = b'{"output": "x"}'
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec)
        c.converse(modelId="amazon.nova-pro-v1:0", messages=[{"role": "user", "content": [nova]}])
        r = c.invoke_model(modelId="amazon.titan-text-express-v1", body=b'{"inputText": "hi"}')
        assert r["body"].read() == data  # not Anthropic's body: untouched
        f = c.invoke_model(modelId=MODEL, body=io.BytesIO(b'{"anthropic_version": "x", "messages": []}'))
        assert f["body"].read() == data  # a file body is not read
        # another model's body with messages and a usage block (Mistral's shape): not Anthropic's
        other = json.dumps({"choices": [{"message": {"content": "x"}}],
                            "usage": {"prompt_tokens": 9, "completion_tokens": 1}}).encode()
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(other), len(other)),
                                           "contentType": "application/json"})
        m = c.invoke_model(modelId="mistral.mistral-large-2407-v1:0",
                           body=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}))
        assert m["body"].read() == other
    assert rows(rec) == []


def test_a_call_that_raises_is_not_recorded_and_the_error_is_the_callers():
    rec = Recorder(":memory:")
    c = client()
    with Stubber(c) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
        prefixwise.wrap(c, recorder=rec)
        with pytest.raises(c.exceptions.ThrottlingException):
            c.converse(modelId=MODEL, messages=[U1])
    assert rows(rec) == []


def test_recording_cannot_break_the_call(monkeypatch, capsys):
    rec = Recorder(":memory:")
    c = client()
    monkeypatch.setattr("prefixwise.bedrock.converse_request", lambda kwargs, model=None: 1 / 0)
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage()))
        stub.add_response("converse", reply(usage()))
        prefixwise.wrap(c, recorder=rec)
        assert c.converse(modelId=MODEL, messages=[U1])["stopReason"] == "end_turn"
        assert c.converse(modelId=MODEL, messages=[U1])["stopReason"] == "end_turn"
    assert capsys.readouterr().err.count("could not record a Bedrock call") == 1  # said once


def test_a_body_that_fails_to_read_fails_for_the_caller_as_it_would_have():
    rec = Recorder(":memory:")
    c = client()

    class Broken(io.BytesIO):
        """Fails once, then reads as empty, as a reset socket can."""
        failed = False

        def read(self, *a):
            if not Broken.failed:
                Broken.failed = True
                raise OSError("connection reset")
            return b""

    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    with Stubber(c) as stub:
        stub.add_response("invoke_model", {"body": StreamingBody(Broken(b"x"), 1), "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec)
        result = c.invoke_model(modelId=MODEL, body=json.dumps(request))
    assert isinstance(result["body"], StreamingBody)
    with pytest.raises(OSError, match="connection reset"):
        result["body"].read()
    with pytest.raises(OSError, match="connection reset"):
        next(iter(result["body"].iter_chunks()))
    assert [r[9] for r in rows(rec)] == ["not_recorded"]  # a row, so error rates stay honest


def test_wrapping_twice_records_once():
    rec = Recorder(":memory:")
    c = client()
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))
        prefixwise.wrap(prefixwise.wrap(c, recorder=rec), recorder=rec)
        c.converse(modelId=MODEL, system=SYS, messages=[U1])
    assert len(rows(rec)) == 1


def test_another_aws_client_is_refused():
    s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="example")
    with pytest.raises(TypeError, match="Bedrock runtime"):
        prefixwise.wrap(s3)


# -- what an independent review found -------------------------------------------------------------

def test_wrapping_does_not_import_litellm():
    import subprocess
    import sys
    code = ("import sys, prefixwise, boto3, openai; prefixwise.wrap(openai.OpenAI(api_key='x')); "
            "prefixwise.wrap(boto3.client('bedrock-runtime', region_name='us-east-1', aws_access_key_id='a', "
            "aws_secret_access_key='b')); print('litellm' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_a_changed_image_reads_as_changed():
    """boto3 takes images and documents as bytes: they are compared by content."""
    rec = Recorder(":memory:")
    c = client()

    def image(data):
        return {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": data}}}, {"text": "what is it"}]}
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))
        prefixwise.wrap(c, recorder=rec)
        with prefixwise.session("img"):
            c.converse(modelId=MODEL, system=SYS, messages=[image(b"\x89PNG one"), A1, U2])
            c.converse(modelId=MODEL, system=SYS, messages=[image(b"\x89PNG two"), A1, U2])
    second = rows(rec)[1]
    assert second[4] not in ("none", "routing_miss", "ttl_expired"), second
    body = converse_request({"modelId": MODEL, "messages": [image(b"x" * 10000)]})
    assert json.dumps(body)  # all of it is JSON now
    assert "sha256:" in json.dumps(body)  # a long blob is kept as its digest


def test_effort_and_output_format_are_read_from_output_config():
    body = converse_request({"modelId": MODEL, "messages": [U1], "outputConfig": {
        "effort": "low", "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"schema": "{}"}}}}})
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    rec = Recorder(":memory:")
    c = client()
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3010)))
        prefixwise.wrap(c, recorder=rec)
        with prefixwise.session("eff"):
            c.converse(modelId=MODEL, system=SYS, messages=[U1], outputConfig={"effort": "high"})
            c.converse(modelId=MODEL, system=SYS, messages=[U1, A1, U2], outputConfig={"effort": "low"})
    assert rows(rec)[1][4] == "param_rendered"


def test_a_prompt_router_is_judged_by_the_model_it_chose():
    router = "arn:aws:bedrock:us-east-1:111122223333:default-prompt-router/anthropic.claude:1"
    sonnet = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"
    haiku = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
    rec = Recorder(":memory:")
    c = client()
    with Stubber(c) as stub:
        stub.add_response("converse", {**reply(usage(fresh=20, read=0, write=3000)),
                                       "trace": {"promptRouter": {"invokedModelId": sonnet}}})
        stub.add_response("converse", {**reply(usage(fresh=20, read=0, write=3000)),
                                       "trace": {"promptRouter": {"invokedModelId": haiku}}})
        stub.add_response("converse", reply(usage(fresh=20, read=0, write=3000)))  # no trace: cannot tell
        prefixwise.wrap(c, recorder=rec)
        with prefixwise.session("router"):
            c.converse(modelId=router, system=SYS, messages=[U1])
            c.converse(modelId=router, system=SYS, messages=[U1, A1, U2])
            c.converse(modelId=router, system=SYS, messages=[U1])
    out = rows(rec)
    assert [r[1] for r in out] == [sonnet, haiku]
    assert out[1][4] == "model_switch" and all(r[8] and r[8] > 0 for r in out)


def test_provisioned_throughput_and_look_alike_ids_are_not_recorded():
    rec = Recorder(":memory:")
    c = client()
    provisioned = "arn:aws:bedrock:us-east-1:111122223333:provisioned-model/abc123"
    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    data = json.dumps({"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
                       "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 1}}).encode()
    with Stubber(c) as stub:
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        stub.add_response("converse", reply(usage()))
        prefixwise.wrap(c, recorder=rec)
        r = c.invoke_model(modelId=provisioned, body=json.dumps(request))
        assert r["body"].read() == data  # billed by the hour: not read, not recorded
        c.converse(modelId="arn:aws:bedrock:us-east-1:111122223333:imported-model/my-claude-distill",
                   messages=[U1])
    assert rows(rec) == []


def test_the_body_handed_back_is_a_streaming_body():
    rec = Recorder(":memory:")
    c = client()
    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    data = json.dumps({"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5-20250929",
                       "content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 1}}).encode()
    with Stubber(c) as stub:
        for _ in range(2):
            stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                               "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec)
        body = c.invoke_model(modelId=MODEL, body=json.dumps(request))["body"]
        assert isinstance(body, StreamingBody)
        body.set_socket_timeout  # noqa: B018 - the attribute exists
        assert next(body) == data
        again = c.invoke_model(modelId=MODEL, body=json.dumps(request))["body"]
        assert b"".join(again.iter_chunks(7)) == data


def test_the_write_is_split_by_the_lifetime_the_request_asked_for_when_bedrock_does_not_split_it():
    assert anthropic_usage(usage(write=40), {"1h"})["cache_creation"] == {
        "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 40}
    assert anthropic_usage(usage(write=40), {"5m"})["cache_creation"]["ephemeral_5m_input_tokens"] == 40
    assert "cache_creation" not in anthropic_usage(usage(write=40), {"5m", "1h"})  # mixed: not known
    partial = anthropic_usage(usage(write=40, details=[{"ttl": "1h", "inputTokens": 10}]), {"1h"})
    assert partial["cache_creation"]["ephemeral_1h_input_tokens"] == 40
    # InvokeModel's names for the counts
    assert anthropic_usage({"inputTokens": 1, "outputTokens": 1, "cacheReadInputTokenCount": 5,
                            "cacheWriteInputTokenCount": 6}) == {
        "input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 6}


def test_what_the_row_carries():
    from prefixwise.bedrock import converse_response
    r = converse_response({**reply(usage(), request_id="abc", stop="tool_use"),
                           "output": {"message": {"role": "assistant", "content": [
                               {"toolUse": {"toolUseId": "t1", "name": "read", "input": {"p": 1}}}]}}}, MODEL)
    assert (r["id"], r["stop_reason"], r["model"]) == ("abc", "tool_use", MODEL)
    assert r["content"] == [{"type": "tool_use", "id": "t1", "name": "read", "input": {"p": 1}}]
    assert converse_response({"output": {}}, MODEL) is None  # no usage, no row
    rec = Recorder(":memory:")
    seen = []
    real = rec.record
    rec.record = lambda *a, **k: (seen.append(k), real(*a, **k))[1]
    c = client()
    request = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    data = json.dumps({"id": "m", "type": "message", "role": "assistant", "content": [], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 5, "output_tokens": 1}}).encode()
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage(), request_id="conv-1"))
        stub.add_response("invoke_model", {"body": StreamingBody(io.BytesIO(data), len(data)),
                                           "contentType": "application/json"})
        prefixwise.wrap(c, recorder=rec, agent_id="planner")
        conv = c.converse(modelId=MODEL, messages=[U1])
        c.invoke_model(modelId=MODEL, body=json.dumps(request))
    assert [k["genai_provider"] for k in seen] == ["aws.bedrock", "aws.bedrock"]
    assert [k["agent_id"] for k in seen] == ["planner", "planner"]
    assert all(k["latency_ms"] >= 0 for k in seen)
    with rec.ledger.reader() as conn:
        got = conn.execute("SELECT agent_id, model FROM entries ORDER BY id").fetchall()
    assert got == [("planner", MODEL), ("planner", MODEL)] and conv["stopReason"] == "end_turn"


def test_a_failure_to_read_the_call_warns_once_and_leaves_a_row(monkeypatch, capsys):
    rec = Recorder(":memory:")
    c = client()
    monkeypatch.setattr("prefixwise.bedrock.converse_request", lambda kwargs, model=None: 1 / 0)
    with Stubber(c) as stub:
        stub.add_response("converse", reply(usage()))
        prefixwise.wrap(c, recorder=rec)
        c.converse(modelId=MODEL, messages=[U1])
    assert [r[9] for r in rows(rec)] == ["not_recorded"]
    assert not getattr(rec, "_warned", False)  # the Recorder's own warning is not used up
