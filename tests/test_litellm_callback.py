"""PrefixwiseCallback: real LiteLLM calls to a mock server; the ledger shows what was recorded."""

import asyncio
import threading
import time

import pytest

litellm = pytest.importorskip("litellm")

from prefixwise.litellm_callback import PrefixwiseCallback
from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler
from prefixwise.sdk import Recorder

SYSTEM = "You are careful. " * 300


@pytest.fixture(scope="module")
def base():
    srv = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(PrefixwiseProxy(mock=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def _registered():
    # LiteLLM copies `callbacks` into its own lists on first use: register once, swap the ledger per test.
    c = PrefixwiseCallback(recorder=Recorder(":memory:"))
    saved = litellm.callbacks
    litellm.callbacks = [c]
    yield c
    litellm.callbacks = saved


@pytest.fixture
def cb(_registered):
    _registered.recorder = Recorder(":memory:", agent_id="litellm")
    _registered._warned = False
    return _registered


def rows(cb, n, timeout=10.0):
    """LiteLLM runs success callbacks off the calling thread: wait for n rows."""
    deadline = time.monotonic() + timeout
    while True:
        with cb.recorder.ledger.reader() as c:
            out = c.execute("SELECT provider, session_id, agent_id, task_id, cause, prompt_tokens, cached_tokens "
                            "FROM entries ORDER BY id").fetchall()
        if len(out) >= n or time.monotonic() > deadline:
            return out
        time.sleep(0.05)


def call(base, model, messages, **kw):
    api_base = base if model.startswith("anthropic/") else base + "/v1"
    return litellm.completion(model=model, api_base=api_base, api_key="sk-test", messages=messages, **kw)


def test_anthropic_through_litellm(base, cb):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "hi"}]
    meta = {"task_id": "t1", "agent_id": "coder"}
    call(base, "anthropic/claude-haiku-5-5", msgs, litellm_session_id="s1", metadata=meta)
    call(base, "anthropic/claude-haiku-5-5", msgs + [{"role": "assistant", "content": "ok"},
                                                    {"role": "user", "content": "go"}],
         litellm_session_id="s1", metadata=meta)
    out = rows(cb, 2)
    assert [r[:5] for r in out] == [("anthropic", "s1", "coder", "t1", "first_write"),
                                    ("anthropic", "s1", "coder", "t1", "none")]
    assert out[1][5:7] == (2000, 1000)  # LiteLLM's prompt_tokens includes cache reads: not counted twice


def test_openai_through_litellm_session_from_metadata(base, cb):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    call(base, "openai/gpt-5.6", msgs, metadata={"session_id": "m1"})
    call(base, "openai/gpt-5.6", msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}],
         metadata={"session_id": "m1"})
    out = rows(cb, 2)
    assert [(r[0], r[1], r[2], r[4]) for r in out] == [("openai", "m1", "litellm", "first_write"),
                                                       ("openai", "m1", "litellm", "none")]


def test_session_falls_back_to_user(base, cb):
    call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}], user="u-7")
    assert rows(cb, 1)[0][1] == "u-7"


def test_injected_breakpoints_are_not_called_missing(base, cb):
    """LiteLLM may add cache_control itself: a call with injection points is never 'no breakpoint'."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]
    call(base, "anthropic/claude-haiku-5-5", msgs, litellm_session_id="inj",
         cache_control_injection_points=[{"location": "message", "role": "system"}])
    assert rows(cb, 1)[0][4] != "breakpoint_missing"


def test_async_path(base, cb):
    async def go():
        await litellm.acompletion(model="openai/gpt-5.6", api_base=base + "/v1", api_key="sk-test",
                                  messages=[{"role": "user", "content": "hi"}], litellm_session_id="as")
        await asyncio.sleep(1)  # LiteLLM logs in a background task; a real app keeps its loop running
    asyncio.run(go())
    assert [r[1] for r in rows(cb, 1)] == ["as"]


def test_streams_pass_through_unrecorded(base, cb):
    chunks = list(call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}], stream=True))
    assert chunks
    time.sleep(0.5)
    assert rows(cb, 1, timeout=0.5) == []


def test_recording_never_breaks_litellm(base, cb, capsys):
    cb.recorder.record = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire"))
    for _ in range(2):
        r = call(base, "openai/gpt-5.6", [{"role": "user", "content": "hi"}])
        assert r.usage.prompt_tokens == 2000
    time.sleep(0.5)
    assert capsys.readouterr().err.count("LiteLLM call not recorded") == 1


def test_proxy_server_config_loads_one_shared_callback(tmp_path, monkeypatch):
    """config.yaml `callbacks: prefixwise.litellm_callback.prefixwise_callback`, loaded the way the
    LiteLLM proxy server loads it."""
    get_instance_fn = pytest.importorskip("litellm.proxy.types_utils.utils").get_instance_fn
    import prefixwise.litellm_callback as mod
    monkeypatch.setenv("PREFIXWISE_LEDGER_PATH", str(tmp_path / "ledger.db"))
    vars(mod).pop("prefixwise_callback", None)  # not delattr: hasattr would create it here
    first = get_instance_fn("prefixwise.litellm_callback.prefixwise_callback")
    assert isinstance(first, PrefixwiseCallback) and first.recorder.agent_id == "litellm"
    assert get_instance_fn("prefixwise.litellm_callback.prefixwise_callback") is first
    assert (tmp_path / "ledger.db").exists()
