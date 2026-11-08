"""Upstream auto-detection: `prefixwise serve --upstream URL` learns --engine (and --kv-events auto)
from what the server says about itself (prefixwise.engines.probe_upstream, cli._detect_engine)."""

import http.server
import json
import threading

import pytest

from prefixwise.cli import _detect_engine
from prefixwise.engines import probe_upstream


class _Server(threading.Thread):
    """Answers GET paths from a dict: path -> (status, json body); anything else 404."""

    def __init__(self, routes: dict):
        super().__init__(daemon=True)
        self.routes, self.hits = routes, []
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                srv.hits.append(self.path)
                status, body = srv.routes.get(self.path, (404, {"error": "not found"}))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.start()

    def run(self):
        self.srv.serve_forever()

    def stop(self):
        self.srv.shutdown()


SGLANG_INFO = {"version": "0.5.23", "model_path": "Qwen/Qwen3-4B", "enable_cache_report": True, "dp_size": 1,
               "kv_events": {"endpoint_host": "*", "endpoint_port_base": 5557, "dp_size": 1}, "frontend": "python"}
VLLM_SOURCES = {"0": {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv-events"}}


def test_sglang_is_detected_from_server_info_with_its_events_and_cache_report():
    s = _Server({"/server_info": (200, SGLANG_INFO)})
    try:
        info = probe_upstream(s.url)
        assert (info["engine"], info["version"], info["publishes_events"], info["cache_report"]) == ("sglang", "0.5.23", True, True)
        assert info["notes"] == [] and s.hits == ["/server_info"]  # the deprecated endpoint is never called when this answers
    finally:
        s.stop()
    # cache report off: a warning; no events: publishes_events False
    s = _Server({"/server_info": (200, {**SGLANG_INFO, "enable_cache_report": False, "kv_events": None})})
    try:
        info = probe_upstream(s.url + "/v1")  # a /v1 suffix is fine
        assert info["engine"] == "sglang" and info["publishes_events"] is False
        assert any("--enable-cache-report off" in n for n in info["notes"])
    finally:
        s.stop()


def test_older_sglang_falls_back_to_the_deprecated_endpoint():
    s = _Server({"/get_server_info": (200, {"version": "0.5.19", "server_args": {"model_path": "m", "enable_cache_report": True}})})
    try:
        info = probe_upstream(s.url)
        assert info["engine"] == "sglang" and info["version"] == "0.5.19" and info["publishes_events"] is False
        assert s.hits == ["/server_info", "/get_server_info"]
    finally:
        s.stop()


def test_vllm_is_detected_from_version_and_its_event_sources():
    s = _Server({"/version": (200, {"version": "0.33.0"}), "/kv_event_sources": (200, VLLM_SOURCES)})
    try:
        info = probe_upstream(s.url)
        assert (info["engine"], info["version"], info["publishes_events"]) == ("vllm", "0.33.0", True)
    finally:
        s.stop()
    s = _Server({"/version": (200, {"version": "0.33.0"}), "/kv_event_sources": (200, {})})
    try:
        assert probe_upstream(s.url)["publishes_events"] is False
    finally:
        s.stop()
    # an older vLLM without /kv_event_sources: engine known, events unknown
    s = _Server({"/version": (200, {"version": "0.29.0"})})
    try:
        info = probe_upstream(s.url)
        assert info["engine"] == "vllm" and info["publishes_events"] is None
    finally:
        s.stop()


def test_models_owner_is_the_last_resort_and_unknown_servers_detect_nothing():
    s = _Server({"/v1/models": (200, {"data": [{"id": "m", "owned_by": "vllm"}]})})
    try:
        assert probe_upstream(s.url)["engine"] == "vllm"
    finally:
        s.stop()
    s = _Server({"/v1/models": (200, {"data": [{"id": "m", "owned_by": "openai"}]})})
    try:
        info = probe_upstream(s.url)
        assert info["engine"] is None and info["notes"] == []
    finally:
        s.stop()
    # unreachable: a note, no engine, and quickly
    info = probe_upstream("http://127.0.0.1:9", timeout=0.5)
    assert info["engine"] is None and "did not answer" in info["notes"][0]
    # hosted APIs are never probed
    assert probe_upstream("https://api.anthropic.com") == {"engine": None, "version": None, "publishes_events": None,
                                                           "cache_report": None, "notes": []}
    assert probe_upstream("https://api.openai.com/v1")["engine"] is None


def test_detect_engine_fills_the_flags_and_explicit_flags_stand(monkeypatch):
    pytest.importorskip("zmq")
    pytest.importorskip("msgspec")
    probe = lambda url, api_key=None: {"engine": "sglang", "version": "0.5.23", "publishes_events": True,  # noqa: E731
                                        "cache_report": True, "notes": []}
    engine, kv, lines = _detect_engine("http://127.0.0.1:30000", None, None, None, probe=probe)
    assert (engine, kv) == ("sglang", "auto")
    assert lines[0].startswith("prefixwise: upstream is SGLang 0.5.23 (detected): --engine sglang, --kv-events auto")
    assert any("--full-attention" in l for l in lines)
    # explicit --engine: nothing probed, nothing printed
    assert _detect_engine("http://127.0.0.1:30000", "vllm", None, None, probe=lambda *a, **k: 1 / 0) == ("vllm", None, [])
    # explicit --kv-events kept; --full-attention given: no hint
    engine, kv, lines = _detect_engine("http://x", None, "tcp://127.0.0.1:5557", True, probe=probe)
    assert kv == "tcp://127.0.0.1:5557" and not any("--full-attention" in l for l in lines)
    # mock mode and no upstream: never probed
    assert _detect_engine("http://x", None, None, None, mock=True, probe=lambda *a, **k: 1 / 0) == (None, None, [])
    assert _detect_engine(None, None, None, None, probe=lambda *a, **k: 1 / 0) == (None, None, [])
    # no events published: inferred, with the pointer
    probe2 = lambda url, api_key=None: {"engine": "vllm", "version": "0.33.0", "publishes_events": False,  # noqa: E731
                                         "cache_report": None, "notes": []}
    engine, kv, lines = _detect_engine("http://x", None, None, None, probe=probe2)
    assert (engine, kv) == ("vllm", None) and any("publishes no KV events" in l for l in lines)
    # the extras missing: engine set, events not subscribed, told how
    import builtins
    real_import = builtins.__import__

    def no_zmq(name, *a, **k):
        if name == "zmq":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_zmq)
    engine, kv, lines = _detect_engine("http://x", None, None, None, probe=probe)
    assert (engine, kv) == ("sglang", None) and any("prefixwise[engines]" in l for l in lines)
