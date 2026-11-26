"""How a request is read and hashed: wrong facts an independent review reproduced (#184, items 3, 8,
9, 14, 15 and 16)."""
import json
import subprocess
import sys

from prefixwise import check
from prefixwise.fingerprint import Fingerprinter, cache_markers, mask_volatiles, without_cache_markers
from prefixwise.fixes import auto_breakpoints
from prefixwise.proxy import PrefixwiseProxy, cache_ttl_seconds, extract_blocks, has_cache_breakpoint

SYS = "You are a careful agent. " * 400
U1 = {"role": "user", "content": "start"}
A1 = {"role": "assistant", "content": "ok"}
GO = {"role": "user", "content": "go"}
S = {"x-prefixwise-session-id": "s"}
OPUS = "claude-opus-5-5"
MARK = {"type": "ephemeral"}


def scripted(usages, **kw):
    proxy = PrefixwiseProxy(upstream_url="http://unused.invalid", api_key="k", **kw)
    queue = list(usages)
    proxy._forward = lambda provider, body, request_id, *a, **k: (200, {"id": "msg_" + request_id[:8],
                                                                         "usage": queue.pop(0)})
    return proxy


def claude(read=0, write=0, fresh=200):
    return {"input_tokens": fresh, "output_tokens": 5, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write, "cache_creation": {"ephemeral_5m_input_tokens": write}}


def marked(text=SYS, **marker):
    return [{"type": "text", "text": text, "cache_control": {**MARK, **marker}}]


# -- 3. a marker is told by where it sits, not by the word ---------------------------------------

def test_the_word_cache_control_in_text_is_not_a_marker():
    """With no marker anywhere, a user who types "cache_control" made the request look marked: the
    missing-marker verdict became a routing miss."""
    for text in ("why is my bill high?", "why is cache_control not helping my bill?"):
        proxy = scripted([claude(fresh=3000), claude(fresh=3100)])
        msgs = [{"role": "user", "content": text}]
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": msgs}, S)
        entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": SYS, "messages": msgs + [A1, GO]}, S)[1]
        assert (entry.cause, entry.avoidable_tokens) == ("breakpoint_misplaced", 3000), text


def test_the_ci_gate_fails_an_agent_that_dropped_its_markers_whatever_its_prompt_says():
    for system in ("You are a billing agent. " * 300, "You review LLM code; look for missing cache_control markers. " * 100):
        proxy = PrefixwiseProxy(mock=True)
        proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system, "messages": [U1]}, S)
        entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": system, "messages": [U1, A1, GO]}, S)[1]
        assert entry.cause == "breakpoint_misplaced"
        assert check.judge(check.rows_since(proxy.ledger, 0), check.DEFAULT_FAIL_ON).exit_code == 2


def test_a_tool_argument_named_ttl_is_not_the_cache_lifetime():
    """A DNS tool called with {"ttl": "1h"} turned a 5-minute cache into a 1-hour one, so an expiry
    after ten minutes was a routing miss."""
    def call(arg):
        return [U1, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "dns_set",
                                                       "input": {"name": "a.example", "ttl": arg}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]}]
    for arg in ("300", "1h"):
        body = {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": call(arg)}
        assert cache_ttl_seconds("anthropic", body) == 300.0
    assert cache_ttl_seconds("anthropic", {"model": OPUS, "system": marked(ttl="1h"), "messages": [U1]}) == 3600.0
    assert cache_ttl_seconds("anthropic", {"model": OPUS, "cache_control": {**MARK, "ttl": "1h"}, "messages": [U1]}) == 3600.0


def test_markers_are_found_where_the_apis_define_them():
    tool = {"name": "t", "input_schema": {"type": "object", "properties": {"cache_control": {"type": "string"}}}}
    assert cache_markers([tool]) == [] and not has_cache_breakpoint({"tools": [tool], "messages": [U1]})
    places = {
        "a tool": {"tools": [{**tool, "cache_control": MARK}]},
        "a tool's function (LiteLLM)": {"tools": [{"type": "function", "function": {"name": "t", "cache_control": MARK}}]},
        "a system block": {"system": marked("x")},
        "a message (LiteLLM)": {"messages": [{"role": "user", "content": "x", "cache_control": MARK}]},
        "a content block": {"messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": MARK}]}]},
        "a block inside a tool result": {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "x", "cache_control": MARK}]}]}]},
    }
    for where, body in places.items():
        assert has_cache_breakpoint(body), where
        assert auto_breakpoints({"model": OPUS, "system": "s " * 4000, "messages": [U1], **body}, 512) is None, where
    # the conversation of a Responses request is under `input`: markers there are still found
    assert cache_markers({"input": [{"role": "user", "content": [{"type": "input_text", "text": "x", "cache_control": MARK}]}]})


# -- 14. only markers are left out of the hash ---------------------------------------------------

def test_a_schema_property_or_tool_input_named_cache_control_is_part_of_the_prompt():
    """Stripped at every depth, so a tool whose schema gained such a property hashed the same."""
    before = {"name": "t", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}
    after = {"name": "t", "input_schema": {"type": "object", "properties": {"q": {"type": "string"},
                                                                           "cache_control": {"type": "string"}}}}
    fp = Fingerprinter()
    hashes = [fp.fingerprint_blocks(extract_blocks({"model": OPUS, "tools": [t], "messages": [U1]}, "anthropic"))[0].canonical
              for t in (before, after)]
    assert hashes[0] != hashes[1]
    call = {"type": "tool_use", "id": "t1", "name": "set", "input": {"cache_control": "no-store"}}
    assert without_cache_markers(call) == call
    block = {"type": "text", "text": "x", "cache_control": MARK}
    assert without_cache_markers([block]) == [{"type": "text", "text": "x"}]          # a marker still goes
    proxy = scripted([claude(write=3000), claude(write=3100)])
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "tools": [before], "messages": [U1]}, S)
    entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "tools": [after],
                                       "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "tool_catalog_change"


def test_a_marker_that_moves_to_the_newest_message_is_still_not_a_change():
    proxy = scripted([claude(write=3000), claude(read=3000, write=100)])
    last = lambda text: {"role": "user", "content": [{"type": "text", "text": text, "cache_control": MARK}]}  # noqa: E731
    proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(), "messages": [last("start")]}, S)
    entry = proxy.handle("anthropic", {"model": OPUS, "max_tokens": 9, "system": marked(),
                                       "messages": [U1, A1, last("go")]}, S)[1]
    assert entry.cause == "none"


# -- 8. Anthropic's output_config ----------------------------------------------------------------

def test_a_change_of_output_config_is_seen():
    """Effort and the structured-output format live in `output_config`; only the older
    `output_format` was compared, so an effort change that missed was a routing miss."""
    proxy = scripted([claude(write=3000), claude(write=3100)])
    base = {"model": OPUS, "max_tokens": 9, "system": marked()}
    proxy.handle("anthropic", {**base, "output_config": {"effort": "high"}, "messages": [U1]}, S)
    entry = proxy.handle("anthropic", {**base, "output_config": {"effort": "low"}, "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "param_rendered" and "output_config" in entry.evidence
    hit = scripted([claude(write=3000), claude(read=3000, write=100)])
    hit.handle("anthropic", {**base, "output_config": {"effort": "high"}, "messages": [U1]}, S)
    entry = hit.handle("anthropic", {**base, "output_config": {"effort": "low"}, "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "none"       # the cache served it all: the change did not reach the prompt


# -- 9. a value at the start of a line is masked -------------------------------------------------

def test_a_date_time_or_id_at_the_start_of_a_line_is_volatile():
    """The masks run on JSON text, where a line break is backslash-n: the n sat right before the value
    and the patterns' word boundary failed. A timestamp on its own line read as an edit of the wording."""
    for a, b in (("2026-11-26T09:00:01Z", "2026-11-26T09:00:09Z"),
                 ("Thu Nov 26 09:00:01 UTC 2026", "Thu Nov 26 09:05:44 UTC 2026"),
                 ("Thu Nov 26 2026", "Fri Nov 27 2026"),          # the weekday starts the line
                 ("Nov 26", "Dec 1"),                               # a month with no year
                 ("09:00:01", "09:05:44"),
                 ("0b0f7c2e-58a1-4c1d-9d5e-2f6f1f0e9a11", "9a7d3b10-1f2e-4a6b-8c9d-0e1f2a3b4c5d"),
                 ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "ffffffff-aaaa-bbbb-cccc-dddddddddddd")):
        for sep in ("\n", "\t", "\r\n", " "):
            one, two = (json.dumps(f"Rules.{sep}{v}{sep}End.") for v in (a, b))
            assert mask_volatiles(one) == mask_volatiles(two), (a, sep)
    proxy = scripted([claude(write=3000), claude(write=3100)])
    stamp = lambda s: {"model": OPUS, "max_tokens": 9, "messages": [U1],  # noqa: E731
                       "system": marked(SYS + f"\n2026-11-26T09:00:{s}Z\n")}
    proxy.handle("anthropic", stamp("01"), S)
    entry = proxy.handle("anthropic", {**stamp("09"), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "volatile_system"


def test_a_literal_backslash_n_in_the_prompt_is_left_alone():
    """A prompt that shows the two characters backslash and n (code, a regex) is not a line break."""
    assert mask_volatiles(json.dumps(r"split on \n2026")) == mask_volatiles(json.dumps(r"split on \n2026"))
    assert mask_volatiles(json.dumps(r"use \node")) == json.dumps(r"use \node")
    assert mask_volatiles(json.dumps("a\nb")) != mask_volatiles(json.dumps(r"a\nb"))


# -- 15. each hash is keyed by its own text ------------------------------------------------------

def test_with_a_key_set_a_timestamp_that_crosses_256_bytes_is_still_volatile():
    """Blocks of 256 bytes or less are keyed. A value that grew moved a block across that limit, the
    two masked hashes were made with different keys, and a volatile value read as an edit."""
    for key in (None, b"per-install-key"):
        fp = Fingerprinter(hmac_key=key)
        dates = ("2026-11-26T09:00:01Z", "Thu Nov 26 09:00:01 UTC 2026")   # one date, in two formats and lengths
        text = "You are a careful agent. " * 9
        text = text[:250 - len(json.dumps(f"Now: {dates[0]}."))]             # the shorter one lands just under 256
        short, long = (text + f"Now: {d}." for d in dates)
        a, b = fp.fingerprint_block("system", 0, short), fp.fingerprint_block("system", 0, long)
        assert len(json.dumps(short)) <= 256 < len(json.dumps(long))
        assert a.masked == b.masked and a.raw != b.raw and a.canonical != b.canonical, key
    keyed, bare = Fingerprinter(hmac_key=b"k"), Fingerprinter()
    assert keyed.fingerprint_block("system", 0, "short").canonical != bare.fingerprint_block("system", 0, "short").canonical
    long_text = "x" * 2000   # long blocks stay plain SHA-256, key or not
    assert keyed.fingerprint_block("system", 0, long_text).canonical == bare.fingerprint_block("system", 0, long_text).canonical


# -- 16. a request logged twice is one request ---------------------------------------------------

def test_import_gateway_counts_a_request_logged_twice_once(tmp_path):
    def line(msgs, prompt, cached, t, rid):
        return json.dumps({"start_time": t, "end_time": t + 2, "headers": {"x-session-id": "conv-1"},
                           "request": {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}] + msgs},
                           "response": {"id": rid, "model": "gpt-5.6-terra",
                                        "usage": {"prompt_tokens": prompt, "completion_tokens": 50,
                                                  "prompt_tokens_details": {"cached_tokens": cached}}}})
    first = line([U1], 3000, 0, 1795000000, "chatcmpl-AAA")
    second = line([U1, A1, GO], 3100, 2944, 1795000010, "chatcmpl-BBB")
    got = {}
    for name, lines in (("clean", [first, second]), ("twice", [first, first, second])):
        log, db = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.db"
        log.write_text("\n".join(lines) + "\n")
        out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "gateway", "--path", str(log),
                              "--ledger", str(db), "--json"], capture_output=True, text=True,
                             env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
        got[name] = json.loads(out.stdout)
    assert (got["clean"]["imported"], got["clean"]["duplicates"]) == (2, 0)
    assert (got["twice"]["imported"], got["twice"]["duplicates"]) == (2, 1)
