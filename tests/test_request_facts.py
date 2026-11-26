"""How a request is read and hashed: wrong facts an independent review reproduced (#184, items 3, 8,
9, 14, 15 and 16), and what a second review found in the first fix."""
import json
import subprocess
import sys

from prefixwise import check
import hashlib
import unicodedata

from prefixwise import details
from prefixwise.fingerprint import (Fingerprinter, cache_markers, mask_only, mask_volatiles, mask_volatiles_kinds,
                                    without_cache_markers)
from prefixwise.fixes import auto_breakpoints
from prefixwise.proxy import PrefixwiseProxy, cache_ttl_seconds, extract_blocks, has_cache_breakpoint
from prefixwise.records import param_fingerprint

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

def jmask(text):
    return mask_volatiles(json.dumps(text), json_text=True)


def test_a_date_time_or_id_at_the_start_of_a_line_is_volatile():
    """The fingerprinter masks JSON text, where a line break is backslash-n: the n sat right before the
    value. A date led by a weekday or a month name, or an id with no digit in its first group, kept
    its value there, so a timestamp on its own line read as an edit of the wording."""
    for a, b in (("2026-11-26T09:00:01Z", "2026-11-26T09:00:09Z"),
                 ("Thu Nov 26 09:00:01 UTC 2026", "Thu Nov 26 09:05:44 UTC 2026"),
                 ("Thu Nov 26 2026", "Fri Nov 27 2026"),          # the weekday starts the line
                 ("Nov 26", "Dec 1"),                               # a month with no year
                 ("09:00:01", "09:05:44"),
                 ("0b0f7c2e-58a1-4c1d-9d5e-2f6f1f0e9a11", "9a7d3b10-1f2e-4a6b-8c9d-0e1f2a3b4c5d"),
                 ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "ffffffff-aaaa-bbbb-cccc-dddddddddddd")):
        for sep in ("\n", "\t", "\r\n", " "):
            one, two = (f"Rules.{sep}{v}{sep}End." for v in (a, b))
            assert jmask(one) == jmask(two), (a, sep)
    # the dates-and-times-only mask, which names what kind of value changed, sees them too
    only = lambda t: mask_only(json.dumps(t), frozenset({"date", "time"}), json_text=True)  # noqa: E731
    assert only("Rules.\nThu Nov 26 2026\nEnd.") == only("Rules.\nFri Nov 27 2026\nEnd.")
    assert "path" in mask_volatiles_kinds(json.dumps("Work in:\n/home/dev/repo\n"), json_text=True)[1]
    proxy = scripted([claude(write=3000), claude(write=3100)])
    stamp = lambda day: {"model": OPUS, "max_tokens": 9, "messages": [U1],  # noqa: E731
                         "system": marked(SYS + f"\n{day} Nov 26 2026\n")}
    proxy.handle("anthropic", stamp("Thu"), S)
    entry = proxy.handle("anthropic", {**stamp("Fri"), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "volatile_system"


def test_a_prompt_that_differs_only_in_its_line_breaks_is_still_an_edit():
    """The line breaks stay in the masked text: re-wrapping a system prompt is an edit of it, not a
    volatile value. (A first version of the fix turned every line break into a space.)"""
    assert jmask("Rules:\n- be terse") != jmask("Rules: - be terse")
    assert jmask("a\tb") != jmask("a b") and jmask("a\r\nb") != jmask("a\nb")
    assert jmask("Rules:\n- be terse\n- 3 steps") == json.dumps("Rules:\n- be terse\n- <N> steps")
    words = ("Follow the house style in every answer and keep each reply short. " * 60).split()
    wrap = lambda n: "\n".join(" ".join(words[i:i + n]) for i in range(0, len(words), n))  # noqa: E731
    proxy = scripted([claude(write=3000), claude(write=3100)])
    body = lambda n: {"model": OPUS, "max_tokens": 9, "messages": [U1], "system": marked(wrap(n))}  # noqa: E731
    proxy.handle("anthropic", body(12), S)
    entry = proxy.handle("anthropic", {**body(15), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "system_edit"


def test_a_literal_backslash_n_in_the_prompt_is_left_alone():
    """A prompt that shows the two characters backslash and n (code, a regex) is not a line break."""
    for text in (r"split on \n2026", r"use \node", r"C:\new\table", r"a regex: \d+\n\t"):
        assert jmask(text) == mask_volatiles(json.dumps(text)), text      # the mark changes nothing here
    assert jmask(r"use \node") == json.dumps(r"use \node")
    assert jmask("a\nb") != jmask(r"a\nb")


def test_a_windows_path_in_a_plain_prompt_is_still_seen_by_the_gate():
    """`prefixwise check` looks at a string system prompt as it is, not as JSON: the backslash-r and
    backslash-t inside a path are part of the path."""
    for path in (r"C:\repos\tools", r"D:\tmp\run", r"C:\new\table"):
        watch = check.VolatileWatch()
        watch("openai", "s", [("system", f"Work only inside {path}.")])
        assert watch.seen == {"s": {"path"}}, path
        watch = check.VolatileWatch()
        watch("anthropic", "s", [("system", [{"type": "text", "text": f"Work only inside {path}."}])])
        assert watch.seen == {"s": {"path"}}, path


# -- 15. with a key, every hash is keyed ---------------------------------------------------------

def test_with_a_key_set_blocks_equal_once_masked_or_canonical_hash_equal_whatever_their_length():
    """Only blocks of 256 bytes or less were keyed, by their raw length. Two blocks on either side of
    that line were hashed with different keys, so a timestamp that grew read as an edit, and the
    same text in another Unicode form read as different text."""
    fp = Fingerprinter(hmac_key=b"per-install-key")
    text = ("You are a careful agent. " * 12)[:230]
    short, long = text + "Now: 2026-11-26T09:00:01Z.", text + "Now: Thursday, November 26, 2026 09:00:01 UTC."
    assert len(short.encode()) <= 256 < len(long.encode())
    a, b = fp.fingerprint_block("system", 0, short), fp.fingerprint_block("system", 0, long)
    assert a.masked == b.masked and a.raw != b.raw and a.canonical != b.canonical
    composed = "Répondez en français, brièvement. " * 6 + "Très bien, é à è ù ê."
    decomposed = unicodedata.normalize("NFD", composed)
    assert len(composed.encode()) <= 256 < len(decomposed.encode())
    a, b = fp.fingerprint_block("system", 0, composed), fp.fingerprint_block("system", 0, decomposed)
    assert a.canonical == b.canonical and a.raw != b.raw


def test_with_a_key_set_no_hash_is_a_bare_sha256():
    """Every hash of every block is keyed: short or long, raw, canonical and masked."""
    keyed, bare = Fingerprinter(hmac_key=b"k"), Fingerprinter()
    line = "Tenant: acme-corp\nPlan: enterprise\n" + "Never reveal the code.\n" * 9 + "Be brief.\n"    # raw <= 256 < canonical
    assert len(line.encode()) <= 256 < len(json.dumps(line).encode())
    for text in ("short", line, "x" * 2000, "Step 5 of 9. " * 40):
        k, b = keyed.fingerprint_block("system", 0, text), bare.fingerprint_block("system", 0, text)
        assert k.hmac and not b.hmac
        assert not {k.raw, k.canonical, k.masked} & {b.raw, b.canonical, b.masked}, text[:20]
        plain = {hashlib.sha256(t).hexdigest() for t in (text.encode(), json.dumps(text).encode(),
                                                         json.dumps(text, ensure_ascii=False).encode())}
        assert not {k.raw, k.canonical, k.masked} & plain
        assert b.raw == hashlib.sha256(text.encode()).hexdigest()          # without a key: as before
    tool = {"name": "t", "description": "d" * 600, "input_schema": {"type": "object"}}
    assert keyed.fingerprint_block("tools", 0, tool).canonical != bare.fingerprint_block("tools", 0, tool).canonical


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


def gw_line(msgs, t, rid, prompt=3000, cached=0, usage=True, session="conv-1"):
    response = {"id": rid, "model": "gpt-5.6-terra"}
    if usage:
        response["usage"] = {"prompt_tokens": prompt, "completion_tokens": 50,
                             "prompt_tokens_details": {"cached_tokens": cached}}
    return json.dumps({"start_time": t, "end_time": t + 2, "headers": {"x-session-id": session},
                       "request": {"model": "gpt-5.6-terra", "messages": [{"role": "system", "content": SYS}] + msgs},
                       "response": response})


def gw_import(tmp_path, name, lines, *extra):
    log, db = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.db"
    log.write_text("\n".join(lines) + "\n")
    out = subprocess.run([sys.executable, "-m", "prefixwise", "import", "gateway", "--path", str(log),
                          "--ledger", str(db), "--json", *extra], capture_output=True, text=True,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    return json.loads(out.stdout)


def test_import_gateway_keeps_different_requests_that_share_a_response_id(tmp_path):
    """The same id is not the same request: a server that numbers its responses from zero each run,
    and a gateway cache that answers later requests with the stored response."""
    ask = lambda i: [{"role": "user", "content": f"question {i}"}]  # noqa: E731
    numbered = [gw_line(ask(i), 1795000000 + 10 * i, f"chatcmpl-{i % 3}", session=f"c{i}") for i in range(9)]
    got = gw_import(tmp_path, "numbered", numbered)
    assert (got["imported"], got["duplicates"]) == (9, 0)
    cached = [gw_line([U1], 1795000000 + 3600 * i, "chatcmpl-STORED") for i in range(3)]   # one request, an hour apart
    got = gw_import(tmp_path, "cached", cached)
    assert (got["imported"], got["duplicates"]) == (3, 0)


def test_import_gateway_keeps_the_line_with_usage_and_counts_repeats_in_the_window(tmp_path):
    t = 1795000000
    bare, full = gw_line([U1], t, "chatcmpl-A", usage=False), gw_line([U1], t, "chatcmpl-A")
    for name, lines in (("bare-first", [bare, full]), ("full-first", [full, bare])):
        got = gw_import(tmp_path, name, lines + [gw_line([U1, A1, GO], t + 10, "chatcmpl-B", 3100, 2944)])
        assert (got["imported"], got["duplicates"]) == (2, 1), name
        from prefixwise.ledger import Ledger
        with Ledger(str(tmp_path / f"{name}.db")).reader() as conn:
            assert [r[0] for r in conn.execute("SELECT prompt_tokens FROM entries ORDER BY id")] == [3000, 3100], name
    old = [gw_line([U1], t, "chatcmpl-A")] * 3                      # repeats before the window are not this import's
    new = [gw_line([U1, A1, GO], t + 7200, "chatcmpl-B", 3100, 2944)] * 2
    got = gw_import(tmp_path, "window", old + new, "--since", str(t + 3600))
    assert (got["imported"], got["duplicates"]) == (1, 1)


# -- what else counts as part of the prompt, and what the other callers see -----------------------

def test_cache_control_inside_any_schema_or_tool_arguments_is_not_a_marker():
    schema = {"type": "object", "properties": {"cache_control": {"type": "object"}}}
    bodies = {
        "an OpenAI tool's parameters": {"tools": [{"type": "function", "function": {"name": "t", "parameters": schema}}]},
        "a Responses tool's parameters": {"tools": [{"type": "function", "name": "t", "parameters": schema}]},
        "a structured-output schema": {"response_format": {"type": "json_schema", "json_schema": {"name": "o", "schema": schema}}},
        "an OpenAI tool call's arguments": {"messages": [{"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": {"cache_control": {"type": "ephemeral"}}}}]}]},
    }
    for where, body in bodies.items():
        assert cache_markers(body) == [] and not has_cache_breakpoint({"messages": [U1], **body}), where
        assert without_cache_markers(body) == body, where
    # ... so a request with only such a field still gets its markers placed, and the comparison keys
    # of the report's detail keep the field
    tool = {"name": "t", "input_schema": schema}
    placed = auto_breakpoints({"model": OPUS, "system": "s " * 4000, "tools": [tool],
                               "messages": [{"role": "user", "content": "what does cache_control do?"}]}, 512)
    assert placed is not None and cache_markers(placed)
    assert details._strip_markers([tool]) == [tool]
    assert details._strip_markers([{**tool, "cache_control": MARK}]) == [tool]


def test_a_change_of_context_management_is_a_changed_parameter():
    """Anthropic's own diagnosis answers `unavailable` when it differs (with tool_choice, thinking and
    output_config): the blocks are equal and something that is not a block changed."""
    proxy = scripted([claude(write=3000), claude(write=3100)])
    base = {"model": OPUS, "max_tokens": 9, "system": marked()}
    edits = lambda n: {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": n}}]}  # noqa: E731
    proxy.handle("anthropic", {**base, "context_management": edits(30000), "messages": [U1]}, S)
    entry = proxy.handle("anthropic", {**base, "context_management": edits(50000), "messages": [U1, A1, GO]}, S)[1]
    assert entry.cause == "param_rendered" and "context_management" in entry.evidence



# -- a third look: what the second review of the fix found ----------------------------------------

def test_with_a_key_set_the_digests_in_a_record_are_keyed_too():
    """A record also carries a digest of each message, of the tools and system prompt, of a cache key
    and of each request parameter's value. They were bare SHA-256 whatever the key, so a short message
    or a tool name in tool_choice could be guessed from the record a collector receives."""
    body = {"model": OPUS, "max_tokens": 9, "system": "You are a bot.", "prompt_cache_key": "tenant-7",
            "tool_choice": {"type": "tool", "name": "lookup"}, "messages": [{"role": "user", "content": "yes"}]}
    bare = hashlib.sha256(json.dumps({"role": "user", "content": "yes"}).encode()).hexdigest()[:16]
    plain, keyed = details.body_facts("anthropic", body), details.body_facts("anthropic", body, hmac_key=b"k")
    assert plain.msgs == [bare] and keyed.msgs != [bare]                     # without a key: as before
    for name in ("msgs", "nothink", "prefix_key", "prefix_key_canonical", "prefix_key_masked", "cache_key"):
        assert getattr(plain, name) != getattr(keyed, name), name
    choice = body["tool_choice"]
    assert param_fingerprint(choice) == hashlib.sha256(
        json.dumps(choice, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    assert param_fingerprint(choice, b"k") != param_fingerprint(choice)
    seen = {}
    for key in (None, b"per-install-key"):
        proxy = PrefixwiseProxy(mock=True, hmac_key=key)
        ctx = proxy.prepare("anthropic", dict(body), dict(S))
        seen[key] = (ctx["facts"].msgs, ctx["facts"].cache_key)
    assert seen[None][0] == [bare] and seen[b"per-install-key"][0] != [bare]
    assert seen[None][1] != seen[b"per-install-key"][1]
    # equal requests still compare equal under one key: an appended turn is a hit, not an edit
    proxy = scripted([claude(write=3000), claude(read=3000, write=100)], hmac_key=b"per-install-key")
    base = {"model": OPUS, "max_tokens": 9, "system": marked(), "tool_choice": {"type": "auto"}}
    proxy.handle("anthropic", {**base, "messages": [U1]}, S)
    assert proxy.handle("anthropic", {**base, "messages": [U1, A1, GO]}, S)[1].cause == "none"


def test_the_report_details_count_only_real_markers():
    """A 1-hour marker and a schema property named cache_control: every marker asks for an hour."""
    tool = {"name": "t", "input_schema": {"type": "object", "properties": {"cache_control": {"type": "object"}}}}
    body = {"model": OPUS, "system": marked(ttl="1h"), "tools": [tool], "messages": [U1]}
    assert details.body_facts("anthropic", body).all_1h


def test_the_report_details_mask_a_date_at_the_start_of_a_line_too():
    keys = lambda day: details._prefix_keys(f"k-{day}", "anthropic", OPUS, "", [],  # noqa: E731
                                            f"Rules.\n{day} Nov 26 2026\nEnd.")
    thu, fri = keys("Thu"), keys("Fri")
    assert thu[0] != fri[0] and thu[1] == fri[1] and thu[2] == fri[2]        # canonical differs; masked equal


def test_the_marks_never_reach_the_output_and_the_fast_path_agrees_with_the_slow_one():
    from prefixwise import fingerprint
    texts = ["Rules.\nThu Nov 26 2026\n\t42 items\r\nEnd.", r"C:\new\table and a real" + "\nline 7",
             "no escapes at all 2026-11-26", r"a \\n literal then" + "\n9"]
    for text in texts:
        doc = json.dumps(text)
        assert fingerprint._mark_breaks(doc) == fingerprint._ESCAPED_SPACE_RE.sub(
            r"\1\2" + fingerprint._BREAK_MARK, doc), text
        for out in (jmask(text), mask_only(doc, frozenset({"date", "time"}), json_text=True),
                    mask_only(doc, frozenset({"id"}), json_text=True)):
            assert "\x00" not in out and json.loads(out) is not None, text
    assert mask_only(json.dumps("a\nThu Nov 26 2026"), frozenset({"date"}), json_text=True) == json.dumps("a\n<DATE>")


def test_the_gate_sees_a_path_at_the_start_of_a_line_in_a_block_list():
    watch = check.VolatileWatch()
    watch("anthropic", "s", [("system", [{"type": "text", "text": "Work only inside:\n/home/dev/repo\n"}])])
    assert watch.seen == {"s": {"path"}}


def test_import_gateway_same_id_and_start_with_another_body_is_another_request(tmp_path):
    t = 1795000000
    ask = lambda q: [{"role": "user", "content": q}]  # noqa: E731
    got = gw_import(tmp_path, "bodies", [gw_line(ask("one"), t, "chatcmpl-0", session="a"),
                                         gw_line(ask("two"), t, "chatcmpl-0", session="b")])
    assert (got["imported"], got["duplicates"]) == (2, 0)


def test_import_gateway_counts_a_repeat_by_when_its_request_ended(tmp_path):
    """The window is the requests that ended at or after --since; a repeat of one of them is in it,
    even when the request started before."""
    t = 1795000000
    long_one = json.dumps({**json.loads(gw_line([U1], t - 10, "chatcmpl-L")), "end_time": t + 50})
    got = gw_import(tmp_path, "straddle", [long_one, long_one], "--since", str(t))
    assert (got["imported"], got["duplicates"]) == (1, 1)


def test_every_edge_reads_the_key_not_only_serve(monkeypatch):
    """`PREFIXWISE_HMAC_KEY` was read by `prefixwise serve` alone: `run`, `prefixwise.wrap`, the
    LiteLLM callback and the vLLM middleware build the same proxy without it, and sent bare hashes."""
    from prefixwise import gateway_import
    from prefixwise.sdk import Recorder

    monkeypatch.delenv("PREFIXWISE_HMAC_KEY", raising=False)
    assert PrefixwiseProxy(mock=True).fingerprinter.hmac_key is None
    monkeypatch.setenv("PREFIXWISE_HMAC_KEY", "per-install-key")
    assert PrefixwiseProxy(mock=True).fingerprinter.hmac_key == b"per-install-key"
    assert Recorder(":memory:").proxy.fingerprinter.hmac_key == b"per-install-key"
    assert PrefixwiseProxy(mock=True, hmac_key=b"other").fingerprinter.hmac_key == b"other"   # a given key wins
    assert PrefixwiseProxy(mock=True, hmac_key=b"").fingerprinter.hmac_key is None            # b"": no key
    made = []
    real = gateway_import.__dict__.get("PrefixwiseProxy")
    import prefixwise.proxy as proxy_module

    class Spy(proxy_module.PrefixwiseProxy):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self.fingerprinter.hmac_key)

    monkeypatch.setattr(proxy_module, "PrefixwiseProxy", Spy)
    gateway_import.replay([], ":memory:")           # a replay takes no setting from the environment
    assert made == [None] and real is None


def test_with_a_key_set_what_the_proxy_sends_is_keyed_at_every_step(monkeypatch):
    """The record a collector receives, and the facts recomputed when a fix rewrites the request."""
    body = {"model": OPUS, "max_tokens": 9, "system": SYS, "tool_choice": {"type": "tool", "name": "lookup"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}], "messages": [U1]}
    sent = {}
    for key in (b"", b"per-install-key"):
        proxy = scripted([claude(fresh=3000), claude(write=3100)], hmac_key=key, fixes=["auto_breakpoints"])
        records, build = [], proxy._record
        monkeypatch.setattr(proxy, "_record", lambda *a, _b=build, _r=records, **kw: _r.append(_b(*a, **kw)) or _r[-1])
        proxy.handle("anthropic", dict(body), S)
        second = proxy.handle("anthropic", {**body, "messages": [U1, A1, GO]}, S)[1]
        assert "auto_breakpoints" in (second.fixes or ""), key      # the fix rewrote the request: facts recomputed
        sent[key] = records
    bare, keyed = sent[b""], sent[b"per-install-key"]
    choice = param_fingerprint(body["tool_choice"])
    assert [r.params["tool_choice"] for r in bare] == [choice, choice]
    assert all(r.params["tool_choice"] != choice for r in keyed)
    for b, k in zip(bare, keyed):
        assert b.facts.msgs and not set(b.facts.msgs) & set(k.facts.msgs)
        assert b.facts.prefix_key != k.facts.prefix_key
        assert not {x.raw for x in b.blocks} & {x.raw for x in k.blocks}
