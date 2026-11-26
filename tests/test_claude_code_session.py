"""Claude Code names its conversation two ways (read from its published 2.1.x build): the
X-Claude-Code-Session-Id header on every request, and metadata.user_id as a JSON string holding
device_id, account_uuid and session_id. Prefixwise keys on session_id only, so the device and account
ids never reach the ledger."""
import json

from prefixwise.proxy import PrefixwiseProxy, _claude_code_session

USER_ID = json.dumps({"device_id": "dev-abc123", "account_uuid": "acct-9f8e", "session_id": "sess-42"})


def _turn(n, **extra):
    msgs = [{"role": "user", "content": "fix the bug"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": "go on"}]
    return {"model": "claude-sonnet-5", "max_tokens": 8, "system": "You are Claude Code.", "messages": msgs, **extra}


def test_the_session_comes_from_the_metadata_json_and_nothing_else_is_kept():
    assert _claude_code_session(USER_ID) == "sess-42"
    assert _claude_code_session("plain-user") is None and _claude_code_session("{not json") is None
    assert _claude_code_session(json.dumps({"device_id": "d"})) is None
    p = PrefixwiseProxy(mock=True)
    _, e = p.handle("anthropic", _turn(0, metadata={"user_id": USER_ID}))
    assert e.session_id == "sess-42"
    row = json.dumps(p.ledger.recent(1)[0])
    assert "dev-abc123" not in row and "acct-9f8e" not in row  # the device and account ids are not stored


def test_the_header_names_the_session_and_wins_over_the_body():
    p = PrefixwiseProxy(mock=True)
    _, e = p.handle("anthropic", _turn(0, metadata={"user_id": USER_ID}), {"x-claude-code-session-id": "hdr-7"})
    assert e.session_id == "hdr-7"
    _, e = p.handle("anthropic", _turn(0), {"x-claude-code-session-id": "hdr-8",
                                            "x-prefixwise-session-id": "mine"})
    assert e.session_id == "mine"  # Prefixwise's own header still wins


def test_turns_of_one_claude_code_session_compare_with_each_other():
    p = PrefixwiseProxy(mock=True)
    p.handle("anthropic", _turn(0, metadata={"user_id": USER_ID}))
    r, e = p.handle("anthropic", _turn(1, metadata={"user_id": USER_ID}))
    assert e.session_id == "sess-42" and r["cause"] != "first_write"


def test_a_plain_user_id_is_still_a_user():
    p = PrefixwiseProxy(mock=True)
    ctx = p.prepare("anthropic", _turn(0, metadata={"user_id": "person-1"}), {})
    assert ctx["session_is_user"] is True
    ctx = p.prepare("anthropic", _turn(0, metadata={"user_id": USER_ID}), {})
    assert ctx["session_is_user"] is False


def test_an_older_ledger_is_scrubbed_of_the_device_and_account_ids(tmp_path):
    """Rows written before the fix keyed the session on the whole JSON; opening the ledger rewrites them
    to the session id alone, and leaves every other session id as it was."""
    import sqlite3

    from prefixwise.ledger import Ledger, LedgerEntry

    path = str(tmp_path / "old.db")
    led = Ledger(path)
    for sid in (USER_ID, "plain-session", '{"not":"claude"}', '{"session_id":"main","run":1}'):
        led.record(LedgerEntry(session_id=sid, agent_id="a", provider="anthropic", model="m", cause="none",
                               legit=None, prompt_tokens=1, cached_tokens=0, avoidable_tokens=0,
                               avoidable_cost=0.0, task_success=None))
    led.conn.close()
    with sqlite3.connect(path) as c:  # as an older version left it
        c.execute("UPDATE entries SET session_id = ? WHERE session_id = 'sess-42'", (USER_ID,))
    Ledger(path)
    with sqlite3.connect(path) as c:
        ids = [r[0] for r in c.execute("SELECT session_id FROM entries ORDER BY id")]
    assert ids == ["sess-42", "plain-session", '{"not":"claude"}', '{"session_id":"main","run":1}']  # only Claude Code's shape


def test_the_scrub_leaves_no_trace_of_the_old_ids_in_the_file(tmp_path, capsys):
    """Found by review: rewriting the rows left the old values in SQLite's freed pages (and in the
    write-ahead log). After a scrub the file is rebuilt, so the ids are gone from its bytes."""
    import os

    from prefixwise.ledger import Ledger, LedgerEntry

    path = str(tmp_path / "old.db")
    led = Ledger(path)
    for i in range(50):
        led.record(LedgerEntry(session_id=USER_ID, agent_id="a", provider="anthropic", model="m", cause="none",
                               legit=None, prompt_tokens=i, cached_tokens=0, avoidable_tokens=0,
                               avoidable_cost=0.0, task_success=None))
    led.conn.close()
    Ledger(path).conn.close()

    def raw():
        return b"".join(open(p, "rb").read() for p in (path, path + "-wal") if os.path.exists(p))
    data = raw()
    assert b"dev-abc123" not in data and b"acct-9f8e" not in data and b"sess-42" in data
    assert "removed Claude Code's device and account ids from 50 ledger row(s)" in capsys.readouterr().err
    Ledger(path).conn.close()  # nothing left to do: no message, no rebuild
    assert "removed" not in capsys.readouterr().err
