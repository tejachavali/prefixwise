"""`--optimize` (docs/specs/check-and-optimize.md section 2): every proven fix in one flag."""

from prefixwise.fixes import AVAILABLE_FIXES, FIX_STATUS, PROVEN_FIXES, parse_fixes
from prefixwise.proxy import PrefixwiseProxy

A = {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}}
B = {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}}
MSGS = [{"role": "user", "content": "q"}]
MSGS2 = MSGS + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]


def test_optimize_stands_for_the_proven_fixes_plus_what_the_user_names():
    assert PROVEN_FIXES == ("canonical_tools",)
    assert all(FIX_STATUS[f]["status"] == "proven" for f in PROVEN_FIXES)
    assert set(PROVEN_FIXES) <= set(AVAILABLE_FIXES)
    assert parse_fixes("optimize") == ("canonical_tools",)
    assert parse_fixes(["optimize", "canonical_tools"]) == ("canonical_tools",)
    assert parse_fixes("canonical_tools,optimize") == ("canonical_tools",)
    assert parse_fixes(None) == () and parse_fixes("") == ()


def test_optimize_through_the_proxy_records_the_fix_it_applied_not_the_word():
    p = PrefixwiseProxy(mock=True, fixes=["optimize"])
    assert p.fixes == ("canonical_tools",)
    p.handle("openai", {"model": "m", "session_id": "s", "tools": [A, B], "messages": MSGS})
    result, entry = p.handle("openai", {"model": "m", "session_id": "s", "tools": [B, A], "messages": MSGS2})
    assert result["cause"] == "none" and entry.fixes == "canonical_tools"


def test_cli_flags_and_middleware_env_expand_optimize(monkeypatch):
    import prefixwise.cli as cli
    parse = cli.build_parser().parse_args
    assert cli._fixes(parse(["serve", "--optimize"])) == ["optimize"]
    assert cli._fixes(parse(["run", "--optimize", "--fix", "canonical_tools", "--", "x"])) == ["canonical_tools", "optimize"]
    assert cli._fixes(parse(["serve"])) is None  # the environment decides
    from prefixwise.asgi import PrefixwiseMiddleware

    def mw():
        return PrefixwiseMiddleware(lambda scope, receive, send: None, ledger=":memory:", engine="vllm", kv_events="")

    for name in ("PREFIXWISE_FIX", "PREFIXWISE_FIXES", "PREFIXWISE_OPTIMIZE"):
        monkeypatch.delenv(name, raising=False)
    assert mw().proxy.fixes == ()
    monkeypatch.setenv("PREFIXWISE_OPTIMIZE", "1")
    assert mw().proxy.fixes == ("canonical_tools",)
    monkeypatch.setenv("PREFIXWISE_OPTIMIZE", "0")
    assert mw().proxy.fixes == ()
    monkeypatch.setenv("PREFIXWISE_FIX", "optimize")
    assert mw().proxy.fixes == ("canonical_tools",)


def test_response_format_is_a_prompt_affecting_param():
    """A changed Chat Completions schema re-renders the prompt: param_rendered, not an unexplained miss."""
    rf = lambda name: {"type": "json_schema", "json_schema": {"name": name, "schema": {"type": "object"}}}  # noqa: E731
    p = PrefixwiseProxy(mock=True)
    p.handle("openai", {"model": "m", "session_id": "s", "messages": MSGS, "response_format": rf("one")})
    assert p.handle("openai", {"model": "m", "session_id": "s", "messages": MSGS2, "response_format": rf("one")})[0]["cause"] == "none"
    assert p.handle("openai", {"model": "m", "session_id": "s", "messages": MSGS2 + MSGS, "response_format": rf("two")})[0]["cause"] == "param_rendered"
