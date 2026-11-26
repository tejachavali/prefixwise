"""`prefixwise demo`: break the cache three ways, then fix it. No key, no network, no cost.

    prefixwise demo               # mock provider; runs straight through in a few seconds
    prefixwise demo --step        # press Enter between acts (for showing it to someone)
    prefixwise demo --live        # the real API: ANTHROPIC_API_KEY (or OPENAI_API_KEY)

One scripted coding agent, nine turns, through the Prefixwise proxy:

    act 1  three clean turns: a first write, then hits
    act 2  a plugin reload sends the same tools in another order    -> tool_reserialize
    act 3  a timestamp lands in the system prompt                   -> volatile_system
    act 4  an earlier message is rewritten                          -> history_edit
    the report: each miss priced, the fixes ranked
    act 5  the same agent again with `--optimize` on the proxy and the two one-line changes the
           report asked for in the agent                            -> every turn after the first hits

Nothing is simulated in the live run: the usage comes from the provider. The mock provider reports a
hit when the prompt's tools, system prompt and earlier messages are unchanged since the session's
previous turn, as a cache that keeps one entry per conversation does; a live provider that keeps
older entries too can turn a reorder back into a hit (turn 6, which returns to the first order). The
mock's token counts and dollar figures are therefore illustrative; the causes are what the proxy
names on real traffic. The ledger holds hashes and token counts, never prompt text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-5.6-luna"}
TOOL_NAMES = ("bash", "edit", "glob", "grep", "read", "write", "web_fetch", "task")
BOLD, DIM, GREEN, RED, END = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
_COLOUR = True


def _paint(text: str, style: str) -> str:
    return f"{style}{text}{END}" if style and _COLOUR else text


def say(text: str = "", style: str = "") -> None:
    print(_paint(text, style), flush=True)


def pause(args, prompt: str) -> None:
    if not args.step:
        time.sleep(args.pause)
        return
    try:
        input(_paint(f"{prompt}  [Enter]", DIM))
    except EOFError:
        pass


class DemoError(Exception):
    """A request failed: the demo stops with the provider's own words, not a traceback."""


class Agent:
    """A coding agent that resends its whole conversation every turn, as agents do."""

    def __init__(self, base: str, provider: str, key: str, session: str):
        self.base, self.provider, self.key, self.session = base, provider, key, session
        # A prefix well above every provider's minimum cacheable size (about 10,000 tokens), tagged
        # per run so the fixed run cannot read the broken run's cache on a live provider.
        self.system = (
            "You are a coding agent working in a large repository. "
            + "Rules, conventions and the repository map follow. " * 900
            + f"Run tag {session}."
        )
        self.messages = [
            {
                "role": "user",
                "content": "In one short sentence, how would you add a --json flag?",
            }
        ]
        self.turn = 0

    def tool(self, name: str) -> dict:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "args": {"type": "string"}},
            "required": ["path"],
        }
        if self.provider == "anthropic":
            return {
                "name": name,
                "description": f"The {name} tool.",
                "input_schema": schema,
            }
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"The {name} tool.",
                "parameters": schema,
            },
        }

    def request(self, tool_order: tuple[str, ...], clock: str | None = None) -> dict:
        self.turn += 1
        tools = [self.tool(n) for n in tool_order]
        # The broken agent keeps the current time in its system prompt (a common habit); the demo
        # holds the clock still until act 3, so the first acts show only their own cause.
        system = self.system + (f" Current time: {clock}." if clock else "")
        msgs = self.messages
        if self.provider == "anthropic":
            cc = {"cache_control": {"type": "ephemeral"}}
            last = {
                **msgs[-1],
                "content": [{"type": "text", "text": msgs[-1]["content"], **cc}],
            }
            body = {
                "model": MODEL["anthropic"],
                "max_tokens": 48,
                "tools": tools,
                "tool_choice": {"type": "none"},
                "system": [{"type": "text", "text": system, **cc}],
                "messages": msgs[:-1] + [last],
            }
            url, headers = (
                self.base + "/v1/messages",
                {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            )
        else:
            body = {
                "model": MODEL["openai"],
                "max_completion_tokens": 48,
                "tools": tools,
                "tool_choice": "none",
                "reasoning_effort": "none",  # OpenAI rejects function tools with reasoning on
                "messages": [{"role": "system", "content": system}, *msgs],
            }
            url, headers = (
                self.base + "/v1/chat/completions",
                {"authorization": f"Bearer {self.key}"},
            )
        headers.update(
            {
                "content-type": "application/json",
                "x-prefixwise-session-id": self.session,
            }
        )
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise DemoError(
                f"turn {self.turn}: the provider answered {exc.code}: {detail}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DemoError(
                f"turn {self.turn}: no answer from the provider: {exc}"
            ) from None
        # The reply is not used: a fixed line keeps both runs' prompts identical, turn for turn.
        self.messages += [
            {
                "role": "assistant",
                "content": f"Step {self.turn} is done; next, step {self.turn + 1}.",
            },
            {
                "role": "user",
                "content": f"And step {self.turn + 1}, in one short sentence?",
            },
        ]
        return body


def run_session(
    args,
    proxy_port: int,
    provider: str,
    key: str,
    session: str,
    fixed: bool,
    ledger: str,
) -> None:
    agent = Agent(f"http://127.0.0.1:{proxy_port}", provider, key, session)
    base_order = TOOL_NAMES
    reloaded = (
        TOOL_NAMES[4:] + TOOL_NAMES[:4]
    )  # the same eight tools, another order: a plugin reload
    # The clock is text in the prompt, so the demo ticks it by hand and never has to wait a second.
    now = time.time()
    clock = (
        None if fixed else time.strftime("%H:%M:%S", time.localtime(now))
    )  # the fixed agent keeps the time out of the prefix
    shown = 0

    def act(title: str, note: str) -> None:
        if fixed:  # the fixed run does none of this: its nine turns are shown as one list
            return
        say()
        say(title, BOLD)
        say(note, DIM)

    def show() -> None:
        nonlocal shown
        if fixed and agent.turn < 9:
            return
        deadline = time.monotonic() + 5.0  # the proxy writes a row just after it answers
        while True:
            turns = session_turns(ledger, session)
            if len(turns) >= agent.turn or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        for t in turns[shown:]:
            print_turn(t)
        shown = len(turns)

    act(
        "act 1: three clean turns",
        "the first turn writes the cache; the next two read it",
    )
    for _ in range(3):
        agent.request(base_order, clock)
    show()

    pause(args, "act 2: a plugin reload sends the same eight tools in another order")
    act(
        "act 2: the same tools, another order",
        "turn 4 reordered, turn 5 reordered, turn 6 back to the first order",
    )
    for order in (reloaded, reloaded, base_order):
        agent.request(order, clock)
    show()

    pause(args, "act 3: someone adds the current time to the system prompt")
    act(
        "act 3: a timestamp in the system prompt",
        "every character that changed is inside the time",
    )
    for tick in (1, 2):
        if not fixed:
            clock = time.strftime(
                "%H:%M:%S", time.localtime(now + tick)
            )  # the clock ticks: a new system prompt every turn
        agent.request(base_order, clock)
    show()

    pause(args, "act 4: the agent rewrites an earlier message to save tokens")
    act(
        "act 4: an earlier message rewritten",
        "turn 2's reply is shortened in place; everything after it is resent uncached",
    )
    if not fixed:
        agent.messages[1] = {"role": "assistant", "content": "Step 1 done."}
    agent.request(base_order, clock)
    show()


def session_turns(ledger: str, session: str) -> list[dict]:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "prefixwise",
            "report",
            "--ledger",
            ledger,
            "--session",
            session,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise DemoError(
            f"prefixwise report failed: {out.stderr.strip() or out.stdout.strip()}"
        )
    return json.loads(out.stdout)["turns"]


def print_turn(t: dict) -> None:
    cause = t.get("cause") or "-"
    hit = t.get("hit_rate")
    hit_s = f"{hit * 100:5.1f}%" if hit is not None else "    ?"
    colour = GREEN if cause in ("none", "first_write") else RED
    waste = t.get("avoidable_cost") or 0.0
    where = (
        f"  changed at {t['first_changed']}"
        if t.get("first_changed") and cause not in ("none", "first_write")
        else ""
    )
    waste_s = f"  wasted ${waste:.4f}" if waste else ""
    say(
        f"  turn {t['turn']:>2}  {t.get('cached_tokens') or 0:>6,} of {t.get('prompt_tokens') or 0:>6,} from cache "
        f"{hit_s}  {_paint(cause, colour)}{where}{waste_s}"
    )


def totals(ledger: str, session: str) -> dict:
    turns = session_turns(ledger, session)
    prompt = sum(t.get("prompt_tokens") or 0 for t in turns)
    cached = sum(t.get("cached_tokens") or 0 for t in turns)
    return {
        "requests": len(turns),
        "hit": cached / prompt if prompt else 0.0,
        "cost": sum(t.get("cost") or 0 for t in turns),
        "avoidable": sum(t.get("avoidable_tokens") or 0 for t in turns),
        "avoidable_cost": sum(t.get("avoidable_cost") or 0 for t in turns),
    }


def start_proxy(ledger: str, provider: str, live: bool, optimize: bool):
    from prefixwise.proxy import PrefixwiseHTTPServer, PrefixwiseProxy, make_handler

    upstreams = {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com",
    }
    proxy = PrefixwiseProxy(
        ledger_path=ledger,
        mock=not live,
        fixes=["optimize"] if optimize else None,
        upstreams=upstreams if live else None,
    )
    proxy.api_key = None  # the agent's own key passes through; the demo never holds one
    server = PrefixwiseHTTPServer(("127.0.0.1", 0), make_handler(proxy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None, prog: str = "prefixwise demo") -> int:
    global _COLOUR
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Break the prompt cache three ways, name and price each miss, then fix it. "
        "A scripted agent against a built-in mock provider: no key, no network, no cost.",
    )
    ap.add_argument(
        "--step",
        action="store_true",
        help="press Enter between acts instead of running straight through",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="use the real API (ANTHROPIC_API_KEY or OPENAI_API_KEY): 18 small requests, billed to your key",
    )
    ap.add_argument(
        "--provider",
        choices=("anthropic", "openai"),
        help="which API's request format (default: whichever key is set, else anthropic)",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.8,
        metavar="SECONDS",
        help="without --step, the wait between acts (default 0.8)",
    )
    ap.add_argument(
        "--keep", metavar="DIR", help="keep the two ledgers in this directory"
    )
    ap.add_argument("--no-color", action="store_true", help="plain text")
    args = ap.parse_args(argv)
    _COLOUR = (
        not args.no_color
        and not os.environ.get("NO_COLOR")
        and sys.stdout.isatty()
    )
    if args.pause < 0:
        ap.error("--pause cannot be negative")

    provider = args.provider or (
        "anthropic"
        if os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("OPENAI_API_KEY")
        else "openai"
    )
    key = (
        os.environ.get(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        )
        or "demo"
    )
    if args.live and key == "demo":
        raise DemoError(
            "--live needs ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment"
        )
    workdir = args.keep or tempfile.mkdtemp(prefix="prefixwise-demo-")
    run = time.strftime(
        "%H%M%S"
    )  # a new tag per run: a second run cannot read the first's cache
    sessions = (f"demo-before-{run}", f"demo-after-{run}")
    os.makedirs(workdir, exist_ok=True)
    for name in ("before.db", "after.db"):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(os.path.join(workdir, name + suffix))
            except FileNotFoundError:
                pass

    say("Prefixwise demo", BOLD)
    say(
        f"provider: {provider}{' (live)' if args.live else ' (mock: no key, no network, no cost)'}; model {MODEL[provider]}",
        DIM,
    )
    say("one coding agent, nine turns, the whole conversation resent each turn", DIM)

    before = os.path.join(workdir, "before.db")
    server = start_proxy(before, provider, args.live, optimize=False)
    try:
        run_session(
            args,
            server.server_address[1],
            provider,
            key,
            sessions[0],
            fixed=False,
            ledger=before,
        )
    finally:
        server.shutdown()
        server.drain(5)
        server.server_close()

    pause(args, "the report: every miss priced, the fixes ranked")
    say()
    out = subprocess.run(
        [sys.executable, "-m", "prefixwise", "report", "--ledger", before],
        capture_output=True,
        text=True,
        check=False,
    )
    print(out.stdout.rstrip() or out.stderr.rstrip())

    pause(
        args,
        "act 5: the same agent with the proxy's --optimize on, and the two changes the report asked for",
    )
    say()
    say("act 5: the fix", BOLD)
    say(
        "the same nine turns again. proxy: --optimize (tools in a fixed order, cache markers placed). "
        "agent: no timestamp in the system prompt, append instead of rewriting.",
        DIM,
    )
    after = os.path.join(workdir, "after.db")
    server = start_proxy(after, provider, args.live, optimize=True)
    try:
        auto = argparse.Namespace(step=False, pause=0.0)
        run_session(
            auto,
            server.server_address[1],
            provider,
            key,
            sessions[1],
            fixed=True,
            ledger=after,
        )
    finally:
        server.shutdown()
        server.drain(5)
        server.server_close()

    b, a = totals(before, sessions[0]), totals(after, sessions[1])
    say()
    say("before and after", BOLD)
    say(f"  {'':22}{'before':>12}{'after':>12}")
    say(f"  {'served from cache':22}{b['hit'] * 100:>11.1f}%{a['hit'] * 100:>11.1f}%")
    say(
        f"  {'cost of the run':22}{'$' + format(b['cost'], '.4f'):>12}{'$' + format(a['cost'], '.4f'):>12}"
    )
    say(f"  {'avoidable tokens':22}{b['avoidable']:>12,}{a['avoidable']:>12,}")
    saved = b["cost"] - a["cost"]
    if b["cost"] and saved > 0:
        say(
            f"\n  saved ${saved:.4f}, {saved / b['cost'] * 100:.0f}% of the run's cost "
            f"(the report had called ${b['avoidable_cost']:.4f} avoidable)",
            GREEN,
        )
    elif b["cost"]:
        say(
            f"\n  no saving this time (before ${b['cost']:.4f}, after ${a['cost']:.4f}): see the two ledgers",
            RED,
        )
    if not args.live:
        say(
            "\nThe token counts and dollars above come from the mock provider and are illustrative; "
            "the causes are the ones the proxy names on real traffic.",
            DIM,
        )
    if args.keep:
        say(f"\nledgers kept in {workdir}  (hashes and counts only)", DIM)
        say(
            f"turn by turn: prefixwise report --ledger {os.path.join(workdir, 'before.db')} --session {sessions[0]}",
            DIM,
        )
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    say()
    say("next, on your own traffic", BOLD)
    say("  prefixwise import claude-code     # the Claude Code logs already on this machine; no key")
    say("  prefixwise run -- <your agent>    # any agent that reads ANTHROPIC_BASE_URL or OPENAI_BASE_URL")
    return 0


def run(argv: list[str] | None = None, prog: str = "prefixwise demo") -> int:
    """`main` with the demo's own failures as one line, not a traceback."""
    try:
        return main(argv, prog)
    except KeyboardInterrupt:
        print(f"\n{prog}: stopped", file=sys.stderr)
        return 130
    except DemoError as exc:
        print(f"\n{prog}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
