"""Prefixwise production benchmark: overhead, throughput, memory, soak, faults, ledger.

Runs the real `prefixwise serve` (a separate process) in front of a fake
upstream (benchmarks/upstream.py, another process), drives it with realistic
agent sessions (a long system prompt and tools, then turns appended one by one)
and measures what the proxy adds. No API keys, no network.

    python benchmarks/bench.py --quick         # about 3 minutes
    python benchmarks/bench.py                 # about 10 minutes
    python benchmarks/bench.py --soak 1800     # add a 30-minute soak

Writes benchmarks/runs/<date>-<host>.json and .md.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import random
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORDS = ("the agent reads file src main tool result returns value error index cache prefix token "
         "model session request response stream config build test patch diff line function class "
         "import module path search query user system assistant message content block hash").split()


# ---------------------------------------------------------------- processes

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port: int, path="/health", timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", path)
            c.getresponse().read()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"port {port} never came up")


class Upstream:
    def __init__(self, delay_ms=0, chunks=20, chunk_delay_ms=0, slow_s=5):
        self.port = free_port()
        env = {**os.environ, "BENCH_SLOW_SECONDS": str(slow_s)}
        self.proc = subprocess.Popen([sys.executable, str(HERE / "upstream.py"), str(self.port), str(delay_ms),
                                      str(chunks), str(chunk_delay_ms)], stdout=subprocess.PIPE, env=env)
        self.proc.stdout.readline()
        wait_ready(self.port, "/v1/models")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(5)


class Proxy:
    def __init__(self, upstream: int | str, ledger: str, env: dict | None = None, extra: list | None = None):
        """upstream: a local port, or a base URL (live_overhead.py)."""
        self.port = free_port()
        url = upstream if isinstance(upstream, str) else f"http://127.0.0.1:{upstream}"
        self.ledger = ledger
        e = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
        e.pop("PREFIXWISE_API_KEY", None)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "prefixwise", "serve", "--port", str(self.port),
             "--upstream", url, "--ledger", ledger, *(extra or [])],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=e)
        wait_ready(self.port)

    def stats(self) -> dict:
        pid = self.proc.pid
        st = Path(f"/proc/{pid}/status").read_text()
        val = {l.split(":")[0]: l.split(":")[1].strip() for l in st.splitlines() if ":" in l}
        return {"rss_mb": int(val["VmRSS"].split()[0]) / 1024, "threads": int(val["Threads"]),
                "fds": len(os.listdir(f"/proc/{pid}/fd"))}

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> str:
        self.proc.terminate()
        try:
            _, err = self.proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            _, err = self.proc.communicate()
        return (err or b"").decode(errors="replace")


# ---------------------------------------------------------------- workload

def text(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(tokens))  # ~1 token per word


class Session:
    """An agent session: a fixed system prompt and tools, then one turn
    (user, assistant tool call, tool result) appended per request."""

    def __init__(self, sid: str, provider: str, base_tokens: int, turn_tokens: int, seed: int):
        self.sid, self.provider, self.turn_tokens = sid, provider, turn_tokens
        self.rng = random.Random(seed)
        self.system = text(self.rng, int(base_tokens * 0.6))
        self.tools = [{"name": f"tool_{i}", "description": text(self.rng, int(base_tokens * 0.4 / 8)),
                       "params": {"type": "object", "properties": {"path": {"type": "string"}}}} for i in range(8)]
        self.messages: list[dict] = []
        self.turn = 0

    def next_body(self, stream=False, model=None) -> dict:
        self.turn += 1
        if self.messages:
            self.messages.append({"role": "assistant", "content": text(self.rng, 60)})
        self.messages.append({"role": "user", "content": text(self.rng, self.turn_tokens)})
        if self.provider == "anthropic":
            body = {"model": model or "claude-sonnet-5-5", "max_tokens": 1024, "system": self.system,
                    "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["params"]}
                              for t in self.tools],
                    "messages": list(self.messages), "metadata": {"user_id": self.sid}}
        else:
            body = {"model": model or "gpt-5.6-terra", "user": self.sid,
                    "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                "parameters": t["params"]}} for t in self.tools],
                    "messages": [{"role": "system", "content": self.system}, *self.messages]}
        if stream:
            body["stream"] = True
        return body

    @property
    def path(self) -> str:
        return "/v1/messages" if self.provider == "anthropic" else "/v1/chat/completions"


def call(port: int, path: str, body: dict | bytes, headers=None, timeout=60.0, read=True) -> dict:
    """One request on a fresh connection. Returns status, ttfb, total, bytes, headers."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("POST", path, body=data, headers=h)
        r = c.getresponse()
        first = r.read(1) if read else b""
        ttfb = time.perf_counter() - t0
        rest = r.read() if read else b""
        total = time.perf_counter() - t0
        return {"status": r.status, "ttfb": ttfb, "total": total, "bytes": len(first) + len(rest),
                "headers": {k.lower(): v for k, v in r.getheaders()}, "body": first + rest}
    except Exception as exc:
        return {"status": None, "error": f"{type(exc).__name__}: {exc}", "total": time.perf_counter() - t0}
    finally:
        c.close()


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ms(x: float) -> float:
    return round(x * 1000, 2)


# ---------------------------------------------------------------- scenarios

SIZES = {"small (2k tok)": (2000, 150), "agent (30k tok)": (12000, 900), "large (120k tok)": (60000, 3000)}


def overhead(n: int) -> list[dict]:
    """Sequential latency, direct vs. through the proxy, same session shape.
    Upstream answers instantly, so the difference is the proxy's own cost."""
    up = Upstream(delay_ms=0, chunks=20)
    tmp = tempfile.mkdtemp()
    px = Proxy(up.port, f"{tmp}/ledger.db")
    out = []
    try:
        for label, (base, turn) in SIZES.items():
            for stream in (False, True):
                for provider in ("openai", "anthropic"):
                    res = {"direct": [], "proxy": [], "direct_ttfb": [], "proxy_ttfb": []}
                    sd = Session(f"d-{label}-{stream}-{provider}", provider, base, turn, 1)
                    sp = Session(f"p-{label}-{stream}-{provider}", provider, base, turn, 1)
                    sizes = []
                    for _ in range(n):
                        bd, bp = sd.next_body(stream), sp.next_body(stream)
                        sizes.append(len(json.dumps(bp)))
                        rd = call(up.port, sd.path, bd)
                        rp = call(px.port, sp.path, bp, {"X-Prefixwise-Session-Id": sp.sid})
                        assert rd["status"] == 200 and rp["status"] == 200, (rd.get("error"), rp.get("status"))
                        res["direct"].append(rd["total"]); res["proxy"].append(rp["total"])
                        res["direct_ttfb"].append(rd["ttfb"]); res["proxy_ttfb"].append(rp["ttfb"])
                    added = [p - d for p, d in zip(res["proxy"], res["direct"])]
                    out.append({"size": label, "provider": provider, "stream": stream, "n": n,
                                "body_kb_median": round(statistics.median(sizes) / 1024, 1),
                                "direct_p50_ms": ms(pct(res["direct"], .5)),
                                "proxy_p50_ms": ms(pct(res["proxy"], .5)),
                                "added_p50_ms": ms(pct(added, .5)), "added_p95_ms": ms(pct(added, .95)),
                                "added_p99_ms": ms(pct(added, .99)),
                                "ttfb_added_p50_ms": ms(pct(res["proxy_ttfb"], .5) - pct(res["direct_ttfb"], .5))})
                    print(f"  overhead {label:17} {provider:9} stream={stream!s:5} "
                          f"+{out[-1]['added_p50_ms']} ms p50, +{out[-1]['added_p95_ms']} ms p95", flush=True)
    finally:
        px.stop(); up.stop()
    return out


def throughput(levels: list[int], turns: int, delay_ms: int, label: str) -> list[dict]:
    """Each worker runs its own agent session; upstream takes delay_ms per call."""
    up = Upstream(delay_ms=delay_ms)
    out = []
    for c in levels:
        tmp = tempfile.mkdtemp()
        px = Proxy(up.port, f"{tmp}/ledger.db")
        lat, errors = [], []

        def worker(i):
            s = Session(f"t{c}-{i}", "openai" if i % 2 else "anthropic", 12000, 900, i)
            for _ in range(turns):
                r = call(px.port, s.path, s.next_body(), {"X-Prefixwise-Session-Id": s.sid})
                if r["status"] == 200:
                    lat.append(r["total"])
                else:
                    errors.append(r.get("status") or r.get("error"))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as pool:
            list(pool.map(worker, range(c)))
        wall = time.perf_counter() - t0
        rows = sqlite3.connect(px.ledger).execute("select count(*) from entries").fetchone()[0]
        st = px.stats()
        px.stop()
        out.append({"workload": label, "concurrency": c, "requests": c * turns, "ok": len(lat),
                    "errors": len(errors), "error_kinds": sorted({str(e) for e in errors})[:5],
                    "ledger_rows": rows, "req_per_s": round(len(lat) / wall, 1),
                    "p50_ms": ms(pct(lat, .5)), "p99_ms": ms(pct(lat, .99)),
                    "upstream_ms": delay_ms, "rss_mb": round(st["rss_mb"], 1)})
        print(f"  throughput {label:12} c={c:<4} {out[-1]['req_per_s']:>7} req/s  p50 {out[-1]['p50_ms']} ms  "
              f"p99 {out[-1]['p99_ms']} ms  errors {len(errors)}  ledger {rows}/{c * turns}", flush=True)
    up.stop()
    return out


def memory(sessions: int, max_sessions: int | None) -> dict:
    """Many short sessions; RSS should plateau once the session map is full."""
    up = Upstream()
    tmp = tempfile.mkdtemp()
    env = {"PREFIXWISE_MAX_SESSIONS": str(max_sessions)} if max_sessions else {}
    px = Proxy(up.port, f"{tmp}/ledger.db", env=env)
    samples = [{"sessions": 0, **px.stats()}]
    step = max(1, sessions // 10)

    def one(i):
        s = Session(f"m{i}", "openai" if i % 2 else "anthropic", 3000, 300, i)
        for _ in range(3):
            call(px.port, s.path, s.next_body(), {"X-Prefixwise-Session-Id": s.sid})

    with ThreadPoolExecutor(16) as pool:
        for start in range(0, sessions, step):
            list(pool.map(one, range(start, min(sessions, start + step))))
            samples.append({"sessions": min(sessions, start + step), **px.stats()})
    px.stop(); up.stop()
    first_half = samples[len(samples) // 2]["rss_mb"]
    res = {"max_sessions": max_sessions or 10000, "sessions": sessions,
           "rss_start_mb": round(samples[0]["rss_mb"], 1), "rss_mid_mb": round(first_half, 1),
           "rss_end_mb": round(samples[-1]["rss_mb"], 1),
           "kb_per_session_until_cap": round((samples[-1]["rss_mb"] - samples[0]["rss_mb"]) * 1024
                                             / min(sessions, max_sessions or 10000), 1),
           "samples": samples}
    print(f"  memory sessions={sessions} cap={res['max_sessions']}: RSS {res['rss_start_mb']} -> "
          f"{res['rss_mid_mb']} -> {res['rss_end_mb']} MB", flush=True)
    return res


def soak(seconds: int, concurrency: int) -> dict:
    """Mixed traffic for a fixed time: providers, streaming, growing sessions,
    task outcomes. Tracks RSS, threads and fds for drift, and every error."""
    up = Upstream(delay_ms=50, chunks=20, chunk_delay_ms=2)
    tmp = tempfile.mkdtemp()
    px = Proxy(up.port, f"{tmp}/ledger.db", env={"PREFIXWISE_MAX_SESSIONS": "500"})
    stop = time.time() + seconds
    counts = {"ok": 0, "errors": 0}
    errs: list[str] = []
    lock = threading.Lock()
    samples = []

    def worker(i):
        n = 0
        while time.time() < stop:
            n += 1
            s = Session(f"s{i}-{n}", "openai" if (i + n) % 2 else "anthropic", 6000, 500, i * 1000 + n)
            task = f"task-{i}-{n}"
            for t in range(random.Random(n).randint(5, 25)):
                if time.time() >= stop:
                    break
                r = call(px.port, s.path, s.next_body(stream=t % 3 == 0),
                         {"X-Prefixwise-Session-Id": s.sid, "X-Prefixwise-Task-Id": task})
                with lock:
                    if r["status"] == 200:
                        counts["ok"] += 1
                    else:
                        counts["errors"] += 1
                        errs.append(str(r.get("status") or r.get("error")))
            call(px.port, "/tasks", {"task_id": task, "success": n % 4 != 0})

    def sampler():
        while time.time() < stop:
            samples.append({"t": round(seconds - (stop - time.time())), **px.stats()})
            time.sleep(max(1, seconds // 30))

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    th.join()
    final = px.stats()
    db = sqlite3.connect(px.ledger)
    rows = db.execute("select count(*) from entries").fetchone()[0]
    db.close()
    ledger_bytes = sum(p.stat().st_size for p in Path(tmp).glob("ledger.db*"))
    t0 = time.perf_counter()
    rep = subprocess.run([sys.executable, "-m", "prefixwise", "report", "--ledger", px.ledger],
                         capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    report_s = time.perf_counter() - t0
    stderr = px.stop()
    up.stop()
    warm = samples[min(2, len(samples) - 1)]
    res = {"seconds": seconds, "concurrency": concurrency, "ok": counts["ok"], "errors": counts["errors"],
           "error_kinds": sorted(set(errs))[:5], "ledger_rows": rows, "ledger_mb": round(ledger_bytes / 2**20, 2),
           "ledger_bytes_per_request": round(ledger_bytes / max(rows, 1)),
           "report_seconds": round(report_s, 2), "report_ok": rep.returncode == 0,
           "rss_warm_mb": round(warm["rss_mb"], 1), "rss_end_mb": round(final["rss_mb"], 1),
           "threads_end": final["threads"], "fds_end": final["fds"],
           "proxy_stderr": stderr.strip().splitlines()[-5:], "samples": samples}
    print(f"  soak {seconds}s c={concurrency}: {counts['ok']} ok, {counts['errors']} errors, ledger {rows} rows "
          f"{res['ledger_mb']} MB, RSS {res['rss_warm_mb']} -> {res['rss_end_mb']} MB, "
          f"threads {final['threads']}, fds {final['fds']}", flush=True)
    return res


def faults() -> list[dict]:
    """Each case: what the client gets, and whether the proxy is still healthy after."""
    up = Upstream(slow_s=3)
    tmp = tempfile.mkdtemp()
    px = Proxy(up.port, f"{tmp}/ledger.db", env={"PREFIXWISE_CLIENT_TIMEOUT": "2"})
    S = lambda model, stream=False, provider="openai": Session("f", provider, 500, 50, 1).next_body(stream, model)
    cases = []

    def case(name, expect, fn):
        try:
            got, ok, note = fn()
        except Exception as exc:
            got, ok, note = f"{type(exc).__name__}: {exc}", False, ""
        healthy = call(px.port, "/v1/chat/completions", S("gpt-5.6-terra"))["status"] == 200 and px.alive()
        cases.append({"case": name, "expect": expect, "got": got, "pass": bool(ok and healthy),
                      "proxy_healthy_after": healthy, "note": note})
        print(f"  fault {'PASS' if ok and healthy else 'FAIL'} {name}: {got}", flush=True)

    def status_is(model, codes, stream=False, provider="openai"):
        def fn():
            path = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
            r = call(px.port, path, S(model, stream, provider))
            note = r.get("error", "") or r.get("body", b"")[:120].decode(errors="replace")
            return r.get("status") or r.get("error"), r.get("status") in codes, note
        return fn

    case("upstream 500", "500 passed through", status_is("bench-fault-500", {500}))
    case("upstream 429", "429 passed through", status_is("bench-fault-429", {429}))
    case("upstream closes connection", "502", status_is("bench-fault-reset", {502}))
    case("upstream returns HTML", "502", status_is("bench-fault-badjson", {502}))
    case("upstream omits usage", "200, cause unknown", status_is("bench-fault-nousage", {200}))
    case("upstream slow (3 s)", "200 after the delay", status_is("bench-fault-slow", {200}))

    def midstream(provider):
        def fn():
            path = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
            r = call(px.port, path, S("bench-fault-midstream", True, provider))
            return (r.get("status") or r.get("error"),
                    r.get("status") in (200, 502) or "IncompleteRead" in str(r.get("error")), "")
        return fn

    case("upstream drops mid-stream (openai)", "stream ends; proxy survives", midstream("openai"))
    case("upstream drops mid-stream (anthropic)", "stream ends; proxy survives", midstream("anthropic"))

    def client_abort():
        # Client hangs up after the first bytes of a stream.
        for _ in range(20):
            c = http.client.HTTPConnection("127.0.0.1", px.port, timeout=5)
            c.request("POST", "/v1/chat/completions", body=json.dumps(S("gpt-5.6-terra", True)),
                      headers={"Content-Type": "application/json"})
            c.getresponse().read(10)
            c.close()
        return "20 aborted streams", True, ""

    case("client disconnects mid-stream x20", "proxy survives", client_abort)
    case("client sends invalid JSON", "400", lambda: (lambda r: (r["status"], r["status"] == 400, ""))(
        call(px.port, "/v1/chat/completions", b"{not json")))
    case("client sends a JSON array", "400", lambda: (lambda r: (r["status"], r["status"] == 400, ""))(
        call(px.port, "/v1/chat/completions", b"[1,2]")))
    case("client body over the limit", "413", lambda: (lambda r: (r["status"], r["status"] == 413, ""))(
        call(px.port, "/v1/chat/completions", b"{}", {"Content-Length": str(64 * 2**20)}, read=True)))

    def idle_connections():
        # Clients that connect and send nothing (or half a request) must not
        # starve everyone else, and must eventually be dropped.
        socks = []
        for i in range(300):
            s = socket.create_connection(("127.0.0.1", px.port))
            if i % 2:
                s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 1000\r\n\r\n{")
            socks.append(s)
        time.sleep(0.5)
        r = call(px.port, "/v1/chat/completions", S("gpt-5.6-terra"), timeout=10)
        held = px.stats()["threads"]
        time.sleep(3)  # past PREFIXWISE_CLIENT_TIMEOUT=2
        after = px.stats()["threads"]
        for s in socks:
            s.close()
        return (f"request {r.get('status')} while 300 idle; threads {held} -> {after} after the client timeout",
                r.get("status") == 200 and after < 10, "")

    case("300 idle / half-sent connections", "others served; threads released", idle_connections)
    stderr = px.stop()
    up.stop()
    cases.append({"case": "proxy stderr", "got": stderr.strip().splitlines()[-8:], "pass": True})
    return cases


# ---------------------------------------------------------------- report

def write_report(res: dict, path: Path):
    L = [f"# Prefixwise benchmark: {res['date']}", "",
         f"Prefixwise {res['version']}, Python {res['python']}, {res['cpus']} CPUs, {res['machine']}. "
         "Fake upstream on localhost in its own process; proxy is the real `prefixwise serve` with an on-disk "
         "ledger. Overhead = proxy latency minus direct latency to the same upstream, same request shape.", ""]
    if res.get("overhead"):
        L += ["## Added latency (sequential, upstream answers instantly)", "",
              "| prompt | provider | stream | body KB | direct p50 | proxy p50 | added p50 | added p95 | added p99 | TTFB added p50 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in res["overhead"]:
            L.append(f"| {r['size']} | {r['provider']} | {'yes' if r['stream'] else 'no'} | {r['body_kb_median']} | "
                     f"{r['direct_p50_ms']} ms | {r['proxy_p50_ms']} ms | **{r['added_p50_ms']} ms** | "
                     f"{r['added_p95_ms']} ms | {r['added_p99_ms']} ms | {r['ttfb_added_p50_ms']} ms |")
        L.append("")
    if res.get("throughput"):
        L += ["## Throughput", "", "Each client runs its own agent session (~12k-token base, +900 tokens a turn).", "",
              "| upstream | concurrency | requests | errors | ledger rows | req/s | p50 | p99 | RSS |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in res["throughput"]:
            L.append(f"| {r['workload']} | {r['concurrency']} | {r['requests']} | {r['errors']} | {r['ledger_rows']} | "
                     f"{r['req_per_s']} | {r['p50_ms']} ms | {r['p99_ms']} ms | {r['rss_mb']} MB |")
        L.append("")
    if res.get("memory"):
        L += ["## Memory", "", "| sessions | session cap | RSS start | RSS midway | RSS end | KB per session |",
              "|---:|---:|---:|---:|---:|---:|"]
        for r in res["memory"]:
            L.append(f"| {r['sessions']} | {r['max_sessions']} | {r['rss_start_mb']} MB | {r['rss_mid_mb']} MB | "
                     f"{r['rss_end_mb']} MB | {r['kb_per_session_until_cap']} |")
        L.append("")
    if res.get("soak"):
        s = res["soak"]
        L += ["## Soak", "",
              f"{s['seconds']} s at concurrency {s['concurrency']}, mixed providers, a third streamed, task outcomes "
              f"posted: **{s['ok']} ok, {s['errors']} errors**{(' ' + str(s['error_kinds'])) if s['errors'] else ''}. "
              f"Ledger {s['ledger_rows']} rows, {s['ledger_mb']} MB ({s['ledger_bytes_per_request']} bytes/request); "
              f"`prefixwise report` on it: {s['report_seconds']} s. RSS {s['rss_warm_mb']} MB warm -> "
              f"{s['rss_end_mb']} MB at the end; threads {s['threads_end']}, fds {s['fds_end']}.", ""]
    if res.get("faults"):
        L += ["## Failure handling", "", "| case | expected | got | pass |", "|---|---|---|---|"]
        for c in res["faults"]:
            if c["case"] == "proxy stderr":
                continue
            L.append(f"| {c['case']} | {c['expect']} | {c['got']} | {'yes' if c['pass'] else '**NO**'} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--soak", type=int, help="soak seconds (default 60, quick 30)")
    ap.add_argument("--only", nargs="*", choices=["overhead", "throughput", "memory", "soak", "faults"])
    ap.add_argument("--out", default=str(HERE / "runs"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from prefixwise import __version__
    only = set(a.only or ["overhead", "throughput", "memory", "soak", "faults"])
    res = {"date": date.today().isoformat(), "version": __version__, "python": platform.python_version(),
           "cpus": os.cpu_count(), "machine": f"{platform.system()} {platform.machine()}",
           "quick": a.quick}
    if "overhead" in only:
        print("overhead", flush=True)
        res["overhead"] = overhead(15 if a.quick else 40)
    if "throughput" in only:
        print("throughput", flush=True)
        lv = [1, 8, 32] if a.quick else [1, 8, 32, 64, 128]
        res["throughput"] = (throughput(lv, 5 if a.quick else 10, 0, "instant") +
                             throughput(lv, 5 if a.quick else 10, 300, "300 ms"))
    if "memory" in only:
        print("memory", flush=True)
        n = 1500 if a.quick else 6000
        res["memory"] = [memory(n, None), memory(n, 1000)]
    if "soak" in only:
        print("soak", flush=True)
        res["soak"] = soak(a.soak or (30 if a.quick else 60), 16)
    if "faults" in only:
        print("faults", flush=True)
        res["faults"] = faults()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{res['date']}-{platform.node() or 'host'}{'-quick' if a.quick else ''}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str))
    write_report(res, out / f"{stem}.md")
    print(f"\nwrote {out / stem}.md and .json")
    failed = [c["case"] for c in res.get("faults", []) if not c["pass"]]
    errors = sum(r["errors"] for r in res.get("throughput", [])) + res.get("soak", {}).get("errors", 0)
    if failed or errors:
        print(f"PROBLEMS: failed fault cases {failed}; request errors {errors}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
