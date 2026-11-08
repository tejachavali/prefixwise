"""prefixwise check: a CI gate for prompt-cache regressions.

    prefixwise check -- pytest tests/agent_scenarios     # run the agent's scenarios through a mock proxy
    prefixwise check --bodies runs/*.jsonl               # or judge saved trajectories (explain's format)

The costly cache regressions are code changes: a tool list built from a set (its order drifts:
`tool_reserialize`), a timestamp added to the system prompt (`volatile_system`), a `cache_control`
marker moved or dropped (`breakpoint_misplaced`), an earlier tool result re-rendered
(`tool_result_mutation`). They pass every functional test and show up weeks later on the bill.

Within one run, each turn is compared with the previous turn of its own conversation, exactly as in
production, so a regression is a cause inside a single run: nothing is compared with another branch
and no prompt is stored between runs. The mock provider answers with synthetic usage; the gated
causes are decided from the request alone (block hashes, markers), so they do not depend on it.

The mock makes an agent's loop run: a user turn that offers tools is answered with a call to a
tool (the forced one, else the first; never when tool_choice is none), so the agent sends a second
turn (the result), which is the turn the gate judges; the turn after a tool call gets text, however
the agent rendered the result, so the mock never drives a loop. The agent has to accept placeholder
arguments (values by type for the schema's required properties). A
conversation's first response is held for `first_turn_delay` seconds (1.1 by default) so a
per-second timestamp in the prompt changes between the turns; a date or time the fingerprinter's
masks find in a system prompt is reported as a warning, because a per-minute one would not change
inside a fast run.

Findings are keyed (scenario, cause, divergence_role) for the allow-list, so a gate can be adopted
on code with known findings. The scenario is the request's session id (`X-Prefixwise-Session-Id`);
under pytest, the bundled plugin (`prefixwise.pytest_plugin`) names each test's node id to the proxy
so an agent that sends no session id still gets one scenario per test.

The dollar projection is an estimate and says so: with a baseline (`prefixwise export --baseline`,
read from `.prefixwise-baseline.json` when present), affected share of this run's eligible turns
(a turn with a previous turn to reuse: first turns are not eligible) x the baseline's eligible turns
per month on that model x its mean reusable prefix tokens x what a miss costs above a hit per token
(`PriceTable.miss_rate`). Without one, tokens only. Never an invented price.

Exit codes: 0 pass, 2 a gated cause found and not allowed, 1 the run itself failed (the agent
command exited non-zero and nothing was found).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .classifier import MissCause
from .fingerprint import _DATE_RE, _TIME_RE
from .fixes import FIX_ADVICE, FIX_STATUS

DEFAULT_FAIL_ON = ("tool_reserialize", "volatile_system", "breakpoint_misplaced", "tool_result_mutation")
STRICT_EXTRA = ("history_edit", "system_edit", "param_rendered")
CAUSES = tuple(c.value for c in MissCause)
BASELINE_FILE = ".prefixwise-baseline.json"
BASELINE_VERSION = 1
DEFAULT_FIRST_TURN_DELAY = 1.1
MONTH_SECONDS = 30 * 86400
_NOT_ELIGIBLE = {"first_write", "model_switch"}  # no previous turn on this model's cache


def parse_fail_on(value: str | None, strict: bool = False) -> tuple[str, ...]:
    names = list(DEFAULT_FAIL_ON) if not value else [v.strip() for v in value.split(",") if v.strip()]
    if strict:
        names += [c for c in STRICT_EXTRA if c not in names]
    unknown = [n for n in names if n not in CAUSES]
    if unknown:
        raise ValueError(f"unknown cause(s) in --fail-on: {', '.join(unknown)}; causes: {', '.join(CAUSES)}")
    if "none" in names:
        raise ValueError("--fail-on none: a hit is not a finding")
    return tuple(names)


@dataclass
class Finding:
    scenario: str
    turn: int  # 1-based, within its scenario
    cause: str
    model: str
    provider: str
    divergence_role: str | None
    divergence_index: int | None
    evidence: str | None
    detail: str | None
    request_id: str | None
    reusable_tokens: int  # tokens this turn could have read and did not (the ledger's avoidable tokens)
    allowed: bool = False


@dataclass
class Projection:
    cause: str
    model: str
    affected: int
    eligible: int
    share: float
    tokens_in_run: int = 0  # reusable tokens these turns did not read, in this run
    eligible_turns_per_month: float | None = None
    tokens_per_month: float | None = None
    usd_per_month: float | None = None
    note: str | None = None


@dataclass
class Report:
    fail_on: tuple
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    projections: list[Projection] = field(default_factory=list)
    requests: int = 0
    scenarios: int = 0
    eligible: int = 0
    eligible_by_model: dict = field(default_factory=dict)
    volatile_scenarios: set = field(default_factory=set)  # scenarios where volatile_system was seen, gated or not
    mock: bool = True
    agent_exit: int | None = None

    @property
    def failing(self) -> list[Finding]:
        return [f for f in self.findings if not f.allowed]

    @property
    def exit_code(self) -> int:
        if self.failing:
            return 2
        return 1 if self.agent_exit else 0

    def to_dict(self) -> dict:
        return {"version": 1, "fail_on": list(self.fail_on), "exit_code": self.exit_code,
                "summary": {"requests": self.requests, "scenarios": self.scenarios, "eligible_turns": self.eligible,
                            "eligible_turns_by_model": self.eligible_by_model, "findings": len(self.findings),
                            "allowed": sum(f.allowed for f in self.findings), "agent_exit": self.agent_exit,
                            "mock": self.mock},
                "findings": [asdict(f) for f in self.findings], "warnings": list(self.warnings),
                "projections": [asdict(p) for p in self.projections]}


# -- the allow-list ----------------------------------------------------------------------

def load_allow(path: str | Path) -> list[dict]:
    """JSON array or JSONL of {"scenario", "cause", "divergence_role"}; a missing key or "*" matches any."""
    text = Path(path).read_text()
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    items = data if isinstance(data, list) else [data]
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("cause", "*"), str):
            raise ValueError(f"{path}: each allow entry is an object with scenario, cause, divergence_role")
    return items


def allowed(f: Finding, allow: list[dict]) -> bool:
    def match(key, value):
        want = entry.get(key, "*")
        return want in ("*", None) or want == value
    for entry in allow:
        if match("scenario", f.scenario) and match("cause", f.cause) and match("divergence_role", f.divergence_role):
            return True
    return False


# -- judging rows ------------------------------------------------------------------------

ROW_COLUMNS = ("session_id", "model", "provider", "cause", "status", "prompt_tokens", "cached_tokens",
               "avoidable_tokens", "divergence_role", "divergence_index", "evidence", "detail", "request_id")


def rows_since(ledger, since: float) -> list[dict]:
    with ledger.reader() as conn:
        cur = conn.execute(f"SELECT {', '.join(ROW_COLUMNS)} FROM entries WHERE recorded_at >= ? ORDER BY id",
                           (since,))
        return [dict(zip(ROW_COLUMNS, r)) for r in cur]


def judge(rows: list[dict], fail_on: tuple, allow: list[dict] | None = None, mock: bool = True) -> Report:
    """Rows of one run (the ledger's columns), in order, to a report: findings for the gated causes,
    eligible turns per model (a turn with a previous turn to reuse), scenarios and requests."""
    rep = Report(fail_on=fail_on, mock=mock)
    turns: dict[str, int] = {}
    for r in rows:
        scenario = str(r.get("session_id") or "default")
        turns[scenario] = turns.get(scenario, 0) + 1
        if (r.get("status") or 200) >= 400 or r.get("cause") is None:
            continue  # no verdict: a failed request
        rep.requests += 1
        model = str(r.get("model") or "unknown")
        if r["cause"] not in _NOT_ELIGIBLE:
            rep.eligible += 1
            rep.eligible_by_model[model] = rep.eligible_by_model.get(model, 0) + 1
        if r["cause"] == "volatile_system":
            rep.volatile_scenarios.add(scenario)
        if r["cause"] in fail_on:
            f = Finding(scenario=scenario, turn=turns[scenario], cause=r["cause"], model=model,
                        provider=str(r.get("provider") or "unknown"), divergence_role=r.get("divergence_role"),
                        divergence_index=r.get("divergence_index"), evidence=r.get("evidence"),
                        detail=r.get("detail"), request_id=r.get("request_id"),
                        reusable_tokens=int(r.get("avoidable_tokens") or 0))
            f.allowed = allowed(f, allow or [])
            rep.findings.append(f)
    rep.scenarios = len(turns)
    if rep.requests and not rep.eligible:
        rep.warnings.append(
            "0 cache-eligible multi-turn sequences observed: every conversation had one turn, so "
            "tool_reserialize, volatile_system and breakpoint_misplaced could not show. The mock answers a "
            "turn that offers tools with a tool call; does the agent run its loop, and does each scenario "
            "keep one session id (X-Prefixwise-Session-Id, or the pytest plugin)?")
    return rep


# -- the baseline and the projection -----------------------------------------------------

def baseline_from_ledger(ledger, since: float | None = None, now: float | None = None) -> dict:
    """Per model: turns per month, eligible turns per month (those with a previous turn to reuse)
    and the mean reusable prefix tokens of an eligible turn (what it read, or could have), scaled
    from the ledger's span to 30 days. Counts and means only: no ids, no text."""
    where, args = " WHERE status < 400 AND cause IS NOT NULL", []
    if since is not None:
        where, args = where + " AND recorded_at >= ?", [since]
    with ledger.reader() as conn:
        span = conn.execute(f"SELECT MIN(recorded_at), MAX(recorded_at) FROM entries{where}", args).fetchone()
        rows = conn.execute(
            f"SELECT model, provider, COUNT(*), SUM(cause NOT IN ('first_write', 'model_switch')),"
            f" SUM(CASE WHEN cause NOT IN ('first_write', 'model_switch') THEN COALESCE(cached_tokens, 0) + COALESCE(avoidable_tokens, 0)"
            f" ELSE 0 END) FROM entries{where} GROUP BY model, provider ORDER BY COUNT(*) DESC", args).fetchall()
    first, last = span if span else (None, None)
    seconds = max((last or 0) - (first or 0), 86400.0)  # a ledger shorter than a day is scaled as a day
    scale = MONTH_SECONDS / seconds
    models: dict[str, dict] = {}
    for model, provider, n, eligible, reusable in rows:
        if model in models:  # the same model under two providers: the busier one names the provider
            m = models[model]
            m["turns_per_month"] += round(n * scale, 1)
            m["eligible_turns_per_month"] += round((eligible or 0) * scale, 1)
            m["_eligible"] += eligible or 0
            m["_reusable"] += reusable or 0
        else:
            models[model] = {"provider": provider, "turns_per_month": round(n * scale, 1),
                             "eligible_turns_per_month": round((eligible or 0) * scale, 1),
                             "_eligible": eligible or 0, "_reusable": reusable or 0}
    for m in models.values():
        m["mean_reusable_tokens"] = round(m.pop("_reusable") / m["_eligible"], 1) if m["_eligible"] else 0.0
        m.pop("_eligible")
    return {"version": BASELINE_VERSION, "generated_at": now if now is not None else time.time(),
            "window_days": round(seconds / 86400, 2), "models": models}


def load_baseline(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("version") != BASELINE_VERSION or not isinstance(data.get("models"), dict):
        raise ValueError(f"{path}: not a prefixwise baseline (version {BASELINE_VERSION}; prefixwise export --baseline)")
    for model, m in data["models"].items():
        if not isinstance(m, dict) or not all(isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)
                                               for k in ("turns_per_month", "eligible_turns_per_month", "mean_reusable_tokens")):
            raise ValueError(f"{path}: model {model!r} needs numeric turns_per_month, eligible_turns_per_month, "
                             "mean_reusable_tokens")
    return data


def project(rep: Report, baseline: dict | None, prices=None) -> None:
    """One projection per (cause, model) with findings not allowed. Share = affected eligible turns /
    eligible turns on that model in this run (first turns excluded from both)."""
    groups: dict[tuple[str, str], list[Finding]] = {}
    for f in rep.failing:
        groups.setdefault((f.cause, f.model), []).append(f)
    for (cause, model), fs in sorted(groups.items()):
        affected, eligible = len(fs), rep.eligible_by_model.get(model, 0)
        share = affected / eligible if eligible else 0.0
        p = Projection(cause=cause, model=model, affected=affected, eligible=eligible, share=share,
                       tokens_in_run=sum(f.reusable_tokens for f in fs))
        base = (baseline or {}).get("models", {}).get(model) if baseline else None
        if baseline is None:
            p.note = "no baseline: tokens only (prefixwise export --baseline on a production ledger)"
        elif base is None:
            p.note = f"the baseline has no turns on {model}: tokens only"
        else:
            p.eligible_turns_per_month = float(base.get("eligible_turns_per_month") or 0)
            mean = float(base.get("mean_reusable_tokens") or 0)
            p.tokens_per_month = share * p.eligible_turns_per_month * mean
            rate = prices.miss_rate(model, base.get("provider"), int(mean)) if prices is not None else None
            if rate is None:
                p.note = f"no price for {model}: tokens only"
            else:
                p.usd_per_month = p.tokens_per_month * rate
        rep.projections.append(p)


# -- the system-prompt date warning ------------------------------------------------------

class VolatileWatch:
    """Installed as the proxy's request observer: notes, per scenario, whether a system block holds
    a date or time (the masks the fingerprinter applies). The text is looked at and dropped."""

    def __init__(self):
        self.seen: dict[str, set[str]] = {}

    def __call__(self, provider: str, session_id: str, raw_blocks: list) -> None:
        kinds = set()
        for role, value in raw_blocks:
            if role not in ("system", "system_dynamic"):
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if _DATE_RE.search(text):
                kinds.add("date")
            if _TIME_RE.search(text):
                kinds.add("time")
        if kinds:
            self.seen.setdefault(str(session_id), set()).update(kinds)

    def warnings(self, rep: Report) -> list[str]:
        out = []
        for scenario, kinds in sorted(self.seen.items()):
            if scenario in rep.volatile_scenarios:  # it did change: a finding (or not gated), not a warning
                continue
            what = " and ".join(sorted(kinds))
            out.append(f"scenario={scenario}: a {what} is in the system prompt. It did not change within this "
                       "run, so no volatile_system was found; one that changes by the minute or the day would "
                       "cost a cache miss per change in production (every turn, with a per-second one). Move it "
                       "to the end of the prompt; if it is fixed text (a cutoff date), ignore this.")
        return out


# -- rendering ---------------------------------------------------------------------------

def _usd(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def _block(f: Finding) -> str:
    if f.divergence_role is None:
        return "no block differed (markers or parameters)"
    idx = "" if f.divergence_index is None else f"[{f.divergence_index}]"
    return f"first changed block: {f.divergence_role}{idx}"


def render(rep: Report) -> str:
    lines = []
    for f in rep.findings:
        tag = "ALLOW" if f.allowed else "FAIL "
        lines.append(f"{tag} {f.cause}  scenario={f.scenario}  turn={f.turn}  {_block(f)}")
        if f.evidence:
            lines.append(f"      {f.evidence}")
        advice = FIX_ADVICE.get(f.cause, {})
        if advice.get("fix") and not f.allowed:
            status = FIX_STATUS.get(advice.get("toggle") or "", {}).get("status")
            lines.append(f"      fix: {advice['fix']}" + (f" [{status}]" if status else ""))
    for p in rep.projections:
        head = (f"{p.cause} on {p.model}: {p.affected} of {p.eligible} eligible turns in this run "
                f"({p.share:.0%})")
        if p.usd_per_month is not None:
            lines.append(f"projected: ~{_usd(p.usd_per_month)}/month, an estimate: {head}; baseline "
                         f"{p.eligible_turns_per_month:,.0f} eligible turns/month, "
                         f"{p.tokens_per_month:,.0f} reusable tokens/month missed")
        elif p.tokens_per_month is not None:
            lines.append(f"projected: ~{p.tokens_per_month:,.0f} reusable tokens/month missed, an estimate: {head}; "
                         f"{p.note}")
        else:
            lines.append(f"affected: {head}; {p.tokens_in_run:,} reusable tokens not read in this run; {p.note}")
    for w in rep.warnings:
        lines.append(f"WARN {w}")
    n, a = len(rep.findings), sum(f.allowed for f in rep.findings)
    mock = " (mock usage: token counts are synthetic; the causes are real)" if rep.mock and rep.requests else ""
    tail = (f"{n} finding{'s' if n != 1 else ''}" + (f" ({a} allowed)" if a else "")
            + f" in {rep.scenarios} scenario{'s' if rep.scenarios != 1 else ''}, {rep.eligible} eligible of "
            f"{rep.requests} requests{mock}")
    if rep.agent_exit:
        tail += f"; the command exited {rep.agent_exit}"
    verdict = {0: "PASS", 1: "RUN FAILED", 2: "FAIL"}[rep.exit_code]
    lines.append(f"{verdict}: {tail}")
    return "\n".join(lines)


# -- saved trajectories ------------------------------------------------------------------

def rows_from_bodies(files: list[str], provider: str | None = None) -> list[dict]:
    """explain's records (bare request bodies, or {"request", "usage", "session_id", ...}) as rows
    judge() reads: one scenario per session id (the file's name when a file has none)."""
    from .explain import explain_session, load_turns

    rows = []
    for path in files:
        turns = load_turns(path)
        for t in turns:
            if t.session_id == "default":
                t.session_id = Path(path).name
        for t, ex in zip(turns, explain_session(turns, provider)):
            rows.append({"session_id": t.session_id, "model": str(t.body.get("model", "unknown")),
                         "provider": ex.provider, "cause": ex.cause, "status": 200,
                         "prompt_tokens": ex.prompt_tokens or 0, "cached_tokens": ex.cached_tokens or 0,
                         "avoidable_tokens": 0 if ex.legit is not False else
                         max(0, (ex.prompt_tokens or 0) - (ex.cached_tokens or 0)),
                         "divergence_role": (ex.divergence or {}).get("role"),
                         "divergence_index": (ex.divergence or {}).get("index"),
                         "evidence": ex.evidence, "detail": None, "request_id": None})
    return rows



__all__ = ["DEFAULT_FAIL_ON", "STRICT_EXTRA", "BASELINE_FILE", "Finding", "Projection", "Report", "VolatileWatch",
           "parse_fail_on", "load_allow", "allowed", "rows_since", "judge", "baseline_from_ledger", "load_baseline",
           "project", "render", "rows_from_bodies"]
