"""Prometheus metrics at GET /metrics: Prefixwise's facts where operators already look.

Counters cover this process's lifetime (Prometheus keeps the history; rate() and
increase() handle restarts). Labels are bounded on purpose: provider, model, cause
(and error for failures). Session, agent and task ids would make one series per
conversation, so they stay in the ledger and the /api/v1 endpoints.

Cache hit rate, avoidable share and dollars per hour are PromQL over these counters;
docs/prometheus.md has the queries, deploy/grafana/prefixwise.json the dashboard,
deploy/prometheus/prefixwise-rules.yml the alerts. Standard library only.
"""

from __future__ import annotations

import threading

from . import __version__

# (name, help, unit suffix already in the name)
_COUNTERS = (
    ("prefixwise_requests_total", "Requests with a verdict, by the cause of any cache miss (none = a hit)."),
    ("prefixwise_prompt_tokens_total", "Prompt tokens of requests with a verdict."),
    ("prefixwise_cached_tokens_total", "Prompt tokens read from the cache."),
    ("prefixwise_cache_write_tokens_total", "Prompt tokens written to the cache (all TTLs), where reported."),
    ("prefixwise_avoidable_tokens_total", "Prompt tokens that could have been cache reads but were not "
                                          "(waste causes only)."),
    ("prefixwise_completion_tokens_total", "Completion (output) tokens."),
    ("prefixwise_cost_usd_total", "Dollars for priced requests (a model with no rates adds nothing)."),
    ("prefixwise_avoidable_cost_usd_total", "What the avoidable tokens cost as uncached input rather than "
                                            "cache reads (priced requests)."),
)
_FAILED = ("prefixwise_requests_failed_total", "Requests with no verdict: upstream errors, cut-off streams, "
                                               "client disconnects.")
# Seconds. Upstream calls take seconds to minutes; Prefixwise's own time is milliseconds.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320)
_OVERHEAD_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names: tuple[str, ...], values: tuple) -> str:
    return "{" + ",".join(f'{n}="{_esc(v)}"' for n, v in zip(names, values)) + "}" if names else ""


def _num(x: float) -> str:
    return repr(float(x)) if isinstance(x, float) else str(x)


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.series: dict[tuple, list] = {}  # labels -> [bucket counts..., sum, count]

    def observe(self, labels: tuple, value: float) -> None:
        s = self.series.setdefault(labels, [0] * len(self.buckets) + [0.0, 0])
        for i, b in enumerate(self.buckets):
            if value <= b:
                s[i] += 1
        s[-2] += value
        s[-1] += 1


class Metrics:
    """Thread-safe counters fed from each ledger row as the proxy records it."""

    LABELS = ("provider", "model", "cause")

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, float]] = {name: {} for name, _ in _COUNTERS}
        self._failed: dict[tuple, float] = {}
        self._latency = _Histogram(_LATENCY_BUCKETS)
        self._overhead = _Histogram(_OVERHEAD_BUCKETS)

    def observe(self, e) -> None:
        """One recorded LedgerEntry."""
        with self._lock:
            if e.cause is None or (e.status or 200) >= 400:
                key = (e.provider, e.model, e.error or f"http_{e.status}")
                self._failed[key] = self._failed.get(key, 0) + 1
                return
            key = (e.provider, e.model, e.cause)
            c = self._counters
            for name, value in (("prefixwise_requests_total", 1),
                                ("prefixwise_prompt_tokens_total", e.prompt_tokens),
                                ("prefixwise_cached_tokens_total", e.cached_tokens),
                                ("prefixwise_cache_write_tokens_total", e.cache_write_tokens),
                                ("prefixwise_avoidable_tokens_total", e.avoidable_tokens),
                                ("prefixwise_completion_tokens_total", e.completion_tokens),
                                ("prefixwise_cost_usd_total", e.cost),
                                ("prefixwise_avoidable_cost_usd_total", e.avoidable_cost)):
                if value is None:
                    continue
                c[name][key] = c[name].get(key, 0) + value
            if e.latency_ms is not None:
                self._latency.observe((e.provider,), e.latency_ms / 1000)
            if e.overhead_ms is not None:
                self._overhead.observe((), e.overhead_ms / 1000)

    def render(self, engine: str | None = None, kv_events: dict | None = None) -> str:
        """The text exposition format (version 0.0.4)."""
        out: list[str] = []
        with self._lock:
            out += ["# HELP prefixwise_info Prefixwise build and setup.", "# TYPE prefixwise_info gauge",
                    f"prefixwise_info{_labels(('version', 'engine'), (__version__, engine or 'none'))} 1"]
            for name, help_ in _COUNTERS:
                out += [f"# HELP {name} {help_}", f"# TYPE {name} counter"]
                for key, v in sorted(self._counters[name].items()):
                    out.append(f"{name}{_labels(self.LABELS, key)} {_num(v)}")
            name, help_ = _FAILED
            out += [f"# HELP {name} {help_}", f"# TYPE {name} counter"]
            for key, v in sorted(self._failed.items()):
                out.append(f"{name}{_labels(('provider', 'model', 'error'), key)} {_num(v)}")
            for name, help_, hist, names in (
                    ("prefixwise_upstream_latency_seconds", "Upstream time per request with a verdict "
                     "(streams: to the last byte).", self._latency, ("provider",)),
                    ("prefixwise_overhead_seconds", "Time Prefixwise itself added per request.",
                     self._overhead, ())):
                out += [f"# HELP {name} {help_}", f"# TYPE {name} histogram"]
                for key, s in sorted(hist.series.items()):
                    for b, n in zip(hist.buckets, s):
                        out.append(f"{name}_bucket{_labels((*names, 'le'), (*key, _num(float(b))))} {n}")
                    out.append(f"{name}_bucket{_labels((*names, 'le'), (*key, '+Inf'))} {s[-1]}")
                    out.append(f"{name}_sum{_labels(names, key)} {_num(float(s[-2]))}")
                    out.append(f"{name}_count{_labels(names, key)} {s[-1]}")
        if kv_events is not None:
            out += ["# HELP prefixwise_kv_events_connected 1 while the engine's KV-event stream is read.",
                    "# TYPE prefixwise_kv_events_connected gauge",
                    f"prefixwise_kv_events_connected {1 if kv_events.get('connected') else 0}"]
            for field, help_ in (("batches", "Event batches read."),
                                 ("dropped_messages", "Messages lost (sequence gaps): evictions cannot be "
                                                      "ruled out meanwhile."),
                                 ("decode_errors", "Event payloads that could not be decoded."),
                                 ("engine_restarts", "Engine restarts seen (sequence restarted)."),
                                 ("overflows", "Times the block index overflowed (removals were missed)."),
                                 ("disconnects", "Times a KV-event stream's connection dropped (engine gone or "
                                                 "replaced): nothing is claimed across the gap."),
                                 ("wrong_rank", "Event batches stamped with another data-parallel rank than the "
                                                "stream's (crossed endpoints): claims stop.")):
                name = f"prefixwise_kv_events_{field}_total"
                out += [f"# HELP {name} {help_}", f"# TYPE {name} counter",
                        f"{name} {int(kv_events.get(field) or 0)}"]
        return "\n".join(out) + "\n"
