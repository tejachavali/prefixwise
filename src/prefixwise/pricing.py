"""Dollar costs from a dated price table.

Rule (README): if a rate is unknown, the cost is None, never invented.
The built-in table (prices.json) holds only rates read from a provider's
own pricing page for confirmed API model ids, each with its source and
date. Users extend or override it with their own file (negotiated rates,
OpenAI models, anything missing): PREFIXWISE_PRICES=/path.json or
`prefixwise serve --prices /path.json`. Same format as prices.json.

Rates are USD per 1M tokens. Per model:
  input, output, cache_read           required to price a request
  cache_write_5m, cache_write_1h      Anthropic cache writes (5-minute / 1-hour TTL)
  cache_write                         one write rate (providers that report a single
                                      write count); the string "input" means writes are
                                      billed at the input rate, as an explicit choice
  long_prompt                         {"above_prompt_tokens": N, <same rate fields>}:
                                      rates for requests whose whole prompt exceeds N

Request cost = uncached input x input + cache reads x cache_read
             + cache writes x their rate + output x output.
Avoidable cost = avoidable tokens x (paid - cache_read), where paid is the rate this request
was billed for its prompt tokens not read from the cache: on a miss those tokens are written to
the cache again (Anthropic: 1.25x input for 5 minutes, 2x for 1 hour) or sent uncached, and a hit
would have read them. Without the request's usage, paid is the input rate.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from dataclasses import dataclass, field
from importlib import resources

RATE_FIELDS = ("input", "output", "cache_read", "cache_write", "cache_write_5m", "cache_write_1h")


@dataclass
class Breakdown:
    """Billable token counts for one request, from the provider's usage."""

    uncached_input: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_write: int = 0  # single write count (providers that do not split by TTL)
    output: int = 0

    @property
    def prompt(self) -> int:
        return (self.uncached_input + self.cache_read + self.cache_write_5m
                + self.cache_write_1h + self.cache_write)


def breakdown(provider: str, usage: dict) -> Breakdown:
    """Split provider usage into billable parts. Each provider's own fields."""
    usage = usage or {}
    if provider == "anthropic":
        read = int(usage.get("cache_read_input_tokens", 0) or 0)
        total_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        split = usage.get("cache_creation") or {}
        w1h = int(split.get("ephemeral_1h_input_tokens", 0) or 0)
        w5m = int(split.get("ephemeral_5m_input_tokens", 0) or 0) if split else total_write - w1h
        return Breakdown(uncached_input=int(usage.get("input_tokens", 0) or 0), cache_read=read,
                         cache_write_5m=w5m, cache_write_1h=w1h,
                         output=int(usage.get("output_tokens", 0) or 0))
    if "prompt_tokens" not in usage and "input_tokens" in usage:  # OpenAI Responses API
        prompt = int(usage.get("input_tokens", 0) or 0)
        details = usage.get("input_tokens_details") or {}
        out = int(usage.get("output_tokens", 0) or 0)
    else:  # chat completions (OpenAI, vLLM)
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        out = int(usage.get("completion_tokens", 0) or 0)
    cached = int(details.get("cached_tokens", 0) or usage.get("cached_tokens", 0) or 0)
    write = int(details.get("cache_write_tokens", 0) or 0)
    return Breakdown(uncached_input=max(0, prompt - cached - write), cache_read=cached,
                     cache_write=write, output=out)


def normalize_usage(provider: str, usage: dict) -> tuple[int, int]:
    """Return (prompt_tokens, cached_tokens) from a provider usage object.

    Anthropic reports input_tokens EXCLUDING cache reads and writes, so
    the full prompt is input + cache_read + cache_creation.
    OpenAI/vLLM report prompt_tokens with prompt_tokens_details.cached_tokens.
    """
    if provider == "anthropic":
        read = int(usage.get("cache_read_input_tokens", 0) or 0)
        write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        return int(usage.get("input_tokens", 0) or 0) + read + write, read
    if "prompt_tokens" not in usage and "input_tokens" in usage:  # Responses API
        details = usage.get("input_tokens_details") or {}
        return int(usage.get("input_tokens", 0) or 0), int(details.get("cached_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or usage.get("cached_tokens", 0) or 0)
    return prompt, cached


def completion_tokens_of(provider: str, usage: dict) -> int:
    if provider == "anthropic" or ("completion_tokens" not in usage and "output_tokens" in usage):
        return int(usage.get("output_tokens", 0) or 0)
    return int(usage.get("completion_tokens", 0) or 0)



def _builtin() -> dict:
    return json.loads(resources.files("prefixwise").joinpath("prices.json").read_text())


@dataclass
class PriceTable:
    models: dict = field(default_factory=dict)
    as_of: str | None = None
    sources: list[str] = field(default_factory=list)  # files the table was built from

    @classmethod
    def load(cls, extra_path: str | None = None) -> "PriceTable":
        """Built-in rates, then the user's file (env PREFIXWISE_PRICES or extra_path) on top."""
        base = _builtin()
        table = cls(models=dict(base["models"]), as_of=base.get("as_of"), sources=["built-in prices.json"])
        path = extra_path or os.environ.get("PREFIXWISE_PRICES")
        if path:
            with open(path) as f:
                user = json.load(f)
            models = user.get("models", user)
            if not isinstance(models, dict):
                raise ValueError(f"{path}: expected {{'models': {{model_id: rates}}}}")
            for name, rates in models.items():
                _validate(name, rates, path)
            table.models.update(models)
            table.sources.append(path)
        return table

    def rates_for(self, model: str, today: date | None = None) -> dict | None:
        """Rates for a model id: exact, or a table id plus only a dated snapshot
        (-20261007) and/or a cloud version suffix (-v1, -v1:0), optionally behind a
        cloud prefix ('us.anthropic.'). A different model that merely starts with
        a table id (gpt-5.6-sol-pro) is NOT matched. Rates past their
        valid_until date (promotional prices) are unknown."""
        if not model:
            return None
        key = model if model in self.models else None
        if key is None:
            for k in self.models:
                if re.search(rf"(^|[./:]){re.escape(k)}(-\d{{8}})?(-v\d+(:\d+)?)?$", model) and (key is None or len(k) > len(key)):
                    key = k
        if key is None:
            return None
        rates = self.models[key]
        until = rates.get("valid_until")
        if until and (today or date.today()) > date.fromisoformat(until):
            return None
        return rates

    @staticmethod
    def _tier(rates: dict, prompt_tokens: int) -> dict:
        lp = rates.get("long_prompt")
        if lp and prompt_tokens > lp.get("above_prompt_tokens", float("inf")):
            return {**rates, **{k: v for k, v in lp.items() if k in RATE_FIELDS}}
        return rates

    def request_cost(self, model: str, provider: str, usage: dict) -> float | None:
        rates = self.rates_for(model)
        if rates is None:
            return None
        b = breakdown(provider, usage)
        r = self._tier(rates, b.prompt)
        parts = [(b.uncached_input, r.get("input")), (b.cache_read, r.get("cache_read")),
                 (b.output, r.get("output")), (b.cache_write_5m, r.get("cache_write_5m", r.get("cache_write"))),
                 (b.cache_write_1h, r.get("cache_write_1h")), (b.cache_write, r.get("cache_write"))]
        total = 0.0
        for tokens, rate in parts:
            if not tokens:
                continue
            if rate == "input":
                rate = r.get("input")
            if not isinstance(rate, (int, float)):
                return None  # a part we cannot price: unknown, not zero
            total += tokens * rate / 1e6
        return round(total, 8)

    def avoidable_cost(self, model: str, avoidable_tokens: int, prompt_tokens: int,
                       provider: str | None = None, usage: dict | None = None) -> float | None:
        if not avoidable_tokens:
            return 0.0 if self.rates_for(model) else None
        rates = self.rates_for(model)
        if rates is None:
            return None
        r = self._tier(rates, prompt_tokens)
        read = r.get("cache_read")
        if not isinstance(r.get("input"), (int, float)) or not isinstance(read, (int, float)):
            return None
        paid = self._paid_rate(r, breakdown(provider, usage)) if provider and usage else None
        return round(avoidable_tokens * ((r["input"] if paid is None else paid) - read) / 1e6, 8)

    def write_surcharge(self, model: str, prompt_tokens: int, write_5m: int = 0, write_1h: int = 0,
                        cache_write: int = 0) -> float | None:
        """USD paid to write these tokens into the cache above sending them uncached: Anthropic
        0.25x input for 5 minutes, 1x for 1 hour; a provider billing writes at the input rate pays
        none. None when a needed rate is unknown."""
        rates = self.rates_for(model)
        if rates is None:
            return None
        r = self._tier(rates, prompt_tokens)
        inp = r.get("input")
        if not isinstance(inp, (int, float)):
            return None
        total = 0.0
        for n, rate in ((write_5m, r.get("cache_write_5m", r.get("cache_write"))),
                        (write_1h, r.get("cache_write_1h")), (cache_write, r.get("cache_write"))):
            if not n:
                continue
            if rate == "input":
                rate = inp
            if not isinstance(rate, (int, float)):
                return None
            total += n * max(0.0, rate - inp) / 1e6
        return round(total, 8)

    @staticmethod
    def _paid_rate(r: dict, b: Breakdown) -> float | None:
        """USD per 1M tokens this request paid for the prompt tokens it did not read from the
        cache (uncached input and cache writes, at their own rates); None if it read them all or
        a rate is unknown."""
        parts = [(b.uncached_input, r.get("input")), (b.cache_write_5m, r.get("cache_write_5m", r.get("cache_write"))),
                 (b.cache_write_1h, r.get("cache_write_1h")), (b.cache_write, r.get("cache_write"))]
        tokens = cost = 0
        for n, rate in parts:
            if not n:
                continue
            if rate == "input":
                rate = r.get("input")
            if not isinstance(rate, (int, float)):
                return None
            tokens, cost = tokens + n, cost + n * rate
        return cost / tokens if tokens else None


def _validate(name: str, rates: object, path: str) -> None:
    if not isinstance(rates, dict):
        raise ValueError(f"{path}: rates for {name!r} must be an object")
    for k in RATE_FIELDS:
        v = rates.get(k)
        if v is not None and not (isinstance(v, (int, float)) and v >= 0) and not (k == "cache_write" and v == "input"):
            raise ValueError(f"{path}: {name}.{k} must be a non-negative number")
