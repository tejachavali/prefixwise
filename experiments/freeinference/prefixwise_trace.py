#!/usr/bin/env python3
"""Prefixwise's miss verdicts on the FreeInference agentic trace (Harvard MADSys).

The trace holds no text. Per request it holds the input as chained 16-token
block ids, the provider's own `prompt_tokens` and `cache_read_tokens`, times,
model, an anonymous provider letter, o200k tokens per role, and the dataset
authors' own `transition` / `cause` label against step - 1 (computed from the
text before anonymising). This script compares every request after a
session's first with the session's previous step, the way Prefixwise compares
a turn with its previous turn (docs/ALGORITHM.md, sections 1 to 3), names a
verdict, counts the tokens lost, and cross-tabulates the verdicts against the
authors' labels.

    python3 prefixwise_trace.py WEEK_DIR [WEEK_DIR ...] [--out results.json]
    python3 prefixwise_trace.py RELEASE_DIR --week 2026-08-30 --out results.json
    python3 prefixwise_trace.py --selftest [--trace-repo PATH_TO_FREEINFERENCE_CLONE]

Only pyarrow, pandas and numpy are needed. `block_ids` is read with column
selection, a few thousand rows at a time; everything else is read once per
week without it. The decision rules are documented in README.md beside this
file, and in `VERDICT_RULES` below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy
import pandas
import pyarrow
from pyarrow import parquet

# --------------------------------------------------------------------------------------------------------------------
# The release format (a copy of release/schema.py, FORMAT_VERSION 1; the self-test checks the copy against the
# original when --trace-repo points at a clone of the trace repository).

FORMAT_VERSION = 1
SESSIONS_FILE = "sessions.parquet"
REQUESTS_FILE = "requests.parquet"
TOOL_CALLS_FILE = "tool_calls.parquet"
MANIFEST_FILE = "manifest.json"
TABLES_FILE = "tables.json"
BLOCK_TOKENS = 16
ROLES = ("system", "user", "assistant", "tool")
ROLE_TOKEN_FIELDS = ROLES + ("tool_definitions",)
ROLE_TOKENS_TYPE = pyarrow.struct([(field, pyarrow.int64()) for field in ROLE_TOKEN_FIELDS])
TRANSITIONS = ("append", "mutation", "switch")
CAUSES = ("compaction", "tool_definitions", "system_prompt", "dropped_turns", "injection", "tool_result", "other")

SESSIONS_SCHEMA = pyarrow.schema([
    ("session_id", pyarrow.string()),
    ("user_id", pyarrow.string()),
    ("harness", pyarrow.string()),
    ("parent_session_id", pyarrow.string()),
    ("spawned_by_step", pyarrow.int32()),
    ("spawned_by_call_index", pyarrow.int32()),
])
REQUESTS_SCHEMA = pyarrow.schema([
    ("session_id", pyarrow.string()),
    ("step", pyarrow.int32()),
    ("link_reason", pyarrow.string()),
    ("start_ms", pyarrow.int64()),
    ("end_ms", pyarrow.int64()),
    ("ttft_ms", pyarrow.int64()),
    ("model", pyarrow.string()),
    ("provider", pyarrow.string()),
    ("stream", pyarrow.bool_()),
    ("step_trigger", pyarrow.string()),
    ("waited_for", pyarrow.string()),
    ("prompt_tokens", pyarrow.int64()),
    ("completion_tokens", pyarrow.int64()),
    ("cache_read_tokens", pyarrow.int64()),
    ("role_tokens", ROLE_TOKENS_TYPE),
    ("n_messages", pyarrow.int32()),
    ("tool_definition_names", pyarrow.list_(pyarrow.string())),
    ("finish_reason", pyarrow.string()),
    ("transition", pyarrow.string()),
    ("cause", pyarrow.string()),
    ("block_ids", pyarrow.large_list(pyarrow.int64())),
])
TOOL_CALLS_SCHEMA = pyarrow.schema([
    ("session_id", pyarrow.string()),
    ("step", pyarrow.int32()),
    ("call_index", pyarrow.int32()),
    ("name", pyarrow.string()),
    ("category", pyarrow.string()),
    ("programs", pyarrow.list_(pyarrow.string())),
    ("parse_status", pyarrow.string()),
    ("arguments_skeleton", pyarrow.string()),
    ("outcome", pyarrow.string()),
    ("result_tokens", pyarrow.int64()),
    ("latency_ms", pyarrow.int64()),
])

# --------------------------------------------------------------------------------------------------------------------
# Prefixwise's thresholds (docs/ALGORITHM.md section 1, src/prefixwise/classifier.py).

READ_SLACK_TOKENS = 128  # reads are reported in whole 128-token units on every provider but Anthropic
MIN_CACHEABLE_TOKENS = 1024  # OpenAI's (and most Anthropic models') minimum cacheable prefix
CLOCK_SKEW_MS = 1_000  # PREFIXWISE_CLOCK_SKEW: a request sent this much before its previous turn ended is in flight
GAP_BUCKETS = (("<5m", 0.0, 300.0), ("5-60m", 300.0, 3_600.0), (">60m", 3_600.0, math.inf))
REGIONS = ("tool_definitions", "system", "conversation")  # prefix order, from export/log_records/messages.py

VERDICTS = ("first_write", "switch", "hit", "below_min", "prompt_changed", "not_read", "unknown")

VERDICT_RULES = """\
Rules, first match wins (Prefixwise order, ALGORITHM.md 2.1-2.4):
 1 first_write    step 0 of a session (each subagent session is its own conversation)
 2 unknown        step - 1 is missing from the table (no previous turn to compare)
 3 switch         model or provider letter differs from step - 1
 4 unknown        sent > 1 s before step - 1 ended, and read < half of R (ALGORITHM 2.4: could not read its write)
 5 unknown        prompt_tokens missing
 6 hit            read >= R - min(128, R/8)   (served all, ALGORITHM 1; R = min(prompt_prev, prompt_curr))
 7 hit            prefix intact and read >= R/2   (the half-read rule, ALGORITHM 1 / 2.3)
 8 below_min      R < 1024 and nothing read (or no count)
 9 prompt_changed the block ids diverge before the end of the previous input (or the input is a strict prefix of it)
10 unknown        cache_read_tokens is null: the provider reports no cache count (ALGORITHM 2.3)
11 not_read       prefix intact, read < R/2; bucketed by the idle gap: <5m, 5-60m, >60m
"""


# --------------------------------------------------------------------------------------------------------------------
# Finding the weeks.

def week_directories(paths: list[pathlib.Path], weeks: list[str] | None) -> list[pathlib.Path]:
    """Each path is a week folder (it holds requests.parquet) or a release folder of week folders."""
    found = []
    for path in paths:
        path = path.expanduser()
        if (path / REQUESTS_FILE).exists():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(child for child in path.iterdir() if (child / REQUESTS_FILE).exists()))
        else:
            raise SystemExit(f"{path}: not a week folder (no {REQUESTS_FILE}) nor a folder of week folders")
    if weeks:
        found = [path for path in found if path.name[:10] in weeks]
        missing = sorted(set(weeks) - {path.name[:10] for path in found})
        if missing:
            raise SystemExit(f"no week folder starts with {', '.join(missing)}")
    if not found:
        raise SystemExit("no week folder found")
    return found


def file_sha256(path: pathlib.Path) -> str:
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_week(week: pathlib.Path) -> None:
    """The reader's check (release/reader.py check_week): format version, and each table's sha256 in tables.json."""
    manifest = json.loads((week / MANIFEST_FILE).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise SystemExit(f"{week.name}: release format {manifest.get('format_version')}, this script reads {FORMAT_VERSION}")
    record = json.loads((week / TABLES_FILE).read_text())
    if record["manifest_sha256"] != file_sha256(week / MANIFEST_FILE):
        raise SystemExit(f"{week.name}: tables were converted from another manifest")
    for name, facts in record["files"].items():
        if file_sha256(week / name) != facts["sha256"]:
            raise SystemExit(f"{week.name}/{name} differs from the sha256 in tables.json")


# --------------------------------------------------------------------------------------------------------------------
# Reading one week.

META_COLUMNS = ["session_id", "step", "start_ms", "end_ms", "model", "provider", "stream", "waited_for",
                "prompt_tokens", "cache_read_tokens", "role_tokens", "transition", "cause"]


def read_meta(week: pathlib.Path) -> pandas.DataFrame:
    """Every request of the week without its block ids, its role tokens flattened, in file order."""
    present = parquet.read_schema(week / REQUESTS_FILE).names
    missing = [name for name in META_COLUMNS + ["block_ids"] if name not in present]
    if missing:
        raise SystemExit(f"{week.name}/{REQUESTS_FILE} lacks the columns {missing}")
    table = parquet.read_table(week / REQUESTS_FILE, columns=META_COLUMNS).flatten()
    frame = table.to_pandas()
    for column in ["prompt_tokens", "cache_read_tokens", "start_ms", "end_ms"] + \
            [f"role_tokens.{field}" for field in ROLE_TOKEN_FIELDS]:
        frame[column] = frame[column].astype("Int64")  # nullable: null (not reported) stays distinct from 0
    return frame


def read_sessions(week: pathlib.Path) -> pandas.DataFrame:
    return parquet.read_table(week / SESSIONS_FILE, columns=["session_id", "harness", "parent_session_id"]).to_pandas()


def common_prefix_length(first: numpy.ndarray, second: numpy.ndarray) -> int:
    """How many leading values the two arrays share (as simulation/cheaper_reads.py computes it)."""
    length = min(len(first), len(second))
    differs = numpy.flatnonzero(first[:length] != second[:length])
    return int(differs[0]) if len(differs) else length


def scan_blocks(week: pathlib.Path, batch_rows: int) -> dict[str, numpy.ndarray]:
    """Per request in file order: its block count, the previous row's, and the leading blocks they share.

    The file is in session and step order (release/trace_lines.py sorts it so; this is checked), so a request's
    previous step is the row before it, possibly in the batch before. Only `session_id`, `step` and `block_ids`
    are read, a batch at a time.
    """
    source = parquet.ParquetFile(week / REQUESTS_FILE)
    n = source.metadata.num_rows
    n_blocks = numpy.zeros(n, dtype=numpy.int64)
    prev_blocks = numpy.full(n, -1, dtype=numpy.int64)
    shared = numpy.full(n, -1, dtype=numpy.int64)
    has_prev = numpy.zeros(n, dtype=bool)
    previous_key = None
    previous_ids = numpy.zeros(0, dtype=numpy.int64)
    row = 0
    for batch in source.iter_batches(batch_size=batch_rows, columns=["session_id", "step", "block_ids"]):
        lists = batch.column("block_ids")
        offsets = lists.offsets.to_numpy()
        values = lists.flatten().to_numpy(zero_copy_only=False)  # respects a sliced batch, unlike `.values`
        base = offsets[0]
        for position, (session_id, step) in enumerate(zip(batch.column("session_id").to_pylist(),
                                                          batch.column("step").to_pylist())):
            key = (session_id, step)
            if previous_key is not None and key <= previous_key:
                raise SystemExit(f"{week.name}/{REQUESTS_FILE} is not in (session_id, step) order at row {row}")
            ids = values[offsets[position] - base:offsets[position + 1] - base]
            n_blocks[row] = len(ids)
            if previous_key is not None and previous_key[0] == session_id and previous_key[1] == step - 1:
                has_prev[row] = True
                prev_blocks[row] = len(previous_ids)
                shared[row] = common_prefix_length(ids, previous_ids)
            previous_key, previous_ids = key, ids.copy()
            row += 1
    if row != n:
        raise SystemExit(f"{week.name}: read {row} rows of block ids, the file holds {n}")
    return {"n_blocks": n_blocks, "prev_n_blocks": prev_blocks, "shared_blocks": shared, "has_prev": has_prev}


# --------------------------------------------------------------------------------------------------------------------
# The comparison with step - 1, and the verdict.

def region_of(position: numpy.ndarray, tool_end: numpy.ndarray, system_end: numpy.ndarray) -> numpy.ndarray:
    """The region holding an o200k token position: tool definitions, then system, then the conversation."""
    return numpy.where(position < tool_end, "tool_definitions", numpy.where(position < system_end, "system", "conversation"))


def block_region(start: numpy.ndarray, tool_end: numpy.ndarray, system_end: numpy.ndarray) -> numpy.ndarray:
    """The region of a 16-token block: one region, or `a~b` when the block straddles a boundary, since the first
    differing token can be anywhere in the block."""
    first = region_of(start, tool_end, system_end)
    last = region_of(start + BLOCK_TOKENS - 1, tool_end, system_end)
    return numpy.where(first == last, first, numpy.char.add(numpy.char.add(first.astype(str), "~"), last.astype(str)))


def nullable(series: pandas.Series) -> numpy.ndarray:
    """A nullable integer column as floats, NaN for null."""
    return series.astype("Float64").to_numpy(dtype=float, na_value=numpy.nan)


def compare(meta: pandas.DataFrame, blocks: dict[str, numpy.ndarray], sessions: pandas.DataFrame) -> pandas.DataFrame:
    """Per request: the comparison with step - 1, the verdict, its detail, and the tokens lost."""
    frame = meta.copy()
    for name, values in blocks.items():
        frame[name] = values
    frame = frame.merge(sessions.rename(columns={"parent_session_id": "parent"}), on="session_id", how="left")
    frame["harness"] = frame["harness"].fillna("(no session row)")
    frame["is_subagent"] = frame["parent"].notna()

    role_columns = [f"role_tokens.{field}" for field in ROLE_TOKEN_FIELDS]
    o200k = frame[role_columns].astype("Float64").fillna(0).sum(axis=1).to_numpy(dtype=float)
    o200k = numpy.where(o200k > 0, o200k, BLOCK_TOKENS * frame.n_blocks.to_numpy())  # no role tokens: the blocks
    frame["o200k_tokens"] = o200k
    tools = frame["role_tokens.tool_definitions"].astype("Float64").fillna(0).to_numpy(dtype=float)
    system = frame["role_tokens.system"].astype("Float64").fillna(0).to_numpy(dtype=float)

    prev = frame.shift(1)  # the previous row; only used where has_prev says it is step - 1 of the same session
    has_prev = frame.has_prev.to_numpy()
    prompt = nullable(frame.prompt_tokens)
    read = nullable(frame.cache_read_tokens)
    prompt_prev = numpy.where(has_prev, nullable(prev.prompt_tokens.astype("Float64")), numpy.nan)
    o200k_prev = numpy.where(has_prev, numpy.roll(o200k, 1), numpy.nan)
    tools_prev = numpy.roll(tools, 1)
    system_prev = numpy.roll(system, 1)
    end_prev = numpy.where(has_prev, nullable(prev.end_ms.astype("Float64")), numpy.nan)

    shared = frame.shared_blocks.to_numpy().astype(float)
    n_blocks = frame.n_blocks.to_numpy().astype(float)
    prev_n_blocks = frame.prev_n_blocks.to_numpy().astype(float)
    intact = has_prev & (shared == prev_n_blocks)  # every full block of step - 1 is the start of this input
    truncated = has_prev & ~intact & (shared == n_blocks)  # this input is a strict prefix of step - 1's
    frame["prefix"] = numpy.where(~has_prev, None, numpy.where(intact, "intact", numpy.where(truncated, "truncated",
                                                                                              "diverged")))

    # Tokens. The block ids are o200k tokens; the provider counts are in the model's own tokenizer. The reusable
    # prefix is scaled by step - 1's ratio of provider to o200k tokens, as simulation/cheaper_reads.py scales a
    # step's shared share by its prompt tokens.
    frame["reusable_o200k"] = numpy.where(has_prev, BLOCK_TOKENS * shared, numpy.nan)
    frame["prev_tail_o200k"] = numpy.where(has_prev, o200k_prev - BLOCK_TOKENS * prev_n_blocks, numpy.nan)
    ratio_prev = numpy.where((prompt_prev > 0) & (o200k_prev > 0), prompt_prev / numpy.where(o200k_prev > 0, o200k_prev, 1),
                             numpy.nan)
    ratio_curr = numpy.where((prompt > 0) & (o200k > 0), prompt / numpy.where(o200k > 0, o200k, 1), numpy.nan)
    ratio = numpy.where(numpy.isfinite(ratio_prev), ratio_prev, numpy.where(numpy.isfinite(ratio_curr), ratio_curr, 1.0))
    frame["token_ratio"] = numpy.where(has_prev, ratio, numpy.nan)
    big_r = numpy.where(numpy.isfinite(prompt_prev), numpy.fmin(prompt_prev, prompt), prompt)  # ALGORITHM 1: R
    frame["R"] = numpy.where(has_prev, big_r, numpy.nan)
    reusable = numpy.round(BLOCK_TOKENS * shared * ratio)
    reusable = numpy.fmin(reusable, big_r)
    reusable = numpy.where(intact, big_r, reusable)  # intact: all of R (the partial last block is within the slack)
    frame["reusable"] = numpy.where(has_prev, reusable, numpy.nan)

    # Where the input diverged: the start of the first differing block, in o200k tokens, and its region in step - 1's
    # layout (the order of export/log_records/messages.py: tool definitions, top-level system prompt, messages).
    diverged = has_prev & ~intact
    start = BLOCK_TOKENS * numpy.where(diverged, shared, 0)
    region_prev = block_region(start, tools_prev, tools_prev + system_prev)
    region_curr = block_region(start, tools, tools + system)
    frame["divergence_o200k"] = numpy.where(diverged, start, numpy.nan)
    frame["divergence_share"] = numpy.where(diverged, start / numpy.where(o200k_prev > 0, o200k_prev, 1), numpy.nan)
    frame["region"] = numpy.where(diverged, region_prev, None)
    frame["region_curr_layout"] = numpy.where(diverged, region_curr, None)
    frame["tools_changed_size"] = has_prev & (tools != tools_prev)
    frame["system_changed_size"] = has_prev & (system != system_prev)

    gap_s = (nullable(frame.start_ms) - end_prev) / 1_000
    frame["gap_s"] = gap_s
    bucket = numpy.full(len(frame), None, dtype=object)
    for name, low, high in GAP_BUCKETS:
        bucket[(gap_s >= low) & (gap_s < high)] = name
    bucket[gap_s < 0] = "<5m"  # overlapping by less than the clock skew
    frame["gap_bucket"] = bucket
    in_flight = has_prev & ((nullable(frame.start_ms) + CLOCK_SKEW_MS) < end_prev)

    read_known = numpy.isfinite(read)
    r_ok = numpy.isfinite(big_r) & (big_r > 0)
    slack = numpy.minimum(READ_SLACK_TOKENS, numpy.floor(numpy.where(r_ok, big_r, 0) / 8))
    served_all = read_known & r_ok & (read > 0) & ((read >= big_r) | (read + slack > big_r))
    half_read = read_known & r_ok & (read >= big_r / 2)
    nothing_read = ~read_known | (read == 0)
    switched = has_prev & ((frame.model.to_numpy() != prev.model.to_numpy()) |
                           (frame.provider.to_numpy() != prev.provider.to_numpy()))
    step0 = frame.step.to_numpy() == 0

    conditions = [
        (step0, "first_write", "step 0"),
        (~has_prev, "unknown", "no_previous_step"),
        (switched, "switch", "model_or_provider"),
        (in_flight & ~half_read, "unknown", "sent_before_previous_finished"),
        (~numpy.isfinite(prompt), "unknown", "no_prompt_count"),
        (served_all, "hit", "served_all"),
        (intact & half_read, "hit", "half_read"),
        (r_ok & (big_r < MIN_CACHEABLE_TOKENS) & nothing_read, "below_min", "R_under_1024"),
        (diverged, "prompt_changed", None),
        (~read_known, "unknown", "no_cache_count"),
        (intact, "not_read", None),
    ]
    verdict = numpy.full(len(frame), "unknown", dtype=object)
    rule = numpy.full(len(frame), "unmatched", dtype=object)
    decided = numpy.zeros(len(frame), dtype=bool)
    for condition, name, detail in conditions:
        chosen = condition & ~decided
        verdict[chosen] = name
        rule[chosen] = detail if detail is not None else name
        decided |= chosen
    frame["verdict"] = verdict
    changed = verdict == "prompt_changed"
    rule[changed] = numpy.where(truncated[changed], "truncated", region_prev[changed])
    missing = verdict == "not_read"
    rule[missing] = numpy.char.add(numpy.where(read[missing] > 0, "partial_read ", "nothing_read "),
                                   bucket[missing].astype(str))
    frame["detail"] = rule

    # Tokens lost, in provider tokens. `lost_tokens` is the reusable prefix (the shared blocks, scaled) minus the
    # read; `invalidated_tokens` is what a change put beyond reach: R minus the reusable prefix. Their sum is
    # Prefixwise's bound on avoidable tokens, max(0, R - read) (ALGORITHM 5). Null reads give null losses.
    compared = has_prev & ~step0 & ~switched & numpy.isfinite(prompt)
    lost = numpy.where(compared & read_known, numpy.maximum(0, reusable - numpy.where(read_known, read, 0)), numpy.nan)
    invalid = numpy.where(compared & diverged,
                          numpy.maximum(0, big_r - numpy.maximum(reusable, numpy.where(read_known, read, 0))), numpy.nan)
    invalid = numpy.where(compared & ~diverged, 0.0, invalid)
    frame["lost_tokens"] = lost
    frame["invalidated_tokens"] = invalid
    frame["read_beyond_reusable"] = compared & diverged & read_known & (read > reusable + slack + 0.02 * reusable)
    frame["prefixwise_cause"], frame["legitimacy"] = prefixwise_names(frame)
    return frame


def prefixwise_names(frame: pandas.DataFrame) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Prefixwise's nearest cause name and legitimacy for each verdict (ALGORITHM 2.2, 2.3; classifier.py).

    The trace has one hash per block and no text, so Prefixwise's raw/canonical/masked distinction is not available:
    a tool change cannot be split into reserialisation (waste) and a catalog change (legitimate), nor a system change
    into a volatile value (waste) and an edit (not counted). Compaction is the one place the authors' label is used:
    Prefixwise needs a compaction event to call a history rewrite legitimate, and the authors' `compaction` cause is
    the trace's only record of one.
    """
    cause = numpy.full(len(frame), "unknown", dtype=object)
    legit = numpy.full(len(frame), "undetermined", dtype=object)
    verdict, detail = frame.verdict.to_numpy(), frame.detail.to_numpy()
    compaction = (frame.cause == "compaction").to_numpy()

    def put(mask, name, legitimacy):
        cause[mask], legit[mask] = name, legitimacy

    put(verdict == "first_write", "first_write", "legitimate")
    put(verdict == "switch", "model_switch", "legitimate")
    put(verdict == "hit", "none", "n/a")
    put(verdict == "below_min", "below_min_prefix", "legitimate")
    changed = verdict == "prompt_changed"
    put(changed & (detail == "tool_definitions"), "tool_reserialize|tool_catalog_change", "undetermined")
    put(changed & (detail == "system"), "volatile_system|system_edit", "undetermined")
    put(changed & (detail == "conversation"), "history_edit|tool_result_mutation", "avoidable")
    put(changed & (detail == "conversation") & compaction, "compaction_rewrite", "legitimate")
    put(changed & (detail == "truncated"), "history_edit (truncated)", "avoidable")
    put(changed & (detail == "truncated") & compaction, "compaction_rewrite", "legitimate")
    straddle = changed & numpy.array(["~" in str(value) for value in detail])
    put(straddle, "boundary (region ambiguous)", "undetermined")
    missed = verdict == "not_read"
    bucket = frame.gap_bucket.to_numpy()
    nothing = numpy.array([str(value).startswith("nothing_read") for value in detail])
    put(missed & (bucket == "<5m") & nothing, "routing_miss (inferred)", "undetermined")
    # The classifier names these routing_miss (inferred) since #171 (a part read within the TTL is not an
    # expiry). The run records of 2026-11-23 were made with the earlier rule, under this label.
    put(missed & (bucket == "<5m") & ~nothing, "unknown (partial read within 5m)", "undetermined")
    put(missed & (bucket == "5-60m"), "ttl_expired (5-min TTL) | routing_miss", "undetermined")
    put(missed & (bucket == ">60m"), "ttl_expired", "legitimate")
    return cause, legit


# --------------------------------------------------------------------------------------------------------------------
# Summaries.

def as_int(value) -> int:
    return int(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else 0


def verdict_table(frame: pandas.DataFrame) -> dict:
    """Per verdict: requests, prompt tokens, tokens lost, and their share of all prompt tokens."""
    total_prompt = float(frame.prompt_tokens.astype("Float64").fillna(0).sum())
    rows = {}
    for verdict in VERDICTS:
        part = frame[frame.verdict == verdict]
        lost = float(numpy.nansum(part.lost_tokens))
        invalid = float(numpy.nansum(part.invalidated_tokens))
        rows[verdict] = {
            "requests": int(len(part)),
            "share_of_requests": len(part) / len(frame) if len(frame) else 0.0,
            "prompt_tokens": as_int(part.prompt_tokens.astype("Float64").fillna(0).sum()),
            "read_tokens": as_int(part.cache_read_tokens.astype("Float64").fillna(0).sum()),
            "lost_tokens": int(lost),
            "invalidated_tokens": int(invalid),
            "lost_share_of_prompt_tokens": lost / total_prompt if total_prompt else 0.0,
            "lost_plus_invalidated_share_of_prompt_tokens": (lost + invalid) / total_prompt if total_prompt else 0.0,
            "requests_with_unmeasured_loss": int(part.lost_tokens.isna().sum()) if verdict not in ("first_write", "switch") else 0,
        }
    return {"total_requests": int(len(frame)), "total_prompt_tokens": int(total_prompt), "verdicts": rows}


def detail_table(frame: pandas.DataFrame, verdict: str) -> dict:
    part = frame[frame.verdict == verdict]
    out = {}
    for detail, group in part.groupby("detail", dropna=False):
        out[str(detail)] = {"requests": int(len(group)), "lost_tokens": int(numpy.nansum(group.lost_tokens)),
                            "invalidated_tokens": int(numpy.nansum(group.invalidated_tokens))}
    return dict(sorted(out.items(), key=lambda item: -item[1]["requests"]))


def breakdown(frame: pandas.DataFrame, column: str) -> dict:
    """Per value of `column`: requests, verdict counts, prompt tokens, and lost tokens and shares per verdict."""
    out = {}
    for value, group in frame.groupby(column, dropna=False):
        prompt_total = float(group.prompt_tokens.astype("Float64").fillna(0).sum())
        verdicts = {}
        for verdict, part in group.groupby("verdict"):
            lost = float(numpy.nansum(part.lost_tokens)) + float(numpy.nansum(part.invalidated_tokens))
            verdicts[verdict] = {"requests": int(len(part)), "lost_tokens": int(lost),
                                 "lost_share_of_prompt_tokens": lost / prompt_total if prompt_total else 0.0}
        reads = group.cache_read_tokens
        out[str(value)] = {
            "requests": int(len(group)),
            "prompt_tokens": int(prompt_total),
            "read_tokens": as_int(reads.astype("Float64").fillna(0).sum()),
            "requests_with_null_read": int(reads.isna().sum()),
            "requests_read_over_prompt": int(((reads.astype("Float64") > group.prompt_tokens.astype("Float64"))
                                              .fillna(False)).sum()),
            "verdicts": verdicts,
        }
    return dict(sorted(out.items(), key=lambda item: -item[1]["requests"]))


def top_loss_causes(frame: pandas.DataFrame, limit: int = 15) -> list[dict]:
    """The (verdict, detail, Prefixwise cause, authors' label) groups that lost the most tokens."""
    part = frame[frame.verdict.isin(["hit", "prompt_changed", "not_read", "below_min", "unknown"])].copy()
    part["total_lost"] = numpy.nan_to_num(part.lost_tokens) + numpy.nan_to_num(part.invalidated_tokens)
    part["authors"] = part.transition.fillna("-") + "/" + part.cause.fillna("-")
    total_prompt = float(frame.prompt_tokens.astype("Float64").fillna(0).sum())
    total_lost = float(part.total_lost.sum())
    rows = []
    grouped = part.groupby(["verdict", "detail", "prefixwise_cause", "legitimacy", "authors"], dropna=False)
    for key, group in grouped:
        lost = float(group.total_lost.sum())
        if lost <= 0:
            continue
        rows.append({"verdict": key[0], "detail": key[1], "prefixwise_cause": key[2], "legitimacy": key[3],
                     "authors_label": key[4], "requests": int(len(group)), "lost_tokens": int(lost),
                     "share_of_lost": lost / total_lost if total_lost else 0.0,
                     "share_of_prompt_tokens": lost / total_prompt if total_prompt else 0.0})
    rows.sort(key=lambda row: -row["lost_tokens"])
    return rows[:limit]


def crosstab(rows: pandas.Series, columns: pandas.Series) -> dict:
    table = pandas.crosstab(rows.fillna("-"), columns.fillna("-"))
    return {str(row): {str(column): int(table.loc[row, column]) for column in table.columns if table.loc[row, column]}
            for row in table.index}


EXPECTED_REGION = {"tool_definitions": ("tool_definitions",), "system_prompt": ("system",)}


def region_agrees(cause: str, region: str, region_curr: str) -> str:
    """Whether a mutation's region fits its cause: tool definitions and system prompt in their regions, every other
    cause in the conversation. A block straddling a boundary that includes the expected region is `ambiguous`."""
    expected = EXPECTED_REGION.get(cause, ("conversation",))
    if region in expected or (cause == "tool_definitions" and region_curr in expected):
        return "agrees"
    if "~" in str(region) and any(part in expected for part in str(region).split("~")):
        return "ambiguous"
    return "disagrees"


def validation(frame: pandas.DataFrame, samples: int = 20) -> dict:
    """The verdicts against the authors' transition and cause."""
    compared = frame[frame.has_prev].copy()
    out = {"note": "prefix: intact = every full block of step - 1 starts this input; truncated = this input is a "
                   "strict prefix of step - 1's; diverged = otherwise."}
    out["transition_by_prefix"] = crosstab(compared.transition, compared.prefix)
    out["transition_by_verdict"] = crosstab(compared.transition, compared.verdict)
    mutations = compared[compared.transition == "mutation"]
    out["cause_by_prefix"] = crosstab(mutations.cause, mutations.prefix)
    changed = mutations[mutations.prefix != "intact"].copy()
    out["cause_by_region"] = crosstab(changed.cause, changed.region)
    out["cause_by_region_curr_layout"] = crosstab(changed.cause, changed.region_curr_layout)
    changed["fit"] = [region_agrees(cause, region, region_curr) for cause, region, region_curr in
                      zip(changed.cause.fillna("-"), changed.region.fillna("-"), changed.region_curr_layout.fillna("-"))]
    out["cause_region_fit"] = crosstab(changed.cause, changed.fit)
    out["divergence_share_quartiles_by_cause"] = {
        str(cause): [float(value) for value in group.divergence_share.quantile([0.25, 0.5, 0.75]).round(4)]
        for cause, group in changed.groupby("cause")}

    def keys(rows: pandas.DataFrame) -> list[list]:
        return [[session, int(step)] for session, step in zip(rows.session_id.head(samples), rows.step.head(samples))]

    append_diverged = compared[(compared.transition == "append") & (compared.prefix != "intact")]
    mutation_intact = mutations[mutations.prefix == "intact"]
    switch_rows = compared[compared.transition == "switch"]
    verdict_switch = compared[compared.verdict == "switch"]
    tail = mutation_intact.prev_tail_o200k
    out["disagreements"] = {
        "append_with_diverged_prefix": {"requests": int(len(append_diverged)), "examples": keys(append_diverged)},
        "mutation_with_intact_prefix": {
            "requests": int(len(mutation_intact)),
            "by_cause": {str(k): int(v) for k, v in mutation_intact.cause.value_counts().items()},
            "with_partial_last_block_in_previous": int((tail > 0).sum()),
            "note": "a change inside step - 1's last partial block (< 16 o200k tokens, never hashed) is invisible "
                    "to the block ids",
            "examples": keys(mutation_intact)},
        "mutation_region_disagrees": {"requests": int((changed.fit == "disagrees").sum()),
                                      "examples": keys(changed[changed.fit == "disagrees"])},
        "switch_label_vs_model_provider": {
            "authors_switch_requests": int(len(switch_rows)),
            "verdict_switch_requests": int(len(verdict_switch)),
            "authors_switch_not_verdict_switch": int((switch_rows.verdict != "switch").sum()),
            "verdict_switch_not_authors_switch": int((verdict_switch.transition != "switch").sum())},
        "null_transition_after_step_0": int(compared.transition.isna().sum()),
        "mutation_without_cause": int(mutations.cause.isna().sum()),
    }
    out["read_beyond_reusable_by_cause"] = {str(k): int(v) for k, v in
                                            compared[compared.read_beyond_reusable].cause.fillna("-").value_counts().items()}
    return out


def diagnostics(frame: pandas.DataFrame) -> dict:
    expected_blocks = numpy.floor(frame.o200k_tokens.to_numpy() / BLOCK_TOKENS)
    return {
        "requests": int(len(frame)),
        "sessions": int(len(frame[["week", "session_id"]].drop_duplicates())),
        "subagent_sessions": int(len(frame.loc[frame.is_subagent, ["week", "session_id"]].drop_duplicates())),
        "requests_missing_previous_step": int(((frame.step > 0) & ~frame.has_prev).sum()),
        "requests_with_null_cache_read": int(frame.cache_read_tokens.isna().sum()),
        "requests_with_zero_cache_read": int((frame.cache_read_tokens == 0).sum()),
        "requests_with_null_prompt": int(frame.prompt_tokens.isna().sum()),
        "requests_read_over_prompt": int(((frame.cache_read_tokens.astype("Float64") >
                                           frame.prompt_tokens.astype("Float64")).fillna(False)).sum()),
        "blocks_not_floor_of_role_tokens_over_16": int((frame.n_blocks.to_numpy() != expected_blocks).sum()),
        "token_ratio_provider_over_o200k_quartiles": [float(v) for v in frame.token_ratio.dropna()
                                                      .quantile([0.25, 0.5, 0.75]).round(4)] if frame.token_ratio.notna().any() else [],
        "first_write_reading_at_least_half": int(((frame.verdict == "first_write") &
                                                  (frame.cache_read_tokens.astype("Float64") >=
                                                   frame.prompt_tokens.astype("Float64") / 2).fillna(False)).sum()),
        "prompt_changed_with_R_under_1024_region_loss": int(((frame.verdict == "prompt_changed") &
                                                            (frame.reusable < MIN_CACHEABLE_TOKENS)).sum()),
        "prompt_changed_nothing_read_gap_over_5m": int(((frame.verdict == "prompt_changed") &
                                                       (frame.cache_read_tokens == 0).fillna(False) &
                                                       frame.gap_bucket.isin(["5-60m", ">60m"])).sum()),
    }


def summarise(frame: pandas.DataFrame, weeks: list[str]) -> dict:
    return {
        "weeks": weeks,
        "rules": VERDICT_RULES,
        "diagnostics": diagnostics(frame),
        "overall": verdict_table(frame),
        "details": {verdict: detail_table(frame, verdict) for verdict in ("hit", "prompt_changed", "not_read", "unknown")},
        "prefixwise_cause": {str(k): int(v) for k, v in frame.prefixwise_cause.value_counts().items()},
        # Prefixwise (ALGORITHM 2.2) calls a change with nothing read after a gap over the TTL `ttl_expired`: the
        # entry was gone anyway. The TTLs of the anonymous providers are not known, so the gap is shown instead.
        "prompt_changed_region_by_gap_nothing_read": crosstab(
            frame.detail[(frame.verdict == "prompt_changed") & (frame.cache_read_tokens == 0).fillna(False)],
            frame.gap_bucket[(frame.verdict == "prompt_changed") & (frame.cache_read_tokens == 0).fillna(False)]),
        "lost_by_legitimacy": {str(k): int(numpy.nansum(g.lost_tokens) + numpy.nansum(g.invalidated_tokens))
                               for k, g in frame.groupby("legitimacy")},
        "top_loss_causes": top_loss_causes(frame),
        "validation": validation(frame),
        "by_provider": breakdown(frame, "provider"),
        "by_model": breakdown(frame, "model"),
        "by_harness": breakdown(frame, "harness"),
        "by_subagent": breakdown(frame, "is_subagent"),
    }


def print_summary(summary: dict, out=sys.stdout) -> None:
    def p(text=""):
        print(text, file=out)

    overall = summary["overall"]
    p(f"Prefixwise verdicts on {', '.join(summary['weeks'])}: {overall['total_requests']:,} requests, "
      f"{overall['total_prompt_tokens']:,} prompt tokens")
    d = summary["diagnostics"]
    p(f"  sessions {d['sessions']:,} (subagent {d['subagent_sessions']:,}); null cache_read {d['requests_with_null_cache_read']:,}; "
      f"read>prompt {d['requests_read_over_prompt']:,}; missing step-1 {d['requests_missing_previous_step']:,}; "
      f"blocks != floor(role_tokens/16) {d['blocks_not_floor_of_role_tokens_over_16']:,}")
    p()
    p(f"{'verdict':<16}{'requests':>11}{'share':>8}{'lost tok':>15}{'invalidated':>15}{'lost+inv %prompt':>18}")
    for verdict, row in overall["verdicts"].items():
        p(f"{verdict:<16}{row['requests']:>11,}{row['share_of_requests']:>8.1%}{row['lost_tokens']:>15,}"
          f"{row['invalidated_tokens']:>15,}{row['lost_plus_invalidated_share_of_prompt_tokens']:>18.2%}")
    for verdict in ("prompt_changed", "not_read", "unknown", "hit"):
        p(f"\n  {verdict} by detail:")
        for detail, row in list(summary["details"][verdict].items())[:10]:
            p(f"    {detail:<34}{row['requests']:>10,}  lost {row['lost_tokens']:>14,}  invalidated {row['invalidated_tokens']:>14,}")
    p("\nTop causes of lost tokens (lost + invalidated):")
    for row in summary["top_loss_causes"][:10]:
        p(f"  {row['share_of_lost']:>6.1%}  {row['lost_tokens']:>14,}  {row['verdict']}/{row['detail']}  "
          f"[{row['prefixwise_cause']}, {row['legitimacy']}]  authors {row['authors_label']}  n={row['requests']:,}")
    v = summary["validation"]
    p("\nValidation against the authors' labels")
    p("  transition x prefix: " + json.dumps(v["transition_by_prefix"]))
    p("  mutation cause x region (step-1 layout): " + json.dumps(v["cause_by_region"]))
    p("  mutation cause x fit: " + json.dumps(v["cause_region_fit"]))
    dis = v["disagreements"]
    p(f"  append with diverged prefix: {dis['append_with_diverged_prefix']['requests']:,}")
    p(f"  mutation with intact prefix: {dis['mutation_with_intact_prefix']['requests']:,} "
      f"{dis['mutation_with_intact_prefix']['by_cause']}")
    p(f"  mutation whose region does not fit its cause: {dis['mutation_region_disagrees']['requests']:,}")
    p(f"  switch label vs model/provider: {dis['switch_label_vs_model_provider']}")
    for column in ("by_provider", "by_harness", "by_model"):
        p(f"\n{column.replace('_', ' ')} (top 10 by requests):")
        for value, row in list(summary[column].items())[:10]:
            shares = ", ".join(f"{k} {r['lost_share_of_prompt_tokens']:.1%}" for k, r in row["verdicts"].items()
                               if r["lost_tokens"])
            p(f"  {value:<28}{row['requests']:>10,} req  null-read {row['requests_with_null_read']:>8,}  lost: {shares}")


# --------------------------------------------------------------------------------------------------------------------
# Running.

def analyse_week(week: pathlib.Path, batch_rows: int, scope: str) -> pandas.DataFrame:
    started = time.time()
    meta = read_meta(week)
    sessions = read_sessions(week)
    blocks = scan_blocks(week, batch_rows)
    frame = compare(meta, blocks, sessions)
    frame.insert(0, "week", week.name)
    if scope == "agentic":
        frame = frame[frame.harness != "eval:swe-bench"]
    elif scope == "swe-bench":
        frame = frame[frame.harness == "eval:swe-bench"]
    print(f"{week.name}: {len(frame):,} requests in {time.time() - started:.0f}s", file=sys.stderr)
    return frame


PER_REQUEST_COLUMNS = ["week", "session_id", "step", "harness", "is_subagent", "model", "provider", "transition", "cause",
                       "prompt_tokens", "cache_read_tokens", "n_blocks", "prev_n_blocks", "shared_blocks", "prefix",
                       "reusable_o200k", "reusable", "R", "token_ratio", "divergence_o200k", "divergence_share", "region",
                       "region_curr_layout", "gap_s", "gap_bucket", "verdict", "detail", "prefixwise_cause",
                       "legitimacy", "lost_tokens", "invalidated_tokens"]


def run(options: argparse.Namespace) -> dict:
    weeks = week_directories(options.paths, options.week)
    frames = []
    for week in weeks:
        if options.verify:
            verify_week(week)
        frames.append(analyse_week(week, options.batch_rows, options.scope))
    frame = pandas.concat(frames, ignore_index=True)
    summary = summarise(frame, [week.name for week in weeks])
    print_summary(summary)
    if options.out:
        pathlib.Path(options.out).write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print(f"\nJSON written to {options.out}", file=sys.stderr)
    if options.per_request:
        frame[PER_REQUEST_COLUMNS].to_parquet(options.per_request, index=False)
        print(f"per-request rows written to {options.per_request}", file=sys.stderr)
    return summary


# --------------------------------------------------------------------------------------------------------------------
# The self-test: synthetic weeks in exactly the release schema, with known verdicts.

class Synth:
    """Builds requests as token lists in the export's order (tool definitions, system, messages), hashes their
    16-token blocks as export/request_inputs/digests.py does, and numbers the hashes as week-local ids."""

    def __init__(self, seed: int = 7):
        self.random = numpy.random.default_rng(seed)
        self.sessions, self.requests = [], []
        self.ids: dict[bytes, int] = {}
        self.expected: dict[tuple[str, int], tuple[str, str | None]] = {}
        self.checks: list = []

    def tokens(self, count: int) -> list[int]:
        return [int(t) for t in self.random.integers(1, 200_000, size=count)]

    def block_ids(self, tokens: list[int]) -> list[int]:
        data = numpy.asarray(tokens, dtype="<u4").tobytes()
        width = BLOCK_TOKENS * 4
        previous, out = b"", []
        for start in range(0, len(data) - width + 1, width):
            previous = hashlib.blake2b(previous + data[start:start + width], digest_size=8, key=b"synthetic").digest()
            out.append(self.ids.setdefault(previous, len(self.ids) + 1))
        return out

    def session(self, session_id: str, harness: str, parent: str | None = None, spawned_by_step: int | None = None):
        self.sessions.append({"session_id": session_id, "user_id": "u" * 16, "harness": harness,
                              "parent_session_id": parent, "spawned_by_step": spawned_by_step,
                              "spawned_by_call_index": 0 if parent else None})

    def request(self, session_id: str, step: int, entries: list[tuple[str, list[int]]], *, start_s: float, end_s: float,
                read: int | None, model: str = "m1", provider: str = "A", ratio: float = 1.0,
                transition: str | None = None, cause: str | None = None, expect: tuple[str, str | None] | None = None):
        tokens = [t for _, part in entries for t in part]
        role_tokens = {field: 0 for field in ROLE_TOKEN_FIELDS}
        for role, part in entries:
            role_tokens[role] += len(part)
        prompt = int(round(len(tokens) * ratio))
        self.requests.append({
            "session_id": session_id, "step": step, "link_reason": "tool_call_id" if step else None,
            "start_ms": int(start_s * 1000), "end_ms": int(end_s * 1000), "ttft_ms": 100, "model": model,
            "provider": provider, "stream": True, "step_trigger": "tool" if step else "user",
            "waited_for": "tool" if step else None, "prompt_tokens": prompt, "completion_tokens": 50,
            "cache_read_tokens": read, "role_tokens": role_tokens,
            "n_messages": sum(1 for role, _ in entries if role != "tool_definitions"),
            "tool_definition_names": ["read_file"], "finish_reason": "tool_calls",
            "transition": transition if step else None, "cause": cause, "block_ids": self.block_ids(tokens)})
        if expect is not None:
            self.expected[(session_id, step)] = expect
        return prompt

    def write(self, week: pathlib.Path) -> None:
        week.mkdir(parents=True, exist_ok=True)
        requests = sorted(self.requests, key=lambda row: (row["session_id"], row["step"]))
        sessions = sorted(self.sessions, key=lambda row: row["session_id"])
        tables = {SESSIONS_FILE: pyarrow.Table.from_pylist(sessions, schema=SESSIONS_SCHEMA),
                  REQUESTS_FILE: pyarrow.Table.from_pylist(requests, schema=REQUESTS_SCHEMA),
                  TOOL_CALLS_FILE: pyarrow.Table.from_pylist([], schema=TOOL_CALLS_SCHEMA)}
        (week / MANIFEST_FILE).write_text(json.dumps({"format_version": FORMAT_VERSION, "files": {}}) + "\n")
        record = {"manifest_sha256": file_sha256(week / MANIFEST_FILE), "files": {}}
        for name, table in tables.items():
            parquet.write_table(table, week / name, compression="zstd", row_group_size=4)  # tiny row groups: batches
            record["files"][name] = {"bytes": (week / name).stat().st_size, "sha256": file_sha256(week / name),
                                     "rows": table.num_rows}
        (week / TABLES_FILE).write_text(json.dumps(record, indent=2) + "\n")


def floor128(value: float) -> int:
    return int(value // 128 * 128)


def build_synthetic(synth: Synth) -> None:
    t = synth.tokens
    # Session "a": a top-level claude-code session on provider A, ratio 1 (provider tokens = o200k tokens).
    tools, system = t(2000), t(1504)  # 2000 = 125 blocks, 3504 = 219 blocks: the region boundaries fall on blocks
    convo = [("user", t(300))]

    def entries(tool_part=None, system_part=None, messages=None):
        return [("tool_definitions", tool_part or tools), ("system", system_part or system)] + (messages or convo)

    clock = 1_000_000.0
    p = synth.request("a", 0, entries(), start_s=clock, end_s=clock + 5, read=0, expect=("first_write", "step 0"))
    convo = convo + [("assistant", t(200)), ("tool", t(800))]
    clock += 10
    p = synth.request("a", 1, entries(), start_s=clock, end_s=clock + 5, read=floor128(p), transition="append",
                      expect=("hit", "served_all"))
    convo = convo + [("assistant", t(100)), ("tool", t(400))]
    clock += 7  # 2 s after step 1 ended
    p_prev = p
    p = synth.request("a", 2, entries(), start_s=clock, end_s=clock + 5, read=0, transition="append",
                      expect=("not_read", "nothing_read <5m"))
    synth.checks.append((("a", 2), "lost_tokens", float(p_prev)))  # intact: R = min(prev, curr) is all lost
    convo = convo + [("assistant", t(100)), ("user", t(50))]
    clock += 5 + 600  # 10 min idle
    p = synth.request("a", 3, entries(), start_s=clock, end_s=clock + 5, read=0, transition="append",
                      expect=("not_read", "nothing_read 5-60m"))
    convo = convo + [("assistant", t(100)), ("user", t(60))]
    clock += 5 + 7200  # 2 h idle
    p = synth.request("a", 4, entries(), start_s=clock, end_s=clock + 5, read=0, transition="append",
                      expect=("not_read", "nothing_read >60m"))
    # Step 5: the system prompt edited 100 tokens in (o200k position 2100, block 131).
    system2 = list(system)
    system2[100] += 1
    clock += 10
    p = synth.request("a", 5, entries(system_part=system2), start_s=clock, end_s=clock + 5, read=floor128(2096),
                      transition="mutation", cause="system_prompt", expect=("prompt_changed", "system"))
    synth.checks.append((("a", 5), "divergence_o200k", 2096.0))
    synth.checks.append((("a", 5), "reusable", 2096.0))
    synth.checks.append((("a", 5), "lost_tokens", 2096.0 - floor128(2096)))
    system = system2
    # Step 6: a tool definition edited at position 500 (block 31).
    tools2 = list(tools)
    tools2[500] += 1
    clock += 10
    p = synth.request("a", 6, entries(tool_part=tools2), start_s=clock, end_s=clock + 5, read=0,
                      transition="mutation", cause="tool_definitions", expect=("prompt_changed", "tool_definitions"))
    synth.checks.append((("a", 6), "divergence_o200k", 496.0))
    tools = tools2
    # Step 7: an earlier tool result re-rendered (the first tool message, conversation offset 500 + 10).
    clock += 10
    first_tool = convo[2][1]
    convo = convo[:2] + [("tool", first_tool[:10] + [first_tool[10] + 1] + first_tool[11:])] + convo[3:]
    p = synth.request("a", 7, entries(), start_s=clock, end_s=clock + 5, read=floor128(3504 + 500), transition="mutation",
                      cause="tool_result", expect=("prompt_changed", "conversation"))
    synth.checks.append((("a", 7), "divergence_o200k", 3504.0 + 496))
    # Step 8: compaction: the conversation replaced by a summary.
    convo = [("user", t(900))]
    clock += 10
    p = synth.request("a", 8, entries(), start_s=clock, end_s=clock + 5, read=3456, transition="mutation",
                      cause="compaction", expect=("prompt_changed", "conversation"))
    # Step 9: a model switch.
    convo = convo + [("assistant", t(100))]
    clock += 10
    p = synth.request("a", 9, entries(), start_s=clock, end_s=clock + 5, read=0, model="m2", transition="switch",
                      expect=("switch", "model_or_provider"))
    # Step 10: back on m2 (same model as step 9), appended, 60% read: the half-read rule makes it a hit.
    convo = convo + [("tool", t(3000))]
    clock += 10
    p_prev = p
    p = synth.request("a", 10, entries(), start_s=clock, end_s=clock + 5, read=int(0.6 * p_prev), model="m2",
                      transition="append", expect=("hit", "half_read"))
    # Step 11: labelled append, but the input diverges (an injected disagreement).
    convo2 = list(convo)
    convo2[0] = ("user", convo[0][1][:5] + [convo[0][1][5] + 1] + convo[0][1][6:])
    convo = convo2 + [("assistant", t(40))]
    clock += 10
    p = synth.request("a", 11, entries(), start_s=clock, end_s=clock + 5, read=0, model="m2", transition="append",
                      expect=("prompt_changed", "conversation"))
    # Step 12: labelled mutation, the change inside step 11's last partial block: invisible to the blocks.
    total_prev = sum(len(part) for _, part in entries())
    assert total_prev % BLOCK_TOKENS != 0, total_prev
    last_role, last_part = convo[-1]
    convo = convo[:-1] + [(last_role, last_part[:-1] + [last_part[-1] + 1]), ("user", t(30))]
    clock += 10
    p_prev = p
    p = synth.request("a", 12, entries(), start_s=clock, end_s=clock + 5, read=floor128(p_prev), model="m2",
                      transition="mutation", cause="other", expect=("hit", "served_all"))
    # Step 13: a change the provider never saw (read covers R although the blocks diverge): a hit.
    convo2 = list(convo)
    convo2[1] = (convo[1][0], [convo[1][1][0] + 1] + convo[1][1][1:])
    convo = convo2
    clock += 10
    p_prev = p
    p = synth.request("a", 13, entries(), start_s=clock, end_s=clock + 5, read=p_prev, model="m2",
                      transition="mutation", cause="other", expect=("hit", "served_all"))

    # Session "b": a subagent of "a", short prompts: below the minimum.
    synth.session("a", "claude-code")
    synth.session("b", "claude-code", parent="a", spawned_by_step=3)
    small = [("system", t(300)), ("user", t(100))]
    clock = 2_000_000.0
    synth.request("b", 0, small, start_s=clock, end_s=clock + 2, read=0, expect=("first_write", "step 0"))
    small = small + [("assistant", t(100)), ("tool", t(300))]
    synth.request("b", 1, small, start_s=clock + 3, end_s=clock + 5, read=0, transition="append",
                  expect=("below_min", "R_under_1024"))
    small = small + [("assistant", t(100)), ("tool", t(1500))]
    synth.request("b", 2, small, start_s=clock + 6, end_s=clock + 8, read=0, transition="append",
                  expect=("below_min", "R_under_1024"))
    small = small + [("assistant", t(100)), ("tool", t(500))]
    synth.request("b", 3, small, start_s=clock + 9, end_s=clock + 10, read=0, transition="append",
                  expect=("not_read", "nothing_read <5m"))

    # Session "c": provider B reports no cache count (null), opencode.
    synth.session("c", "opencode")
    big = [("tool_definitions", t(1200)), ("system", t(800)), ("user", t(400))]
    clock = 3_000_000.0
    synth.request("c", 0, big, start_s=clock, end_s=clock + 2, read=None, provider="B", expect=("first_write", "step 0"))
    big = big + [("assistant", t(200)), ("tool", t(900))]
    synth.request("c", 1, big, start_s=clock + 3, end_s=clock + 5, read=None, provider="B", transition="append",
                  expect=("unknown", "no_cache_count"))
    big = big[:3] + big[4:]  # the assistant turn dropped: the tool result moves up
    synth.request("c", 2, big, start_s=clock + 6, end_s=clock + 8, read=None, provider="B", transition="mutation",
                  cause="dropped_turns", expect=("prompt_changed", "conversation"))
    synth.checks.append((("c", 2), "lost_tokens", None))  # no count: the loss is not measured
    big = big[:3]  # a strict prefix of step 2: truncated
    synth.request("c", 3, big, start_s=clock + 9, end_s=clock + 10, read=None, provider="B", transition="mutation",
                  cause="other", expect=("prompt_changed", "truncated"))

    # Session "d": provider C, whose tokenizer counts 1.25 tokens per o200k token; the tool definitions end inside a
    # block (1000 is not a multiple of 16).
    synth.session("d", "codex")
    d_tools, d_system = t(1000), t(1000)
    d_convo = [("user", t(600))]
    clock = 4_000_000.0
    p = synth.request("d", 0, [("tool_definitions", d_tools), ("system", d_system)] + d_convo, start_s=clock,
                      end_s=clock + 2, read=0, provider="C", ratio=1.25, expect=("first_write", "step 0"))
    d_convo = d_convo + [("assistant", t(300)), ("tool", t(2000))]
    p = synth.request("d", 1, [("tool_definitions", d_tools), ("system", d_system)] + d_convo, start_s=clock + 3,
                      end_s=clock + 5, read=floor128(p), provider="C", ratio=1.25, transition="append",
                      expect=("hit", "served_all"))
    # A tool result edited at conversation offset 900 + 10 (o200k 2910, block 181, start 2896).
    tool_part = d_convo[2][1]
    d_convo = d_convo[:2] + [("tool", tool_part[:10] + [tool_part[10] + 1] + tool_part[11:])]
    d_convo = d_convo + [("assistant", t(100))]
    p = synth.request("d", 2, [("tool_definitions", d_tools), ("system", d_system)] + d_convo, start_s=clock + 6,
                      end_s=clock + 8, read=floor128(2896 * 1.25), provider="C", ratio=1.25, transition="mutation",
                      cause="tool_result", expect=("prompt_changed", "conversation"))
    synth.checks.append((("d", 2), "reusable", round(2896 * 1.25)))
    # The system prompt edited at its first token (o200k 1000): block 62 spans 992-1007, across the boundary.
    d_system = [d_system[0] + 1] + d_system[1:]
    synth.request("d", 3, [("tool_definitions", d_tools), ("system", d_system)] + d_convo, start_s=clock + 9,
                  end_s=clock + 11, read=0, provider="C", ratio=1.25, transition="mutation", cause="system_prompt",
                  expect=("prompt_changed", "tool_definitions~system"))
    # A provider switch with the same model.
    synth.request("d", 4, [("tool_definitions", d_tools), ("system", d_system)] + d_convo, start_s=clock + 12,
                  end_s=clock + 14, read=0, provider="A", ratio=1.25, transition="switch",
                  expect=("switch", "model_or_provider"))

    # Session "e": sent 5 s before its previous step ended, nothing read: unjudged (ALGORITHM 2.4).
    synth.session("e", "claude-code")
    e = [("tool_definitions", t(2000)), ("user", t(500))]
    clock = 5_000_000.0
    synth.request("e", 0, e, start_s=clock, end_s=clock + 20, read=0, expect=("first_write", "step 0"))
    e = e + [("assistant", t(50)), ("user", t(50))]
    synth.request("e", 1, e, start_s=clock + 15, end_s=clock + 30, read=0, transition="append",
                  expect=("unknown", "sent_before_previous_finished"))


def selftest(trace_repo: pathlib.Path | None, workdir: pathlib.Path | None) -> int:
    failures = []
    with tempfile.TemporaryDirectory(prefix="prefixwise_trace_", dir=workdir) as directory:
        release = pathlib.Path(directory) / "release"
        week = release / "2026-08-30_2026-09-05"
        synth = Synth()
        build_synthetic(synth)
        synth.write(week)
        verify_week(week)

        if trace_repo is not None:
            failures += check_against_trace_repo(trace_repo, release, week)

        options = argparse.Namespace(paths=[week], week=None, out=str(pathlib.Path(directory) / "out.json"),
                                     per_request=None, batch_rows=3, verify=True, scope="all")
        frame = analyse_week(week, batch_rows=3, scope="all")
        by_key = frame.set_index(["session_id", "step"])
        for key, (verdict, detail) in sorted(synth.expected.items()):
            got = (by_key.loc[key, "verdict"], by_key.loc[key, "detail"])
            status = "ok " if got == (verdict, detail) else "BAD"
            if got != (verdict, detail):
                failures.append(f"{key}: expected {verdict}/{detail}, got {got[0]}/{got[1]}")
            print(f"  {status} {key[0]}:{key[1]:<3} expected {verdict}/{detail:<32} got {got[0]}/{got[1]}")
        for key, column, expected in synth.checks:
            got = by_key.loc[key, column]
            good = (expected is None and pandas.isna(got)) or (expected is not None and abs(float(got) - expected) < 1e-6)
            print(f"  {'ok ' if good else 'BAD'} {key[0]}:{key[1]:<3} {column} expected {expected} got {got}")
            if not good:
                failures.append(f"{key} {column}: expected {expected}, got {got}")
        missing = set(zip(frame.session_id, frame.step)) - set(synth.expected)
        if missing:
            failures.append(f"requests without an expectation: {sorted(missing)}")

        summary = run(options)
        dis = summary["validation"]["disagreements"]
        expected_counts = {"append_with_diverged_prefix": 1, "mutation_with_intact_prefix": 1}
        for name, count in expected_counts.items():
            got = dis[name]["requests"]
            print(f"  {'ok ' if got == count else 'BAD'} validation {name}: expected {count}, got {got}")
            if got != count:
                failures.append(f"validation {name}: expected {count}, got {got}")
        fit = summary["validation"]["cause_region_fit"]
        expected_fit = {"compaction": {"agrees": 1}, "dropped_turns": {"agrees": 1}, "other": {"agrees": 2},
                        "system_prompt": {"agrees": 1, "ambiguous": 1}, "tool_definitions": {"agrees": 1},
                        "tool_result": {"agrees": 2}}
        print(f"  {'ok ' if fit == expected_fit else 'BAD'} cause x region fit: {fit}")
        if fit != expected_fit:
            failures.append(f"cause x region fit: expected {expected_fit}, got {fit}")
        json.loads(pathlib.Path(options.out).read_text())
    print(f"\nself-test: {'PASSED' if not failures else 'FAILED'} "
          f"({len(synth.expected)} verdicts, {len(synth.checks)} token checks)")
    for failure in failures:
        print("  " + failure)
    return 1 if failures else 0


def check_against_trace_repo(trace_repo: pathlib.Path, release: pathlib.Path, week: pathlib.Path) -> list[str]:
    """With a clone of the trace repository: the schema copy equals release/schema.py, the trace's own reader loads
    and checks the synthetic week, and its cheaper-reads model counts the same shared blocks as this script."""
    failures = []
    sys.path.insert(0, str(trace_repo))
    try:
        from release import reader as trace_reader, schema as trace_schema
        from simulation import cheaper_reads
    except Exception as error:  # the clone may lack a dependency
        return [f"cannot import the trace repository's code from {trace_repo}: {error!r}"]
    for name, mine in (("SESSIONS_SCHEMA", SESSIONS_SCHEMA), ("REQUESTS_SCHEMA", REQUESTS_SCHEMA),
                       ("TOOL_CALLS_SCHEMA", TOOL_CALLS_SCHEMA)):
        same = getattr(trace_schema, name).equals(mine)
        print(f"  {'ok ' if same else 'BAD'} {name} equals release/schema.py")
        if not same:
            failures.append(f"{name} differs from release/schema.py")
    for name in ("BLOCK_TOKENS", "FORMAT_VERSION", "TRANSITIONS", "CAUSES", "ROLE_TOKEN_FIELDS"):
        if getattr(trace_schema, name) != globals()[name]:
            failures.append(f"{name} differs from release/schema.py")
    loaded = trace_reader.load(release, [week.name[:10]])
    trace_reader.check_week(week)
    print(f"  ok  release/reader.py loads and checks the synthetic week: {len(loaded.requests)} requests, "
          f"{len(loaded.sessions)} sessions")
    theirs = cheaper_reads.shared_prefixes(week).set_index(["session_id", "step"])
    blocks = scan_blocks(week, 3)
    keys = parquet.read_table(week / REQUESTS_FILE, columns=["session_id", "step"]).to_pandas()
    mine_shared = numpy.where(blocks["has_prev"], blocks["shared_blocks"], 0)
    same = (theirs.loc[list(zip(keys.session_id, keys.step)), "shared_blocks"].to_numpy() == mine_shared).all()
    print(f"  {'ok ' if same else 'BAD'} shared blocks equal simulation/cheaper_reads.shared_prefixes")
    if not same:
        failures.append("shared blocks differ from simulation/cheaper_reads.shared_prefixes")
    return failures


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=VERDICT_RULES)
    parser.add_argument("paths", nargs="*", type=pathlib.Path,
                        help="week folders (holding requests.parquet and sessions.parquet), or a release folder of them")
    parser.add_argument("--week", action="append", metavar="SUNDAY", help="only the week starting YYYY-MM-DD; repeatable")
    parser.add_argument("--out", help="write the summary as JSON here")
    parser.add_argument("--per-request", metavar="FILE.parquet", help="also write one row per request")
    parser.add_argument("--batch-rows", type=int, default=2_000, help="requests read per batch of block ids (memory)")
    parser.add_argument("--scope", choices=("all", "agentic", "swe-bench"), default="all",
                        help="all sessions (default); all but the SWE-bench scaffold (the paper's Agentic scope, "
                             "except that subagent sessions are kept here); or only the scaffold")
    parser.add_argument("--verify", action="store_true", help="check each table's sha256 against tables.json first")
    parser.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    parser.add_argument("--trace-repo", type=pathlib.Path,
                        help="with --selftest: a clone of the trace repository, to check the schema copy, its reader "
                             "and its shared-prefix count against this script")
    parser.add_argument("--workdir", type=pathlib.Path,
                        help="with --selftest: where to write the synthetic week (a temporary folder, removed after); "
                             "default: the system's temporary folder")
    options = parser.parse_args(arguments)
    if options.selftest:
        return selftest(options.trace_repo, options.workdir)
    if not options.paths:
        parser.error("give at least one week folder, or --selftest")
    run(options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
