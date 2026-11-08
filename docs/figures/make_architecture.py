"""Draw the architecture flowcharts as static SVG, one light and one dark file each.

    python docs/figures/make_architecture.py

architecture-*.svg: how a request flows through Prefixwise (docs/ARCHITECTURE.md, "The shape
of it" and "A request through the proxy"). decision-*.svg: how the cause is decided, in the
order classifier.py checks (docs/ARCHITECTURE.md, "How a cause is decided"). Colors: the same
tokens as make_figures.py.
"""

from __future__ import annotations

from pathlib import Path

from make_figures import FONT, THEMES, esc

OUT = Path(__file__).resolve().parent
ARROW_ID = "a"

# Roles on top of the shared palette, per mode: neutral decisions, blue verdicts and hero
# nodes, emerald for a cache hit (the one verdict that is not a miss).
ROLES = {
    "light": {"slate": "#94a3b8", "blue": "#2563eb", "verdict_fill": "#f8fafc", "hero_fill": "#f5f8ff",
              "hit": "#059669", "hit_fill": "#ecfdf5"},
    "dark": {"slate": "#64748b", "blue": "#3b82f6", "verdict_fill": "#1f2023", "hero_fill": "#18202f",
             "hit": "#10b981", "hit_fill": "#13261f"},
}


def roles(t: dict) -> dict:
    return ROLES["dark" if t["surface"] == THEMES["dark"]["surface"] else "light"]


def svg(w: int, h: int, t: dict, body: list[str], title: str) -> str:
    marker = (f'<defs><marker id="{ARROW_ID}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
              f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
              f'fill="{t["ink2"]}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{esc(title)}" font-family="{FONT}">'
            f'<title>{esc(title)}</title>{marker}<rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>'
            + "".join(body) + "</svg>\n")


def lines(x, y, rows, fill, size=13, weight=400, anchor="middle", gap=17):
    """Centred multi-line text; the first row is at y."""
    out = []
    for i, row in enumerate(rows):
        w = weight[i] if isinstance(weight, (list, tuple)) else weight
        out.append(f'<text x="{x:.1f}" y="{y + i * gap:.1f}" fill="{fill}" font-size="{size}" '
                   f'text-anchor="{anchor}" font-weight="{w}">{esc(row)}</text>')
    return "".join(out)


def box(t, x, y, w, h, rows, *, accent=None, fill=None, bold_first=True, size=13, dashed=False):
    stroke = accent or t["axis"]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    body = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill or t["surface"]}" '
            f'stroke="{stroke}" stroke-width="{1.6 if accent else 1.2}"{dash}/>')
    gap = 17
    top = y + h / 2 - (len(rows) - 1) * gap / 2 + 4.5
    weights = [600 if (bold_first and i == 0) else 400 for i in range(len(rows))]
    colors = [t["ink"]] + [t["ink2"]] * (len(rows) - 1)
    for i, row in enumerate(rows):
        body += lines(x + w / 2, top + i * gap, [row], colors[i], size=size, weight=weights[i])
    return body


def diamond(t, cx, cy, w, h, rows, stroke=None):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    gap = 16
    top = cy - (len(rows) - 1) * gap / 2 + 4.5
    return (f'<polygon points="{pts}" fill="{t["surface"]}" stroke="{stroke or t["s1"]}" stroke-width="1.6"/>'
            + lines(cx, top, rows, t["ink"], size=12.5, gap=gap))


def cylinder(t, x, y, w, h, rows, stroke=None, fill=None):
    e = 9
    stroke, fill = stroke or t["s1"], fill or t["surface"]
    body = (f'<path d="M{x},{y + e} V{y + h - e} A{w / 2},{e} 0 0 0 {x + w},{y + h - e} V{y + e}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
            f'<ellipse cx="{x + w / 2}" cy="{y + e}" rx="{w / 2}" ry="{e}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.6"/>')
    top = y + h / 2 - (len(rows) - 1) * 8.5 + 9
    return body + lines(x + w / 2, top, rows, t["ink"], weight=[600] + [400] * (len(rows) - 1))


def arrow(t, pts, label=None, lx=None, ly=None, anchor="middle", dashed=False, head=True):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    end = f' marker-end="url(#{ARROW_ID})"' if head else ""
    out = f'<path d="{d}" fill="none" stroke="{t["ink2"]}" stroke-width="1.4"{dash}{end}/>'
    if label:
        out += lines(lx, ly, [label], t["muted"], size=11.5, anchor=anchor)
    return out


def group(t, x, y, w, h, title, right=False, center=False):
    tx, anchor = (x + w / 2, "middle") if center else ((x + w - 14, "end") if right else (x + 14, "start"))
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" stroke="{t["grid"]}" '
            f'stroke-width="1.5"/>' + lines(tx, y + 22, [title], t["muted"], size=12, weight=600,
                                            anchor=anchor))


# -- figure 1: how a request flows -----------------------------------------------------------

def masked(t, x, y, text, size=11.5, anchor="middle"):
    """A label on a plain background, so a line or border behind it never strikes it through."""
    w = len(text) * size * 0.56 + 10
    left = {"middle": x - w / 2, "start": x - 5, "end": x - w + 5}[anchor]
    return (f'<rect x="{left:.1f}" y="{y - size:.1f}" width="{w:.1f}" height="{size + 6:.1f}" rx="4" '
            f'fill="{t["surface"]}"/>' + lines(x, y, [text], t["muted"], size=size, anchor=anchor))


def architecture(t: dict) -> str:
    r = roles(t)
    W, H = 940, 854
    dy = 54  # room between step 3 and step 4 for the turn record, the boundary between the two halves
    b = [lines(W / 2, 34, ["How a request flows through Prefixwise"], t["ink"], size=17, weight=600),
         lines(W / 2, 56, ["An edge where the request is, a collector where the ledger is; one process or many; "
                           "the prompt itself is never stored"], t["muted"], size=12.5)]
    # ways in
    b.append(group(t, 20, 74, W - 40, 112, "WAYS IN"))
    ins = [("Your agent", "through the proxy", "prefixwise serve / run"),
           ("Your own code", "prefixwise.wrap(client)", "Anthropic / OpenAI SDKs"),
           ("LiteLLM", "PrefixwiseCallback", "SDK or proxy server"),
           ("Agent logs", "prefixwise import", "Claude Code, Codex")]
    for x, rows in zip([40, 262, 484, 706], ins):
        b.append(box(t, x, 104, 194, 66, list(rows)))
    # the edge: steps 1 to 3, wherever the request is
    b.append(group(t, 20, 204, W - 40, 272, "EDGE: where the request is", right=True))
    cx, cw = 250, 440
    steps = [(234, ["1  Read the request", "session, agent, task; diagnostic fields added"]),
             (318, ["2  Forward it (proxy only)", "the client's request and auth, as sent"]),
             (402, ["3  Fingerprint the prompt", "blocks in cache order, hashed: never the text"]),
             (486 + dy, ["4  Name one cause", "against this conversation's previous turn; sub-reason; price"])]
    for y, rows in steps:
        hero = y == 486 + dy
        b.append(box(t, cx, y, cw, 58, rows, accent=r["blue"] if hero else None,
                     fill=r["hero_fill"] if hero else None))
    for (y1, _), (y2, _) in zip(steps, steps[1:]):
        b.append(arrow(t, [(cx + cw / 2, y1 + 58), (cx + cw / 2, y2 - 2)]))
    # the turn record: what crosses from the edge to the collector
    b.append(lines(cx + cw / 2 + 12, 486, ["turn record: block hashes, positions, counts, usage; never the text",
                                            "one process: a call. A fleet: HTTP to prefixwise collect, or a Redis",
                                            "Stream partitioned by session, each partition read by one collector"],
                   t["muted"], size=11.5, anchor="start", gap=14))
    # the collector: step 4 and the ledger, one per fleet (or per partition of the log)
    b.append(group(t, 20, 520, W - 40, 216, ""))
    b.append(lines(66, 536, ["COLLECTOR: one per fleet, or one per set of log partitions"],
                   t["muted"], size=12, weight=600, anchor="start"))
    # into step 1: the proxy, and the SDK wrappers (their call already happened: not forwarded)
    b.append(arrow(t, [(150, 170), (150, 252), (cx - 2, 252)]))
    b.append(arrow(t, [(359, 170), (359, 194), (222, 194), (222, 276), (cx - 2, 276)]))
    b.append(arrow(t, [(581, 170), (581, 194), (359, 194)], head=False))  # joins the wrap line
    b.append(lines(214, 300, ["wrap / LiteLLM:", "already sent, so", "recorded, not forwarded"],
                   t["muted"], size=11.5, anchor="end", gap=15))
    # back to the agent: after step 4, so the verdict can ride in the headers (in one process)
    b.append(arrow(t, [(cx - 2, 528 + dy), (46, 528 + dy), (46, 172)], dashed=True))
    b.append(lines(70, 392, ["response, unchanged,", "with the verdict in", "X-Prefixwise-* headers", "(one process;",
                             "an edge says: queued)"], t["muted"], size=11.5, anchor="start", gap=15))
    # the provider or engine (spaced so the labels clear the boxes)
    px = cx + cw + 54
    b.append(box(t, px, 315, 160, 64, ["Provider or engine", "Anthropic, OpenAI,", "vLLM, SGLang"]))
    b.append(arrow(t, [(cx + cw, 334), (px - 2, 334)]))
    b.append(lines((cx + cw + px) / 2, 327, ["request"], t["muted"], size=11.5))
    b.append(arrow(t, [(px - 2, 362), (cx + cw + 2, 362)]))
    b.append(lines((cx + cw + px) / 2, 378, ["response"], t["muted"], size=11.5))
    b.append(box(t, px, 398, 160, 62, ["KV-cache events", "vLLM / SGLang, optional:", "evicted? which replica?"],
                 dashed=True, size=12))
    b.append(arrow(t, [(px - 2, 431), (cx + cw + 2, 431)]))
    # agent logs: their own rules (the logs hold usage and events, never the prompt)
    b.append(box(t, 690, 584 + dy, 200, 64, ["Log rules, own ledger file", "what the log shows changed:",
                                             "model, compaction, TTL, ..."], size=12))
    b.append(arrow(t, [(803, 170), (803, 194), (912, 194), (912, 616 + dy), (892, 616 + dy)]))
    b.append(arrow(t, [(688, 616 + dy), (562, 616 + dy)]))
    # ledger
    b.append(cylinder(t, 380, 584 + dy, 180, 64, ["Ledger", "SQLite, one row per request"], stroke=r["blue"],
                      fill=r["hero_fill"]))
    b.append(arrow(t, [(cx + cw / 2, 544 + dy), (cx + cw / 2, 582 + dy)]))
    # ways out: live per request, read from the ledger, and copies of every row
    ow, gapx, oy = 140, 10, 718 + dy
    x0 = (W - (6 * ow + 5 * gapx)) / 2
    xs = [x0 + i * (ow + gapx) for i in range(6)]
    outs = [["OpenTelemetry", "spans"], ["Prometheus", "/metrics"], ["prefixwise", "report"], ["HTTP API", "/api/v1"],
            ["prefixwise", "export"], ["Sinks: JSONL,", "ClickHouse"]]
    b.append(group(t, 20, 690 + dy, W - 40, 92, "WAYS OUT", center=True))
    for x, name in zip(xs, outs):
        b.append(box(t, x, oy, ow, 44, name, size=12.5, bold_first=False))
    mids = [x + ow / 2 for x in xs]
    bar = 672 + dy
    # Prometheus: the collector's rows, live; OpenTelemetry: spans from the edge, where the request's trace is
    b.append(f'<path d="M300,{544 + dy} V{bar} H{mids[1]}" fill="none" stroke="{t["ink2"]}" '
             f'stroke-width="1.4" stroke-dasharray="5 4"/>')
    b.append(arrow(t, [(mids[1], bar), (mids[1], oy - 2)], dashed=True))
    b.append(lines(308, 552 + dy, ["rows, live, from the collector"], t["muted"], size=11.5, anchor="start"))
    b.append(f'<path d="M{cx - 2},431 H236 V470 H32 V{bar} H{mids[0]}" fill="none" stroke="{t["ink2"]}" '
             f'stroke-width="1.4" stroke-dasharray="5 4"/>')
    b.append(arrow(t, [(mids[0], bar), (mids[0], oy - 2)], dashed=True))
    b.append(lines(40, 596 + dy, ["spans, live,", "from the edge"], t["muted"], size=11.5, anchor="start", gap=15))
    b.append(f'<path d="M{W / 2},{650 + dy} V{bar} M{mids[2]},{bar} H{mids[5]}" fill="none" stroke="{t["ink2"]}" '
             f'stroke-width="1.4"/>')
    for m in mids[2:]:
        b.append(arrow(t, [(m, bar), (m, oy - 2)]))
    b.append(masked(t, mids[3] + 10, bar - 6, "read from the ledger; sinks get a copy of every row", anchor="start"))
    return svg(W, H, t, b, "How a request flows through Prefixwise: four ways in; an edge reads, forwards (proxy "
               "only) and fingerprints the request where it is; a turn record (hashes, counts, usage, never text) "
               "crosses to a collector, in one process or over HTTP or a Redis Stream partitioned by session, which "
               "names one cause and writes one ledger row; the response back with the verdict in headers; the engine's "
               "KV-cache events joined at the edge; agent logs through their own rules into their own ledger file; "
               "OpenTelemetry spans live from the edge, Prometheus live from the collector's rows, report, API and "
               "export from the ledger, sinks a copy of every row")


# -- figure 2: how the cause is decided ------------------------------------------------------

def decision(t: dict) -> str:
    r = roles(t)
    W = 940
    b = [lines(W / 2, 34, ["How Prefixwise names a cache miss"], t["ink"], size=17, weight=600),
         lines(W / 2, 56, ["Each request is compared with the previous turn of its own conversation. "
                           "The first rule that fires wins."], t["muted"], size=12.5)]
    cx, dw, dh = 230, 330, 74
    ox, ow = 470, 440  # the verdict column
    b.append(box(t, cx - 165, 82, 330, 46, ["A request and its response", "plus its thread's previous turn"]))
    b.append(lines(ox + ow / 2, 110, ["VERDICT"], t["muted"], size=12, weight=600))
    rows = [
        (["First turn, new conversation, fork,", "other model, or too short to cache?"],
         ["first_write · fork_boundary · model_switch", "below_min_prefix: nothing could be reused yet"]),
        (["Did a part of the prompt change?", "(tools, system, messages, tool results)"],
         ["The first changed part names it", "tool_reserialize · tool_catalog_change · volatile_system",
          "system_edit · history_edit · tool_result_mutation · compaction_rewrite",
          "unless it had expired anyway (ttl_expired)", "or the cache still served it all (none)"]),
        (["A setting, the size or", "a cache breakpoint?"],
         ["param_rendered · below_min_prefix · breakpoint_misplaced"]),
        (["Was the prefix read", "from the cache?"], ["none: a hit"]),
        (["Do KV events show an eviction", "or a request sent to another cache?"],
         ["evicted · routing_miss", "confirmed (vLLM / SGLang with --kv-events)"]),
        (["The same request again", "within a minute?"], ["retry_replay"]),
        (["Behind a self-hosted engine", "(--engine vllm / sglang)?"],
         ["evicted, inferred (one cache, no event evidence)", "otherwise unknown, saying why"]),
        (["Longer gap than the", "cache keeps entries?"], ["ttl_expired", "the entry expired"]),
        (["Nothing read, and", "within that time?"], ["routing_miss"]),
    ]
    y = 150
    prev_bottom = 128
    for i, (q, outcome) in enumerate(rows):
        cy = y + dh / 2
        b.append(arrow(t, [(cx, prev_bottom), (cx, cy - dh / 2 - 2)], "no" if i else None, cx + 8,
                       prev_bottom + 15, anchor="start"))
        b.append(diamond(t, cx, cy, dw, dh, q, stroke=r["slate"]))
        oh = 22 + 17 * len(outcome)
        hit = outcome == ["none: a hit"]
        b.append(box(t, ox, cy - oh / 2, ow, oh, outcome, accent=r["hit"] if hit else r["blue"],
                     fill=r["hit_fill"] if hit else r["verdict_fill"]))
        b.append(arrow(t, [(cx + dw / 2, cy), (ox - 2, cy)], "yes", cx + dw / 2 + 12, cy - 7, anchor="start"))
        prev_bottom = cy + dh / 2
        y += dh + 26
    # nothing fired: unknown, in the verdict column with the others
    uy = prev_bottom + 40
    b.append(arrow(t, [(cx, prev_bottom), (cx, uy), (ox - 2, uy)], "no", cx + 8, prev_bottom + 15,
                   anchor="start"))
    b.append(box(t, ox, uy - 26, ow, 52, ["unknown", "with the evidence that was missing: never a guess"],
                 accent=r["blue"], fill=r["verdict_fill"]))
    y = uy + 26 + 34
    b.append(lines(W / 2, y + 6, ["Then, for every verdict: a sub-reason where the evidence allows, waste or expected,"],
                   t["muted"], size=12.5))
    b.append(lines(W / 2, y + 24, ["the avoidable tokens and their price, and one row in the ledger"],
                   t["muted"], size=12.5))
    H = int(y + 52)
    return svg(W, H, t, b, "How Prefixwise names a cache miss: a flowchart of the classifier's checks in order, "
               "with every verdict, hits and unknown included, in one column")


def main() -> None:
    for name, fn in (("architecture", architecture), ("decision", decision)):
        for mode, t in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(fn(t))
            print(f"wrote {name}-{mode}.svg")


if __name__ == "__main__":
    main()
