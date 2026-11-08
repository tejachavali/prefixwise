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


def diamond(t, cx, cy, w, h, rows):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    gap = 16
    top = cy - (len(rows) - 1) * gap / 2 + 4.5
    return (f'<polygon points="{pts}" fill="{t["surface"]}" stroke="{t["s1"]}" stroke-width="1.6"/>'
            + lines(cx, top, rows, t["ink"], size=12.5, gap=gap))


def cylinder(t, x, y, w, h, rows):
    e = 9
    body = (f'<path d="M{x},{y + e} V{y + h - e} A{w / 2},{e} 0 0 0 {x + w},{y + h - e} V{y + e}" '
            f'fill="{t["surface"]}" stroke="{t["s1"]}" stroke-width="1.6"/>'
            f'<ellipse cx="{x + w / 2}" cy="{y + e}" rx="{w / 2}" ry="{e}" fill="{t["surface"]}" '
            f'stroke="{t["s1"]}" stroke-width="1.6"/>')
    top = y + h / 2 - (len(rows) - 1) * 8.5 + 9
    return body + lines(x + w / 2, top, rows, t["ink"], weight=[600] + [400] * (len(rows) - 1))


def arrow(t, pts, label=None, lx=None, ly=None, anchor="middle", dashed=False):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = (f'<path d="{d}" fill="none" stroke="{t["ink2"]}" stroke-width="1.4"{dash} '
           f'marker-end="url(#{ARROW_ID})"/>')
    if label:
        out += lines(lx, ly, [label], t["muted"], size=11.5, anchor=anchor)
    return out


def group(t, x, y, w, h, title, right=False):
    tx, anchor = (x + w - 14, "end") if right else (x + 14, "start")
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" stroke="{t["grid"]}" '
            f'stroke-width="1.5"/>' + lines(tx, y + 22, [title], t["muted"], size=12, weight=600,
                                            anchor=anchor))


# -- figure 1: how a request flows -----------------------------------------------------------

def architecture(t: dict) -> str:
    W, H = 940, 800
    b = [lines(W / 2, 34, ["How a request flows through Prefixwise"], t["ink"], size=17, weight=600),
         lines(W / 2, 56, ["Four ways in, one classifier, one row per request; the prompt itself is never stored"],
               t["muted"], size=12.5)]
    # ways in
    b.append(group(t, 20, 74, W - 40, 112, "WAYS IN"))
    ins = [("Your agent", "through the proxy", "prefixwise serve / run"),
           ("Your own code", "prefixwise.wrap(client)", "Anthropic / OpenAI SDKs"),
           ("LiteLLM", "PrefixwiseCallback", "SDK or proxy server"),
           ("Agent logs", "prefixwise import", "Claude Code, Codex")]
    for x, rows in zip([40, 262, 484, 706], ins):
        b.append(box(t, x, 104, 194, 66, list(rows)))
    # the pipeline
    b.append(group(t, 20, 204, W - 40, 356, "PREFIXWISE", right=True))
    cx, cw = 250, 440
    steps = [(234, ["1  Read the request", "session, agent, task; diagnostic fields added"]),
             (318, ["2  Forward it (proxy only)", "the client's request and auth, as sent"]),
             (402, ["3  Fingerprint the prompt", "blocks in cache order, hashed: never the text"]),
             (486, ["4  Name one cause", "against the previous turn; sub-reason; price"])]
    for y, rows in steps:
        b.append(box(t, cx, y, cw, 58, rows, accent=t["s1"] if y == 486 else None))
    for (y1, _), (y2, _) in zip(steps, steps[1:]):
        b.append(arrow(t, [(cx + cw / 2, y1 + 58), (cx + cw / 2, y2 - 2)]))
    # into step 1: the proxy, and the SDK wrappers (their call already happened: not forwarded)
    b.append(arrow(t, [(150, 170), (150, 252), (cx - 2, 252)]))
    b.append(arrow(t, [(359, 170), (359, 194), (222, 194), (222, 276), (cx - 2, 276)]))
    b.append(arrow(t, [(581, 170), (581, 194), (222, 194)]))
    b.append(lines(214, 300, ["wrap / LiteLLM:", "already sent, so", "recorded, not forwarded"],
                   t["muted"], size=11.5, anchor="end", gap=15))
    # back to the agent: after step 4, so the verdict can ride in the headers
    b.append(arrow(t, [(cx - 2, 528), (60, 528), (60, 172)], dashed=True))
    b.append(lines(70, 470, ["response, unchanged,", "with the verdict in", "X-Prefixwise-* headers"],
                   t["muted"], size=11.5, anchor="start", gap=15))
    # the provider or engine
    b.append(box(t, 724, 315, 180, 64, ["Provider or engine", "Anthropic, OpenAI,", "vLLM, SGLang"]))
    b.append(arrow(t, [(cx + cw, 334), (722, 334)], "request", 707, 328))
    b.append(arrow(t, [(722, 362), (cx + cw + 2, 362)], "response", 707, 378))
    b.append(box(t, 724, 486, 180, 58, ["KV-cache events", "vLLM / SGLang (optional)"], dashed=True))
    b.append(arrow(t, [(722, 503), (cx + cw + 2, 503)], "evicted? which replica?", 812, 562))
    # ledger
    b.append(cylinder(t, 380, 584, 180, 64, ["Ledger", "SQLite, one row per request"]))
    b.append(arrow(t, [(cx + cw / 2, 544), (cx + cw / 2, 582)]))
    b.append(arrow(t, [(803, 170), (803, 194), (912, 194), (912, 616), (562, 616)],
                   "usage from the logs", 906, 604, anchor="end"))
    # ways out: live per request, and read from the ledger
    ow, gapx, oy = 168, 10, 718
    x0 = (W - (5 * ow + 4 * gapx)) / 2
    xs = [x0 + i * (ow + gapx) for i in range(5)]
    outs = ["Prometheus /metrics", "OpenTelemetry spans", "prefixwise report", "HTTP API /api/v1",
            "prefixwise export"]
    b.append(group(t, 20, 690, W - 40, 92, "WAYS OUT"))
    for x, name in zip(xs, outs):
        b.append(box(t, x, oy, ow, 44, [name], size=12.5))
    mids = [x + ow / 2 for x in xs]
    bar = 672
    # live: each row as it is recorded (from step 4)
    b.append(f'<path d="M300,546 V{bar} M{mids[0]},{bar} H{mids[1]}" fill="none" stroke="{t["ink2"]}" '
             f'stroke-width="1.4" stroke-dasharray="5 4"/>')
    for m in mids[:2]:
        b.append(arrow(t, [(m, bar), (m, oy - 2)], dashed=True))
    b.append(lines(292, 600, ["live, as each", "row is recorded"], t["muted"], size=11.5, anchor="end", gap=15))
    # read from the ledger
    b.append(f'<path d="M{W / 2},650 V{bar} M{mids[2]},{bar} H{mids[4]}" fill="none" stroke="{t["ink2"]}" '
             f'stroke-width="1.4"/>')
    for m in mids[2:]:
        b.append(arrow(t, [(m, bar), (m, oy - 2)]))
    b.append(lines(mids[3] + 10, bar - 8, ["read from the ledger"], t["muted"], size=11.5, anchor="start"))
    return svg(W, H, t, b, "How a request flows through Prefixwise: four ways in; read, forward (proxy only), "
               "fingerprint and name one cause; the response back with the verdict in headers; one ledger row; "
               "Prometheus and OpenTelemetry live, report, API and export from the ledger")


# -- figure 2: how the cause is decided ------------------------------------------------------

def decision(t: dict) -> str:
    W = 940
    b = [lines(W / 2, 34, ["How Prefixwise names a cache miss"], t["ink"], size=17, weight=600),
         lines(W / 2, 56, ["Each request is compared with the previous one to the same model in the same session. "
                           "The first rule that fires wins."], t["muted"], size=12.5)]
    cx, dw, dh = 230, 330, 74
    ox, ow = 470, 440  # outcome column
    y = 82
    b.append(box(t, cx - 165, y, 330, 46, ["A request and its response", "plus the previous turn"]))
    rows = [
        (["First turn? Another model?", "Previous turn too short to cache?"],
         ["first_write · fork_boundary · model_switch", "below_min_prefix: nothing could be reused yet"]),
        (["Did a part of the prompt change?", "(system, tools, messages, tool results)"],
         ["The first changed part names it", "volatile_system · system_edit · tool_reserialize",
          "tool_catalog_change · history_edit · tool_result_mutation", "compaction_rewrite"]),
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
    y = 128 + 22
    prev_bottom = 128
    for q, outcome in rows:
        cy = y + dh / 2
        b.append(arrow(t, [(cx, prev_bottom), (cx, cy - dh / 2 - 2)],
                       "no" if prev_bottom != 128 else None, cx + 8, prev_bottom + 15, anchor="start"))
        b.append(diamond(t, cx, cy, dw, dh, q))
        oh = 22 + 17 * len(outcome)
        b.append(box(t, ox, cy - oh / 2, ow, oh, outcome, accent=t["s1"]))
        b.append(arrow(t, [(cx + dw / 2, cy), (ox - 2, cy)], "yes", cx + dw / 2 + 12, cy - 7, anchor="start"))
        prev_bottom = cy + dh / 2
        y += dh + 26
    # the end
    b.append(arrow(t, [(cx, prev_bottom), (cx, y - 2)], "no", cx + 8, prev_bottom + 15, anchor="start"))
    b.append(box(t, cx - 165, y, 330, 52, ["unknown", "with the evidence that was missing: never a guess"]))
    y += 52 + 24
    # the after-steps
    b.append(lines(W / 2, y + 6, ["Then, for every verdict: a sub-reason where the evidence allows, waste or expected,"],
                   t["muted"], size=12.5))
    b.append(lines(W / 2, y + 24, ["the avoidable tokens and their price, and one row in the ledger"],
                   t["muted"], size=12.5))
    H = int(y + 52)
    return svg(W, H, t, b, "How Prefixwise names a cache miss: a flowchart of the classifier's rules, in order")


def main() -> None:
    for name, fn in (("architecture", architecture), ("decision", decision)):
        for mode, t in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(fn(t))
            print(f"wrote {name}-{mode}.svg")


if __name__ == "__main__":
    main()
