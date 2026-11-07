"""Draw the README figures as static SVG, one light and one dark file each.

    python docs/figures/make_figures.py

Every number is copied from a run record (linked beside each dataset); change the
record, then the number here. Colors: categorical slots 1-2 and the surface, ink,
grid and axis tokens of the dataviz reference palette, both checked for colour-vision
deficiency and contrast in each mode.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# experiments/e4/runs/2026-11-06-live.md: cached / prompt tokens, 60 paired requests per arm
FIX = [("Claude Haiku 5.5", 114_835 / 225_395, 219_451 / 225_395, "99.9%"),
       ("Qwen3-4B on vLLM", 138_928 / 157_706, 152_800 / 157_706, "97.5%")]
# experiments/e_evict/runs/2026-11-07-colab-l4.md: cached / prompt tokens after the flood (rep 0)
EVICT = [("vLLM", 0 / 3404, "evicted: confirmed from KV events"),
         ("vLLM + LMCache", 3328 / 3404, "hit: restored from CPU"),
         ("SGLang", 3 / 3404, "evicted: inferred"),
         ("SGLang + HiCache", 3382 / 3404, "hit: restored from host memory")]
# benchmarks/runs/2026-11-07-vm.md: added latency, OpenAI, not streamed
OVERHEAD = [(2_000, 2.71, 3.81), (30_000, 5.62, 7.51), (120_000, 13.27, 18.77)]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: int, h: int, t: dict, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{esc(title)}" font-family="{FONT}">'
            f'<title>{esc(title)}</title><rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>'
            + "".join(body) + "</svg>\n")


def text(x, y, s, fill, size=13, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc(s)}</text>')


def col(x, base, w, h, fill, r=4):
    """A column anchored to the baseline with rounded top corners."""
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (f'<path d="M{x:.1f},{base:.1f} V{base - h + r:.1f} Q{x:.1f},{base - h:.1f} {x + r:.1f},{base - h:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{base - h:.1f} {x + w:.1f},{base - h + r:.1f} V{base:.1f} Z" '
            f'fill="{fill}"/>')


def bar(x, y, w, h, fill, r=4):
    """A bar anchored to the left axis with rounded right corners."""
    if w <= 0:
        return ""
    r = min(r, h / 2, w)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>')


def legend(x, y, items, t):
    out = []
    for label, color in items:
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(text(x + 16, y, label, t["ink2"]))
        x += 16 + 7.2 * len(label) + 22
    return "".join(out)


def fix_chart(t: dict) -> str:
    w, h = 720, 300
    left, right, top, base = 56, 700, 48, 250
    body = [legend(left, 24, [("fix off (tools in drifting order)", t["s1"]),
                              ("--fix canonical_tools", t["s2"])], t)]
    for v in (0, 25, 50, 75, 100):
        y = base - (base - top) * v / 100
        body.append(f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
                    f'stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        body.append(text(left - 8, y + 4, f"{v}%", t["muted"], 12, "end"))
    group, bw = (right - left) / len(FIX), 64
    for i, (name, off, on, realised) in enumerate(FIX):
        cx = left + group * (i + 0.5)
        for j, (v, color) in enumerate(((off, t["s1"]), (on, t["s2"]))):
            x = cx - bw - 1 + j * (bw + 2)  # 2px gap between the pair
            hh = (base - top) * v
            body.append(col(x, base, bw, hh, color))
            body.append(text(x + bw / 2, base - hh - 6, f"{v:.0%}", t["ink"], 13, "middle", 600))
        body.append(text(cx, base + 20, name, t["ink"], 13, "middle"))
        body.append(text(cx, base + 38, f"{realised} of the predicted saving realised", t["ink2"], 12, "middle"))
    return svg(w, h, t, body, "Share of prompt tokens served from cache, fix off and on")


def evict_chart(t: dict) -> str:
    w, h = 720, 250
    left, top = 148, 52
    row, bh = 42, 22
    axis_w = 260  # bars on the left; Prefixwise's verdict in a column to their right
    verdict_x = left + axis_w + 70
    n = len(EVICT)
    body = [text(left, 26, "Prefix served from cache after the flood", t["ink2"], 12, "start", 600),
            text(verdict_x, 26, "Prefixwise said (each 5 of 5 runs)", t["ink2"], 12, "start", 600)]
    for v in (0, 50, 100):
        x = left + axis_w * v / 100
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{top + row * n - 14}" '
                    f'stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        body.append(text(x, top + row * n + 2, f"{v}%", t["muted"], 12, "middle"))
    for i, (name, share, verdict) in enumerate(EVICT):
        y = top + i * row
        body.append(text(left - 10, y + bh / 2 + 5, name, t["ink"], 13, "end"))
        body.append(bar(left, y, axis_w * share, bh, t["s1"]))
        body.append(text(left + axis_w * share + 8, y + bh / 2 + 5, f"{share:.0%}", t["ink"], 13, "start", 600))
        body.append(text(verdict_x, y + bh / 2 + 5, verdict, t["ink"], 13))
    body.append(text(left, h - 10, "Qwen3-4B, GPU KV cache of 32,768 tokens flooded with 3x that, then the session's next turn",
                     t["muted"], 12))
    return svg(w, h, t, body, "Prefix served from cache after a GPU flood, and Prefixwise's verdict, per engine")


def overhead_chart(t: dict) -> str:
    w, h = 720, 280
    left, right, top, base = 56, 600, 40, 230
    xmax, ymax = 120_000, 20

    def px(x):
        return left + (right - left) * x / xmax

    def py(y):
        return base - (base - top) * y / ymax
    body = [legend(left, 22, [("median (p50)", t["s1"]), ("p95", t["s2"])], t)]
    for v in (0, 5, 10, 15, 20):
        body.append(f'<line x1="{left}" x2="{right}" y1="{py(v):.1f}" y2="{py(v):.1f}" '
                    f'stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        body.append(text(left - 8, py(v) + 4, f"{v} ms", t["muted"], 12, "end"))
    for v in (0, 30_000, 60_000, 90_000, 120_000):
        body.append(text(px(v), base + 18, f"{v // 1000}k", t["muted"], 12, "middle"))
    body.append(text((left + right) / 2, base + 38, "prompt tokens per request", t["muted"], 12, "middle"))
    for k, color, name in ((1, t["s1"], "p50"), (2, t["s2"], "p95")):
        pts = " ".join(f"{px(p[0]):.1f},{py(p[k]):.1f}" for p in OVERHEAD)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                    f'stroke-linejoin="round" stroke-linecap="round"/>')
        for p in OVERHEAD:
            body.append(f'<circle cx="{px(p[0]):.1f}" cy="{py(p[k]):.1f}" r="4" fill="{color}" '
                        f'stroke="{t["surface"]}" stroke-width="2"/>')
        last = OVERHEAD[-1]
        body.append(text(px(last[0]) + 10, py(last[k]) + 4, f"{name} {last[k]:.1f} ms", t["ink"], 13, "start", 600))
    return svg(w, h, t, body, "Latency Prefixwise adds per request, by prompt size")


if __name__ == "__main__":
    for name, draw in (("fix", fix_chart), ("evict", evict_chart), ("overhead", overhead_chart)):
        for mode, theme in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(draw(theme))
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
