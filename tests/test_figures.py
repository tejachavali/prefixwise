"""The README and docs figures stay true: each SVG is exactly what its script draws now, and the
decision flowchart names every cause the classifier can return. Change the code or the script
without redrawing, and this fails: run `python docs/figures/make_architecture.py` (and
`make_figures.py`), check the pictures, commit them."""

import re
import sys
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
sys.path.insert(0, str(FIGURES))

import make_architecture  # noqa: E402
import make_figures  # noqa: E402

from prefixwise.classifier import MissCause  # noqa: E402

DRAWN = [("architecture", make_architecture.architecture), ("decision", make_architecture.decision),
         ("fix", make_figures.fix_chart), ("evict", make_figures.evict_chart),
         ("overhead", make_figures.overhead_chart)]


@pytest.mark.parametrize("name,draw", DRAWN, ids=[n for n, _ in DRAWN])
@pytest.mark.parametrize("mode", sorted(make_figures.THEMES))
def test_committed_figure_is_what_its_script_draws(name, draw, mode):
    committed = (FIGURES / f"{name}-{mode}.svg").read_text()
    assert committed == draw(make_figures.THEMES[mode]), (
        f"docs/figures/{name}-{mode}.svg is stale: redraw it with the scripts in docs/figures")


def test_the_decision_flowchart_names_every_cause():
    text = " ".join(re.findall(r">([^<]*)</text>", (FIGURES / "decision-light.svg").read_text()))
    missing = [c.value for c in MissCause if not re.search(rf"\b{re.escape(c.value)}\b", text)]
    assert not missing, f"causes missing from the decision flowchart: {missing}"
