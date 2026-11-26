"""The hallway demo, keypress-driven: break the cache three ways, then fix it. About four minutes.

    python experiments/demo/hallway.py            # mock provider: no key, no network, no cost
    python experiments/demo/hallway.py --live     # the real API: ANTHROPIC_API_KEY (or OPENAI_API_KEY)
    python experiments/demo/hallway.py --auto     # no pauses, for a recording

The demo itself ships in the package as `prefixwise demo` (`src/prefixwise/demo.py`); this script
runs it one act per keypress, which is how it is shown in person. Every other flag is the demo's own.
"""

from __future__ import annotations

import sys

from prefixwise.demo import run

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--auto" in argv:
        argv = [a for a in argv if a != "--auto"]
        if "--pause" not in argv and not any(a.startswith("--pause=") for a in argv):
            argv += ["--pause", "2"]
    elif "--step" not in argv:
        argv.append("--step")
    sys.exit(run(argv, prog="hallway"))
