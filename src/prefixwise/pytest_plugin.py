"""pytest plugin: under `prefixwise check -- pytest ...`, each test's node id becomes the scenario.

Registered as the `pytest11` entry point `prefixwise`, so it loads with pytest wherever Prefixwise
is installed, and does nothing unless `PREFIXWISE_CHECK` and `PREFIXWISE_URL` are set, which only
`prefixwise check` does for the command it runs. Before each test it tells the proxy the test's node
id; requests that carry no session id of their own are then recorded under it, so the findings name
the test (`scenario=tests/test_agent.py::test_refactor`) and the allow-list can key on it. An agent
that sets `X-Prefixwise-Session-Id` keeps its own name. Tests run in parallel (pytest-xdist) share
one proxy, so their scenarios can blur: give those agents a session id.

Disable with `-p no:prefixwise`.
"""

from __future__ import annotations

import json
import os
import urllib.request

MAX_NAME = 200


def _tell(name: str) -> None:
    url = os.environ.get("PREFIXWISE_URL")
    if not url or not os.environ.get("PREFIXWISE_CHECK"):
        return
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/check/scenario", method="POST",
                                 data=json.dumps({"name": name[:MAX_NAME]}).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            r.read()
    except Exception:  # noqa: BLE001 - never fail a test over its label
        pass


def pytest_runtest_setup(item) -> None:
    _tell(item.nodeid)
