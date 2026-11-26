"""Keeps tests/ importable from any working directory (tests share helpers, e.g. test_details)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

# No setting of the developer's shell reaches the tests: a PREFIXWISE_HMAC_KEY would key every hash
# they compare with bare ones, a PREFIXWISE_PRICES file would change every dollar. Tests that want a
# setting pass it or set the variable themselves.
# (PREFIXWISE_TEST_* are the suite's own: CI points the record-log tests at its Redis with one.)
for _name in [n for n in os.environ if n.startswith("PREFIXWISE_") and not n.startswith("PREFIXWISE_TEST_")]:
    del os.environ[_name]
