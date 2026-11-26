"""Keeps tests/ importable from any working directory (tests share helpers, e.g. test_details)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

# A developer's own PREFIXWISE_HMAC_KEY would key every hash the tests compare with bare ones (every
# edge reads it). Tests that want a key pass one or set the variable themselves.
os.environ.pop("PREFIXWISE_HMAC_KEY", None)
