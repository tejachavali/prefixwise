"""Keeps tests/ importable from any working directory (tests share helpers, e.g. test_details)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
