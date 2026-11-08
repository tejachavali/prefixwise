"""The release workflow makes the GitHub Release from the CHANGELOG section of the tagged
version (.github/release-notes.awk): that section must exist before the tag is pushed."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def notes(version: str) -> str:
    return subprocess.run(["awk", "-v", f"v={version}", "-f", str(ROOT / ".github" / "release-notes.awk"),
                           str(ROOT / "CHANGELOG.md")], capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not installed")
def test_the_current_version_has_release_notes():
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    text = notes(version)
    assert text.strip(), f"CHANGELOG.md has no '## {version}' section: the GitHub Release would fail"
    assert "\n## " not in text  # stops at the next version


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not installed")
def test_an_unknown_version_has_none():
    assert notes("99.0.0") == ""
