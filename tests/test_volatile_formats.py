"""Reddit feedback, Nov 2026: a timestamp whose format changed between turns was `system_edit` (no fix
shown) because only ISO dates were masked; and "cwd and date in the system prompt" are the injected
bits that kill the prefix. Every common date form now masks to one token, absolute paths are volatile,
and the verdict names what the system prompt carries (never the values)."""

from prefixwise.classifier import MissCause, RequestSnapshot, classify
from prefixwise.fingerprint import Fingerprinter, mask_volatiles, mask_volatiles_kinds
from prefixwise.fixes import FIX_ADVICE

FORMATS = ["2026-11-09T04:10:00Z", "2026-11-09, 04:10:05 UTC", "Nov 9, 2026 04:10:05", "2026/11/09 04:10",
           "Mon, 09 Nov 2026 04:10:00 GMT", "9 November 2026", "11/09/2026 4:10 PM EST", "2026-11-09",
           "Sunday, November 9th, 2026 at 4:10 PM", "2026-11-09T04:10:00.123+01:00", "2026-11-09T04:10:00z",
           "Sun Nov  9 04:10:00 UTC 2026", "2026-11-09 04:10:00.123456", "9. November 2026"]


def test_every_common_timestamp_format_masks_to_the_same_text():
    masked = {mask_volatiles(f"Today is {f}. Be brief.") for f in FORMATS}
    assert masked == {"Today is <DATE>. Be brief."}
    # a bare time, with or without a zone
    assert mask_volatiles("at 04:10 and at 16:10:05 UTC") == "at <TIME> and at <TIME>"
    # words are not zones: "Test" stays
    assert mask_volatiles("Nov 9, 2026 Test") == "<DATE> Test"


def test_words_and_dotted_numbers_are_not_dates():
    """Review findings: month and weekday names must be whole words, and a version or an IP is a number;
    otherwise an edit of a word would read as a volatile value."""
    assert mask_volatiles("Monitor 2026-11-09") != mask_volatiles("Sunset 2026-11-09")
    assert mask_volatiles("Marker 10 2026") == "Marker <N> <N>" and mask_volatiles("Decade 10 2026") == "Decade <N> <N>"
    for text in ("python 3.11.12", "192.168.1.10", "release 2.10.26", "11.09.26"):
        assert "date" not in mask_volatiles_kinds(text)[1], text


def test_paths_are_recognised_but_not_masked():
    """A path in a system prompt is as often an instruction as an injected working directory: it is
    named (so the verdict can say the prompt carries one) and left in the masked text, so an edit of
    an instruction naming a path stays an edit."""
    text, kinds = mask_volatiles_kinds("cwd is /Users/alice/proj/repo (was C:\\Users\\bob\\repo, then ~/w/x). "
                                       "API at https://api.example.com/v1/chat; and/or both")
    assert "path" in kinds and "date" not in kinds and "/Users/alice/proj/repo" in text and "~/w/x" in text
    assert "path" not in mask_volatiles_kinds("use /tmp or ~ for scratch, see https://x.com/a/b and a/b/c")[1]
    assert mask_volatiles("never write to /etc/passwd") != mask_volatiles("never write to /etc/shadow")


def test_kinds_are_recorded_without_the_values():
    fp = Fingerprinter()
    b = fp.fingerprint_block("system", 0, "Date: 2026-11-09. cwd /home/u/repo. Build 7f3a9c1e2b4d6f80. 3 files.")
    assert b.volatile == ["date", "id", "path", "number"]
    assert "2026" not in (b.raw + b.canonical + b.masked) and "repo" not in (b.raw + b.canonical + b.masked)
    assert fp.fingerprint_block("system", 0, "Be careful.").volatile == []


def _snap(system_text, cached=0):
    fp = Fingerprinter()
    blocks = fp.fingerprint_blocks([("system", system_text), ("messages", {"role": "user", "content": "hi"})])
    return RequestSnapshot(model="m", provider="openai", blocks=blocks, prompt_tokens=4000, cached_tokens=cached)


def test_timestamp_format_drift_is_volatile_system_with_a_fix_not_system_edit():
    prev = _snap("You are an agent. Now: 2026-11-09T04:10:00Z. cwd /Users/alice/proj/repo.")
    for f in FORMATS[1:]:
        c = classify(prev, _snap(f"You are an agent. Now: {f}. cwd /Users/alice/proj/repo."))
        assert c.cause is MissCause.VOLATILE_SYSTEM and c.legit is False, f
        assert "carries a date or timestamp" in c.evidence and "also carries a file path" in c.evidence
        assert "a number" not in c.evidence and "a time" not in c.evidence
    assert FIX_ADVICE["volatile_system"]["fix"].startswith("Move dates")
    # a changed working directory is an edit of the prompt (paths are named, not masked): the fix for
    # it is the same, and the evidence of a date change points at the path too
    c = classify(prev, _snap("You are an agent. Now: 2026-11-09T04:10:00Z. cwd /tmp/pytest-of-u/pytest-3/repo."))
    assert c.cause is MissCause.SYSTEM_EDIT
    # a changed word is still an edit
    c = classify(prev, _snap("You are a careful agent. Now: 2026-11-09T04:10:00Z. cwd /Users/alice/proj/repo."))
    assert c.cause is MissCause.SYSTEM_EDIT
