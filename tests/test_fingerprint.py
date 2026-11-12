from prefixwise.fingerprint import Fingerprinter, mask_volatiles


def test_canonical_equal_for_same_tool_different_key_order():
    fp = Fingerprinter()
    a = fp.fingerprint_block("tools", 0, {"b": 1, "a": 2})
    assert a.canonical == fp.fingerprint_block("tools", 0, {"a": 2, "b": 1}).canonical


def test_masked_hash_equal_for_volatile_system():
    fp = Fingerprinter()
    a = fp.fingerprint_block("system", 0, "Today is 2026-11-05, session abc123")
    b = fp.fingerprint_block("system", 0, "Today is 2026-11-06, session xyz789")
    assert a.raw != b.raw
    assert a.masked == b.masked


def test_hmac_replaces_bare_hash_for_short_blocks():
    fp = Fingerprinter(hmac_key=b"test-key")
    block = fp.fingerprint_block("user", 0, "hi")
    assert block.hmac is True
    fp2 = Fingerprinter(hmac_key=b"other-key")
    assert fp2.fingerprint_block("user", 0, "hi").raw != block.raw


def test_soft_tokens_never_hash_content():
    fp = Fingerprinter()
    a = fp.fingerprint_block("soft_tokens", 0, 128)
    b = fp.fingerprint_block("soft_tokens", 0, 128)
    assert a.raw == b.raw
    assert a.raw != fp.fingerprint_block("soft_tokens", 0, 256).raw


def test_mask_volatiles():
    assert mask_volatiles("2026-11-05 and 10:30 and id 123e4567-e89b-12d3-a456-426614174000") == \
        "<DATE> and <TIME> and id <ID>"


def test_dates_without_a_year_are_masked_and_numbers_next_to_month_words_are_not():
    from prefixwise.fingerprint import mask_volatiles
    for text in ("Today is Nov 12.", "12 November", "12. November", "Thursday, November 12", "on 3rd March"):
        assert "<DATE>" in mask_volatiles(text), text
    # Found by review: a number before a month-like word is not a date, so a change to it stays a
    # number change (not "a date or timestamp").
    for text in ("Rule 1 may be skipped", "max 5 mar", "top 40 Aug", "version 2.1 May release"):
        assert "<DATE>" not in mask_volatiles(text), text
