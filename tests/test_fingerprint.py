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
