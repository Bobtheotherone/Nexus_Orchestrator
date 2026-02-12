"""Unit tests for canonical ID helpers."""

from __future__ import annotations

import random

import pytest

from nexus_orchestrator.domain import ids


def _zero_bytes(size: int) -> bytes:
    return b"\x00" * size


def _ff_bytes(size: int) -> bytes:
    return b"\xff" * size


def test_generate_ulid_no_collision_10000() -> None:
    generated = {ids.generate_ulid() for _ in range(10_000)}
    assert len(generated) == 10_000


def test_ulid_charset_length_and_reject_invalid_chars() -> None:
    ulid_value = ids.generate_ulid(timestamp_ms=123_456, randbytes=_ff_bytes)
    assert len(ulid_value) == ids.ULID_LENGTH
    assert ulid_value == ulid_value.upper()
    assert all(char in ids.CROCKFORD_BASE32_ALPHABET for char in ulid_value)

    ids.validate_ulid(ulid_value.lower())

    with pytest.raises(ValueError, match="ulid length must be"):
        ids.validate_ulid("0" * 25)

    for invalid in ["I" + "0" * 25, "l" + "0" * 25, "O" + "0" * 25, "u" + "0" * 25]:
        with pytest.raises(ValueError, match="invalid ULID character"):
            ids.validate_ulid(invalid)

    with pytest.raises(ValueError, match="invalid ULID character"):
        ids.validate_ulid("*" + "0" * 25)


def test_ulid_overflow_and_timestamp_boundaries() -> None:
    max_ulid = "7" + "Z" * 25
    ids.validate_ulid(max_ulid)

    overflow_ulid = "8" + "0" * 25
    with pytest.raises(ValueError, match="overflow"):
        ids.validate_ulid(overflow_ulid)

    min_ulid = "0" * 26
    assert ids.parse_ulid_timestamp_ms(min_ulid) == 0

    top_timestamp_ulid = ids.generate_ulid(
        timestamp_ms=ids.ULID_MAX_TIMESTAMP_MS,
        randbytes=_zero_bytes,
    )
    assert ids.parse_ulid_timestamp_ms(top_timestamp_ulid) == ids.ULID_MAX_TIMESTAMP_MS


def test_prefixed_id_helpers_and_short_id() -> None:
    run_id = ids.generate_run_id(timestamp_ms=1, randbytes=_ff_bytes)
    work_item_id = ids.generate_work_item_id(timestamp_ms=1, randbytes=_ff_bytes)
    attempt_id = ids.generate_attempt_id(timestamp_ms=1, randbytes=_ff_bytes)
    evidence_id = ids.generate_evidence_id(timestamp_ms=1, randbytes=_ff_bytes)
    merge_id = ids.generate_merge_id(timestamp_ms=1, randbytes=_ff_bytes)
    incident_id = ids.generate_incident_id(timestamp_ms=1, randbytes=_ff_bytes)
    event_id = ids.generate_event_id(timestamp_ms=1, randbytes=_ff_bytes)

    ids.validate_run_id(run_id)
    ids.validate_work_item_id(work_item_id)
    ids.validate_attempt_id(attempt_id)
    ids.validate_evidence_id(evidence_id)
    ids.validate_merge_id(merge_id)
    ids.validate_incident_id(incident_id)
    ids.validate_event_id(event_id)

    with pytest.raises(ValueError, match="expected prefix"):
        ids.validate_prefixed_id(run_id, ids.WORK_ITEM_ID_PREFIX)

    short = ids.short_id(run_id)
    assert len(short) == 8
    assert short == run_id[-8:]
    assert ids.short_id(run_id) == short

    with pytest.raises(ValueError, match="at least 8"):
        ids.short_id("short")


def test_requirement_and_constraint_id_validators() -> None:
    ids.validate_requirement_id("REQ-0001")
    ids.validate_constraint_id("CON-SEC-0001")

    with pytest.raises(ValueError, match="REQ-0001"):
        ids.validate_requirement_id("REQ-001")

    with pytest.raises(ValueError, match="sequence"):
        ids.validate_requirement_id("REQ-0000")

    with pytest.raises(ValueError, match="CON-SEC-0001"):
        ids.validate_constraint_id("CON-sec-0001")

    with pytest.raises(ValueError, match="sequence"):
        ids.validate_constraint_id("CON-SEC-0000")


def test_random_seed_does_not_change_ulid_engine() -> None:
    random.seed(123)

    def _boom(_: int) -> int:
        raise AssertionError("global random.getrandbits must not be used")

    previous = random.getrandbits
    random.getrandbits = _boom
    try:
        ulid_value = ids.generate_ulid(timestamp_ms=42)
    finally:
        random.getrandbits = previous

    ids.validate_ulid(ulid_value)

    deterministic_1 = ids.generate_ulid(timestamp_ms=42, randbytes=_zero_bytes)
    deterministic_2 = ids.generate_ulid(timestamp_ms=42, randbytes=_zero_bytes)
    assert deterministic_1 == deterministic_2
