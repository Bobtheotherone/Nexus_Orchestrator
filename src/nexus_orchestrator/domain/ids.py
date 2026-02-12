"""Canonical ID generation and validation for domain entities."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from typing import Final

CROCKFORD_BASE32_ALPHABET: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ULID_LENGTH: Final[int] = 26
ULID_RANDOM_BYTES: Final[int] = 10
ULID_MAX_TIMESTAMP_MS: Final[int] = (1 << 48) - 1
_ULID_MAX_VALUE: Final[int] = (1 << 128) - 1
_PREFIX_SEPARATOR: Final[str] = "-"

# Stable entity ID prefixes.
RUN_ID_PREFIX: Final[str] = "run"
WORK_ITEM_ID_PREFIX: Final[str] = "wi"
ATTEMPT_ID_PREFIX: Final[str] = "att"
EVIDENCE_ID_PREFIX: Final[str] = "ev"
ARTIFACT_ID_PREFIX: Final[str] = "art"
MERGE_ID_PREFIX: Final[str] = "mr"
INCIDENT_ID_PREFIX: Final[str] = "inc"
EVENT_ID_PREFIX: Final[str] = "evt"

REQ_ID_PATTERN_DESCRIPTION: Final[str] = "REQ-0001"
CONSTRAINT_ID_PATTERN_DESCRIPTION: Final[str] = "CON-SEC-0001"

_REQUIREMENT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^REQ-(\d{4})$")
_CONSTRAINT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^CON-([A-Z]{3,10})-(\d{4})$")

_DECODE_TABLE: Final[dict[str, int]] = {
    char: index for index, char in enumerate(CROCKFORD_BASE32_ALPHABET)
}

_RandBytes = Callable[[int], bytes]

__all__ = [
    "ATTEMPT_ID_PREFIX",
    "ARTIFACT_ID_PREFIX",
    "CONSTRAINT_ID_PATTERN_DESCRIPTION",
    "CROCKFORD_BASE32_ALPHABET",
    "EVIDENCE_ID_PREFIX",
    "EVENT_ID_PREFIX",
    "INCIDENT_ID_PREFIX",
    "MERGE_ID_PREFIX",
    "REQ_ID_PATTERN_DESCRIPTION",
    "RUN_ID_PREFIX",
    "ULID_LENGTH",
    "ULID_MAX_TIMESTAMP_MS",
    "WORK_ITEM_ID_PREFIX",
    "generate_attempt_id",
    "generate_artifact_id",
    "generate_evidence_id",
    "generate_event_id",
    "generate_incident_id",
    "generate_merge_id",
    "generate_prefixed_id",
    "generate_run_id",
    "generate_ulid",
    "generate_work_item_id",
    "parse_ulid_timestamp_ms",
    "short_id",
    "validate_attempt_id",
    "validate_artifact_id",
    "validate_constraint_id",
    "validate_evidence_id",
    "validate_event_id",
    "validate_incident_id",
    "validate_merge_id",
    "validate_prefixed_id",
    "validate_requirement_id",
    "validate_run_id",
    "validate_ulid",
    "validate_work_item_id",
]


def generate_ulid(
    *,
    timestamp_ms: int | None = None,
    randbytes: _RandBytes | None = None,
) -> str:
    """Generate a ULID as a 26-character uppercase Crockford Base32 string."""
    ts_ms = _resolve_timestamp_ms(timestamp_ms)
    random_bytes = _resolve_random_bytes(randbytes)
    ulid_value = (ts_ms << 80) | int.from_bytes(random_bytes, "big")
    return _encode_crockford_base32(ulid_value, ULID_LENGTH)


def validate_ulid(s: str) -> None:
    """Validate a ULID and raise ``ValueError`` with precise context on failure."""
    _ = _decode_validated_ulid(s)


def parse_ulid_timestamp_ms(s: str) -> int:
    """Extract the 48-bit millisecond timestamp from a validated ULID."""
    ulid_value = _decode_validated_ulid(s)
    timestamp_ms = ulid_value >> 80
    if not 0 <= timestamp_ms <= ULID_MAX_TIMESTAMP_MS:
        raise ValueError(
            f"ulid timestamp_ms out of range: expected 0..{ULID_MAX_TIMESTAMP_MS}, got {timestamp_ms}"
        )
    return timestamp_ms


def generate_prefixed_id(
    prefix: str,
    *,
    timestamp_ms: int | None = None,
    randbytes: _RandBytes | None = None,
) -> str:
    """Generate a stable prefixed ID in the form ``<prefix>-<ulid>``."""
    _validate_prefix(prefix)
    return f"{prefix}{_PREFIX_SEPARATOR}{generate_ulid(timestamp_ms=timestamp_ms, randbytes=randbytes)}"


def validate_prefixed_id(id_str: str, expected_prefix: str) -> None:
    """Validate ``<prefix>-<ulid>`` format and enforce ``expected_prefix``."""
    _validate_prefix(expected_prefix)
    if not isinstance(id_str, str):
        raise ValueError(f"prefixed id must be a string, got {type(id_str).__name__}")

    expected_lead = f"{expected_prefix}{_PREFIX_SEPARATOR}"
    if not id_str.startswith(expected_lead):
        raise ValueError(f"expected prefix '{expected_lead}'")

    ulid_part = id_str[len(expected_lead) :]
    try:
        validate_ulid(ulid_part)
    except ValueError as exc:
        raise ValueError(f"invalid ULID part for prefix '{expected_prefix}': {exc}") from exc


def short_id(id_str: str) -> str:
    """Return the last 8 characters of an ID for compact display."""
    if not isinstance(id_str, str):
        raise ValueError(f"id must be a string, got {type(id_str).__name__}")
    if len(id_str) < 8:
        raise ValueError("id must be at least 8 characters")
    return id_str[-8:]


def generate_run_id(*, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None) -> str:
    return generate_prefixed_id(RUN_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_run_id(id_str: str) -> None:
    validate_prefixed_id(id_str, RUN_ID_PREFIX)


def generate_work_item_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(WORK_ITEM_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_work_item_id(id_str: str) -> None:
    validate_prefixed_id(id_str, WORK_ITEM_ID_PREFIX)


def generate_attempt_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(ATTEMPT_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_attempt_id(id_str: str) -> None:
    validate_prefixed_id(id_str, ATTEMPT_ID_PREFIX)


def generate_evidence_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(EVIDENCE_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_evidence_id(id_str: str) -> None:
    validate_prefixed_id(id_str, EVIDENCE_ID_PREFIX)


def generate_artifact_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(ARTIFACT_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_artifact_id(id_str: str) -> None:
    validate_prefixed_id(id_str, ARTIFACT_ID_PREFIX)


def generate_merge_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(MERGE_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_merge_id(id_str: str) -> None:
    validate_prefixed_id(id_str, MERGE_ID_PREFIX)


def generate_incident_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(INCIDENT_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_incident_id(id_str: str) -> None:
    validate_prefixed_id(id_str, INCIDENT_ID_PREFIX)


def generate_event_id(
    *, timestamp_ms: int | None = None, randbytes: _RandBytes | None = None
) -> str:
    return generate_prefixed_id(EVENT_ID_PREFIX, timestamp_ms=timestamp_ms, randbytes=randbytes)


def validate_event_id(id_str: str) -> None:
    validate_prefixed_id(id_str, EVENT_ID_PREFIX)


def validate_requirement_id(requirement_id: str) -> None:
    """Validate canonical requirement IDs of the form ``REQ-0001``."""
    if not isinstance(requirement_id, str):
        raise ValueError(f"requirement_id must be a string, got {type(requirement_id).__name__}")

    match = _REQUIREMENT_ID_RE.fullmatch(requirement_id)
    if match is None:
        raise ValueError(
            f"requirement_id must match {REQ_ID_PATTERN_DESCRIPTION} (got {requirement_id!r})"
        )

    number = int(match.group(1))
    if number <= 0:
        raise ValueError("requirement_id sequence must be between 0001 and 9999")


def validate_constraint_id(constraint_id: str) -> None:
    """Validate canonical constraint IDs of the form ``CON-SEC-0001``."""
    if not isinstance(constraint_id, str):
        raise ValueError(f"constraint_id must be a string, got {type(constraint_id).__name__}")

    match = _CONSTRAINT_ID_RE.fullmatch(constraint_id)
    if match is None:
        raise ValueError(
            "constraint_id must match "
            f"{CONSTRAINT_ID_PATTERN_DESCRIPTION} with uppercase category tag (got {constraint_id!r})"
        )

    number = int(match.group(2))
    if number <= 0:
        raise ValueError("constraint_id sequence must be between 0001 and 9999")


# ------------------------
# Internal helper routines
# ------------------------


def _resolve_timestamp_ms(timestamp_ms: int | None) -> int:
    resolved = time.time_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms
    if not isinstance(resolved, int):
        raise ValueError(f"timestamp_ms must be an int, got {type(resolved).__name__}")
    if not 0 <= resolved <= ULID_MAX_TIMESTAMP_MS:
        raise ValueError(
            f"timestamp_ms out of range: expected 0..{ULID_MAX_TIMESTAMP_MS}, got {resolved}"
        )
    return resolved


def _resolve_random_bytes(randbytes: _RandBytes | None) -> bytes:
    provider = secrets.token_bytes if randbytes is None else randbytes
    raw = provider(ULID_RANDOM_BYTES)
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError("randbytes must return a bytes-like object")
    as_bytes = bytes(raw)
    if len(as_bytes) != ULID_RANDOM_BYTES:
        raise ValueError(f"randbytes must return exactly {ULID_RANDOM_BYTES} bytes")
    return as_bytes


def _decode_validated_ulid(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"ulid must be a string, got {type(value).__name__}")
    if len(value) != ULID_LENGTH:
        raise ValueError(f"ulid length must be {ULID_LENGTH}, got {len(value)}")

    decoded = 0
    for index, char in enumerate(value):
        normalized = char.upper()
        digit = _DECODE_TABLE.get(normalized)
        if digit is None:
            raise ValueError(f"invalid ULID character {char!r} at index {index}")
        decoded = (decoded << 5) | digit

    if decoded > _ULID_MAX_VALUE:
        raise ValueError("ulid overflow: value exceeds maximum 128-bit ULID")
    return decoded


def _encode_crockford_base32(value: int, length: int) -> str:
    if value < 0:
        raise ValueError("value must be non-negative")

    mask = 0b11111
    chars = ["0"] * length
    working = value
    for index in range(length - 1, -1, -1):
        chars[index] = CROCKFORD_BASE32_ALPHABET[working & mask]
        working >>= 5

    if working != 0:
        raise ValueError(f"value does not fit into {length} Crockford Base32 characters")
    return "".join(chars)


def _validate_prefix(prefix: str) -> None:
    if not isinstance(prefix, str):
        raise ValueError(f"prefix must be a string, got {type(prefix).__name__}")
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if _PREFIX_SEPARATOR in prefix:
        raise ValueError(f"prefix must not contain '{_PREFIX_SEPARATOR}'")
