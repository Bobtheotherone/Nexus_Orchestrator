"""
nexus-orchestrator — deterministic conflict resolution policy.

File: src/nexus_orchestrator/integration_plane/conflict_resolution.py
Last updated: 2026-02-13

Purpose
- Classify merge conflicts into trivial / non-trivial / contract-level buckets.
- Auto-resolve only provably equivalent trivial conflicts.
- Build deterministic integrator context payloads for non-trivial conflicts.

Functional requirements
- No creative merges: only side selection for proven-equivalent content.
- Contract-level conflicts must be escalated to re-planning.

Non-functional requirements
- Deterministic behavior and stable output ordering.
- Internal audit log with redacted/sanitized metadata (no raw secret payloads).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from enum import StrEnum
else:
    try:
        from enum import StrEnum
    except ImportError:

        class StrEnum(str, Enum):
            """Compatibility fallback for Python < 3.11."""


AuditValue = str | int | bool | None

_MAX_PATH_LEN: Final[int] = 1024
_MAX_BRANCH_LEN: Final[int] = 256
_MAX_LANGUAGE_LEN: Final[int] = 64
_MAX_CONTENT_BYTES: Final[int] = 5 * 1024 * 1024
_MAX_METADATA_ITEMS: Final[int] = 128
_MAX_WORK_ITEMS: Final[int] = 512
_MAX_WORK_ITEM_TEXT: Final[int] = 1024
_BINARY_LIKE_LINE_THRESHOLD: Final[int] = 4096

_SENSITIVE_KEY_TERMS: Final[tuple[str, ...]] = (
    "secret",
    "token",
    "password",
    "credential",
    "private_key",
    "api_key",
)
_REDACTED_AUDIT_VALUE: Final[str] = "***REDACTED***"

_CONTRACT_PATH_PREFIXES_DEFAULT: Final[tuple[str, ...]] = (
    "contract/",
    "contracts/",
    "_contracts/",
    "interfaces/",
)
_CONTRACT_FLAG_KEYS: Final[tuple[str, ...]] = (
    "contract_level",
    "contract_conflict",
    "contract_drift",
    "implicates_contract",
    "interface_contract_conflict",
    "requires_replan",
)
_LANGUAGE_HINT_PYTHON: Final[frozenset[str]] = frozenset({"python", "py"})
_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off"})


class ConflictClassification(StrEnum):
    """Stable conflict classification buckets."""

    TRIVIAL = "trivial"
    NON_TRIVIAL = "non_trivial"
    CONTRACT_LEVEL = "contract_level"


class TrivialConflictProof(StrEnum):
    """Proof class for conservative auto-resolution."""

    NONE = "none"
    EXACT_TEXT_MATCH = "exact_text_match"
    WHITESPACE_ONLY = "whitespace_only"
    PYTHON_IMPORT_ORDER_ONLY = "python_import_order_only"
    PYTHON_AST_EQUIVALENT = "python_ast_equivalent"


class ResolutionStatus(StrEnum):
    """Auto-resolution outcome state."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ConflictInput:
    """Normalized input payload for one merge conflict."""

    path: str
    ours_content: str
    theirs_content: str
    base_content: str | None = None
    ours_branch: str | None = None
    theirs_branch: str | None = None
    language: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_hint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_path(self.path, "ConflictInput.path"))
        object.__setattr__(
            self,
            "ours_content",
            _normalize_content(self.ours_content, "ConflictInput.ours_content"),
        )
        object.__setattr__(
            self,
            "theirs_content",
            _normalize_content(self.theirs_content, "ConflictInput.theirs_content"),
        )
        object.__setattr__(
            self,
            "base_content",
            _normalize_optional_content(self.base_content, "ConflictInput.base_content"),
        )
        object.__setattr__(
            self,
            "ours_branch",
            _normalize_optional_short_text(
                self.ours_branch,
                "ConflictInput.ours_branch",
                max_len=_MAX_BRANCH_LEN,
            ),
        )
        object.__setattr__(
            self,
            "theirs_branch",
            _normalize_optional_short_text(
                self.theirs_branch,
                "ConflictInput.theirs_branch",
                max_len=_MAX_BRANCH_LEN,
            ),
        )
        object.__setattr__(
            self,
            "language",
            _normalize_optional_short_text(
                self.language,
                "ConflictInput.language",
                max_len=_MAX_LANGUAGE_LEN,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_normalize_metadata(self.metadata, "ConflictInput.metadata")),
        )
        object.__setattr__(self, "contract_hint", _coerce_bool(self.contract_hint, "contract_hint"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ConflictInput:
        """Parse common conflict payload shapes into a normalized dataclass."""

        path = _require_one_of(
            payload,
            ("path", "file_path", "target_path", "relative_path"),
            field_name="ConflictInput.path",
        )
        ours_content = _require_one_of(
            payload,
            ("ours_content", "ours", "current", "local", "left"),
            field_name="ConflictInput.ours_content",
        )
        theirs_content = _require_one_of(
            payload,
            ("theirs_content", "theirs", "incoming", "remote", "right"),
            field_name="ConflictInput.theirs_content",
        )

        base_content = _optional_str_one_of(
            payload,
            ("base_content", "base", "ancestor", "original"),
            field_name="ConflictInput.base_content",
        )
        ours_branch = _optional_str_one_of(
            payload,
            ("ours_branch", "source_branch", "left_branch"),
            field_name="ConflictInput.ours_branch",
        )
        theirs_branch = _optional_str_one_of(
            payload,
            ("theirs_branch", "target_branch", "right_branch"),
            field_name="ConflictInput.theirs_branch",
        )
        language = _optional_str_one_of(
            payload,
            ("language", "file_language"),
            field_name="ConflictInput.language",
        )
        metadata_raw = payload.get("metadata", {})
        contract_hint_raw = _optional_one_of(
            payload,
            ("contract_hint", "contract_level", "implicates_contract"),
        )
        contract_hint = _coerce_optional_bool(contract_hint_raw, default=False)

        return cls(
            path=path,
            ours_content=ours_content,
            theirs_content=theirs_content,
            base_content=base_content,
            ours_branch=ours_branch,
            theirs_branch=theirs_branch,
            language=language,
            metadata={} if metadata_raw is None else _as_mapping(metadata_raw, "metadata"),
            contract_hint=contract_hint,
        )


@dataclass(frozen=True, slots=True)
class ConflictResolutionResult:
    """Typed resolution result payload."""

    path: str
    classification: ConflictClassification
    proof: TrivialConflictProof
    status: ResolutionStatus
    reason: str
    requires_replan: bool
    resolved_content: str | None = None
    patch: str | None = None
    patch_from: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _normalize_relative_path(self.path, "ConflictResolutionResult.path"),
        )
        object.__setattr__(
            self,
            "classification",
            _coerce_enum(
                self.classification,
                ConflictClassification,
                "ConflictResolutionResult.classification",
            ),
        )
        object.__setattr__(
            self,
            "proof",
            _coerce_enum(self.proof, TrivialConflictProof, "ConflictResolutionResult.proof"),
        )
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, ResolutionStatus, "ConflictResolutionResult.status"),
        )
        object.__setattr__(
            self,
            "reason",
            _normalize_short_text(
                self.reason,
                "ConflictResolutionResult.reason",
                max_len=512,
            ),
        )
        object.__setattr__(
            self,
            "requires_replan",
            _coerce_bool(self.requires_replan, "ConflictResolutionResult.requires_replan"),
        )
        object.__setattr__(
            self,
            "resolved_content",
            _normalize_optional_content(
                self.resolved_content,
                "ConflictResolutionResult.resolved_content",
            ),
        )
        object.__setattr__(
            self,
            "patch",
            _normalize_optional_content(self.patch, "ConflictResolutionResult.patch"),
        )
        object.__setattr__(
            self,
            "patch_from",
            _normalize_optional_short_text(
                self.patch_from,
                "ConflictResolutionResult.patch_from",
                max_len=32,
            ),
        )

        if self.status is ResolutionStatus.RESOLVED:
            if self.classification is not ConflictClassification.TRIVIAL:
                raise ValueError(
                    "ConflictResolutionResult: resolved status requires trivial classification"
                )
            if self.resolved_content is None:
                raise ValueError(
                    "ConflictResolutionResult: resolved status requires resolved_content"
                )
        else:
            if self.resolved_content is not None:
                raise ValueError(
                    "ConflictResolutionResult: unresolved status must not include resolved_content"
                )
            if self.patch is not None or self.patch_from is not None:
                raise ValueError(
                    "ConflictResolutionResult: unresolved status must not include patch data"
                )

        if self.patch_from is not None and self.patch is None:
            raise ValueError("ConflictResolutionResult: patch_from requires patch content")


@dataclass(frozen=True, slots=True)
class ConflictAuditEntry:
    """Internal, append-only audit log record."""

    sequence: int
    action: str
    path: str
    classification: ConflictClassification | None
    proof: TrivialConflictProof | None
    details: Mapping[str, AuditValue]

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("ConflictAuditEntry.sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("ConflictAuditEntry.sequence must be >= 1")

        object.__setattr__(
            self,
            "action",
            _normalize_short_text(self.action, "ConflictAuditEntry.action", max_len=128),
        )
        object.__setattr__(
            self,
            "path",
            _normalize_relative_path(self.path, "ConflictAuditEntry.path"),
        )
        if self.classification is not None:
            object.__setattr__(
                self,
                "classification",
                _coerce_enum(
                    self.classification,
                    ConflictClassification,
                    "ConflictAuditEntry.classification",
                ),
            )
        if self.proof is not None:
            object.__setattr__(
                self,
                "proof",
                _coerce_enum(self.proof, TrivialConflictProof, "ConflictAuditEntry.proof"),
            )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_normalize_audit_details(self.details)),
        )

    def to_dict(self) -> dict[str, AuditValue]:
        return {
            "sequence": self.sequence,
            "action": self.action,
            "path": self.path,
            "classification": None if self.classification is None else self.classification.value,
            "proof": None if self.proof is None else self.proof.value,
            "details": str(dict(self.details)),
        }


class ConflictResolver:
    """Conservative conflict classifier and trivial auto-resolver."""

    def __init__(
        self,
        *,
        contract_path_prefixes: Sequence[str] | None = None,
    ) -> None:
        prefixes_raw = (
            _CONTRACT_PATH_PREFIXES_DEFAULT
            if contract_path_prefixes is None
            else tuple(contract_path_prefixes)
        )
        self._contract_path_prefixes = _normalize_contract_path_prefixes(prefixes_raw)
        self._audit_log: list[ConflictAuditEntry] = []

    @property
    def audit_log(self) -> tuple[ConflictAuditEntry, ...]:
        """Return a read-only snapshot of the internal audit log."""

        return tuple(self._audit_log)

    def classify_conflict(
        self,
        conflict: ConflictInput | Mapping[str, object],
    ) -> ConflictClassification:
        conflict_input = _coerce_conflict_input(conflict)
        classification, proof = self._classify(conflict_input)
        self._append_audit(
            action="classify_conflict",
            conflict=conflict_input,
            classification=classification,
            proof=proof,
            details={
                "requires_replan": classification is ConflictClassification.CONTRACT_LEVEL,
                "ours_fp": _content_fingerprint(conflict_input.ours_content),
                "theirs_fp": _content_fingerprint(conflict_input.theirs_content),
            },
        )
        return classification

    def auto_resolve_trivial(
        self,
        conflict: ConflictInput | Mapping[str, object],
    ) -> ConflictResolutionResult:
        conflict_input = _coerce_conflict_input(conflict)
        classification, proof = self._classify(conflict_input)

        if classification is not ConflictClassification.TRIVIAL:
            result = ConflictResolutionResult(
                path=conflict_input.path,
                classification=classification,
                proof=proof,
                status=ResolutionStatus.UNRESOLVED,
                reason="conflict is not provably trivial",
                requires_replan=classification is ConflictClassification.CONTRACT_LEVEL,
            )
            self._append_audit(
                action="auto_resolve_trivial",
                conflict=conflict_input,
                classification=classification,
                proof=proof,
                details={"resolved": False, "reason": result.reason},
            )
            return result

        resolved_content, selected_side = _select_deterministic_content(conflict_input)
        patch, patch_from = _build_patch_for_resolution(conflict_input, resolved_content)
        result = ConflictResolutionResult(
            path=conflict_input.path,
            classification=classification,
            proof=proof,
            status=ResolutionStatus.RESOLVED,
            reason=f"resolved by deterministic side selection ({selected_side})",
            requires_replan=False,
            resolved_content=resolved_content,
            patch=patch,
            patch_from=patch_from,
        )
        self._append_audit(
            action="auto_resolve_trivial",
            conflict=conflict_input,
            classification=classification,
            proof=proof,
            details={
                "resolved": True,
                "selected_side": selected_side,
                "patch_from": patch_from or "none",
                "resolved_fp": _content_fingerprint(resolved_content),
            },
        )
        return result

    def prepare_integrator_context(
        self,
        conflict: ConflictInput | Mapping[str, object],
        relevant_work_items: Sequence[object],
    ) -> dict[str, object]:
        conflict_input = _coerce_conflict_input(conflict)
        work_items = _normalize_work_items(relevant_work_items)
        classification, proof = self._classify(conflict_input)
        replan_needed = classification is ConflictClassification.CONTRACT_LEVEL

        context: dict[str, object] = {
            "conflict_path": conflict_input.path,
            "classification": classification.value,
            "trivial_proof": None if proof is TrivialConflictProof.NONE else proof.value,
            "requires_replan": replan_needed,
            "branches": {
                "ours": conflict_input.ours_branch,
                "theirs": conflict_input.theirs_branch,
            },
            "content_fingerprints": {
                "base": (
                    None
                    if conflict_input.base_content is None
                    else _content_fingerprint(conflict_input.base_content)
                ),
                "ours": _content_fingerprint(conflict_input.ours_content),
                "theirs": _content_fingerprint(conflict_input.theirs_content),
            },
            "diff_stats": {
                "ours_vs_theirs_changed_lines": _changed_line_count(
                    conflict_input.ours_content,
                    conflict_input.theirs_content,
                ),
                "base_vs_ours_changed_lines": (
                    None
                    if conflict_input.base_content is None
                    else _changed_line_count(
                        conflict_input.base_content, conflict_input.ours_content
                    )
                ),
                "base_vs_theirs_changed_lines": (
                    None
                    if conflict_input.base_content is None
                    else _changed_line_count(
                        conflict_input.base_content,
                        conflict_input.theirs_content,
                    )
                ),
            },
            "relevant_work_items": list(work_items),
            "recommended_actions": _recommended_actions(
                classification=classification,
                has_base=conflict_input.base_content is not None,
            ),
        }
        self._append_audit(
            action="prepare_integrator_context",
            conflict=conflict_input,
            classification=classification,
            proof=proof,
            details={
                "requires_replan": replan_needed,
                "work_item_count": len(work_items),
            },
        )
        return context

    def requires_replan(self, conflict: ConflictInput | Mapping[str, object]) -> bool:
        conflict_input = _coerce_conflict_input(conflict)
        classification, proof = self._classify(conflict_input)
        needs_replan = classification is ConflictClassification.CONTRACT_LEVEL
        self._append_audit(
            action="requires_replan",
            conflict=conflict_input,
            classification=classification,
            proof=proof,
            details={"requires_replan": needs_replan},
        )
        return needs_replan

    def _classify(
        self, conflict: ConflictInput
    ) -> tuple[ConflictClassification, TrivialConflictProof]:
        if _is_contract_level_conflict(
            conflict,
            contract_path_prefixes=self._contract_path_prefixes,
        ):
            return ConflictClassification.CONTRACT_LEVEL, TrivialConflictProof.NONE

        proof = _trivial_proof(conflict)
        if proof is TrivialConflictProof.NONE:
            return ConflictClassification.NON_TRIVIAL, proof
        return ConflictClassification.TRIVIAL, proof

    def _append_audit(
        self,
        *,
        action: str,
        conflict: ConflictInput,
        classification: ConflictClassification | None,
        proof: TrivialConflictProof | None,
        details: Mapping[str, object],
    ) -> None:
        sanitized_details = _normalize_audit_details(details)
        entry = ConflictAuditEntry(
            sequence=len(self._audit_log) + 1,
            action=action,
            path=conflict.path,
            classification=classification,
            proof=proof,
            details=sanitized_details,
        )
        self._audit_log.append(entry)


def _coerce_conflict_input(conflict: ConflictInput | Mapping[str, object]) -> ConflictInput:
    if isinstance(conflict, ConflictInput):
        return conflict
    if isinstance(conflict, Mapping):
        return ConflictInput.from_mapping(conflict)
    raise TypeError("conflict must be ConflictInput or Mapping[str, object]")


def _normalize_contract_path_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, value in enumerate(prefixes):
        path = _normalize_relative_path(value, f"contract_path_prefixes[{index}]")
        if not path.endswith("/"):
            path = f"{path}/"
        normalized.append(path.casefold())
    if not normalized:
        raise ValueError("contract_path_prefixes must include at least one prefix")
    return tuple(sorted(set(normalized)))


def _is_contract_level_conflict(
    conflict: ConflictInput,
    *,
    contract_path_prefixes: Sequence[str],
) -> bool:
    if conflict.contract_hint:
        return True

    metadata_flags = [
        _coerce_optional_bool(conflict.metadata.get(key), default=False)
        for key in _CONTRACT_FLAG_KEYS
    ]
    if any(metadata_flags):
        return True

    path_lower = conflict.path.casefold()
    return any(path_lower.startswith(prefix) for prefix in contract_path_prefixes)


def _trivial_proof(conflict: ConflictInput) -> TrivialConflictProof:
    ours = conflict.ours_content
    theirs = conflict.theirs_content
    python_pair: tuple[ast.Module, ast.Module] | None = None

    if _looks_binary_like(ours) or _looks_binary_like(theirs):
        return TrivialConflictProof.NONE

    if _is_python_conflict(conflict):
        python_pair = _parse_python_pair(ours, theirs)
        if python_pair is None:
            return TrivialConflictProof.NONE

    if ours == theirs:
        return TrivialConflictProof.EXACT_TEXT_MATCH
    if _is_whitespace_only_change(ours, theirs):
        return TrivialConflictProof.WHITESPACE_ONLY
    if python_pair is not None:
        left_tree, right_tree = python_pair
        if _is_python_import_order_only_trees(left_tree, right_tree):
            return TrivialConflictProof.PYTHON_IMPORT_ORDER_ONLY
        if _is_python_ast_equivalent_trees(left_tree, right_tree):
            return TrivialConflictProof.PYTHON_AST_EQUIVALENT
    return TrivialConflictProof.NONE


def _is_whitespace_only_change(left: str, right: str) -> bool:
    if left == right:
        return False

    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if not _is_whitespace_segment(left[i1:i2]):
            return False
        if not _is_whitespace_segment(right[j1:j2]):
            return False
    return True


def _is_whitespace_segment(value: str) -> bool:
    return value == "" or value.isspace()


def _is_python_conflict(conflict: ConflictInput) -> bool:
    if conflict.path.endswith(".py"):
        return True
    if conflict.language is None:
        return False
    return conflict.language.casefold() in _LANGUAGE_HINT_PYTHON


def _is_python_ast_equivalent(left: str, right: str) -> bool:
    pair = _parse_python_pair(left, right)
    if pair is None:
        return False
    return _is_python_ast_equivalent_trees(*pair)


def _is_python_ast_equivalent_trees(left_tree: ast.Module, right_tree: ast.Module) -> bool:
    return _ast_dump(left_tree) == _ast_dump(right_tree)


def _is_python_import_order_only(left: str, right: str) -> bool:
    pair = _parse_python_pair(left, right)
    if pair is None:
        return False
    return _is_python_import_order_only_trees(*pair)


def _is_python_import_order_only_trees(left_tree: ast.Module, right_tree: ast.Module) -> bool:
    left_prefix, left_imports, left_rest = _split_initial_import_section(left_tree.body)
    right_prefix, right_imports, right_rest = _split_initial_import_section(right_tree.body)
    if not left_imports or not right_imports:
        return False

    if _dump_stmt_list(left_prefix) != _dump_stmt_list(right_prefix):
        return False
    if _dump_stmt_list(left_rest) != _dump_stmt_list(right_rest):
        return False

    left_import_signatures = sorted(_canonical_import_signature(node) for node in left_imports)
    right_import_signatures = sorted(_canonical_import_signature(node) for node in right_imports)
    if left_import_signatures != right_import_signatures:
        return False

    # Ensure there is an actual ordering delta; exact match is handled separately.
    left_ordered = tuple(
        _canonical_import_signature(node, sort_aliases=False) for node in left_imports
    )
    right_ordered = tuple(
        _canonical_import_signature(node, sort_aliases=False) for node in right_imports
    )
    return left_ordered != right_ordered


def _parse_python_pair(left: str, right: str) -> tuple[ast.Module, ast.Module] | None:
    try:
        left_tree = ast.parse(left)
        right_tree = ast.parse(right)
    except SyntaxError:
        return None
    return left_tree, right_tree


def _looks_binary_like(content: str) -> bool:
    if "\x00" not in content:
        return False
    return any(
        len(line) >= _BINARY_LIKE_LINE_THRESHOLD for line in content.splitlines(keepends=False)
    )


def _split_initial_import_section(
    body: Sequence[ast.stmt],
) -> tuple[tuple[ast.stmt, ...], tuple[ast.stmt, ...], tuple[ast.stmt, ...]]:
    prefix: list[ast.stmt] = []
    imports: list[ast.stmt] = []

    index = 0
    if body and _is_docstring_stmt(body[0]):
        prefix.append(body[0])
        index = 1

    while index < len(body) and isinstance(body[index], (ast.Import, ast.ImportFrom)):
        imports.append(body[index])
        index += 1

    rest = list(body[index:])
    return tuple(prefix), tuple(imports), tuple(rest)


def _is_docstring_stmt(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    if not isinstance(node.value, ast.Constant):
        return False
    return isinstance(node.value.value, str)


def _canonical_import_signature(
    node: ast.stmt,
    *,
    sort_aliases: bool = True,
) -> tuple[str, str, int, tuple[tuple[str, str], ...]]:
    if isinstance(node, ast.Import):
        aliases = tuple((item.name, item.asname or "") for item in node.names)
        if sort_aliases:
            aliases = tuple(sorted(aliases))
        return ("import", "", 0, aliases)
    if isinstance(node, ast.ImportFrom):
        aliases = tuple((item.name, item.asname or "") for item in node.names)
        if sort_aliases:
            aliases = tuple(sorted(aliases))
        return ("from", node.module or "", int(node.level), aliases)
    raise TypeError("canonical import signature requires ast.Import or ast.ImportFrom")


def _dump_stmt_list(nodes: Sequence[ast.stmt]) -> tuple[str, ...]:
    return tuple(_ast_dump(node) for node in nodes)


def _ast_dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _select_deterministic_content(conflict: ConflictInput) -> tuple[str, str]:
    if conflict.base_content is not None:
        if (
            conflict.ours_content == conflict.base_content
            and conflict.theirs_content != conflict.base_content
        ):
            return conflict.ours_content, "ours_matches_base"
        if (
            conflict.theirs_content == conflict.base_content
            and conflict.ours_content != conflict.base_content
        ):
            return conflict.theirs_content, "theirs_matches_base"

    ours_len = len(conflict.ours_content)
    theirs_len = len(conflict.theirs_content)
    if ours_len < theirs_len:
        return conflict.ours_content, "shorter_content_ours"
    if theirs_len < ours_len:
        return conflict.theirs_content, "shorter_content_theirs"

    ours_hash = hashlib.sha256(_utf8_bytes(conflict.ours_content)).hexdigest()
    theirs_hash = hashlib.sha256(_utf8_bytes(conflict.theirs_content)).hexdigest()
    if ours_hash <= theirs_hash:
        return conflict.ours_content, "sha256_tiebreak_ours"
    return conflict.theirs_content, "sha256_tiebreak_theirs"


def _build_patch_for_resolution(
    conflict: ConflictInput,
    resolved_content: str,
) -> tuple[str | None, str | None]:
    if conflict.base_content is not None and conflict.base_content != resolved_content:
        patch = _unified_patch(
            path=conflict.path,
            before=conflict.base_content,
            after=resolved_content,
            from_label="base",
            to_label="resolved",
        )
        if patch is not None:
            return patch, "base"

    if conflict.ours_content != resolved_content:
        patch = _unified_patch(
            path=conflict.path,
            before=conflict.ours_content,
            after=resolved_content,
            from_label="ours",
            to_label="resolved",
        )
        if patch is not None:
            return patch, "ours"

    if conflict.theirs_content != resolved_content:
        patch = _unified_patch(
            path=conflict.path,
            before=conflict.theirs_content,
            after=resolved_content,
            from_label="theirs",
            to_label="resolved",
        )
        if patch is not None:
            return patch, "theirs"

    return None, None


def _unified_patch(
    *,
    path: str,
    before: str,
    after: str,
    from_label: str,
    to_label: str,
) -> str | None:
    if before == after:
        return None

    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{from_label}/{path}",
            tofile=f"{to_label}/{path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return None
    return "\n".join(diff_lines) + "\n"


def _changed_line_count(left: str, right: str) -> int:
    changed = 0
    matcher = difflib.SequenceMatcher(
        a=left.splitlines(),
        b=right.splitlines(),
        autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _normalize_work_items(relevant_work_items: Sequence[object]) -> tuple[dict[str, object], ...]:
    if isinstance(relevant_work_items, (str, bytes, bytearray)):
        raise TypeError("relevant_work_items must be a sequence of work item objects")
    if len(relevant_work_items) > _MAX_WORK_ITEMS:
        raise ValueError(f"relevant_work_items must contain <= {_MAX_WORK_ITEMS} items")

    summaries: list[dict[str, object]] = []
    for index, item in enumerate(relevant_work_items):
        summary = _normalize_single_work_item(item, index)
        summaries.append(summary)

    summaries.sort(key=lambda payload: str(payload["id"]))
    return tuple(summaries)


def _normalize_single_work_item(item: object, index: int) -> dict[str, object]:
    item_map = _as_mapping_or_object_dict(item, f"relevant_work_items[{index}]")

    work_item_id = _normalize_short_text(
        _read_first_existing(
            item_map, ("id", "work_item_id"), field_path=f"work_items[{index}].id"
        ),
        f"work_items[{index}].id",
        max_len=_MAX_WORK_ITEM_TEXT,
    )
    title = _normalize_optional_short_text(
        item_map.get("title"),
        f"work_items[{index}].title",
        max_len=_MAX_WORK_ITEM_TEXT,
    )
    status = _normalize_optional_short_text(
        item_map.get("status"),
        f"work_items[{index}].status",
        max_len=128,
    )
    risk_tier = _normalize_optional_short_text(
        item_map.get("risk_tier"),
        f"work_items[{index}].risk_tier",
        max_len=64,
    )
    scope = _normalize_optional_str_sequence(
        item_map.get("scope"),
        f"work_items[{index}].scope",
    )
    dependencies = _normalize_optional_str_sequence(
        item_map.get("dependencies"),
        f"work_items[{index}].dependencies",
    )
    requirement_links = _normalize_optional_str_sequence(
        item_map.get("requirement_links"),
        f"work_items[{index}].requirement_links",
    )

    return {
        "id": work_item_id,
        "title": title,
        "status": status,
        "risk_tier": risk_tier,
        "scope": scope,
        "dependencies": dependencies,
        "requirement_links": requirement_links,
    }


def _as_mapping_or_object_dict(item: object, field_path: str) -> Mapping[str, object]:
    if isinstance(item, Mapping):
        return item

    attributes = {}
    for key in (
        "id",
        "work_item_id",
        "title",
        "status",
        "risk_tier",
        "scope",
        "dependencies",
        "requirement_links",
    ):
        if hasattr(item, key):
            attributes[key] = getattr(item, key)
    if attributes:
        return attributes
    raise TypeError(f"{field_path} must be a mapping or object with work-item fields")


def _recommended_actions(
    *,
    classification: ConflictClassification,
    has_base: bool,
) -> tuple[str, ...]:
    if classification is ConflictClassification.CONTRACT_LEVEL:
        return (
            "Escalate to architect re-plan before attempting file-level merge.",
            "Freeze interface contract edits until re-plan output is approved.",
            "Record impacted work-item IDs and contract paths for planner input.",
        )
    if classification is ConflictClassification.TRIVIAL:
        return (
            "Auto-resolution is safe; apply deterministic resolved content.",
            "Run quick verification for this path before merge queue admission.",
        )
    if has_base:
        return (
            "Review ours/theirs against base to identify true intent deltas.",
            "Preserve both work-item constraint envelopes while resolving.",
            "Limit edits to the conflicted file unless constraints require otherwise.",
        )
    return (
        "Review both sides directly because no merge base content was provided.",
        "Preserve both work-item constraint envelopes while resolving.",
        "Limit edits to the conflicted file unless constraints require otherwise.",
    )


def _content_fingerprint(content: str) -> str:
    payload = _utf8_bytes(content)
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}:bytes:{len(payload)}"


def _normalize_audit_details(details: Mapping[str, object]) -> dict[str, AuditValue]:
    out: dict[str, AuditValue] = {}
    for key in sorted(details):
        normalized_key = _normalize_short_text(key, f"audit.details[{key!r}].key", max_len=128)
        raw_value = details[key]
        out[normalized_key] = _sanitize_audit_value(normalized_key, raw_value)
    return out


def _sanitize_audit_value(key: str, value: object) -> AuditValue:
    key_lower = key.casefold()
    if any(term in key_lower for term in _SENSITIVE_KEY_TERMS):
        return _REDACTED_AUDIT_VALUE

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        if len(value) <= 96:
            return value
        digest = hashlib.sha256(_utf8_bytes(value)).hexdigest()[:16]
        return f"<sha256:{digest}:len={len(value)}>"
    if isinstance(value, Mapping):
        return f"<mapping:{len(value)}>"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return f"<sequence:{len(value)}>"
    return f"<{type(value).__name__}>"


def _normalize_relative_path(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.replace("\\", "/").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > _MAX_PATH_LEN:
        raise ValueError(f"{field_name} must be <= {_MAX_PATH_LEN} characters")
    if "\x00" in text:
        raise ValueError(f"{field_name} must not contain NUL bytes")

    pure = PurePosixPath(text)
    if pure.is_absolute():
        raise ValueError(f"{field_name} must be a relative path")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"{field_name} must not contain parent traversal")
    canonical = pure.as_posix()
    if canonical in {"", "."}:
        raise ValueError(f"{field_name} must point to a concrete file path")
    return canonical


def _normalize_content(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    byte_size = len(_utf8_bytes(value))
    if byte_size > _MAX_CONTENT_BYTES:
        raise ValueError(f"{field_name} exceeds max size {_MAX_CONTENT_BYTES} bytes")
    return value


def _utf8_bytes(value: str) -> bytes:
    return value.encode("utf-8", errors="surrogatepass")


def _normalize_optional_content(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_content(value, field_name)


def _normalize_short_text(value: object, field_name: str, *, max_len: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > max_len:
        raise ValueError(f"{field_name} must be <= {max_len} characters")
    if "\x00" in text:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return text


def _normalize_optional_short_text(
    value: object,
    field_name: str,
    *,
    max_len: int,
) -> str | None:
    if value is None:
        return None
    return _normalize_short_text(value, field_name, max_len=max_len)


def _normalize_metadata(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if len(value) > _MAX_METADATA_ITEMS:
        raise ValueError(f"{field_name} must contain <= {_MAX_METADATA_ITEMS} entries")

    normalized: dict[str, object] = {}
    for key in sorted(value):
        normalized_key = _normalize_short_text(
            key,
            f"{field_name}.key",
            max_len=128,
        )
        normalized[normalized_key] = value[key]
    return normalized


def _normalize_optional_str_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings, not a scalar string")
    if not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")

    out: list[str] = []
    for index, raw in enumerate(value):
        out.append(
            _normalize_short_text(
                raw,
                f"{field_name}[{index}]",
                max_len=_MAX_WORK_ITEM_TEXT,
            )
        )
    return tuple(sorted(set(out)))


def _coerce_enum(value: object, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be {enum_type.__name__} or str")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name}: invalid value {value!r}; allowed: {allowed}") from exc


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _coerce_optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise TypeError("boolean flag values must be bool or bool-like strings")


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return value


def _require_one_of(payload: Mapping[str, object], keys: Sequence[str], *, field_name: str) -> str:
    value = _optional_one_of(payload, keys)
    if value is None:
        choices = ", ".join(keys)
        raise ValueError(f"{field_name} is required (accepted keys: {choices})")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _optional_one_of(payload: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _optional_str_one_of(
    payload: Mapping[str, object],
    keys: Sequence[str],
    *,
    field_name: str,
) -> str | None:
    value = _optional_one_of(payload, keys)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _read_first_existing(
    item_map: Mapping[str, object],
    keys: Sequence[str],
    *,
    field_path: str,
) -> object:
    for key in keys:
        if key in item_map:
            return item_map[key]
    joined = ", ".join(keys)
    raise ValueError(f"{field_path} is required (accepted keys: {joined})")


__all__ = [
    "ConflictAuditEntry",
    "ConflictClassification",
    "ConflictInput",
    "ConflictResolutionResult",
    "ConflictResolver",
    "ResolutionStatus",
    "TrivialConflictProof",
]
