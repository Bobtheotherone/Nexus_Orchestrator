<!--
nexus-orchestrator — audit command guidance

File: docs/quality/AUDIT.md
Last updated: 2026-02-13

Purpose
- Define deterministic, offline-friendly audit commands for placeholder scanning.

What should be included in this file
- WSL-friendly copy/paste commands.
- Canonical placeholder scan command usage (text, JSON, strict modes).
- A minimal local demo that deterministically emits one warning and one error.

Functional requirements
- Replace primitive `rg` placeholder checks with semantic scanning via `scripts/audit_placeholders.py`.
- Keep all examples runnable from repository root without network access.

Non-functional requirements
- Commands should be stable and shell-portable for WSL bash usage.
-->

# Audit Commands

Run all commands from repository root in WSL bash.

## Placeholder Scan (Semantic, Canonical)

Do not use primitive grep/regex checks such as:

```bash
rg -n "TODO|FIXME|NotImplementedError|skeleton" src/
```

Use the semantic scanner instead:

```bash
python scripts/audit_placeholders.py --roots src scripts --format text
```

## Default Audit Command

Use this for normal local/CI placeholder checks:

```bash
python scripts/audit_placeholders.py --roots src scripts --format text
```

## Output Modes

Text mode:

```bash
python scripts/audit_placeholders.py --roots src scripts --format text
```

JSON mode:

```bash
python scripts/audit_placeholders.py --roots src scripts --format json
```

Strict mode (fail on warnings and errors):

```bash
python scripts/audit_placeholders.py --roots src scripts --format text --fail-on-warn
```

## Paranoia Mode (Expanded Warning Surface)

Enable optional warning classes for string markers and audit-tool self-references:

```bash
python scripts/audit_placeholders.py --roots src scripts --format text --warn-on-string-markers --warn-on-audit-tool-self-references
```

Paranoia + strict fail policy:

```bash
python scripts/audit_placeholders.py --roots src scripts --format text --warn-on-string-markers --warn-on-audit-tool-self-references --fail-on-warn
```

Paranoia JSON output:

```bash
python scripts/audit_placeholders.py --roots src scripts --format json --warn-on-string-markers --warn-on-audit-tool-self-references
```

## Deterministic Temp-File Demo

```bash
TMP_DIR="$(mktemp -d)"
mkdir -p "$TMP_DIR/src"

cat > "$TMP_DIR/src/warn_case.py" <<'PY'
MESSAGE = "TODO appears in string data"
PY

cat > "$TMP_DIR/src/error_case.py" <<'PY'
# TODO: replace implementation

def run() -> None:
    return None
PY

python scripts/audit_placeholders.py --repo-root "$TMP_DIR" --roots src --format text
python scripts/audit_placeholders.py --repo-root "$TMP_DIR" --roots src --format text --warn-on-string-markers
python scripts/audit_placeholders.py --repo-root "$TMP_DIR" --roots src --format json --warn-on-string-markers
python scripts/audit_placeholders.py --repo-root "$TMP_DIR" --roots src --format text --warn-on-string-markers --fail-on-warn
echo "strict_exit_code=$?"

rm -rf "$TMP_DIR"
```

Expected result shape:
- Default mode reports only `error_case.py` as `ERROR` (`todo_fixme_comment`).
- With `--warn-on-string-markers`, `warn_case.py` is a `WARNING` (`placeholder_string_literal`).
- Strict mode (`--fail-on-warn`) returns non-zero when warnings or errors exist.

## Warning Interpretation Guidance

- `WARNING` means "review required", not an automatic failure, unless `--fail-on-warn` is set.
- `--warn-on-string-markers` surfaces placeholder marker text found in string data; this is useful for conservative audits but can include intentional literals.
- `--warn-on-audit-tool-self-references` surfaces marker patterns found inside the audit tooling/docs itself; treat these as hygiene signals and verify intent before suppressing.
- If a warning is intentional, document the reason near the code/doc line so future audits stay deterministic.
