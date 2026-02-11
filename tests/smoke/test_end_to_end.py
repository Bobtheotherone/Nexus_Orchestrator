"""
Smoke test: full pipeline with mocked providers.

Purpose:
- Verify the end-to-end lifecycle: ingest → plan → dispatch → verify → merge.
- Uses samples/specs/minimal_design_doc.md as input.
- All provider calls are mocked (no real API calls).
- Verifies that evidence is recorded and the merge queue produces correct state.

This test should pass before any phase is considered complete.
"""
