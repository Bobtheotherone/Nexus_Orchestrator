"""
Integration test: spec ingestion through planning.

Purpose:
- Verify that minimal_design_doc.md produces a valid SpecMap.
- Verify that the SpecMap compiles into a valid TaskGraph with 3 work items.
- Verify constraint envelopes are complete and non-overlapping scopes.
"""
