"""
Exploration mode — parallel candidate evaluation.

Behavior:
- For ambiguous requirements, dispatch multiple agents with different strategies.
- Each candidate independently verified against constraint gate.
- Best candidate selected (or tradeoffs presented to operator).
- Only the selected candidate merges. Others are discarded with evidence preserved.
"""
