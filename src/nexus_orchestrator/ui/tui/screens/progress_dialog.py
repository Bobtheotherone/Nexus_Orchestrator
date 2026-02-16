"""Progress dialog — select an existing ECO (Engineering Change Order) to continue.

File: src/nexus_orchestrator/ui/tui/screens/progress_dialog.py

Shows recent runs grouped by spec path (each unique spec = one ECO).
Filters to actionable ECOs (failed, paused, running — not completed/cancelled).
On selection, dismisses with ``/run run <spec_path> --resume``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


# Actionable run statuses (can be resumed/continued)
_ACTIONABLE_STATUSES = frozenset({"failed", "paused", "running", "planning", "created"})


class ProgressDialog(ModalScreen[str | None]):
    """Modal: select an ECO (by spec path) to continue working on."""

    DEFAULT_CSS = """
    ProgressDialog {
        align: center middle;
    }
    #progress-box {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: #0b1020;
        border: solid #3fa9f5;
        padding: 2;
    }
    #progress-title {
        text-style: bold;
        color: #3fa9f5;
        padding: 0 0 1 0;
    }
    #progress-empty {
        color: #7f8aa3;
        padding: 1 0;
    }
    .eco-btn {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: transparent;
        border: none;
        content-align: left middle;
    }
    .eco-btn:hover {
        background: #111830;
    }
    .eco-btn:focus {
        background: #111830;
        text-style: bold;
    }
    #progress-cancel {
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, state_db_path: str | Path) -> None:
        super().__init__()
        self._state_db_path = Path(state_db_path)

    def compose(self) -> ComposeResult:
        ecos = self._load_ecos()

        with Vertical(id="progress-box"):
            yield Static("Progress \u2014 Continue an ECO", id="progress-title")

            if not ecos:
                yield Static(
                    "No actionable ECOs found.\n"
                    "Run a plan first to create work items.",
                    id="progress-empty",
                )
            else:
                for i, eco in enumerate(ecos):
                    spec_short = eco["spec_path"]
                    if len(spec_short) > 50:
                        spec_short = "\u2026" + spec_short[-48:]
                    status = eco["status"]
                    work_info = eco["work_info"]
                    label = f"{spec_short}\n  Status: {status}  {work_info}"
                    yield Button(
                        label,
                        id=f"eco-{i}",
                        classes="eco-btn",
                    )

            yield Button("Cancel", id="progress-cancel", variant="default")

        self._ecos = ecos

    def on_mount(self) -> None:
        # Focus first ECO button if available, otherwise cancel
        ecos = getattr(self, "_ecos", [])
        if ecos:
            try:
                self.query_one("#eco-0", Button).focus()
            except Exception:
                pass
        else:
            try:
                self.query_one("#progress-cancel", Button).focus()
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("eco-"):
            idx = int(btn_id.split("-", 1)[1])
            ecos = getattr(self, "_ecos", [])
            if idx < len(ecos):
                spec_path = ecos[idx]["spec_path"]
                self.dismiss(f"/run run {spec_path} --resume")
                return
        if btn_id == "progress-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _load_ecos(self) -> list[dict[str, str]]:
        """Load ECOs from the state DB, grouped by spec path."""
        try:
            from nexus_orchestrator.persistence.state_db import StateDB
            from nexus_orchestrator.persistence.repositories import RunRepo, WorkItemRepo

            if not self._state_db_path.exists():
                return []

            db = StateDB(self._state_db_path)
            run_repo = RunRepo(db)
            work_item_repo = WorkItemRepo(db)

            all_runs = run_repo.list(limit=50)
            if not all_runs:
                return []

            # Group by spec path, keep latest run per spec
            by_spec: dict[str, object] = {}
            for run in all_runs:
                if run.spec_path not in by_spec:
                    by_spec[run.spec_path] = run

            # Filter to actionable statuses
            ecos: list[dict[str, str]] = []
            for spec_path, run in by_spec.items():
                status_value = run.status.value if hasattr(run.status, "value") else str(run.status)
                if status_value not in _ACTIONABLE_STATUSES:
                    continue

                # Count work items
                work_items = work_item_repo.list_for_run(run.id, limit=1000)
                total = len(work_items)
                merged = sum(1 for wi in work_items if hasattr(wi, "status") and wi.status.value == "merged")
                failed = sum(1 for wi in work_items if hasattr(wi, "status") and wi.status.value == "failed")

                work_info = f"({merged}/{total} merged"
                if failed:
                    work_info += f", {failed} failed"
                work_info += ")"

                ecos.append({
                    "spec_path": spec_path,
                    "status": status_value,
                    "work_info": work_info,
                })

            return ecos

        except Exception:
            return []


__all__ = ["ProgressDialog"]
