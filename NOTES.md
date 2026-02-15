# TUI UX Overhaul — Implementation Notes

## Architecture

Refactored from a monolithic 1,847-line `tui_app.py` into a 3-layer architecture:

```
ui/tui/
  __init__.py     — Entrypoint (tui_available, run_tui, tui_entrypoint)
  app.py          — Textual App (view layer — renders state, emits intents)
  controller.py   — Controller (owns AppState, translates intents → service calls)
  state.py        — Pure data models (NO Textual imports, mypy strict)
  runner.py       — Streaming command runner (subprocess + in-process, NO Textual)
  theme.tcss      — Extracted CSS theme
  widgets/
    transcript.py — Incremental O(1) append transcript (ring buffer)
    composer.py   — Multiline command input with history
    statusline.py — Always-visible status bar (workspace, git, runner state)
    sidebar.py    — Quick action Buttons with real activation events
  screens/
    help.py       — F1 help overlay
    plan_dialog.py— Modal for selecting spec files (replaces hardcoded paths)
```

## Design Decisions & Assumptions

1. **Streaming via InProcessRunner (default)**: Since `nexus` may not be on PATH
   in all environments, the default runner calls `run_cli()` in-process via an
   executor. SubprocessRunner is used when `nexus` is on PATH, providing true
   line-by-line streaming and SIGINT-based cancellation.

2. **Ring buffer transcript**: The transcript uses a deque(maxlen=5000) for the
   event log in state, and mounts each event as a separate Textual Static widget
   for O(1) append. Old widgets are removed when the limit is exceeded.

3. **Quick actions are Buttons**: Replaced ListView with Button widgets that have
   real `on_button_pressed` handlers. "Plan..." opens a PlanDialog modal instead
   of hardcoding `samples/specs/minimal_design_doc.md`.

4. **Recent spec paths**: Persisted in `~/.config/nexus_orchestrator/recent_specs.json`
   (up to 5 entries). No secrets stored.

5. **Legacy tui_app.py preserved**: The old monolithic file is kept for backward
   compatibility (OnboardingWidget, SplashScreen are still referenced from the
   new app.py). It can be fully removed once all onboarding/splash logic is
   migrated.

6. **mypy strict on core modules**: `state.py`, `runner.py`, `controller.py` all
   pass mypy strict. Widget/screen files are excluded (Textual type stubs are
   incomplete).

7. **Backend detection off UI thread**: `detect_backends()` and
   `detect_workspace_info()` run in executor threads to avoid blocking the UI.

8. **Cancel protocol**: SIGINT → wait 2s → SIGTERM → wait 3s → SIGKILL. The UI
   shows RUNNING → CANCEL_REQUESTED → CANCELLED states.

## Keybindings

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Cycle focus: sidebar <-> transcript <-> input |
| Ctrl+P | Open command palette |
| F1 | Help overlay |
| Up / Down | Command history (when input focused) |
| Enter | Execute command / activate quick action |
| Ctrl+C | Cancel task (1st) / exit (2nd within 2s) |
| Ctrl+E | Export transcript to file |
| Ctrl+Q | Quit immediately |

## Before / After

### Before
- Monolithic 1,847-line `tui_app.py` excluded from mypy
- Transcript: O(n) `"\n".join()` + `Static.update()` on every append
- Commands: blocking `run_cli()` with captured stdout/stderr blob
- Cancel: fake (toggled a flag, didn't stop anything)
- Quick actions: ListView items that only prefilled input, no activation
- Backend detection: synchronous on UI thread (could hang startup)
- No status line showing workspace/git/runner state
- Hardcoded `samples/specs/minimal_design_doc.md` in plan action
- 49 tests covering onboarding only

### After
- 3-layer architecture: state/runner/controller (0 Textual deps) + widgets/screens
- Transcript: O(1) append via individual Static widgets, ring buffer (5000 max)
- Commands: streaming runner (SubprocessRunner with async line-by-line output)
- Cancel: real SIGINT/SIGTERM/SIGKILL cascade on subprocess
- Quick actions: Button widgets with immediate execution on press
- Backend/workspace detection: async off UI thread
- Status line: workspace path, git branch/dirty, runner state, always visible
- Plan dialog: modal with recent specs list + input, persists last-used paths
- Core modules pass mypy strict (state.py, runner.py, controller.py)

## Production Blocker Fixes (Phase 2)

### 1. ASCII-only icons by default
- All emoji/Unicode icons (`\U0001f50d`, `\u25b6`, etc.) replaced with ASCII equivalents
- `icon()` and `star_sep()` functions in `sidebar.py` centralize all icon rendering
- `NEXUS_TUI_ICONS` env var: `emoji` for Unicode, `nerd` for Nerd Font glyphs
- Default (unset): pure ASCII (`>`, `v`, `x`, `*`, etc.)
- Sidebar, status line, header, and help screen all use the icon system

### 2. Composer no longer clipped at 80x24
- Changed Composer height from fixed `3` to `auto` with `max-height: 5`
- Removed `border-top` from Composer and StatusLine in theme.tcss
- Removed Footer widget (keybinding hints redundant with status line)
- Reordered bottom dock: StatusLine (1 row) then Composer (auto)
- Total bottom consumption: ~2 rows (status) + ~1 row (input) = ~3 rows

### 3. Plan dialog no longer crashes with NoActiveWorker
- Replaced `push_screen_wait(PlanDialog())` with `push_screen(PlanDialog(), callback=...)`
- `_on_plan_dialog_result()` async callback handles the result
- No worker context required for message handlers

### 4. Copy/export transcript
- `/copy` now uses best-effort clipboard (pbcopy, wl-copy, xclip, xsel)
- `/export [path]` writes transcript to file (auto-named if no path given)
- `Ctrl+E` keybinding for quick export
- `_transcript_as_text()` serializes transcript to plain text format

### Test results (Phase 2)
- 635 tests pass (77 TUI-specific, up from 65)
- mypy strict: 102 source files, 0 errors
- Added: 6 icon theme tests, 6 transcript export tests, updated copy/sidebar tests

## Production Blocker Fixes (Phase 3)

### 1. Copy/Paste + Ctrl+C Confusion

**Problem**: Ctrl+C immediately cancels a running task, making it easy to
accidentally nuke a long-running operation when the user intended to copy.
Ctrl+Shift+C often doesn't work in Textual apps due to mouse capture.

**Solution — double-press Ctrl+C to cancel** (chosen over confirm dialog):
- When a task is running, first Ctrl+C press shows a warning:
  *"Press Ctrl+C again within 2s to cancel the running task (or Esc to dismiss)."*
- Second Ctrl+C within the 2s window actually cancels.
- When idle, double-press Ctrl+C to exit (unchanged).
- Justification: A confirm dialog would steal focus and block keyboard input.
  The double-press pattern is the same as the exit pattern, so it's consistent
  and doesn't interrupt flow. It also works the same in all terminal emulators.

**New keybindings**:
- `Ctrl+Y` — Copy last output / full transcript to system clipboard.
  Falls back to auto-export if clipboard unavailable.
- `Ctrl+E` — Export transcript to `~/.nexus/logs/nexus_transcript_<ts>.txt`.
- `/copy` and `/export [path]` slash commands also available.
- Ctrl+Shift+C is left unbound in-app so it works as terminal-native copy.

**Clipboard fallback chain**:
1. Attempt clipboard copy (pbcopy → wl-copy → xclip → xsel)
2. If all fail, auto-export to `~/.nexus/logs/` and show the file path.
3. Never crash — clipboard failure is always handled gracefully.

### 2. Zero Real-Time Updates During Runs

**Problem**: The TUI showed no incremental output during long-running operations.
The UI appeared frozen until the task completed.

**Root causes identified**:
1. SubprocessRunner did not set `PYTHONUNBUFFERED=1`, so Python subprocesses
   buffered their stdout when not attached to a TTY.
2. The `_reduce_runner_event()` method called `_notify()` (which calls
   `_on_state_change()`) synchronously from the asyncio task. Without an
   explicit yield back to the event loop, Textual had no opportunity to render
   between events.

**Fixes**:
- SubprocessRunner now passes `PYTHONUNBUFFERED=1` in the subprocess env.
- `_reduce_runner_event()` calls `await asyncio.sleep(0)` after each
  `_notify()` to yield to the Textual event loop, allowing it to render
  the newly appended transcript event before processing the next one.

### 3. No In-Progress Cursor / Spinner Animation

**Problem**: No visual indicator that work is happening. The status line showed
"running..." as static text, but there was no animation.

**Solution**: New `RunnerSpinner` widget (`widgets/spinner.py`):
- Braille-dot spinner animation (⠋⠙⠹⠸⠼⠴⠦⠧) at 100ms per frame.
- Shows contextual activity text: "Running: plan...", "Generating: ...",
  "Cancelling: ...".
- Driven by Textual's `set_interval` timer (paused when idle, resumed when
  active). Low CPU cost — only updates one Static widget per frame.
- Placed in the dock between the transcript and the status line.
- Visibility controlled by CSS class toggle (`display: none` / `display: block`).
- State-driven: `update_from_state()` checks `runner_status` and shows/hides.

### Keybindings (updated)

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Cycle focus: sidebar ↔ transcript ↔ input |
| Ctrl+P | Open command palette |
| F1 | Help overlay |
| Up / Down | Command history (when input focused) |
| Enter | Execute command / activate quick action |
| Ctrl+C | Cancel task (double-press when running) / exit (double-press when idle) |
| Ctrl+E | Export transcript to ~/.nexus/logs/ |
| Ctrl+Y | Copy transcript to clipboard (fallback: export) |
| Ctrl+Q | Quit immediately |
