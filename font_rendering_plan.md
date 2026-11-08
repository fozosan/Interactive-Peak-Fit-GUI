# Improved Plan: Tkinter Font Rendering / Antialiasing

## Codebase Context (What Was Found)

Before reading the plan, these are the **exact current font touch-points** in the codebase:

| Location | File | Lines | Current Hardcode |
|---|---|---|---|
| `matplotlib.rcParams["font.family"]` | `ui/app.py` | 217 | `"Arial"` — module-level |
| `FontProperties(family="Arial")` | `ui/app.py` | 7084 | Legend font, in `refresh_plot()` |
| `tkfont.nametofont("TkDefaultFont")` | `ui/app.py` | 1235 | Reads system default, stores as `self.default_font` |
| `tkfont.Font(family="Ubuntu", size=10)` | `ui/app.py` | 1240 | Fallback if system default fails — Ubuntu-only |
| `style.configure("TButton", font=("Ubuntu", 9))` | `ui/app.py` | 1230 | Hardcoded Ubuntu, no fallback |
| `style.configure("Fit.TButton", font=("Ubuntu", 10, "bold"))` | `ui/app.py` | 1630 | Hardcoded Ubuntu, no fallback |
| `log_console.configure(font=self.default_font)` | `ui/app.py` | 3274 | Uses stored default font |
| `apply_theme(root)` → `sv_ttk` / `ttkthemes` / default | `ui/theme.py` | 15–61 | Themes may override fonts |
| Config file | `~/.gl_peakfit_config.json` | — | No font keys exist yet |
| Config DEFAULTS dict | `ui/app.py` | 408–446 | No font keys exist yet |

**Critical gap in the original plan:** It did not account for the fact that `"Ubuntu"` is hardcoded in **two** `ttk.Style` calls (`ui/app.py:1230` and `ui/app.py:1630`) that fire unconditionally at init, *before* any font selection. These will silently fall back to a system font on non-Ubuntu systems but will not use the chosen fallback family. The plan must fix these.

**Second gap:** The `main()` function in `ui/app.py` calls `apply_theme(root)` *before* creating `PeakFitApp`. Font configuration should happen between `apply_theme()` and `PeakFitApp(root)`, or at the start of `PeakFitApp.__init__`, but *not inside `_build_ui()`* (where `style.configure("Fit.TButton", ...)` currently fires at line 1630 — too late).

**Third gap (Matplotlib):** `matplotlib.rcdefaults()` is called at **module import time** (line 216), resetting all rcParams, then `font.family = "Arial"` is set. Any guard for `text.antialiased` etc. must come *after* line 216 in the same block.

---

## Phase 1 — Diagnostics (`ui/font_diagnostics.py`)

### What to implement

New file `ui/font_diagnostics.py`. Keep it importable with no side-effects; side-effects only happen when `run_diagnostics(root)` is called.

```
ui/font_diagnostics.py
  run_diagnostics(root, *, log_fn=print) -> dict
  show_visual_sample(root) -> None          # opens a Toplevel
```

#### `run_diagnostics(root, *, log_fn=print) -> dict`

Collect and log all of the following. Return everything as a dict (for tests).

**Platform / session:**
- `platform.system()`, `platform.release()`, `platform.machine()`
- `os.environ.get("XDG_SESSION_TYPE")`, `os.environ.get("DISPLAY")`, `os.environ.get("WAYLAND_DISPLAY")`
- WSL detection: `"microsoft" in Path("/proc/version").read_text().lower()` (guard with `Path("/proc/version").exists()`)
- `os.environ.get("SSH_CLIENT")` / `"SSH_TTY"` to detect remote X

**Tk runtime:**
- `tkinter.TkVersion`, `tkinter.TclVersion`
- `root.tk.call("tk", "windowingsystem")` — wrap in try/except
- `root.tk.call("tk", "scaling")` — wrap in try/except

**Fonts:**
- `tkfont.nametofont("TkDefaultFont").actual()` — wrap in try/except
- `len(tkfont.families(root))`
- Check presence (case-insensitive) of:
  - Sans: `"Ubuntu"`, `"Noto Sans"`, `"DejaVu Sans"`, `"Liberation Sans"`, `"Cantarell"`, `"Segoe UI"`, `"Arial"`
  - Mono: `"Ubuntu Mono"`, `"DejaVu Sans Mono"`, `"Liberation Mono"`, `"Courier New"`, `"Consolas"`

**Logging format:** prefix all lines with `[FONT-DIAG]` so they are filterable in the app's log console.

#### `show_visual_sample(root) -> None`

Opens a `tk.Toplevel`. Renders the test strings in each candidate family (plus TkDefaultFont as baseline). Use `tk.Label` with explicit `font=(family, 10)`. Below each label, show which family name was actually resolved via `tkfont.Font(family=..., size=10).actual()["family"]`.

Test strings:
- `"The quick brown fox jumps over the lazy dog 0123456789"`
- `"AaBbYyZz — %$@ (hinting test)"`

Include a "Close" button at the bottom.

#### Hook into startup

In `main()` (`ui/app.py`), check for a `--font-diagnostics` CLI flag (via `sys.argv`) **or** a config key `"font_diagnostics": true`. If either is set, call `run_diagnostics(root, log_fn=print)` and `show_visual_sample(root)` after `apply_theme()` but before `PeakFitApp(root)`.

---

## Phase 2 — Central Font Policy (`ui/fonts.py`)

### What to implement

New file `ui/fonts.py`.

```python
SANS_CANDIDATES  = ["Ubuntu", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Cantarell", "Segoe UI", "Arial"]
MONO_CANDIDATES  = ["Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono", "Consolas", "Courier New"]

DEFAULT_UI_SIZE  = 10   # match existing app defaults

def pick_first_available_family(root, candidates) -> str | None: ...
def configure_named_fonts(root, *, sans_family, mono_family, ui_size) -> None: ...
def configure_ttk_styles(style, *, sans_family, mono_family, ui_size) -> None: ...
```

#### `pick_first_available_family`
- Call `tkfont.families(root)` once (cache result in a module-level variable keyed by `id(root)` to avoid repeated calls; `tkfont.families()` is slow).
- Case-insensitive match: `candidate.lower() == available.lower()`.
- Return the string from `families()` (preserve Tk's casing), or `None`.

#### `configure_named_fonts`
Named fonts to configure: `TkDefaultFont`, `TkTextFont`, `TkMenuFont`, `TkFixedFont`, `TkHeadingFont`, `TkSmallCaptionFont`, `TkIconFont`, `TkTooltipFont`.

- For each, call `tkfont.nametofont(name)` inside try/except (some may not exist).
- Apply `family=sans_family` on all except `TkFixedFont` (which gets `mono_family`).
- Apply `size=ui_size` to all (use `ui_size - 1` for `TkSmallCaptionFont`).
- Do not crash on any individual name.

#### `configure_ttk_styles`
Styles to configure (if `sans_family` is not None):
```
TLabel, TButton, TCheckbutton, TRadiobutton, TEntry, TCombobox,
TMenubutton, TSpinbox, TLabelframe.Label, TNotebook.Tab, Treeview, Treeview.Heading
```
- Use `style.configure(name, font=(sans_family, ui_size))` for all.
- Keep **special styles** but use same family:
  - `"Fit.TButton"` → `(sans_family, ui_size, "bold")` (was `("Ubuntu", 10, "bold")`)
  - `"Danger.TButton"` / `"Success.TButton"` → font added if not already set
  - TkFixedFont-based widgets (log console) → handled via named font, not style

### Integration into `ui/app.py`

**In `main()` function (lines 7116–7122):**
```python
def main():
    root = tk.Tk()
    theme_applied, theme_name = apply_theme(root)
    log_theme_status(theme_applied, theme_name)

    # --- NEW: font policy ---
    from ui.fonts import (pick_first_available_family,
                           configure_named_fonts, configure_ttk_styles,
                           DEFAULT_UI_SIZE)
    from tkinter import ttk
    _style = ttk.Style(root)
    _cfg   = load_config()  # already imported
    _sans  = pick_first_available_family(root, ...)  # or from config
    _mono  = pick_first_available_family(root, ...)
    _size  = _cfg.get("ui_font_size", DEFAULT_UI_SIZE)
    configure_named_fonts(root, sans_family=_sans, mono_family=_mono, ui_size=_size)
    configure_ttk_styles(_style, sans_family=_sans, mono_family=_mono, ui_size=_size)
    # --- END NEW ---

    app = PeakFitApp(root)
    root.mainloop()
```

**In `PeakFitApp.__init__`:**
- Remove the hardcoded `self._style.configure("TButton", font=("Ubuntu", 9))` at line 1230.
- Remove / update the hardcoded `self._style.configure("Fit.TButton", font=("Ubuntu", 10, "bold"))` at line 1630 — this is now handled by `configure_ttk_styles` in `main()`, but `"Fit.TButton"` isn't configured there until `_build_ui` is called; solution: configure `"Fit.TButton"` in `configure_ttk_styles`, which fires before `PeakFitApp()` is constructed, so line 1630 only needs `self.fit_btn.configure(style="Fit.TButton")` (no `.configure(font=...)` call).
- Update the fallback at line 1240: change `family="Ubuntu"` to use the selected `sans_family` (pass it in, or read from a module-level variable set in `main()`). **Simplest approach:** add `ui/fonts.py` module-level `_selected_sans: str | None = None` set by `configure_named_fonts`, and read it in the except branch.

### Config keys to add to `DEFAULTS` dict (ui/app.py line 408)
```python
"ui_font_family":      "auto",   # "auto" or explicit family name
"ui_font_size":        10,
"ui_mono_font_family": "auto",
"font_diagnostics":    False,
```
And in `load_config()`, honor these keys. When `"auto"`, use `pick_first_available_family`.

---

## Phase 3 — Matplotlib rcParams (Safe Tweaks Only)

**Location:** `ui/app.py` lines 216–221. The existing block is:
```python
matplotlib.rcdefaults()
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["savefig.dpi"] = 150
```

**Add immediately after line 221 (same module-level block):**
```python
for _key in ("text.antialiased", "lines.antialiased", "patch.antialiased"):
    if _key in matplotlib.rcParams:
        matplotlib.rcParams[_key] = True
```

**Do NOT** touch `font.family = "Arial"` or `FontProperties(family="Arial")` in `refresh_plot()` — keep as-is. The plan's constraint is met: plot fonts stay Arial.

**Also guard the legend call at line 7084** — it currently does not guard against FontProperties failure. Already in try/except, so it's fine, but add `family` fallback:
```python
self.ax.legend(loc="best", prop=FontProperties(family="Arial"))
```
No change needed here; it's already safe.

---

## Phase 4 — Environment Guidance

Add a "Font Rendering" section to the Help window (in `ui/helptext.py`, `build_help()` function). This avoids creating a separate doc file and makes it discoverable via the existing Help button.

Content to add:
- "If text looks jagged on Linux" checklist (same content as original plan)
- Mention `~/.config/fontconfig/conf.d/99-font-rendering.conf` with example
- Xresources knobs: `Xft.antialias: 1`, `Xft.hinting: 1`, `Xft.rgba: rgb`, `Xft.lcdfilter: lcddefault`
- Note: conda-forge `tk` package has Xft enabled; system Python Tk may not

---

## Phase 5 — Tests

### New test file: `tests/test_fonts.py`

```python
# tests/test_fonts.py
import pytest
import sys
import tkinter as tk
from unittest.mock import MagicMock, patch

class TestPickFirstAvailableFamily:
    def test_returns_none_if_no_candidates_available(self): ...
    def test_case_insensitive_match(self): ...
    def test_returns_actual_tk_casing(self): ...
    def test_returns_first_available_from_ordered_list(self): ...

class TestConfigureNamedFonts:
    def test_no_crash_when_named_font_missing(self): ...
    def test_sets_family_when_provided(self): ...
    def test_skips_family_when_none(self): ...

class TestConfigureTtkStyles:
    def test_no_crash_when_family_none(self): ...
    def test_configures_fit_button_bold(self): ...
```

All tests using real Tk root must be decorated with `@skip_if_no_display` from `conftest.py`, or use mocks.

**Mock strategy for headless tests:** Mock `tkfont.families()` to return a known list; mock `tkfont.nametofont()` to return a mock font object. This allows `pick_first_available_family` and `configure_named_fonts` to be tested without a display.

### Update existing test: `tests/test_theme_application.py`

The test `test_app_can_be_instantiated_with_theme` at line 194 currently only checks `PeakFitApp is not None`. After the change, add a test that font selection runs without crashing after `apply_theme()`.

---

## Implementation Order

1. **`ui/fonts.py`** — pure functions first, testable without display
2. **Tests: `tests/test_fonts.py`** — write alongside; confirm all pass headlessly
3. **`main()` integration** — add `configure_named_fonts` + `configure_ttk_styles` calls, pass config keys
4. **Remove hardcoded Ubuntu** from `ui/app.py:1230` and update `:1240` fallback; update `:1630` to not re-set font
5. **Add config keys** to `DEFAULTS` dict and `load_config()` in `ui/app.py`
6. **Matplotlib rcParams guards** — 3-line addition after line 221 in `ui/app.py`
7. **`ui/font_diagnostics.py`** — diagnostics module + startup hook + `--font-diagnostics` flag
8. **Help text** — add Linux guidance section to `ui/helptext.py`

---

## Things the Original Plan Missed / Got Wrong

| Issue | Original Plan | Correction |
|---|---|---|
| `"Ubuntu"` hardcoded in two style calls | Not mentioned | Must be replaced at `app.py:1230` and `:1630` |
| `_build_ui()` sets `"Fit.TButton"` font after init | Not mentioned | Move to `configure_ttk_styles` in `main()` |
| `matplotlib.rcdefaults()` resets everything | Not mentioned | rcParam guards must come *after* line 216 in the same block |
| Fallback `tkfont.Font(family="Ubuntu")` at line 1240 | Not mentioned | Must use selected family, not hardcode Ubuntu |
| `TkFixedFont` for log console | Not mentioned | Should use `mono_family` not `sans_family` |
| `font.antialiased` was the bad key | Correctly identified | Guarded with `if _key in matplotlib.rcParams` |
| `ttk.Style(root)` must be created before configuring | Not mentioned | Must pass same `Style` instance to avoid duplicate |
| `load_config()` must be called before `PeakFitApp()` | Not mentioned | `main()` should call `load_config()` for font prefs |
| Help guidance location | Suggested README | Better: `ui/helptext.py` (in-app Help window) |
