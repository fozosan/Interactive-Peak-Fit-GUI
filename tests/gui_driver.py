"""Drive the real PeakFitApp the way a user does, for tests under Xvfb.

Only the operating system's file pickers and the message boxes are answered;
every click runs the app's own command. Fit, Export and Run Batch hand their
work to the app's worker threads, so the driver pumps Tk events until the job
reports back, as the main loop would.
"""

from __future__ import annotations

import csv
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox


class Driver:
    def __init__(self, app, monkeypatch):
        self.app = app
        self.root = app.root
        self.dialogs: list[tuple[str, str, str]] = []
        self.open_answers: list[str] = []
        self.save_answers: list[str] = []
        self.folder_answers: list[str] = []
        monkeypatch.setattr(filedialog, "askopenfilename", lambda **k: self.open_answers.pop(0))
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: self.save_answers.pop(0))
        monkeypatch.setattr(filedialog, "askdirectory", lambda **k: self.folder_answers.pop(0))
        for kind in ("showinfo", "showwarning", "showerror"):
            monkeypatch.setattr(
                messagebox,
                kind,
                lambda title="", message="", _kind=kind, **k: self.dialogs.append((_kind, str(title), str(message))),
            )

    # --- widgets ---------------------------------------------------------
    def button(self, text: str, within: str | None = None):
        top = self._labelframe(self.root, within) if within else self.root
        found = self._find(top, lambda w: w.winfo_class() in ("TButton", "Button") and w.cget("text") == text)
        if found is None:
            raise LookupError(f"no button {text!r}" + (f" in {within!r}" if within else ""))
        return found

    def _labelframe(self, widget, text):
        found = self._find(widget, lambda w: w.winfo_class() == "TLabelframe" and w.cget("text") == text)
        if found is None:
            raise LookupError(f"no labelled frame {text!r}")
        return found

    def _find(self, widget, match):
        for child in widget.winfo_children():
            try:
                if match(child):
                    return child
            except tk.TclError:
                pass
            hit = self._find(child, match)
            if hit is not None:
                return hit
        return None

    def click(self, text: str, within: str | None = None) -> None:
        self.button(text, within).invoke()

    # --- waiting -----------------------------------------------------------
    def pump_until(self, done, timeout: float = 180.0, what: str = "the job") -> None:
        """Run the Tk main loop until ``done()``, as the app's own loop would.

        A nested mainloop, not update(): Tk accepts a call from another thread only
        while the main thread is inside mainloop, so an update() loop would hide how
        the app behaves for real."""

        t0 = time.monotonic()
        timed_out = []

        def check():
            if done():
                self.root.quit()
            elif time.monotonic() - t0 > timeout:
                timed_out.append(True)
                self.root.quit()
            else:
                self.root.after(10, check)

        self.root.after(0, check)
        self.root.mainloop()
        if timed_out:
            raise TimeoutError(f"{what} did not finish in {timeout:.0f} s; dialogs: {self.dialogs[-3:]}")
        self.root.update()

    def _log_count(self, text: str) -> int:
        return sum(text in line for line in self.app._log_buffer)

    def _dialog_count(self, title: str) -> int:
        return sum(t == title for _, t, _ in self.dialogs)

    def _raise_on_error(self, title: str, start: int) -> None:
        errors = [m for k, t, m in self.dialogs[start:] if t == title and k in ("showerror", "showwarning")]
        if errors:
            raise AssertionError(f"{title} failed: {errors[-1]}")

    # --- what a user does --------------------------------------------------
    def open_file(self, path) -> None:
        self.open_answers.append(str(path))
        self.click("Open")
        self.root.update()

    def set_window(self, lo: float, hi: float) -> None:
        self.app.fit_min_var.set(str(lo))
        self.app.fit_max_var.set(str(hi))
        self.app.apply_fit_range_from_fields()
        self.root.update()

    def fit(self) -> None:
        start, n0, d0 = len(self.dialogs), self._log_count("Fit finished"), self._dialog_count("Fit")
        self.app.fit_btn.invoke()
        self.pump_until(
            lambda: self._log_count("Fit finished") > n0 or self._dialog_count("Fit") > d0, what="Fit"
        )
        self._raise_on_error("Fit", start)

    UNC_OUTCOMES = ("Computed ", "Uncertainty failed", "Uncertainty not computed")

    def run_uncertainty(self) -> str:
        """Press Run in the Uncertainty panel; return the log line that ends it."""

        n0 = len(self.app._log_buffer)
        self.click("Run", within="Uncertainty")

        def outcome():
            return [ln for ln in self.app._log_buffer[n0:] if any(k in ln for k in self.UNC_OUTCOMES)]

        self.pump_until(lambda: bool(outcome()), what="Uncertainty")
        return outcome()[0]

    def export(self, path) -> None:
        start, d0 = len(self.dialogs), self._dialog_count("Export")
        self.save_answers.append(str(path))
        self.click("Export")
        self.pump_until(lambda: self._dialog_count("Export") > d0, what="Export")
        self._raise_on_error("Export", start)

    def run_batch(self, folder, summary_csv) -> None:
        start, d0 = len(self.dialogs), self._dialog_count("Batch")
        self.folder_answers.append(str(folder))
        self.save_answers.append(str(summary_csv))
        self.click("Run Batch…")
        self.pump_until(lambda: self._dialog_count("Batch") > d0, timeout=300, what="Run Batch")
        self._raise_on_error("Batch", start)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _num(text):
    try:
        return float(text)
    except ValueError:
        return None


def table_differences(a, b, rel: float = 1e-9, absolute: float = 1e-12) -> list[str]:
    """Every cell where two CSV tables differ beyond floating-point tolerance."""

    ha, ra = read_csv(a)
    hb, rb = read_csv(b)
    out = []
    if ha != hb:
        out.append(f"header: {ha} != {hb}")
        return out
    if len(ra) != len(rb):
        out.append(f"rows: {len(ra)} != {len(rb)}")
        return out
    for i, (row_a, row_b) in enumerate(zip(ra, rb)):
        for name, va, vb in zip(ha, row_a, row_b):
            if va == vb:
                continue
            fa, fb = _num(va), _num(vb)
            if fa is not None and fb is not None:
                if math.isnan(fa) and math.isnan(fb):
                    continue
                if math.isclose(fa, fb, rel_tol=rel, abs_tol=absolute):
                    continue
            out.append(f"row {i} {name}: {va!r} != {vb!r}")
    return out


def text_differences(a, b) -> list[str]:
    la = Path(a).read_text(encoding="utf-8").splitlines()
    lb = Path(b).read_text(encoding="utf-8").splitlines()
    if len(la) != len(lb):
        return [f"lines: {len(la)} != {len(lb)}"]
    return [f"line {i}: {x!r} != {y!r}" for i, (x, y) in enumerate(zip(la, lb)) if x != y]
