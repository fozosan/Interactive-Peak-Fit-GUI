"""Tkinter + Matplotlib based user interface for Peakfit 3.x.

Only a minimal entry point is provided; UI wiring follows the blueprint but
is not yet implemented.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

from core import data_io, models, peaks, signals
from fit import classic, step_engine


class App(tk.Tk):
    """Minimal Tkinter/Matplotlib interface for Peakfit 3.x scaffolding."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Peakfit 3.x")

        self.x = None
        self.y = None
        self.baseline = None
        self.peaks: list[peaks.Peak] = []

        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control = tk.Frame(self)
        control.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(control, text="Open", command=self.open_file).pack(fill=tk.X)
        tk.Button(control, text="Add Peak", command=self.add_peak).pack(fill=tk.X)
        tk.Button(control, text="Step", command=self.step_once).pack(fill=tk.X)
        tk.Button(control, text="Fit", command=self.run_fit).pack(fill=tk.X)

    # ------------------------------------------------------------------
    def open_file(self) -> None:
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            self.x, self.y = data_io.load_xy(path)
            self.baseline = signals.als_baseline(self.y)
        except Exception as exc:  # pragma: no cover - user feedback
            messagebox.showerror("Error", str(exc))
            return
        self.peaks.clear()
        self.refresh_plot()

    def add_peak(self) -> None:
        if self.x is None:
            return
        try:
            center = float(simpledialog.askstring("Center", "Center"))
            height = float(simpledialog.askstring("Height", "Height"))
            fwhm = float(simpledialog.askstring("FWHM", "FWHM"))
            eta = float(simpledialog.askstring("Eta", "Eta", initialvalue="0.5"))
        except (TypeError, ValueError):  # user cancelled or invalid
            return
        self.peaks.append(peaks.Peak(center, height, fwhm, eta))
        self.refresh_plot()

    def run_fit(self) -> None:
        if self.x is None or not self.peaks:
            return
        res = classic.solve(self.x, self.y, self.peaks, "add", self.baseline, {})
        if res["ok"]:
            # update peaks with fitted heights
            new = []
            theta = res["theta"]
            for i in range(len(self.peaks)):
                c, h, w, e = theta[4 * i : 4 * (i + 1)]
                new.append(peaks.Peak(c, h, w, e))
            self.peaks = new
            self.refresh_plot()
        else:  # pragma: no cover - user feedback
            messagebox.showerror("Fit failed", res["message"])

    def step_once(self) -> None:
        """Perform a single Gauss-Newton step and update the plot."""
        if self.x is None or not self.peaks:
            return
        theta, _cost = step_engine.step_once(
            self.x,
            self.y,
            self.peaks,
            "add",
            self.baseline,
            "linear",
            None,
            0.0,
            np.inf,
            None,
        )
        new = []
        for i in range(len(self.peaks)):
            c, h, w, e = theta[4 * i : 4 * (i + 1)]
            new.append(peaks.Peak(c, h, w, e))
        self.peaks = new
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self.ax.clear()
        if self.x is not None and self.y is not None:
            self.ax.plot(self.x, self.y, label="data")
        if self.baseline is not None:
            self.ax.plot(self.x, self.baseline, label="baseline")
        if self.peaks:
            model = models.pv_sum(self.x, self.peaks)
            self.ax.plot(self.x, model + (self.baseline or 0), label="model")
        self.ax.legend()
        self.canvas.draw()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch only
    main()