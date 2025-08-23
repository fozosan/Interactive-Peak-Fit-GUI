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
from fit import classic, modern, lmfit_backend, step_engine


class App(tk.Tk):
    """Minimal Tkinter/Matplotlib interface for Peakfit 3.x scaffolding."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Peakfit 3.x")

        self.x = None
        self.y = None
        self.baseline = None
        self.peaks: list[peaks.Peak] = []
        self.file_path: str | None = None
        self.last_result: dict | None = None

        # baseline configuration
        self.mode_var = tk.StringVar(value="add")
        self.lam_var = tk.DoubleVar(value=1e5)
        self.p_var = tk.DoubleVar(value=0.001)
        self.niter_var = tk.IntVar(value=10)

        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control = tk.Frame(self)
        control.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(control, text="Open", command=self.open_file).pack(fill=tk.X)
        tk.Button(control, text="Export Peaks", command=self.export_peaks).pack(fill=tk.X)
        tk.Button(control, text="Export Trace", command=self.export_trace).pack(fill=tk.X)
        tk.Button(control, text="Add Peak", command=self.add_peak).pack(fill=tk.X)
        tk.Button(control, text="Step", command=self.step_once).pack(fill=tk.X)

        # baseline controls
        base_frame = tk.LabelFrame(control, text="Baseline")
        base_frame.pack(fill=tk.X, pady=4)
        tk.OptionMenu(base_frame, self.mode_var, "add", "subtract").pack(fill=tk.X)
        tk.Label(base_frame, text="λ").pack()
        tk.Entry(base_frame, textvariable=self.lam_var).pack(fill=tk.X)
        tk.Label(base_frame, text="p").pack()
        tk.Entry(base_frame, textvariable=self.p_var).pack(fill=tk.X)
        tk.Label(base_frame, text="iters").pack()
        tk.Entry(base_frame, textvariable=self.niter_var).pack(fill=tk.X)
        tk.Button(base_frame, text="Recompute", command=self.recompute_baseline).pack(fill=tk.X)

        self.snr_label = tk.Label(control, text="S/N: -")
        self.snr_label.pack(fill=tk.X)

        self.solver_var = tk.StringVar(value="classic")
        tk.OptionMenu(control, self.solver_var, "classic", "modern", "lmfit").pack(fill=tk.X)

        tk.Button(control, text="Fit", command=self.run_fit).pack(fill=tk.X)

    # ------------------------------------------------------------------
    def open_file(self) -> None:
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            self.x, self.y = data_io.load_xy(path)
            self.recompute_baseline()
        except Exception as exc:  # pragma: no cover - user feedback
            messagebox.showerror("Error", str(exc))
            return
        self.file_path = path
        self.last_result = None
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

    def recompute_baseline(self) -> None:
        """Recalculate baseline using ALS parameters."""
        if self.y is None:
            return
        lam = self.lam_var.get()
        p = self.p_var.get()
        niter = self.niter_var.get()
        self.baseline = signals.als_baseline(self.y, lam=lam, p=p, niter=niter)
        sn = signals.snr_estimate(self.y - self.baseline)
        self.snr_label.config(text=f"S/N: {sn:.2f}")
        self.refresh_plot()

    def run_fit(self) -> None:
        if self.x is None or not self.peaks:
            return
        solver = self.solver_var.get()
        mode = self.mode_var.get()
        if solver == "classic":
            res = classic.solve(self.x, self.y, self.peaks, mode, self.baseline, {})
        elif solver == "modern":
            res = modern.solve(
                self.x,
                self.y,
                self.peaks,
                mode,
                self.baseline,
                {"loss": "linear"},
            )
        else:  # lmfit
            res = lmfit_backend.solve(
                self.x,
                self.y,
                self.peaks,
                mode,
                self.baseline,
                {},
            )

        if res["ok"]:
            # update peaks with fitted heights
            new = []
            theta = res["theta"]
            for i in range(len(self.peaks)):
                c, h, w, e = theta[4 * i : 4 * (i + 1)]
                new.append(peaks.Peak(c, h, w, e))
            self.peaks = new
            self.last_result = res
            self.refresh_plot()
        else:  # pragma: no cover - user feedback
            messagebox.showerror("Fit failed", res["message"])

    def step_once(self) -> None:
        """Perform a single Gauss-Newton step and update the plot."""
        if self.x is None or not self.peaks:
            return
        mode = self.mode_var.get()
        theta, _cost = step_engine.step_once(
            self.x,
            self.y,
            self.peaks,
            mode,
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
        self.last_result = None
        self.refresh_plot()

    def export_peaks(self) -> None:
        """Export fitted peaks to a CSV peak table."""
        if self.x is None or not self.peaks:
            return
        mode = self.mode_var.get()
        model = models.pv_sum(self.x, self.peaks)
        base = self.baseline if self.baseline is not None else 0.0
        if mode == "add":
            resid = model + base - self.y
        else:
            resid = model - (self.y - base)
        rmse = float(np.sqrt(np.mean(resid**2))) if self.y is not None else 0.0
        areas = [models.pv_area(p.height, p.fwhm, p.eta) for p in self.peaks]
        total = sum(areas) if areas else 1.0
        records = []
        for i, (p, area) in enumerate(zip(self.peaks, areas), start=1):
            records.append(
                {
                    "file": self.file_path or "",
                    "peak": i,
                    "center": p.center,
                    "height": p.height,
                    "fwhm": p.fwhm,
                    "eta": p.eta,
                    "lock_width": p.lock_width,
                    "lock_center": p.lock_center,
                    "area": area,
                    "area_pct": 100.0 * area / total,
                    "rmse": rmse,
                    "fit_ok": True,
                    "mode": mode,
                    "als_lam": "",
                    "als_p": "",
                    "fit_xmin": float(self.x[0]),
                    "fit_xmax": float(self.x[-1]),
                }
            )

        csv_text = data_io.build_peak_table(records)
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(csv_text)

    def export_trace(self) -> None:
        """Export a trace table capturing baseline, model and peaks."""
        if self.x is None or self.y is None:
            return
        mode = self.mode_var.get()
        csv_text = data_io.build_trace_table(
            self.x, self.y, self.baseline, self.peaks, mode=mode
        )
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(csv_text)

    def refresh_plot(self) -> None:
        self.ax.clear()
        if self.x is not None and self.y is not None:
            self.ax.plot(self.x, self.y, label="data")
        if self.baseline is not None:
            self.ax.plot(self.x, self.baseline, label="baseline")
        if self.peaks:
            model = models.pv_sum(self.x, self.peaks)
            base = self.baseline if self.baseline is not None else 0.0
            if self.mode_var.get() == "add":
                self.ax.plot(self.x, model + base, label="model")
            else:
                self.ax.plot(self.x, model, label="model")
        self.ax.legend()
        self.canvas.draw()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch only
    main()
