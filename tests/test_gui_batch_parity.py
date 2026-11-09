"""The GUI's Fit and Export write what batch writes for the same spectrum and settings.

Game plan 2.1: one settings object, one row builder, one writer and one fit
call, shared by the GUI and batch. The 2026-11-07 review found 9 of the 43 fit
columns differed between the two paths. These tests drive the real app under
Xvfb (Open, a fit window, seed peaks, Fit, Export), then press its Run Batch
button on the same file. The button's settings reach batch.runner.run_batch
unchanged; the test only starts batch from the same seeds and, as decision D2
requires for a comparison, in the GUI's fixed-baseline mode with the GUI's
unweighted fit. Every exported file must match its batch twin to
floating-point tolerance.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.gui_driver import Driver, table_differences, text_differences
from tests.synthetic_raman import TEMPLATE, WINDOW, mos2_spectrum, write_spectrum

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    pytest.skip("drives the real GUI: needs a display (run under xvfb-run)", allow_module_level=True)

tk = pytest.importorskip("tkinter")

from core import uncertainty as core_uncertainty  # noqa: E402
from ui import app as ui_app  # noqa: E402

# A dotted name, as the user's exposure series have.
SPECTRUM = "MoS2_0.5mW 07.csv"
STEM = SPECTRUM[: -len(".csv")]
SAME_MODE = {"baseline_fit": "fixed", "noise_model": "none"}


@pytest.fixture
def gui(monkeypatch):
    root = tk.Tk()
    app = ui_app.PeakFitApp(root)
    driver = Driver(app, monkeypatch)
    try:
        yield driver
    finally:
        root.destroy()


@pytest.fixture
def spectrum(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / SPECTRUM
    write_spectrum(path, *mos2_spectrum(7))
    return path


def _seed(gui):
    gui.app.peaks = [ui_app.Peak(**dict(p)) for p in TEMPLATE]
    gui.app.refresh_tree()


def _batch_like_the_gui(gui, monkeypatch, spectrum, out_dir, compute_uncertainty):
    """Press Run Batch; batch starts from the GUI's seeds, in the GUI's baseline mode."""

    seen = []
    real = ui_app.batch_runner.run_batch

    def same_seeds_same_mode(patterns, config, **kw):
        seen.append(dict(config))
        return real(
            patterns,
            {**config, "peaks": [dict(p) for p in TEMPLATE], "source": "template", "reheight": False,
             "save_traces": True, **SAME_MODE},
            **kw,
        )

    monkeypatch.setattr(ui_app.batch_runner, "run_batch", same_seeds_same_mode)
    gui.app.batch_source.set("current")
    gui.app.batch_unc_enabled.set(compute_uncertainty)
    gui.run_batch(spectrum.parent, out_dir / "batch_summary.csv")
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("solver", ["modern_vp", "classic"])
@pytest.mark.parametrize("use_baseline", [True, False], ids=["baseline", "no_baseline"])
def test_fit_and_export_write_the_batch_fit_table_and_trace(gui, monkeypatch, tmp_path, spectrum, solver, use_baseline):
    gui.open_file(spectrum)
    gui.app.solver_choice.set(solver)
    gui.set_window(*WINDOW)
    gui.app.use_baseline.set(use_baseline)
    _seed(gui)
    gui.fit()
    (tmp_path / "gui").mkdir()
    gui.export(tmp_path / "gui" / SPECTRUM)

    _batch_like_the_gui(gui, monkeypatch, spectrum, tmp_path / "batch", compute_uncertainty=False)

    for suffix in ("_fit.csv", "_trace.csv"):
        gui_file, batch_file = tmp_path / "gui" / f"{STEM}{suffix}", tmp_path / "batch" / f"{STEM}{suffix}"
        assert gui_file.exists() and batch_file.exists(), suffix
        assert table_differences(gui_file, batch_file) == [], suffix


def _canned_bootstrap(*, theta, x_all, alpha=0.05, **_):
    """One uncertainty result, the same for whatever path asks: stats from theta,
    a band on the x it is given (always: which paths ask for a band is the
    uncertainty settings' business, not the writer's)."""

    theta = np.asarray(theta, float)
    sd = 0.01 * np.abs(theta) + 0.1
    stats = {
        f"p{i}": {"est": float(t), "sd": float(s), "p2.5": float(t - 2 * s), "p97.5": float(t + 2 * s)}
        for i, (t, s) in enumerate(zip(theta, sd))
    }
    x = np.asarray(x_all, float)
    band = (x, 500.0 + 0.001 * x, 520.0 + 0.001 * x)
    return core_uncertainty.UncertaintyResult(
        method="bootstrap", label="Bootstrap", stats=stats,
        diagnostics={"alpha": float(alpha), "n_boot": 20, "n_fail": 0}, band=band,
    )


def test_one_uncertainty_result_is_written_as_batch_writes_it(gui, monkeypatch, tmp_path, spectrum):
    # The writer, isolated: both paths get the same uncertainty result.
    import batch.runner as runner

    for module in (core_uncertainty, runner):
        monkeypatch.setattr(module, "bootstrap_ci", _canned_bootstrap, raising=False)
    try:
        import core.pipeline as pipeline
    except ImportError:
        pass
    else:
        monkeypatch.setattr(pipeline, "bootstrap_ci", _canned_bootstrap, raising=False)

    gui.open_file(spectrum)
    gui.set_window(*WINDOW)
    gui.app.unc_method.set("Bootstrap")
    _seed(gui)
    gui.fit()
    (tmp_path / "gui").mkdir()
    gui.export(tmp_path / "gui" / SPECTRUM)

    _batch_like_the_gui(gui, monkeypatch, spectrum, tmp_path / "batch", compute_uncertainty=True)

    gui_dir, batch_dir = tmp_path / "gui", tmp_path / "batch"
    for suffix in ("_uncertainty.csv", "_uncertainty_band.csv"):
        assert (gui_dir / f"{STEM}{suffix}").exists() and (batch_dir / f"{STEM}{suffix}").exists(), suffix
        assert table_differences(gui_dir / f"{STEM}{suffix}", batch_dir / f"{STEM}{suffix}") == [], suffix
    assert text_differences(gui_dir / f"{STEM}_uncertainty.txt", batch_dir / f"{STEM}_uncertainty.txt") == []
    # One writer: neither path writes a file for this spectrum that the other does not.
    def spectrum_files(folder):
        return sorted(p.name for p in folder.iterdir() if p.name.startswith(STEM))

    assert spectrum_files(gui_dir) == spectrum_files(batch_dir)


# --- the uncertainty, computed: one context for the GUI and batch -------------

METHOD_LABELS = {"asymptotic": "Asymptotic", "bootstrap": "Bootstrap", "bayesian": "Bayesian"}


@pytest.mark.parametrize("method", ["asymptotic", "bootstrap", "bayesian"])
def test_fit_export_and_uncertainty_write_what_batch_writes(gui, monkeypatch, tmp_path, spectrum, method):
    """Fit, then Export with the uncertainty computed, against Run Batch with
    uncertainty: every file of the spectrum matches, uncertainty files included.
    Seed all makes the bootstrap and the chain repeatable; alpha 0.1 on the
    asymptotic case checks that the level reaches the context and the labels."""

    app = gui.app
    gui.open_file(spectrum)
    gui.set_window(*WINDOW)
    app.perf_seed_all.set(True)
    app.seed_var.set("11")
    app.unc_method.set(METHOD_LABELS[method])
    app.boot_n_var.set(40)
    app.bayes_burn_var.set(100)
    app.bayes_steps_var.set(400)
    if method == "asymptotic":
        app.alpha_var.set(0.1)
    _seed(gui)
    gui.fit()
    (tmp_path / "gui").mkdir()
    gui.export(tmp_path / "gui" / SPECTRUM)

    _batch_like_the_gui(gui, monkeypatch, spectrum, tmp_path / "batch", compute_uncertainty=True)

    gui_dir, batch_dir = tmp_path / "gui", tmp_path / "batch"
    files = sorted(p.name for p in gui_dir.iterdir() if p.name.startswith(STEM))
    assert files == sorted(p.name for p in batch_dir.iterdir() if p.name.startswith(STEM))
    assert {f"{STEM}_uncertainty.csv", f"{STEM}_uncertainty.txt"} <= set(files)
    for name in files:
        diff = text_differences if name.endswith(".txt") else table_differences
        assert diff(gui_dir / name, batch_dir / name) == [], name
    if method == "asymptotic":
        header = (gui_dir / f"{STEM}_uncertainty.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "p5" in header.split(",") and "p95" in header.split(",")
        assert "90% CI" in (gui_dir / f"{STEM}_uncertainty.txt").read_text(encoding="utf-8")
