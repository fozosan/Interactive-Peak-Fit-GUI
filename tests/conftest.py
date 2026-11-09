import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Keep the suite away from the user's settings. ui.app and gl_decon_gui_batch.py
# set CONFIG_PATH to ui/ui_config/gl_peakfit_config.json in this repo when they
# are imported, and read the old ~/.gl_peakfit_config.json while that file is
# absent, so a test that saves a setting would write the user's real file.
# Pytest imports this conftest before any test module, so GL_PEAKFIT_CONFIG set
# here, ahead of every other import, sends every import of either module, fresh
# re-imports included, to a throwaway file with no old file. HOME moves too, for
# code that joins paths to it; Windows reads USERPROFILE instead of HOME.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="peakfit-tests-home-"))
_TEST_CONFIG = _TEST_HOME / ".gl_peakfit_config.json"
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)
os.environ["HOME"] = os.environ["USERPROFILE"] = str(_TEST_HOME)
os.environ["GL_PEAKFIT_CONFIG"] = str(_TEST_CONFIG)
_REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "ui" / "ui_config"

from tkinter import filedialog, messagebox, simpledialog  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Answer Tk dialogs without opening them. A real dialog waits for a click that
# never comes in an unattended run: bare pytest hung in messagebox.showinfo inside
# test_baseline_method_persistence. Message boxes return their default button, as
# if Enter were pressed; the text prompt and the file pickers return what Cancel
# returns. A test that needs another answer monkeypatches the function itself.
_DIALOG_ANSWERS = [
    (messagebox, ("showinfo", "showwarning", "showerror"), messagebox.OK),
    (messagebox, ("askokcancel", "askyesno", "askyesnocancel", "askretrycancel"), True),
    (messagebox, ("askquestion",), messagebox.YES),
    (simpledialog, ("askstring",), None),
    (filedialog, ("askopenfilename", "asksaveasfilename", "askdirectory"), ""),
]


def _answer(value):
    return lambda *args, **kwargs: value


def _answer_dialogs(set_attribute=setattr):
    for module, names, value in _DIALOG_ANSWERS:
        for name in names:
            set_attribute(module, name, _answer(value))


_answer_dialogs()


def _repo_config_state():
    """The user's real settings folder in this repo: file names, sizes, mtimes."""
    if not _REPO_CONFIG_DIR.is_dir():
        return None
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in _REPO_CONFIG_DIR.iterdir())


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch):
    """Start every test with no saved settings and the stock dialog answers.

    Otherwise what one test leaves behind reaches every later test in the run:
    test_ui_classic_defaults saves the classic solver, the re-import in
    test_config_migration_baseline_defaults points ui.app at that test's legacy
    file, and test_single_export_wide_toggle replaces showinfo and
    asksaveasfilename without monkeypatch. After the test, ui/ui_config must be
    as it was: a test that wrote there changed the user's real settings.
    """
    _TEST_CONFIG.unlink(missing_ok=True)
    for name in ("ui.app", "gl_decon_gui_batch"):
        module = sys.modules.get(name)
        if module is not None:
            monkeypatch.setattr(module, "CONFIG_PATH", _TEST_CONFIG)
    _answer_dialogs(monkeypatch.setattr)
    before = _repo_config_state()
    yield
    assert _repo_config_state() == before, f"this test changed the user's settings in {_REPO_CONFIG_DIR}"


# Use non-interactive backend when headless
HEADLESS = (os.environ.get("DISPLAY", "") == "" and os.name != "nt")
if HEADLESS:
    matplotlib.use("Agg")

# ensure project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

# deterministic RNG fixture
@pytest.fixture
def rng():
    return np.random.default_rng(123)

# synthetic two-peak data fixture for core-level tests
@pytest.fixture
def two_peak_data(rng):
    from core import peaks, models

    x = np.linspace(-5.0, 5.0, 201)
    seeds = [
        peaks.Peak(-0.8, 1.0, 0.6, 0.5),
        peaks.Peak(0.9, 0.8, 0.5, 0.4),
    ]
    y = models.pv_sum(x, seeds)
    mask = np.ones_like(x, bool)
    cfg = {
        "solver": "modern_vp",
        "solver_loss": "linear",
        "solver_weight": "none",
        "solver_restarts": 1,
        "perf_seed_all": True,
    }
    data = {
        "x": x,
        "y": y,
        "peaks_in": seeds,
        "cfg": cfg,
        "baseline": None,
        "mode": "add",
        "fit_mask": mask,
        "rng_seed": 123,
    }
    return data

# marker for GUI dependent tests
skip_if_no_display = pytest.mark.skipif(HEADLESS, reason="requires display")

# numeric comparison helper

def close_to(a, b, rtol=5e-5, atol=5e-8):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def bayes_knobs(*, walkers=0, burn=1000, steps=4000, thin=1):
    """
    Canonical knobs for strict router:
    - walkers: 0 means 'auto'
    """
    return {
        "bayes_walkers": int(walkers),
        "bayes_burn": int(burn),
        "bayes_steps": int(steps),
        "bayes_thin": int(thin),
    }


def bootstrap_cfg(n=200, jitter=0.0):
    return {
        "bootstrap_n": int(n),
        "bootstrap_jitter": float(jitter),
    }


def ensure_unc_common(cfg: dict) -> dict:
    """
    Fill uncertainty flags used by GUI/Batch parity and bands/diagnostics.
    Keeps tests explicit but DRY.
    """
    out = dict(cfg)
    out.setdefault("bayes_diagnostics", False)
    out.setdefault("bayes_band_enabled", False)
    out.setdefault("bayes_band_force", False)
    out.setdefault("bayes_band_max_draws", 512)
    out.setdefault("bayes_diag_ess_min", 200.0)
    out.setdefault("bayes_diag_rhat_max", 1.05)
    out.setdefault("bayes_diag_mcse_mean", float("inf"))
    out.setdefault("perf_parallel_strategy", "outer")
    out.setdefault("perf_blas_threads", 0)
    out.setdefault("unc_center_resid", True)
    return out

# helper to ensure CSVs have no blank lines

@pytest.fixture
def no_blank_lines():
    def _check(path: Path) -> bool:
        text = Path(path).read_text()
        return "\n\n" not in text
    return _check


def _maybe_read_unc_files(basedir: Path, stem: str):
    """
    Return (wide_df, long_df, used_path) where either or both can be None.
    Finds <stem>_uncertainty_wide.csv first, else <stem>_uncertainty.csv.
    """
    base = basedir / stem
    wide = base.with_name(base.name + "_uncertainty_wide.csv")
    long = base.with_name(base.name + "_uncertainty.csv")

    wide_df = pd.read_csv(wide) if wide.exists() else None
    long_df = pd.read_csv(long) if long.exists() else None
    used = wide if wide_df is not None else (long if long_df is not None else None)
    return wide_df, long_df, used


def _pivot_long_to_wide(long_df: "pd.DataFrame"):
    """
    Convert long schema:
      file, peak, param, value, stderr, ci_lo, ci_hi, method, rmse, dof, p2_5, p97_5, ...
    into a wide per-peak DataFrame with legacy columns:
      file, peak, method, rmse, dof, backend, n_draws, n_boot, ess, rhat,
      center, center_stderr, center_ci_lo, center_ci_hi, center_p2_5, center_p97_5,
      height, ...
    """
    if long_df is None or long_df.empty:
        return None

    meta_cols = ["file","peak","method","rmse","dof","backend","n_draws","n_boot","ess","rhat"]
    for c in meta_cols:
        if c not in long_df.columns:
            long_df[c] = ""

    blocks = {}
    for valcol, suffix in [
        ("value", ""), ("stderr","_stderr"), ("ci_lo","_ci_lo"),
        ("ci_hi","_ci_hi"), ("p2_5","_p2_5"), ("p97_5","_p97_5"),
    ]:
        if valcol in long_df.columns:
            pvt = long_df.pivot_table(
                index=["file","peak"],
                columns="param",
                values=valcol,
                aggfunc="first",
            )
            pvt.columns = [f"{p}{suffix}" for p in pvt.columns]
            blocks[valcol] = pvt

    wide = None
    for pvt in blocks.values():
        wide = pvt if wide is None else wide.join(pvt, how="outer")

    meta = (
        long_df
        .sort_values(["file","peak"])
        .groupby(["file","peak"], as_index=True)[meta_cols]
        .first()
    )
    if wide is None:
        wide = meta.copy()
    else:
        wide = meta.join(wide, how="left")

    wide = wide.reset_index()
    ordered = [
        "file","peak","method","rmse","dof","backend","n_draws","n_boot","ess","rhat",
        "center","center_stderr","center_ci_lo","center_ci_hi","center_p2_5","center_p97_5",
        "height","height_stderr","height_ci_lo","height_ci_hi","height_p2_5","height_p97_5",
        "fwhm","fwhm_stderr","fwhm_ci_lo","fwhm_ci_hi","fwhm_p2_5","fwhm_p97_5",
        "eta","eta_stderr","eta_ci_lo","eta_ci_hi","eta_p2_5","eta_p97_5",
    ]
    final_cols = [c for c in ordered if c in wide.columns] + [c for c in wide.columns if c not in ordered]
    return wide[final_cols]
