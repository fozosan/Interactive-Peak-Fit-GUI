"""One bad file costs a batch only that file (game plan Phase 0.3).

Each case puts one bad file, ``m_bad.csv``, between two good ones. The batch
must still fit ``z_good.csv`` after it, give every matched file a row in the
status table ``<output_base>_files.csv``, and say why the bad file was not
fitted. Before this was fixed, six of these cases aborted the whole batch with
no summary, and the too-few-points case was reported as a good fit.
"""
import csv
import os
import threading

import numpy as np
import pytest

from batch import runner
from core import pipeline
from core import fit_api, models
from core.peaks import Peak
from fit import classic
from tests.conftest import bayes_knobs, bootstrap_cfg, ensure_unc_common

STATUS_COLUMNS = [
    "file", "path", "status", "flags", "message", "n_points_window", "n_peaks",
    "rmse", "fit_ok", "baseline_fit", "baseline_order", "noise_s0", "noise_g", "noise_n", "noise_method",
    "unc_method", "unc_status", "unc_message",
]
TRUE_PEAKS = [Peak(40.0, 100.0, 5.0, 0.5), Peak(60.0, 80.0, 6.0, 0.5)]
WINDOW = (20.0, 80.0)


def _spectrum(n=201):
    x = np.linspace(0.0, 100.0, n)
    noise = np.random.default_rng(0).normal(0.0, 0.5, x.size)
    return x, models.pv_sum(x, TRUE_PEAKS) + 10.0 + noise


def _write(path, x, y):
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")


def _config(out_dir, solver="modern_vp"):
    cfg = {
        "peaks": [
            {"center": 41.0, "height": 90.0, "fwhm": 4.5, "eta": 0.5,
             "lock_center": False, "lock_width": False},
            {"center": 59.0, "height": 70.0, "fwhm": 5.5, "eta": 0.5,
             "lock_center": False, "lock_width": False},
        ],
        "solver": solver,
        "mode": "add",
        "baseline": {"method": "als", "lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": False,
        "source": "template",
        "reheight": False,
        "auto_max": 5,
        solver: {"centers_in_window": True},
        "baseline_uses_fit_range": True,
        "perf_seed_all": False,
        "perf_max_workers": 0,
        "fit_xmin": WINDOW[0],
        "fit_xmax": WINDOW[1],
        "output_dir": str(out_dir),
        "output_base": "batch",
    }
    cfg.update(ensure_unc_common({}))
    cfg.update(bootstrap_cfg(n=50))
    cfg.update(bayes_knobs())
    return cfg


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# Each case writes the bad file, patches what it needs, and returns the status
# the bad file should get and whether its fit produced peak rows.

def _window_outside_data(path, monkeypatch, cfg):
    x, y = _spectrum()
    _write(path, x + 1000.0, y)
    return "skipped_no_points", False


def _nan_in_window(path, monkeypatch, cfg):
    x, y = _spectrum()
    y[np.argmin(np.abs(x - 50.0))] = np.nan
    _write(path, x, y)
    return "fit_error", False


def _unreadable_file(path, monkeypatch, cfg):
    _write(path, *_spectrum())
    path.chmod(0)
    if os.access(path, os.R_OK):  # root, or a filesystem without permission bits
        path.chmod(0o644)
        pytest.skip("cannot make a file unreadable here")
    return "skipped_unreadable", False


def _directory_named_csv(path, monkeypatch, cfg):
    path.mkdir()
    return "skipped_unreadable", False


def _classic_not_converging(path, monkeypatch, cfg):
    # The bad file has a finer grid, so its fit window has more points; only its
    # fit runs out of function evaluations, through scipy's own failure.
    x, y = _spectrum(301)
    _write(path, x, y)
    cfg["solver"] = "classic"
    cfg["classic"] = {"centers_in_window": True}
    n_bad = int(np.count_nonzero((x >= WINDOW[0]) & (x <= WINDOW[1])))
    real_curve_fit = classic.curve_fit

    def curve_fit(f, xdata, *args, **kwargs):
        if len(xdata) == n_bad:
            kwargs["maxfev"] = 1
        return real_curve_fit(f, xdata, *args, **kwargs)

    monkeypatch.setattr(classic, "curve_fit", curve_fit)
    return "not_converged", False


def _fit_not_ok_with_uncertainty(path, monkeypatch, cfg):
    x, y = _spectrum(301)
    _write(path, x, y)
    real_fit = fit_api.run_fit_consistent

    def run_fit_consistent(x_in, *args, **kwargs):
        res = real_fit(x_in, *args, **kwargs)
        if len(x_in) == len(x):
            res["fit_ok"] = False
        return res

    monkeypatch.setattr(fit_api, "run_fit_consistent", run_fit_consistent)
    return "not_converged", True


def _too_few_points_in_window(path, monkeypatch, cfg):
    # Four points inside the window for two peaks with six free parameters.
    x, y = _spectrum()
    keep = (x < WINDOW[0]) | (x > WINDOW[1]) | (np.abs(x - 40.0) < 0.6) | (x == 60.0)
    assert np.count_nonzero(keep & (x >= WINDOW[0]) & (x <= WINDOW[1])) == 4
    _write(path, x[keep], y[keep])
    return "skipped_no_points", False


CASES = {
    "window_outside_data": _window_outside_data,
    "nan_in_window": _nan_in_window,
    "unreadable_file": _unreadable_file,
    "directory_named_csv": _directory_named_csv,
    "classic_not_converging": _classic_not_converging,
    "fit_not_ok_with_uncertainty": _fit_not_ok_with_uncertainty,
    "too_few_points_in_window": _too_few_points_in_window,
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_bad_file_does_not_stop_the_batch(tmp_path, monkeypatch, case):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    x, y = _spectrum()
    _write(in_dir / "a_good.csv", x, y)
    _write(in_dir / "z_good.csv", x, y)
    cfg = _config(out_dir)
    expected, bad_has_rows = CASES[case](in_dir / "m_bad.csv", monkeypatch, cfg)

    ok, n = runner.run_batch(
        [str(in_dir / "*.csv")], cfg,
        compute_uncertainty=True, unc_method="Asymptotic", log=lambda m: None,
    )

    assert (ok, n) == (2, 3)
    status = _rows(out_dir / "batch_files.csv")
    assert [r["file"] for r in status] == ["a_good.csv", "m_bad.csv", "z_good.csv"]
    assert [r["status"] for r in status] == ["ok", expected, "ok"]
    bad = status[1]
    assert bad["message"], "the status table must say why the file was not fitted"
    assert bad["fit_ok"] == ("" if expected.startswith("skipped") else "False")
    assert bad["unc_status"] == "skipped"
    assert status[2]["unc_status"] == "ok"
    assert (out_dir / "z_good_uncertainty.csv").exists()
    assert not (out_dir / "m_bad_uncertainty.csv").exists()

    peak_files = [r["file"] for r in _rows(out_dir / "batch_fit.csv")]
    assert peak_files.count("z_good.csv") == 2
    assert ("m_bad.csv" in peak_files) is bad_has_rows


def test_the_status_table_has_the_documented_columns(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    _write(in_dir / "a.csv", *_spectrum())

    runner.run_batch([str(in_dir / "*.csv")], _config(out_dir), compute_uncertainty=False)

    with open(out_dir / "batch_files.csv", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == STATUS_COLUMNS
    (row,) = _rows(out_dir / "batch_files.csv")
    assert row["status"] == "ok" and row["fit_ok"] == "True" and row["flags"] == ""
    assert row["path"] == str(in_dir / "a.csv")
    assert int(row["n_points_window"]) == 121 and int(row["n_peaks"]) == 2
    assert float(row["rmse"]) < 1.0
    assert row["unc_method"] == row["unc_status"] == row["unc_message"] == ""


def test_the_count_is_the_files_matched(tmp_path):
    # The second number used to be the position of the last file fitted, so a
    # skipped last file made it 1 here.
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    _write(in_dir / "a.csv", *_spectrum())
    (in_dir / "z.csv").write_text("not a spectrum\n")

    ok, n = runner.run_batch([str(in_dir / "*.csv")], _config(out_dir), compute_uncertainty=False)

    assert (ok, n) == (1, 2)
    assert [r["status"] for r in _rows(out_dir / "batch_files.csv")] == ["ok", "skipped_format"]


def test_an_abort_writes_a_partial_summary(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    for name in ("a.csv", "b.csv", "c.csv"):
        _write(in_dir / name, *_spectrum())
    abort = threading.Event()

    def progress(i, total, path):
        if i == 1:  # the user presses Abort while the first file is being fitted
            abort.set()

    res = runner.run_batch(
        [str(in_dir / "*.csv")], _config(out_dir),
        compute_uncertainty=False, progress=progress, abort_event=abort,
    )

    assert isinstance(res, dict) and res["aborted"]
    status = _rows(out_dir / "batch_files.csv")
    assert [(r["file"], r["status"]) for r in status] == [
        ("a.csv", "ok"), ("b.csv", "not_processed"), ("c.csv", "not_processed"),
    ]
    assert {r["file"] for r in _rows(out_dir / "batch_fit.csv")} == {"a.csv"}


def test_stale_uncertainty_files_are_removed_when_uncertainty_now_fails(tmp_path, monkeypatch):
    # A rerun into the same folder whose uncertainty fails used to leave the
    # previous run's files in place, looking current.
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    _write(in_dir / "a.csv", *_spectrum())
    cfg = _config(out_dir)
    runner.run_batch([str(in_dir / "a.csv")], cfg, compute_uncertainty=True, unc_method="Asymptotic")
    assert (out_dir / "a_uncertainty.csv").exists()
    (out_dir / "a_uncertainty_wide.csv").write_text("old\n")
    (out_dir / "a_band.csv").write_text("old\n")

    def failing_router(*args, **kwargs):
        raise RuntimeError("synthetic uncertainty failure")

    monkeypatch.setattr(pipeline, "route_uncertainty", failing_router)  # batch uncertainty runs there
    ok, n = runner.run_batch([str(in_dir / "a.csv")], cfg, compute_uncertainty=True, unc_method="Asymptotic")

    assert (ok, n) == (1, 1)
    (row,) = _rows(out_dir / "batch_files.csv")
    assert (row["status"], row["unc_status"]) == ("ok", "failed")
    assert "synthetic uncertainty failure" in row["unc_message"]
    for suffix in ("_uncertainty.csv", "_uncertainty_wide.csv", "_uncertainty.txt",
                   "_uncertainty_band.csv", "_band.csv"):
        assert not (out_dir / f"a{suffix}").exists(), suffix
