"""Instrument ``.csv`` exports must load, and their results must stay separate.

The byte layouts are copied from WITec "Exported ASCII-File" spectra. Despite the
extension they are not comma-separated: tab-delimited, CRLF, Latin-1 (the micro
sign in ``PositionUnit``), with numbers in the metadata block.
"""
from __future__ import annotations

import numpy as np
import pytest

from batch import runner
from core.data_io import load_xy, write_uncertainty_csvs
from core.models import pv_sum
from core.peaks import Peak
from tests.conftest import ensure_unc_common

WITEC_HEADER = (
    b"//Exported ASCII-File\r\n"
    b"[Header]\r\n"
    b"FileName = \r\n"
    b"GraphName = Mono_MoS2_800g_standrad_50x_5sec_10a_0p6\r\n"
    b"SizeX = 1\r\n"
    b"SizeY = 1\r\n"
    b"SizeGraph = 1650\r\n"
    b"PositionUnit = \xb5m\r\n"
    b"PositionX = -13342.15\r\n"
    b"PositionY = -2695.075\r\n"
    b"PositionZ = 0\r\n"
    b"XAxisUnit = rel. 1/cm\r\n"
    b"DataUnit = CCD cts\r\n"
    b"\r\n"
    b"[Data]\r\n"
    b"X-Axis\tMono_MoS2_800g_standrad_50x_5sec_10a_0p6\r\n"
    b"rel. 1/cm\tCCD cts\r\n"
)


def _write_spectrum(path) -> None:
    x = np.linspace(-5, 5, 101)
    y = pv_sum(x, [Peak(0.0, 1.0, 1.0, 0.5)])
    np.savetxt(path, np.column_stack([x, y]), delimiter=",")


def _batch_config(out_dir) -> dict:
    return {
        "peaks": [{"center": 0.0, "height": 1.0, "fwhm": 1.0, "eta": 0.5}],
        "solver": "classic",
        "mode": "add",
        "baseline": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
        "save_traces": True,
        "source": "template",
        "output_dir": str(out_dir),
        "output_base": "batch",
        **ensure_unc_common({}),
    }


def test_witec_export_keeps_only_the_data_rows(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_bytes(
        WITEC_HEADER
        + b"-1.32994E+02\t 5.05700E+02\r\n"
        + b"-1.32181E+02\t 5.04300E+02\r\n"
        + b" 1.06886E+03\t 5.03200E+02\r\n"
    )

    x, y = load_xy(str(path))

    assert x.tolist() == [-132.994, -132.181, 1068.86]
    assert y.tolist() == [505.7, 504.3, 503.2]


@pytest.mark.parametrize(
    "rows",
    [
        b'"-1.32994E+02\t 5.04200E+02"\r\n"-1.32181E+02\t 5.04500E+02"\r\n',
        b'"-1.32994E+02","5.04200E+02"\r\n"-1.32181E+02","5.04500E+02"\r\n',
    ],
    ids=["whole-row-quoted", "each-field-quoted"],
)
def test_quotes_are_not_part_of_the_numbers(tmp_path, rows):
    path = tmp_path / "spectrum.csv"
    path.write_bytes(rows)

    x, y = load_xy(str(path))

    assert x.tolist() == [-132.994, -132.181]
    assert y.tolist() == [504.2, 504.5]


def test_three_numeric_columns_after_metadata_are_still_rejected(tmp_path):
    path = tmp_path / "map.csv"
    path.write_bytes(WITEC_HEADER + b"1.0\t2.0\t3.0\r\n4.0\t5.0\t6.0\r\n")

    # run_batch keys its "skip this file" branch on this wording.
    with pytest.raises(ValueError, match="more than two"):
        load_xy(str(path))


def test_rerunning_a_batch_in_its_own_output_folder_fits_only_the_spectrum(tmp_path):
    _write_spectrum(tmp_path / "a.csv")
    pattern = str(tmp_path / "*.csv")

    for _ in range(2):  # the second pass also matches every table the first wrote
        ok, _ = runner.run_batch(
            [pattern],
            _batch_config(tmp_path),
            compute_uncertainty=True,
            unc_method="Asymptotic",
        )
        assert ok == 1


def test_uncertainty_csvs_stay_separate_for_dotted_spectrum_names(tmp_path):
    # Same exposure series as the real data: the names differ only after the dot.
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _write_spectrum(in_dir / "MoS2_50x_0.10sec_10a 10.csv")
    _write_spectrum(in_dir / "MoS2_50x_0.20sec_10a 11.csv")
    out_dir = tmp_path / "out"

    runner.run_batch(
        [str(in_dir / "*.csv")],
        _batch_config(out_dir),
        compute_uncertainty=True,
        unc_method="Asymptotic",
    )

    assert (out_dir / "MoS2_50x_0.10sec_10a 10_uncertainty.csv").exists()
    assert (out_dir / "MoS2_50x_0.20sec_10a 11_uncertainty.csv").exists()


@pytest.mark.parametrize(
    "base, written",
    [
        ("out.csv", "out_uncertainty.csv"),
        ("OUT.CSV", "OUT_uncertainty.csv"),
        ("a_0.20sec_10a", "a_0.20sec_10a_uncertainty.csv"),
    ],
)
def test_only_a_csv_extension_is_dropped_from_the_export_base(tmp_path, base, written):
    unc_norm = {
        "label": "Asymptotic",
        "method": "Asymptotic",
        "stats": [{"peak": 1, "param": "center", "value": 1.0, "stderr": 0.1}],
    }

    write_uncertainty_csvs(tmp_path / base, "s.csv", unc_norm, write_wide=False)

    assert [p.name for p in tmp_path.iterdir()] == [written]
