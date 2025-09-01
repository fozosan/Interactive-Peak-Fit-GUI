import sys
import pathlib

# make project modules importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from batch import runner as batch_runner


def test_batch_uncertainty_exports(tmp_path):
    src_dir = pathlib.Path(__file__).resolve().parent / "fixtures" / "batch_small"
    # ensure fixtures exist
    assert src_dir.exists()

    cfg = {
        'peaks': [ {'center':10.0,'height':1.0,'fwhm':1.0,'eta':0.5} ],
        'solver': 'classic',
        'mode': 'add',
        'baseline': {'lam':1e5, 'p':0.001, 'niter':10, 'thresh':0.0},
        'save_traces': True,
        'output_dir': str(tmp_path),
        'output_base': 'batch',
        'unc_method': 'Asymptotic (JᵀJ)',
    }
    # patterns for input files
    patterns = [str(src_dir / '*.csv')]
    batch_runner.run(patterns, cfg)

    outs = sorted(p for p in tmp_path.glob('*_uncertainty.csv') if not p.name.startswith('batch_uncertainty'))
    assert len(outs) == 2, 'Expected per-file uncertainty CSV(s) in out_dir'
    for csv_path in outs:
        stem = csv_path.name.replace('_uncertainty.csv','')
        txt = csv_path.with_name(f'{stem}_uncertainty.txt')
        band = csv_path.with_name(f'{stem}_uncertainty_band.csv')
        assert txt.exists(), f'Missing TXT report for {stem}'
        assert band.exists(), f'Missing band CSV for {stem}'
