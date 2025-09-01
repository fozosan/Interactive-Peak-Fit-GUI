import pathlib
import sys

import pytest

# make package importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.data_io import canonical_unc_label


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("covariance", "Asymptotic (JᵀJ)"),
        ("J^T J", "Asymptotic (JᵀJ)"),
        ("residual bootstrap", "Bootstrap (residual)"),
        ("percentile resampling", "Bootstrap (residual)"),
        ("emcee sampler", "Bayesian (MCMC)"),
        ("MCMC chain", "Bayesian (MCMC)"),
    ],
)
def test_unc_label_aliases(alias, expected):
    """All supported aliases should resolve to canonical labels."""
    assert canonical_unc_label(alias) == expected

