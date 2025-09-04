import pytest
from core.data_io import canonical_unc_label as canon


@pytest.mark.parametrize("alias, expected", [
    ("asym", "Asymptotic (JᵀJ)"),
    ("jtj", "Asymptotic (JᵀJ)"),
    ("j^t j", "Asymptotic (JᵀJ)"),
    ("jᵀ j", "Asymptotic (JᵀJ)"),
    ("gauss", "Asymptotic (JᵀJ)"),
    ("hessian", "Asymptotic (JᵀJ)"),
    ("linearized", "Asymptotic (JᵀJ)"),
    ("curvature", "Asymptotic (JᵀJ)"),
    ("cov", "Asymptotic (JᵀJ)"),
    ("covariance", "Asymptotic (JᵀJ)"),
    ("covmatrix", "Asymptotic (JᵀJ)"),
    ("boot", "Bootstrap"),
    ("bootstrap", "Bootstrap"),
    ("resample", "Bootstrap"),
    ("resampling", "Bootstrap"),
    ("resid", "Bootstrap"),
    ("residual", "Bootstrap"),
    ("percentile", "Bootstrap"),
    ("perc", "Bootstrap"),
    ("bayes", "Bayesian (MCMC)"),
    ("bayesian", "Bayesian (MCMC)"),
    ("mcmc", "Bayesian (MCMC)"),
    ("emcee", "Bayesian (MCMC)"),
    ("pymc", "Bayesian (MCMC)"),
    ("numpyro", "Bayesian (MCMC)"),
    ("hmc", "Bayesian (MCMC)"),
    ("nuts", "Bayesian (MCMC)"),
    ("posterior", "Bayesian (MCMC)"),
    ("chain", "Bayesian (MCMC)"),
])
def test_canonical_unc_label_aliases(alias, expected):
    assert canon(alias) == expected


@pytest.mark.parametrize("alias", ["", "foobar", None])
def test_canonical_unc_label_unknown(alias):
    assert canon(alias) == "unknown"
