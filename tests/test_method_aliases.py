from core import data_io


def test_canonical_unc_label_aliases():
    assert data_io.canonical_unc_label("asymptotic") in (
        "Asymptotic",
        "Asymptotic (J^T J)",
        "Asymptotic (JᵀJ)",
    )
    assert data_io.canonical_unc_label("ASYM") in (
        "Asymptotic",
        "Asymptotic (J^T J)",
        "Asymptotic (JᵀJ)",
    )
    assert data_io.canonical_unc_label("bootstrap") in (
        "Bootstrap",
        "Bootstrap (residual)",
    )
    assert data_io.canonical_unc_label("bayes") in ("Bayesian", "MCMC", "Bayesian (MCMC)")

