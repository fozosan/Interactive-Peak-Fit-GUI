import numpy as np
import pytest
from core import fit_api, uncertainty


def test_bayesian_optional(two_peak_data):
    try:
        res_bayes = uncertainty.bayesian_ci(
            engine=fit_api.run_fit_consistent,
            data=two_peak_data,
            walkers=16,
            steps=50,
            burn=10,
            thin=1,
            band=False,
        )
    except uncertainty.NotAvailable:
        pytest.skip("emcee not installed")

    res_asym = fit_api.run_fit_consistent(**two_peak_data, return_jacobian=True)
    asym = uncertainty.asymptotic_ci(
        res_asym["theta"], res_asym["residual_fn"], res_asym["jacobian"], res_asym["ymodel_fn"],
    )
    assert np.all(np.isfinite(res_bayes["param_std"]))
    assert np.allclose(res_bayes["param_std"], asym["param_std"], rtol=0.3)
