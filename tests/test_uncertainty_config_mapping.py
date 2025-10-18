from ui.uncertainty_utils import method_display_to_key


def test_method_display_to_key_bootstrap_label():
    assert method_display_to_key("Bootstrap (percentile)") == "bootstrap"


def test_method_display_to_key_bayesian_prefix():
    assert method_display_to_key("Bayesian") == "bayesian"


def test_method_display_to_key_asymptotic_default():
    assert method_display_to_key("") == "asymptotic"
    assert method_display_to_key(None) == "asymptotic"


def test_method_display_to_key_passthrough_for_unknown():
    assert method_display_to_key("custom") == "custom"
