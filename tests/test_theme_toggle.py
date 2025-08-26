import pytest

from ui.app import PeakFitApp


def test_no_theme_hooks():
    assert not hasattr(PeakFitApp, "apply_theme")
