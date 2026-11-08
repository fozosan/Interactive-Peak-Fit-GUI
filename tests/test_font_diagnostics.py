"""--font-diagnostics: the report and the sample window."""
import os
from unittest.mock import MagicMock, patch

import pytest

from ui.font_diagnostics import run_diagnostics, show_visual_sample


def test_report_survives_a_failed_family_listing():
    with patch("tkinter.font.families", side_effect=RuntimeError("no font list")):
        results = run_diagnostics(MagicMock(), log_fn=lambda line: None)
    assert results["num_families"] is None
    assert results["found_sans_fonts"] == []
    assert results["found_mono_fonts"] == []


@pytest.mark.skipif(os.environ.get("DISPLAY", "") == "" and os.name != "nt", reason="requires display")
def test_sample_window_starts_with_tk_default_font():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        show_visual_sample(root)
        window = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
        frame = window.winfo_children()[0]
        samples = [w for w in frame.winfo_children() if isinstance(w, tk.Label)]
        assert str(samples[0].cget("font")) == "TkDefaultFont"
    finally:
        root.destroy()
