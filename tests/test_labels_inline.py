import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import ui.app as app

def f(s, enabled=True):
    return app.format_axis_label_inline(s, enabled)

def test_plain_text_passthrough():
    assert f("Raman shift") == "Raman shift"

def test_existing_math_preserved():
    assert f("A $k^{-1}$ cm") == "A $k^{-1}$ cm"

def test_superscripts():
    assert f("cm^-1") == "cm$^{-1}$"
    assert f("cm^{ -1 }") == "cm$^{-1}$"

def test_subscripts():
    assert f("E_g") == "E$_g$"
    assert f("k_B") == "k$_B$"
    assert f("E_sub3") == "E$_{sub3}$"

def test_mixed():
    s = "I_0/I^ref cm^-1"
    expect = "I$_0$/I$^{ref}$ cm$^{-1}$"
    assert f(s) == expect

def test_escapes():
    assert f(r"Raman\_shift cm\^-1") == "Raman_shift cm^-1"

def test_toggle_off():
    s = "I_0 cm^-1"
    assert f(s, enabled=False) == s
