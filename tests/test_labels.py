import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ui import app


def f(s: str) -> str:
    return app.format_axis_label_inline(s)


def test_plain_text_passthrough():
    assert f("Raman shift") == "Raman shift"


def test_superscript_fragment_braced_spaces_trimmed():
    assert f("Raman shift cm^{ -1 }") == "Raman shift cm$^{-1}$"


def test_subscript_fragment():
    assert f("E_{g}") == "E$_g$"


def test_existing_math_passthrough():
    assert f("$k^{-1}$ cm") == "$k^{-1}$ cm"


def test_escaped_literals_not_wrapped():
    assert f(r"Raman\_shift cm^{ -1 }") == "Raman_shift cm$^{-1}$"


def test_multiple_fragments_mixed():
    s = "I_D/I_G (a.u.) cm^{ -1 } Å_{ -1 }"
    expect = "I$_D$/I$_G$ (a.u.) cm$^{-1}$ Å$_{-1}$"
    assert f(s) == expect

