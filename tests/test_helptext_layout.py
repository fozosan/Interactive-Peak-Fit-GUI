"""The Help window shows build_help() as it comes, so the text must start at column 0.

build_help() dedents one indented triple-quoted string. A single line typed at
column 0 empties the common margin, and every other line then keeps its indent.
"""
from ui.helptext import build_help


def _headings(text):
    """(title, underline) pairs: a line of only '-' or '=' under its title."""
    lines = text.splitlines()
    return [
        (lines[i - 1], line)
        for i, line in enumerate(lines)
        if i > 0 and len(line.strip()) >= 3 and set(line.strip()) <= set("-=")
    ]


def test_section_headings_start_at_column_zero():
    headings = _headings(build_help({}))
    titles = [title.strip() for title, _ in headings]
    assert "Troubleshooting" in titles
    assert "Font Rendering on Linux (jagged text?)" in titles
    indented = [pair for pair in headings if any(s != s.lstrip() for s in pair)]
    assert indented == []
