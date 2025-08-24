"""User interface package for Peakfit 3.x.

This module intentionally avoids importing submodules on package import.
Doing so prevents premature import errors when the repository root has not
yet been added to ``sys.path`` (e.g., when launching the UI from an IDE).
Submodules like :mod:`ui.app` and :mod:`ui.helptext` should be imported
explicitly by clients as needed.
"""

__all__ = ["app", "helptext"]
