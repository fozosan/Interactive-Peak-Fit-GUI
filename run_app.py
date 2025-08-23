"""Convenience launcher for the Peakfit 3.x prototype UI.

Ensures that the package imports resolve when executed from an IDE like
PyCharm or from the command line.
"""
from ui.app import main

if __name__ == "__main__":
    main()
