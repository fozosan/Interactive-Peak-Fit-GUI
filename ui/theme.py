"""
Theme management for Interactive Peak Fit GUI.

Attempts to apply modern themes with graceful fallback to default Tkinter styling.
Priority: sv-ttk (Microsoft Fluent design) > ttkthemes (Arc theme) > Default Tkinter

This module is non-fatal: if all theme libraries are missing or fail to apply,
the app will continue running with the default Tkinter appearance.
"""

import sys
from typing import Tuple


def apply_theme(root) -> Tuple[bool, str]:
    """
    Apply the best available modern theme to the Tkinter root window.
    
    Attempts to apply themes in this order:
    1. sv-ttk (Microsoft Fluent design - dark theme)
    2. ttkthemes with Arc theme (clean and modern)
    3. Default Tkinter (no changes if both unavailable)
    
    Args:
        root: The Tkinter root window (tk.Tk() instance)
    
    Returns:
        Tuple[bool, str]: (success, theme_name)
            - success: True if a theme was applied, False if using default
            - theme_name: Name of the applied theme or "default"
    
    This function is non-fatal and designed to never crash the app.
    """
    
    # Priority 1: Try sv-ttk (Microsoft Fluent Design)
    try:
        import sv_ttk
        sv_ttk.set_theme("dark")
        return True, "sv-ttk (Fluent Dark)"
    except ImportError:
        pass  # sv-ttk not installed, try next
    except Exception as e:
        print(f"[WARNING] Failed to apply sv-ttk theme: {e}", file=sys.stderr)
    
    # Priority 2: Try ttkthemes with Arc theme
    try:
        from ttkthemes import ThemedTk
        # If we're here, sv-ttk failed, so we need to apply Arc to the existing root
        # Note: ThemedTk wraps root creation, so we use a different approach
        # We'll set the theme style on the existing root via ttk.Style
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use("arc")
        return True, "ttkthemes (Arc)"
    except ImportError:
        pass  # ttkthemes not installed
    except Exception as e:
        print(f"[WARNING] Failed to apply ttkthemes Arc theme: {e}", file=sys.stderr)
    
    # Fallback: Use default Tkinter theme
    return False, "default (Tkinter)"


def log_theme_status(theme_applied: bool, theme_name: str) -> None:
    """
    Log the theme application status to stdout.
    
    Args:
        theme_applied: Whether a custom theme was successfully applied
        theme_name: Name of the applied theme
    """
    if theme_applied:
        print(f"[INFO] Theme applied: {theme_name}")
    else:
        print(f"[INFO] Using default Tkinter theme (custom themes not available)")
        print(f"[TIP] Install 'sv-ttk' or 'ttkthemes' for modern appearance:")
        print(f"      pip install sv-ttk")
        print(f"      # or")
        print(f"      pip install ttkthemes")
