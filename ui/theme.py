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
        sv_ttk.set_theme("dark", root)
        # sv-ttk recolours classic Tk widgets on <<ThemeChanged>>, which Tk sends
        # when idle and drops for a root that has no window yet. Deliver it now, so
        # the widgets the app builds next start with the dark palette: the
        # matplotlib toolbar picks black or light icons from its button background
        # when it is created.
        root.winfo_id()  # creates the root's window without mapping it
        root.event_generate("<<ThemeChanged>>")
        return True, "sv-ttk (Fluent Dark)"
    except ImportError:
        pass  # sv-ttk not installed, try next
    except Exception as e:
        print(f"[WARNING] Failed to apply sv-ttk theme: {e}", file=sys.stderr)

    # Priority 2: Try ttkthemes with Arc theme
    try:
        from ttkthemes import ThemedStyle
        # If we're here, sv-ttk failed, so we need to apply Arc to the existing root.
        # ThemedStyle loads the ttkthemes packages into this root's interpreter;
        # a plain ttk.Style cannot find "arc".
        ThemedStyle(root).set_theme("arc")
        return True, "ttkthemes (Arc)"
    except ImportError:
        pass  # ttkthemes not installed
    except Exception as e:
        print(f"[WARNING] Failed to apply ttkthemes Arc theme: {e}", file=sys.stderr)
    
    # Fallback: Use default Tkinter theme
    return False, "default (Tkinter)"


def paint_icon(image, color) -> None:
    """
    Paint every opaque pixel of a one-colour icon in ``color``.

    Pass the theme's button text colour, so the icon reads like the button
    labels on a dark theme and a light one alike (black on the default theme).
    """
    color = str(color or "")
    if not color:
        return
    for y in range(image.height()):
        for x in range(image.width()):
            if not image.transparency_get(x, y):
                image.put(color, (x, y))


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
