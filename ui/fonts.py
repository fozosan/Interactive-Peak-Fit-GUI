"""
Central font policy for Interactive Peak Fit GUI.

Provides functions to:
1. Select available font families from a prioritized list
2. Configure Tk named fonts consistently across the app
3. Configure ttk styles with selected fonts

This module ensures consistent, portable font configuration that works on
Linux (X11/Wayland), Windows, and macOS without hardcoding distro-specific fonts.
"""

from typing import TYPE_CHECKING, Dict, Any, Callable, Optional

if TYPE_CHECKING:
    from tkinter import ttk

import tkinter as tk
import tkinter.font as tkfont


# Font candidates in priority order (sans-serif)
SANS_CANDIDATES = [
    "Ubuntu", 
    "Noto Sans", 
    "DejaVu Sans", 
    "Liberation Sans", 
    "Cantarell", 
    "Segoe UI", 
    "Arial"
]

# Font candidates in priority order (monospace)
MONO_CANDIDATES = [
    "Ubuntu Mono", 
    "DejaVu Sans Mono", 
    "Liberation Mono", 
    "Consolas", 
    "Courier New"
]

# Default UI font size (matches existing app defaults)
DEFAULT_UI_SIZE: int = 10

# Module-level cache of lookups, keyed by (id(root), candidates) so the sans
# and mono lookups on one root each keep their own answer
_font_cache: Dict[Any, Optional[str]] = {}


def pick_first_available_family(
    root: tk.Tk, 
    candidates: list[str]
) -> Optional[str]:
    """
    Find the first available font family from a prioritized list.
    
    Args:
        root: The Tkinter root window (tk.Tk() instance)
        candidates: List of candidate font names in priority order
        
    Returns:
        The actual family name as returned by tkfont.families(), preserving 
        Tk's casing, or None if no candidates are available.
        
    Note:
        This function caches results per root window and candidate list to
        avoid repeated calls to tkfont.families(), which can be slow.
    """

    # Check cache first (keyed by root id and candidate list)
    key = (id(root), tuple(candidates))
    if key in _font_cache:
        return _font_cache[key]

    # Get available families (case-insensitive matching)
    try:
        all_families = tkfont.families(root)

        for candidate in candidates:
            for available in all_families:
                if available.lower() == candidate.lower():
                    result = available  # preserve Tk's casing
                    _font_cache[key] = result
                    return result

        # No match found
        _font_cache[key] = None
        return None

    except Exception:
        # If enumeration fails, cache None to avoid repeated failures
        _font_cache[key] = None
        return None


def configure_named_fonts(
    root: tk.Tk, 
    *, 
    sans_family: Optional[str], 
    mono_family: Optional[str], 
    ui_size: int
) -> None:
    """
    Configure Tk named fonts with selected families.
    
    Sets the following named fonts on the provided root window:
    - TkDefaultFont (general UI, uses sans_family)
    - TkTextFont (text widgets, uses sans_family)
    - TkMenuFont (menus, uses sans_family)
    - TkFixedFont (monospace text like logs, uses mono_family)
    - TkHeadingFont (if available, uses sans_family)
    - TkSmallCaptionFont (if available, uses sans_family with size-1)
    - TkIconFont (if available, uses sans_family)
    - TkTooltipFont (if available, uses sans_family)
    
    Args:
        root: The Tkinter root window (tk.Tk() instance)
        sans_family: Sans-serif font family name (or None to skip changes)
        mono_family: Monospace font family name (or None to skip changes)
        ui_size: Base UI font size in points
        
    Side-effects:
        Modifies named fonts on the root window. Does not crash if a
        named font doesn't exist or can't be configured.
        
    Note:
        This function stores the selected sans_family in module-level
        variable _selected_sans for fallback use in PeakFitApp.__init__.
    """
    
    global _selected_sans  # type: ignore
    
    # Configure each named font with try/except per name
    named_fonts_config = [
        ("TkDefaultFont", sans_family, ui_size),
        ("TkTextFont", sans_family, ui_size),
        ("TkMenuFont", sans_family, ui_size),
        ("TkFixedFont", mono_family, ui_size),  # monospace for logs/console
    ]
    
    # Additional named fonts that may not exist on all systems
    optional_fonts = [
        "TkHeadingFont",
        "TkSmallCaptionFont", 
        "TkIconFont",
        "TkTooltipFont"
    ]
    
    for name, family, size in named_fonts_config:
        _configure_named_font(root, name, family, size)
    
    # Configure optional fonts
    for name in optional_fonts:
        try:
            font_obj = tkfont.nametofont(name)
            actual_props = font_obj.actual()
            
            if name == "TkSmallCaptionFont":
                # Smaller size for small captions
                _configure_named_font(root, name, sans_family, max(8, ui_size - 1))
            else:
                _configure_named_font(root, name, sans_family, ui_size)
        except Exception:
            # Font doesn't exist on this system; skip silently
            pass
    
    # Store selected sans for fallback use
    _selected_sans = sans_family


def _configure_named_font(
    root: tk.Tk, 
    name: str, 
    family: Optional[str], 
    size: int
) -> None:
    """Helper to configure a single named font with error handling."""
    
    try:
        font_obj = tkfont.nametofont(name)
        
        props = {}
        if family is not None:
            props["family"] = family
        props["size"] = size
        
        # Update only the properties we want to change
        for key, value in props.items():
            font_obj.configure(**{key: value})
            
    except Exception:
        # Named font doesn't exist on this system; skip silently
        pass


def configure_ttk_styles(
    style: "ttk.Style", 
    *, 
    sans_family: Optional[str], 
    mono_family: Optional[str], 
    ui_size: int
) -> None:
    """
    Configure ttk widget styles with selected font families.
    
    Applies the selected sans-serif family to all standard ttk widgets for
    consistent appearance across the application.
    
    Args:
        style: The ttk.Style instance (should be created from root before calling)
        sans_family: Sans-serif font family name (or None to skip changes)
        mono_family: Monospace font family name (not used here; handled via TkFixedFont)
        ui_size: Font size in points
        
    Side-effects:
        Modifies the provided Style instance. Does not crash if any style
        configuration fails.
        
    Styles configured:
        - TLabel, TButton, TCheckbutton, TRadiobutton
        - TEntry, TCombobox, TMenubutton, TSpinbox
        - TLabelframe.Label, TNotebook.Tab
        - Treeview, Treeview.Heading (for peak tables)
        - Fit.TButton (bold variant for fit button)
        - Danger.TButton, Success.TButton (font added if not already set)
    """
    
    if sans_family is None:
        # No font selected; let Tk defaults stand
        return
    
    font_tuple = (sans_family, ui_size)
    bold_font_tuple = (sans_family, ui_size, "bold")
    
    # Standard widgets - all use same family and size
    standard_styles = [
        "TLabel",
        "TButton", 
        "TCheckbutton",
        "TRadiobutton",
        "TEntry",
        "TCombobox",
        "TMenubutton",
        "TSpinbox",
        "TLabelframe.Label",
        "TNotebook.Tab"
    ]
    
    for style_name in standard_styles:
        try:
            style.configure(style_name, font=font_tuple)
        except Exception:
            # Some styles may not exist; skip silently
            pass
    
    # Treeview widgets (used for peak tables)
    try:
        style.configure("Treeview", font=font_tuple)
        style.configure("Treeview.Heading", font=(sans_family, ui_size + 1, "bold"))
    except Exception:
        pass
    
    # Special styles that need bold variant
    special_styles = [
        ("Fit.TButton", bold_font_tuple),
    ]
    
    for style_name, font_config in special_styles:
        try:
            style.configure(style_name, font=font_config)
        except Exception:
            pass
    
    # Danger/Success buttons - add font if not already set
    # (colors are typically defined elsewhere; we just ensure consistent fonts)
    for style_name in ["Danger.TButton", "Success.TButton"]:
        try:
            current_config = style.configure(style_name)
            if "font" not in current_config:
                style.configure(style_name, font=font_tuple)
        except Exception:
            pass


# Module-level variable to store selected sans-serif family for fallback use
_selected_sans: Optional[str] = None
