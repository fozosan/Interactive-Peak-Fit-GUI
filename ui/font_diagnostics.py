"""
Font diagnostics module for Interactive Peak Fit GUI.

Provides diagnostic functions to detect font rendering issues on Linux systems.
This module is importable without side-effects; all actions only occur when
explicitly called via run_diagnostics() or show_visual_sample().
"""

import os
import platform
from pathlib import Path
from typing import Callable, Dict, Any

import tkinter as tk
import tkinter.font as tkfont


# Font families to check (case-insensitive matching)
SANS_CANDIDATES = [
    "Ubuntu", "Noto Sans", "DejaVu Sans", "Liberation Sans", 
    "Cantarell", "Segoe UI", "Arial"
]

MONO_CANDIDATES = [
    "Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono", 
    "Courier New", "Consolas"
]


def _log(msg: str, log_fn: Callable[[str], None] = print) -> None:
    """Log message with FONT-DIAG prefix."""
    log_fn(f"[FONT-DIAG] {msg}")


def run_diagnostics(root: tk.Tk, *, log_fn: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    Run comprehensive font diagnostics and return results as a dict.
    
    Args:
        root: The Tkinter root window (tk.Tk() instance)
        log_fn: Function to call for each diagnostic line (default: print)
        
    Returns:
        Dict containing all diagnostic data for programmatic use/testing
        
    Side-effects:
        Logs all diagnostics with [FONT-DIAG] prefix to log_fn
    """
    
    results: Dict[str, Any] = {}
    
    # --- Platform / session info ---
    _log("=== PLATFORM & SESSION ===", log_fn)
    try:
        system_name = platform.system()
        release = platform.release()
        machine = platform.machine()
        results["platform"] = {"system": system_name, "release": release, "machine": machine}
        _log(f"Platform: {system_name} {release} ({machine})", log_fn)
    except Exception as e:
        _log(f"Failed to get platform info: {e}", log_fn)
    
    # Environment variables
    xdg_session = os.environ.get("XDG_SESSION_TYPE", "not set")
    display = os.environ.get("DISPLAY", "not set")
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "not set")
    ssh_client = os.environ.get("SSH_CLIENT", None)
    ssh_tty = os.environ.get("SSH_TTY", None)
    
    results["environment"] = {
        "XDG_SESSION_TYPE": xdg_session,
        "DISPLAY": display,
        "WAYLAND_DISPLAY": wayland_display,
        "SSH_CLIENT": ssh_client,
        "SSH_TTY": ssh_tty,
    }
    
    _log(f"XDG_SESSION_TYPE: {xdg_session}", log_fn)
    _log(f"DISPLAY: {display}", log_fn)
    _log(f"WAYLAND_DISPLAY: {wayland_display}", log_fn)
    
    # Remote X detection
    is_remote = bool(ssh_client or ssh_tty)
    results["is_remote_x"] = is_remote
    if is_remote:
        _log("WARNING: Running over remote X (SSH/X11 forwarding)", log_fn)
        _log("         Font rendering may be limited by X server settings", log_fn)
    
    # WSL detection
    try:
        proc_version_path = Path("/proc/version")
        if proc_version_path.exists():
            version_text = proc_version_path.read_text().lower()
            is_wsl = "microsoft" in version_text
            results["is_wsl"] = is_wsl
            if is_wsl:
                _log("DETECTED: Running under WSL", log_fn)
                _log("          Font rendering depends on Windows host configuration", log_fn)
    except Exception as e:
        results["is_wsl"] = False
        _log(f"Failed to detect WSL: {e}", log_fn)
    
    # --- Tk runtime info ---
    _log("\n=== TK RUNTIME ===", log_fn)
    try:
        tk_version = tk.TkVersion
        tcl_version = tk.TclVersion
        results["tk_runtime"] = {"TkVersion": float(tk_version), "TclVersion": str(tcl_version)}
        _log(f"TkVersion: {tk_version}, TclVersion: {tcl_version}", log_fn)
    except Exception as e:
        _log(f"Failed to get Tk version: {e}", log_fn)
    
    # Windowing system
    try:
        windowing_system = root.tk.call("tk", "windowingsystem")
        results["windowing_system"] = windowing_system
        _log(f"Windowing system: {windowing_system}", log_fn)
        
        if windowing_system == "x11":
            _log("  -> X11 detected. Font rendering depends on Xft/fontconfig", log_fn)
        elif windowing_system == "wayland":
            _log("  -> Wayland detected. Font rendering is typically good by default", log_fn)
        elif windowing_system == "aqua":
            _log("  -> macOS (Aqua). Font rendering uses Core Text", log_fn)
    except Exception as e:
        results["windowing_system"] = None
        _log(f"Failed to detect windowing system: {e}", log_fn)
    
    # Tk scaling factor
    try:
        scaling = root.tk.call("tk", "scaling")
        results["scaling"] = float(scaling)
        _log(f"Tk scaling: {scaling}", log_fn)
    except Exception as e:
        results["scaling"] = None
        _log(f"Failed to get Tk scaling: {e}", log_fn)
    
    # --- Font info ---
    _log("\n=== FONT CONFIGURATION ===", log_fn)
    
    # Try to read default font
    try:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_props = default_font.actual()
        results["tk_default_font"] = {
            "family": default_props.get("family"),
            "size": default_props.get("size"),
            "weight": default_props.get("weight"),
        }
        _log(f"TkDefaultFont: {default_props.get('family')} {default_props.get('size')}", log_fn)
    except Exception as e:
        results["tk_default_font"] = None
        _log(f"Failed to read TkDefaultFont: {e}", log_fn)
    
    # Available font families (count and check candidates)
    all_families, found_sans, found_mono = [], [], []
    try:
        all_families = tkfont.families(root)
        results["num_families"] = len(all_families)
        _log(f"\nTotal available font families: {len(all_families)}", log_fn)
        
        # Check sans-serif candidates (case-insensitive)
        families_lower = [f.lower() for f in all_families]
        
        found_sans = []
        for candidate in SANS_CANDIDATES:
            found = None
            for available in all_families:
                if available.lower() == candidate.lower():
                    found = available  # preserve Tk's casing
                    break
            if found:
                found_sans.append(found)
        
        results["found_sans_fonts"] = found_sans
        _log(f"\nSans-serif candidates found: {', '.join(found_sans)}", log_fn)
        not_found_sans = [c for c in SANS_CANDIDATES if c.lower() not in families_lower]
        if not_found_sans:
            _log(f"  Not found: {', '.join(not_found_sans)}", log_fn)
        
        # Check monospace candidates (case-insensitive)
        found_mono = []
        for candidate in MONO_CANDIDATES:
            found = None
            for available in all_families:
                if available.lower() == candidate.lower():
                    found = available  # preserve Tk's casing
                    break
            if found:
                found_mono.append(found)
        
        results["found_mono_fonts"] = found_mono
        _log(f"\nMonospace candidates found: {', '.join(found_mono)}", log_fn)
        not_found_mono = [c for c in MONO_CANDIDATES if c.lower() not in families_lower]
        if not_found_mono:
            _log(f"  Not found: {', '.join(not_found_mono)}", log_fn)
            
    except Exception as e:
        results["num_families"] = None
        results["found_sans_fonts"] = []
        results["found_mono_fonts"] = []
        _log(f"\nFailed to enumerate font families: {e}", log_fn)
    
    # --- Summary & recommendations ---
    _log("\n=== SUMMARY ===", log_fn)

    if results.get("is_remote_x"):
        _log("- Running over remote X: font rendering depends on X server settings", log_fn)
        _log("  Consider using ssh -X (may be slow) or local display", log_fn)
    
    if results.get("is_wsl"):
        _log("- Running under WSL: check Windows host font configuration", log_fn)
        _log("  Install fonts on Windows, then access via WSL", log_fn)
    
    if len(all_families) < 10:
        _log(f"- Very few fonts available ({len(all_families)}): consider installing more fonts", log_fn)
    
    return results


def show_visual_sample(root: tk.Tk) -> None:
    """
    Open a Toplevel window showing sample text in various font families.
    
    Args:
        root: The Tkinter root window (tk.Tk() instance)
        
    Side-effects:
        Creates and displays a new Toplevel window with font samples
    """
    
    sample_window = tk.Toplevel(root)
    sample_window.title("Font Rendering Sample")
    sample_window.geometry("700x600")
    sample_window.transient(root)  # Keep on top of main window
    
    sample_text_1 = "The quick brown fox jumps over the lazy dog 0123456789"
    sample_text_2 = "AaBbYyZz — %$@ (hinting test)"
    
    # Frame for all labels
    main_frame = tk.Frame(sample_window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add TkDefaultFont as baseline first
    _add_font_sample(main_frame, "TkDefaultFont (system default)", sample_text_1, sample_text_2)
    
    # Add each candidate font family
    all_candidates = list(SANS_CANDIDATES) + list(MONO_CANDIDATES)
    for candidate in all_candidates:
        _add_font_sample(main_frame, candidate, sample_text_1, sample_text_2)
    
    # Close button at bottom
    close_btn = ttk.Button(
        main_frame, 
        text="Close", 
        command=sample_window.destroy
    )
    close_btn.pack(pady=10)


def _add_font_sample(parent: tk.Frame, family_name: str, text_1: str, text_2: str) -> None:
    """Helper to add a font sample label with resolved family info."""
    
    try:
        # Try TkDefaultFont first (special case - no explicit family; the caller
        # labels it "TkDefaultFont (system default)")
        if family_name.startswith("TkDefaultFont"):
            font_obj = tkfont.nametofont("TkDefaultFont")
            actual_family = font_obj.actual().get("family", "unknown")
            size = font_obj.actual().get("size", 10)
            resolved_as_asked = True
        else:
            # Try to create font with this family, fallback gracefully
            try:
                font_obj = tkfont.Font(family=family_name, size=10)
                actual_family = font_obj.actual()["family"]
                size = 10
            except Exception:
                # Family doesn't exist, use a sensible default
                font_obj = tkfont.Font(size=10)
                actual_family = "NOT FOUND"
                size = 10
            resolved_as_asked = actual_family == family_name
        
        label_frame = tk.Frame(parent)
        label_frame.pack(fill=tk.X, pady=2)
        
        family_label = tk.Label(
            label_frame, 
            text=family_name,
            font=("Helvetica", 8),
            fg="gray"
        )
        family_label.pack(side=tk.LEFT)
        
        resolved_label = tk.Label(
            label_frame,
            text=f"[resolved: {actual_family}]",
            font=("Helvetica", 8),
            fg="blue" if resolved_as_asked else "red"
        )
        resolved_label.pack(side=tk.RIGHT)
        
        sample_label = tk.Label(
            parent,
            text=text_1,
            font=font_obj
        )
        sample_label.pack(anchor=tk.W, pady=(0, 2))
        
        sample_label_2 = tk.Label(
            parent,
            text=text_2,
            font=font_obj
        )
        sample_label_2.pack(anchor=tk.W)
        
    except Exception as e:
        # If anything fails, show error but continue
        error_label = tk.Label(
            parent,
            text=f"{family_name}: ERROR - {e}",
            fg="red"
        )
        error_label.pack(anchor=tk.W, pady=2)


# Import ttk here to avoid circular imports if possible
try:
    from tkinter import ttk
except ImportError:
    ttk = None  # type: ignore
