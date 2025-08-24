"""Configuration handling for Peakfit 3.x."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "3.0",
    "batch": {
        "patterns": "*.csv;*.txt;*.dat",
        "source": "template",
        "reheight": False,
        "save_traces": False,
        "auto_max": 5,
    },
    "classic": {
        "maxfev": 20000,
        "bound_centers_to_window": True,
        "margin_frac": 0.0,
        "fwhm_min_factor": 2.0,
        "fwhm_max_factor": 0.5,
        "height_factor": 3.0,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return ``base``."""

    for key, val in override.items():
        if (
            isinstance(val, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            _merge(base[key], val)
        else:
            base[key] = val
    return base


def load(path: str | Path) -> Dict[str, Any]:
    """Load configuration from *path* or return defaults if missing."""

    p = Path(path)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if not p.exists():
        return cfg
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            _merge(cfg, data)
            classic = cfg.setdefault("classic", {})
            # Migrate legacy keys if present
            if "fwhm_min_dx_factor" in classic and "fwhm_min_factor" not in classic:
                classic["fwhm_min_factor"] = classic.pop("fwhm_min_dx_factor")
            if "fwhm_max_span_factor" in classic and "fwhm_max_factor" not in classic:
                classic["fwhm_max_factor"] = classic.pop("fwhm_max_span_factor")
            if "max_height_factor" in classic and "height_factor" not in classic:
                classic["height_factor"] = classic.pop("max_height_factor")
            classic.setdefault("margin_frac", 0.0)
            classic.setdefault("fwhm_min_factor", 2.0)
            classic.setdefault("fwhm_max_factor", 0.5)
            classic.setdefault("height_factor", 3.0)
    except json.JSONDecodeError:  # pragma: no cover - corrupted file
        pass
    return cfg


def save(path: str | Path, cfg: Dict[str, Any]) -> None:
    """Persist ``cfg`` to ``path``."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
