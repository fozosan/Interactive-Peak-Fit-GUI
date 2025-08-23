"""Configuration handling for Peakfit 3.x."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "3.0",
}


def load(path: str | Path) -> Dict[str, Any]:
    """Load configuration from *path* or return defaults if missing."""
    raise NotImplementedError("Config loading not yet implemented")


def save(path: str | Path, cfg: Dict[str, Any]) -> None:
    """Persist ``cfg`` to ``path``."""
    raise NotImplementedError("Config saving not yet implemented")
