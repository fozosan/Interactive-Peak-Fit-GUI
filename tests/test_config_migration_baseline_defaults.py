import json
import sys
from pathlib import Path

import pytest

# We import after monkeypatching CONFIG_PATH, so keep helper to import fresh
def _import_app():
    if "ui.app" in sys.modules:
        del sys.modules["ui.app"]
    import ui.app as app
    return app

def test_legacy_keys_are_migrated_to_baseline_defaults(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".gl_peakfit_config.json"
    # Legacy-only keys; no baseline_defaults present
    legacy = {
        "als_lam": 2e5,
        "als_asym": 0.005,
        "als_niter": 15,
        "als_thresh": 0.01,
        "solver_choice": "modern_vp",
        "saved_peaks": [],
        "templates": {},
    }
    cfg_path.write_text(json.dumps(legacy))
    monkeypatch.setenv("HOME", str(tmp_path))  # guard for code that joins home
    app = _import_app()
    # Redirect CONFIG_PATH surgically
    monkeypatch.setattr(app, "CONFIG_PATH", cfg_path, raising=True)

    cfg = app.load_config()
    bd = cfg.get("baseline_defaults", {})
    assert bd, "baseline_defaults should exist after migration"
    assert bd.get("method") == "als"
    als = bd.get("als", {})
    assert als.get("lam") == 2e5
    assert als.get("p") == 0.005
    assert als.get("niter") == 15
    assert als.get("thresh") == 0.01

def test_roundtrip_persists_method_and_per_method_defaults(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".gl_peakfit_config.json"
    base_cfg = {
        "baseline_defaults": {
            "method": "polynomial",
            "als": {"lam": 1e5, "p": 0.001, "niter": 10, "thresh": 0.0},
            "polynomial": {"degree": 3, "normalize_x": False},
        },
        "templates": {},
        "saved_peaks": [],
    }
    cfg_path.write_text(json.dumps(base_cfg))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = _import_app()
    monkeypatch.setattr(app, "CONFIG_PATH", cfg_path, raising=True)

    cfg_loaded = app.load_config()
    assert cfg_loaded["baseline_defaults"]["method"] == "polynomial"
    assert cfg_loaded["baseline_defaults"]["polynomial"] == {"degree": 3, "normalize_x": False}

    # Modify and save; then reload
    cfg_loaded["baseline_defaults"]["method"] = "als"
    cfg_loaded["baseline_defaults"]["als"]["lam"] = 3.3e5
    app.save_config(cfg_loaded)

    cfg2 = app.load_config()
    assert cfg2["baseline_defaults"]["method"] == "als"
    assert cfg2["baseline_defaults"]["als"]["lam"] == 3.3e5
