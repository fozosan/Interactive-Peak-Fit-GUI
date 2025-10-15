import re
import sys
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
P_CORE = ROOT / "core" / "uncertainty.py"
P_BATCH = ROOT / "batch" / "runner.py"
P_UI = ROOT / "ui" / "app.py"


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort logging
        return f"<<READ_ERROR {exc}>>"


def check_ui(text: str):
    ok = True
    notes = []
    if "_unc_change_busy" not in text:
        ok = False
        notes.append("ui: missing self._unc_change_busy guard")
    if "def _on_uncertainty_method_changed" not in text:
        ok = False
        notes.append("ui: missing _on_uncertainty_method_changed")
    else:
        body = text.split("def _on_uncertainty_method_changed", 1)[1]
        if "if getattr(self, \"_unc_change_busy\", False):" not in body and "if getattr(self, '_unc_change_busy', False):" not in body:
            ok = False
            notes.append("ui: missing early return when busy")
        if "_unc_change_busy = True" not in body:
            ok = False
            notes.append("ui: no set busy True in handler")
        if "finally:" not in body or "_unc_change_busy = False" not in body:
            ok = False
            notes.append("ui: missing finally reset busy False")
        if re.search(r"def\\s+_on_uncertainty_method_changed[^\n]*\n\\s+pass\\b", text):
            ok = False
            notes.append("ui: handler body replaced by stub 'pass'")
    return ok, notes


def check_core(text: str):
    ok = True
    notes = []
    for snippet, label in (
        ("def _norm_solver_and_sharing(", "_norm_solver_and_sharing"),
        ("def _build_residual_vector(", "_build_residual_vector"),
        ("def _relabel_by_center(", "_relabel_by_center"),
        ("def _validate_vector_length(", "_validate_vector_length"),
    ):
        if snippet not in text:
            ok = False
            notes.append(f"core: missing helper {label}")
    if "def bayesian_ci" not in text:
        ok = False
        notes.append("core: missing bayesian_ci")
    else:
        bay = text.split("def bayesian_ci", 1)[1]
        if "_norm_solver_and_sharing" not in bay:
            ok = False
            notes.append("core: bayesian_ci not using _norm_solver_and_sharing")
        if "fit[\"share_fwhm\"]" not in bay and "fit['share_fwhm']" not in bay:
            ok = False
            notes.append("core: bayesian_ci not setting fit['share_fwhm']")
        if "fit[\"share_eta\"]" not in bay and "fit['share_eta']" not in bay:
            ok = False
            notes.append("core: bayesian_ci not setting fit['share_eta']")
        if "lmfit_share_fwhm" in bay and "share_fwhm" not in bay:
            notes.append("core: bayesian_ci still reading lmfit_* directly (warn)")
    if "def bootstrap_ci" not in text:
        ok = False
        notes.append("core: missing bootstrap_ci")
    else:
        boot = text.split("def bootstrap_ci", 1)[1]
        if "bootstrap_residual_mode" not in boot or '"raw"' not in boot:
            ok = False
            notes.append("core: bootstrap_ci missing default residual_mode='raw'")
        if "relabel_by_center" not in boot:
            ok = False
            notes.append("core: bootstrap_ci missing relabel_by_center default")
        if "_build_residual_vector" not in boot:
            ok = False
            notes.append("core: bootstrap_ci not using _build_residual_vector")
        if "run_fit_consistent" in boot and "solver=_solver_name" not in boot:
            ok = False
            notes.append("core: bootstrap_ci refit not passing solver parity")
        if "_validate_vector_length" not in boot:
            ok = False
            notes.append("core: bootstrap_ci missing _validate_vector_length check")
        if "return b, np.asarray(th_new" in boot and "_relabel_by_center" not in boot:
            notes.append("core: bootstrap_ci may not relabel after refit (warn)")
    return ok, notes


def check_batch(text: str):
    ok = True
    notes = []
    if "fit_ctx.setdefault(\"solver\", solver_choice)" not in text:
        ok = False
        notes.append("batch: missing solver propagation to fit_ctx")
    if "fit_ctx.setdefault(\"bootstrap_residual_mode\", \"raw\")" not in text:
        ok = False
        notes.append("batch: missing bootstrap_residual_mode='raw'")
    if "fit_ctx.setdefault(\"relabel_by_center\", True)" not in text:
        ok = False
        notes.append("batch: missing relabel_by_center=True")
    sentinel_a = 'if not str(solver_choice).lower().startswith("lmfit"):'
    sentinel_b = "if not str(solver_choice).lower().startswith('lmfit'):"
    if sentinel_a not in text and sentinel_b not in text:
        ok = False
        notes.append("batch: missing strip of lmfit_share_* for non-LMFit")
    return ok, notes


def main() -> None:
    report = {}
    core_text = read(P_CORE)
    batch_text = read(P_BATCH)
    ui_text = read(P_UI)
    ok_ui, notes_ui = check_ui(ui_text)
    ok_core, notes_core = check_core(core_text)
    ok_batch, notes_batch = check_batch(batch_text)
    report["ui"] = {"ok": ok_ui, "notes": notes_ui}
    report["core"] = {"ok": ok_core, "notes": notes_core}
    report["batch"] = {"ok": ok_batch, "notes": notes_batch}
    print(json.dumps(report, indent=2))
    if not (ok_ui and ok_core and ok_batch):
        sys.exit(1)


if __name__ == "__main__":
    main()

