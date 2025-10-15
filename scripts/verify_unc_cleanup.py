import re, sys, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
P_CORE = ROOT / "core" / "uncertainty.py"
P_BATCH = ROOT / "batch" / "runner.py"

def read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _contains(hay: str, needle: str) -> bool:
    return needle.replace(" ", "") in hay.replace(" ", "")


def check_core(text: str):
    ok, notes = True, []
    # bayesian_ci: normalized flags present and tying OFF by default
    if "def bayesian_ci" not in text:
        return False, ["core: missing bayesian_ci"]
    bay = text.split("def bayesian_ci", 1)[1]
    want = [
        "_norm_solver_and_sharing",
        'fit["solver"]',
        'fit["share_fwhm"]',
        'fit["share_eta"]',
        'if not bool(fit.get("bayes_respect_sharing", False))',
        'share_fwhm = bool(fit_ctx.get("share_fwhm", False))',
        'share_eta = bool(fit_ctx.get("share_eta", False))',
    ]
    for s in want:
        if not _contains(bay, s):
            ok = False; notes.append(f"core: bayesian_ci missing {s}")
    return ok, notes

def check_batch(text: str):
    ok, notes = True, []
    for s in [
        "_predict_full_from_peaks",
        'fit_ctx.setdefault("bootstrap_residual_mode", "raw")',
        'fit_ctx.setdefault("relabel_by_center", True)',
        'fit_ctx.setdefault("center_residuals", True)',
        'fit_ctx["peaks_out"]',
        'fit_ctx["bounds"]',
        'fit_ctx["locked_mask"]',
        'fit_ctx["theta0"]',
    ]:
        if s not in text:
            ok = False; notes.append(f"batch: missing {s}")
    # predict_full length guard
    guard_ok = ("callable(model_eval)" in text and "if _probe.size != x_fit.size" in text)
    if not guard_ok:
        ok = False; notes.append("batch: missing predict_full length guard")
    return ok, notes

def main():
    core_ok, core_notes = check_core(read(P_CORE))
    batch_ok, batch_notes = check_batch(read(P_BATCH))
    report = {"core": {"ok": core_ok, "notes": core_notes},
              "batch": {"ok": batch_ok, "notes": batch_notes}}
    print(json.dumps(report, indent=2))
    sys.exit(0 if (core_ok and batch_ok) else 1)

if __name__ == "__main__":
    main()
