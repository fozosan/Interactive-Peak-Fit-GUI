# Changelog

## Unreleased
- Deprecated: `bootstrap_ci(..., predict_fn=...)` — use `predict_full`.
- Robust: prediction bands are skipped when no model is supplied; see `diagnostics['band_disabled_no_model']`.
- Normalization now mirrors stats to both `stats` and `param_stats`.
- Fixed: instrument `.csv` exports now load. A data row is a line starting with two numbers, so metadata such as WITec's `SizeX = 1` is ignored rather than mistaken for a third column; quotes around rows or fields are dropped.
- Fixed: `_uncertainty.csv` files no longer overwrite each other for spectrum names containing dots (`0.10sec`, `0.20sec`).
- Batch: files with no two-column data are skipped and logged instead of aborting the run.
