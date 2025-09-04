# Changelog

## Unreleased
- Deprecated: `bootstrap_ci(..., predict_fn=...)` — use `predict_full`.
- Robust: prediction bands are skipped when no model is supplied; see `diagnostics['band_disabled_no_model']`.
- Normalization now mirrors stats to both `stats` and `param_stats`.
