# Changelog

## [0.6.2] - 2026-03-24

### Added
- `ConformalisedQuantileRegression`: split Conformalized Quantile Regression (CQR)
  from Romano, Patterson & Candès (NeurIPS 2019, arXiv:1905.03222). Wraps a pair of
  pre-fitted quantile models (lower and upper quantile) and applies a conformal
  calibration correction so that the final intervals achieve marginal coverage
  >= 1 - alpha regardless of quantile model misspecification.

  Why CQR rather than RAPS: RAPS (Angelopoulos et al. 2020) was designed for
  multi-class classification — its regularisation penalty acts on the size of a
  discrete prediction set, which has no direct analogue in the regression setting.
  CQR is the regression counterpart: it operates on quantile model outputs and
  produces adaptive (heteroscedastic) intervals that are wider for high-variance
  risks and narrower for stable ones. For insurance this is more useful than RAPS
  because claim severity is genuinely heteroscedastic — young drivers, high-value
  vehicles, and CAT-exposed properties all have fatter tails than low-risk policies.

  The class accepts any sklearn-compatible quantile model: CatBoost
  `Quantile:alpha=`, LightGBM `objective="quantile"`, sklearn
  `GradientBoostingRegressor(loss="quantile")`. The four output columns are
  `lower`, `q_lo`, `q_hi`, `upper` — the raw quantile outputs are preserved
  as diagnostics alongside the conformally corrected bounds.

## [0.6.1] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.6.0 (2026-03-22) [unreleased]
- Add pytest to dev dependencies — fixes test collection in isolated venv
- Remove emoji from discussion CTA
- Quality review: promote GBM benchmark to primary, move Ridge to examples
- docs: fix README review issues
- docs: add Liang Hong and Charpentier academic citations
- Add Liang Hong (2025, 2026) citations to README and docstrings

## v0.6.0 (2026-03-21)
- fix: replace ragged numpy array in entropy test with per-distribution checks
- fix: pandas is a core dep, not optional; remove duplicate [dependency-groups] dev section
- fix: replace unattributed ~30% interval width claim with actual benchmark numbers
- Add cross-links to related libraries in README
- feat: add RetroAdj benchmark notebook and README section
- feat: add RetroAdj — online conformal inference with retrospective adjustment (v0.6.0)
- Add GBM/heteroskedastic benchmark; update README with actual numbers
- docs: document v0.5.1 features — LightGBM backend and FrequencySeverityConformal
- feat: add LightGBM backend for LocallyWeightedConformal and FrequencySeverityConformal
- Add Solvency II internal model validation section
- Add SelectiveConformalRC — two-stage selective conformal risk control (v0.5.0)
- security: pin pillow>=12.1.1 to close Dependabot alert #1
- fix: update quickstart output to match actual code execution (scale=500 Gamma DGP)
- fix: update license badge from BSD-3 to MIT
- Add discussions link and star CTA
- Add benchmark section documenting conformal vs naive parametric results
- Fix four reviewer-identified issues in README
- Add PyPI classifiers for financial/insurance audience
- Add Google Colab quickstart notebook and Open-in-Colab badge
- Add CONTRIBUTING.md with bug reporting, feature request, and dev setup guidance
- fix: update tests to match post-refactor Anscombe guard and lwc rename
- Fix flaky P1-5 regression test for HongTransformConformal warning
- Fix P1-5 regression test: use switching model to trigger degenerate upper warning
- Fix P0/P1 bugs: Anscombe exponent, apply_exposure, CRC n_sel, B required, lwc rename, Hong dead code
- Fix docs workflow: use pdoc not pdoc3 syntax (no --html flag)
- Add pdoc API documentation workflow with GitHub Pages deployment
- Add benchmark: conformal intervals vs naive parametric for insurance pricing
