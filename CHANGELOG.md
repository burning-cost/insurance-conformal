# Changelog

## [0.7.0] - 2026-03-31

### Added
- `ConditionalCoverageERT`: full ERT (Excess Risk of Target Coverage) test for
  conditional coverage violations, based on Braun et al. (arXiv:2512.11779).

  The existing `ert_coverage_gap()` function is a pragmatic approximation that bins
  by predicted value. `ConditionalCoverageERT` is the full test: it trains a LightGBM
  binary classifier on coverage indicators Z_i = 1{y_i in [lo_i, hi_i]} using the
  features X_i directly, with KFold CV to prevent in-sample overfitting. ERT is the
  difference between the constant predictor loss (always predicting 1-alpha) and the
  CV classifier loss — positive values indicate detectable conditional miscoverage.

  Three loss variants:
  - `loss="l1"`: L1-ERT (mean absolute error). Linear penalty, easy to explain.
  - `loss="l2"`: L2-ERT (Brier score). Quadratic, more sensitive to large gaps.
  - `loss="kl"`: KL-ERT (log loss). Information-theoretic, penalises confident errors.

  Three direction variants:
  - `direction="under"`: only penalise predicted undercoverage (p < marginal rate).
    The most relevant variant for insurance — FCA/TCF concerns focus on systematic
    undercoverage of specific segments, not overcoverage.
  - `direction="over"`: only penalise predicted overcoverage (efficiency concern).
  - `direction="both"`: symmetric, for general diagnostics.

  Bootstrap CIs via `n_bootstraps` resamples with configurable confidence level.

  `subgroup_coverage()` method bins each feature into quantile bins and reports
  empirical coverage per bin — structured for FCA reporting where you need to
  demonstrate that coverage is uniform across policyholder segments.

  See `insurance_conformal.conditional_coverage` for `ConditionalCoverageERT`
  and `ERTResult`.

## [0.6.3] - 2026-03-25

### Added
- `solvency_capital_range()`: lightweight functional API for computing Solvency II
  SCR bounds from any fitted conformal predictor. Returns a `SolvencyCapitalRange`
  dataclass with `scr_estimate`, `lower_bound`, `upper_bound`, `interval_width`,
  `coverage_level`, `n_risks`, `alpha`, `total_scr`, and `mean_interval_width`.

  The upper_bound at alpha=0.005 is a valid 99.5% prediction bound under split conformal
  theory — distribution-free, with finite-sample coverage guarantees. The SCR component
  is max(0, upper_bound - expected_loss), matching the economic definition of required
  capital as the excess of the tail bound over the expected value.

  Designed as the pipeline-friendly companion to SCRReport: use SCRReport when producing
  regulatory submissions with coverage tables and markdown output; use
  solvency_capital_range() when you need SCR estimates inside a larger modelling workflow
  (reserving systems, reinsurance optimisers, stress-testing loops).

  Accepts optional `exposure` weights for policies with non-unit exposure periods.
  Compatible with InsuranceConformalPredictor, LocallyWeightedConformal,
  HongTransformConformal, and ConformalisedQuantileRegression.

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
