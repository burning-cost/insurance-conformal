# Contributing to insurance-conformal

This library provides distribution-free prediction intervals for insurance pricing models. The core guarantee — marginal coverage regardless of model specification — is only useful if the implementation is correct and the calibration split is handled properly. Contributions that sharpen correctness and usability are welcome.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_conformal; print(insurance_conformal.__version__)`)
- The model type (CatBoost, GLM, other) and the prediction target (frequency, severity, pure premium)
- A minimal reproducible example — the synthetic data generators in the library work well for this
- Observed coverage vs. nominal coverage on your holdout set, if relevant

Coverage failures on temporal splits are the most common class of bug. If you are seeing systematic under-coverage, include details about the temporal structure of your data.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Useful areas: conditional coverage methods (coverage that holds within subgroups, not just marginally), integration with additional model types, and severity-specific calibration approaches for the frequency/severity split.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-conformal.git
cd insurance-conformal
uv sync --dev
uv run pytest
```

The library uses `uv` for dependency management. Python 3.10+ is required. Slow tests (fitting CatBoost on synthetic data) are excluded from the default run:

```bash
uv run pytest --run-slow
```

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation
- Docstrings follow NumPy format and note coverage guarantees explicitly — state whether coverage is marginal or conditional
- Tests must verify actual coverage on synthetic data, not just that the code runs without error
