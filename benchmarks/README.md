# Benchmarks — insurance-conformal

**Headline:** Parametric Tweedie intervals undercover the top two risk deciles by 10–15 percentage points (75–82% actual vs 90% target); conformal prediction meets the 90% target across all deciles by construction.

---

## Comparison table

50,000 synthetic UK motor policies. Heteroskedastic Gamma DGP: residual variance grows faster than Tweedie(p=1.5) predicts in the high-mean tail. CatBoost Tweedie(p=1.5) point forecast used for both methods.

| Metric | Parametric Tweedie intervals | Conformal (pearson\_weighted) | Locally-weighted conformal |
|---|---|---|---|
| Aggregate coverage (90% target) | ~88–91% | ≥90% | ≥90% |
| Coverage — top decile (high risk) | ~75–82% | ~88–92% | ~90–93% |
| Coverage — bottom decile (low risk) | ~94–96% | ~90–92% | ~90–92% |
| Mean interval width (90%) | Narrower | Slightly wider (aggregate) | Narrower than standard conformal |
| Assumes constant dispersion | Yes (fails in tail) | No | No |
| Calibration set required | No | Yes (20% holdout) | Yes |
| Valid coverage guarantee | No | Yes (exchangeability) | Approximately |

When the DGP is well-matched to the Tweedie variance function, parametric intervals work fine — the simpler `benchmark.py` (Ridge/null scenario) demonstrates this. The `benchmark_gbm.py` benchmark targets the realistic failure mode: a heteroskedastic portfolio where high-risk policies are genuinely more dispersed than their mean alone would predict.

The undercoverage in the top decile matters because that is where reserving, treaty pricing, and regulatory capital calculations concentrate. A 90% interval that is actually 78% is a systematic understatement of uncertainty for the worst risks.

---

## How to run

### CatBoost scenario (the benchmark with the interesting result)

```bash
uv run python benchmarks/benchmark_gbm.py
```

### Ridge/null scenario (well-matched DGP — baseline reference)

```bash
uv run python benchmarks/benchmark.py
```

### Databricks

```bash
databricks workspace import-dir benchmarks /Workspace/insurance-conformal/benchmarks
```

The CatBoost benchmark requires `catboost`:

```bash
uv add 'insurance-conformal[catboost]'
```

Dependencies: `insurance-conformal`, `catboost` (optional for GBM benchmark), `numpy`.

The `benchmark_gbm.py` runs in approximately 3–5 minutes (CatBoost fit + conformal calibration on 50k policies).
