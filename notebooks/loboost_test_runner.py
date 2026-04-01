# Databricks notebook source
# MAGIC %md
# MAGIC # LoBoostCP Test Runner
# MAGIC
# MAGIC Runs the full test suite for `LoBoostCP` (insurance-conformal v1.0.0).
# MAGIC
# MAGIC Santos, Izbicki & Stern (2025). "LoBoost: Locally Boosted Conformal Prediction."
# MAGIC arXiv:2602.22432.

# COMMAND ----------

# MAGIC %pip install -q "insurance-conformal>=1.0.0" pytest pytest-cov 2>&1 | tail -5

# COMMAND ----------

# MAGIC %pip install -q scikit-learn polars numpy 2>&1 | tail -3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-conformal/tests/test_loboost.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
assert result.returncode == 0, "Tests failed — see output above"
