# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-conformal v0.2.0 — Test Runner
# MAGIC
# MAGIC Runs the full test suite on Databricks serverless compute.
# MAGIC Covers all new modules: LocallyWeightedConformal, Hong, SCRReport, diagnostics_ext.

# COMMAND ----------
# Install the package from local wheel or from PyPI
%pip install --quiet insurance-conformal==0.2.0 catboost

# COMMAND ----------
# If testing from source (before PyPI publish), install from the workspace upload:
# %pip install --quiet /Workspace/insurance_conformal_src/dist/insurance_conformal-0.2.0-py3-none-any.whl catboost

# COMMAND ----------
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance_conformal_src/tests/",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])

print(f"\nReturn code: {result.returncode}")
assert result.returncode == 0, "Tests failed — see output above"
