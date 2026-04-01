# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-conformal v1.1.0 — KMMConformalPredictor Test Runner
# MAGIC
# MAGIC Runs the full test suite from source (workspace upload). Focuses on
# MAGIC the new KMMConformalPredictor in covariate_shift.py, but runs all tests
# MAGIC to catch any regressions.

# COMMAND ----------
%pip install --quiet scipy scikit-learn polars numpy pandas pyarrow

# COMMAND ----------
import subprocess, sys, os

# Install the package from the workspace upload (editable-style)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-conformal-kmm/",
     "--quiet"],
    capture_output=True, text=True, timeout=300
)
print(result.stdout[-2000:] if result.stdout else "(no stdout)")
if result.returncode != 0:
    print("INSTALL ERROR:", result.stderr[-3000:])

# COMMAND ----------
# Run just the covariate shift tests first
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-conformal-kmm/tests/test_covariate_shift.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.returncode != 0 and result.stderr:
    print("STDERR:", result.stderr[-3000:])

# COMMAND ----------
# Run full test suite
result_full = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-conformal-kmm/tests/",
     "-v", "--tb=short", "--no-header", "-q"],
    capture_output=True, text=True, timeout=900
)
print(result_full.stdout[-8000:])
if result_full.returncode != 0 and result_full.stderr:
    print("STDERR:", result_full.stderr[-3000:])
print("Exit code:", result_full.returncode)
