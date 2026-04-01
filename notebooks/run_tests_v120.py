# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-conformal v1.2.0 — ConformalisedSurvival Test Runner
# MAGIC
# MAGIC Runs the full test suite from workspace upload. Focuses on the new
# MAGIC ConformalisedSurvival class (survival.py) and the refactored utils.py,
# MAGIC but runs all tests to catch regressions from the utils.py change.

# COMMAND ----------
%pip install --quiet scipy scikit-learn polars numpy pandas pyarrow

# COMMAND ----------
import subprocess, sys, os

# Install the package from workspace upload
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-conformal-v120/",
     "--quiet"],
    capture_output=True, text=True, timeout=300
)
print(result.stdout[-2000:] if result.stdout else "(no stdout)")
if result.returncode != 0:
    print("INSTALL ERROR:", result.stderr[-3000:])

# COMMAND ----------
# Run survival-specific tests first
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-conformal-v120/tests/test_survival.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True, timeout=600
)
print(result.stdout)
if result.returncode != 0 and result.stderr:
    print("STDERR:", result.stderr[-3000:])

# COMMAND ----------
# Run full test suite — catches any regressions from utils.py refactor
result_full = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-conformal-v120/tests/",
     "-v", "--tb=short", "--no-header", "-q"],
    capture_output=True, text=True, timeout=900
)
print(result_full.stdout[-10000:])
if result_full.returncode != 0 and result_full.stderr:
    print("STDERR:", result_full.stderr[-3000:])
print("Exit code:", result_full.returncode)
