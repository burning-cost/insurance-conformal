# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-conformal test runner
# MAGIC Run full test suite after uploading source from workspace.

# COMMAND ----------

# Install required packages
%pip install polars pandas scipy pyarrow scikit-learn catboost lightgbm matplotlib pillow pytest pytest-cov --quiet

# COMMAND ----------

import subprocess, sys

# Copy source from workspace to a local importable path
result = subprocess.run(
    ["cp", "-r", "/Workspace/insurance-conformal/src/insurance_conformal", "/tmp/insurance_conformal"],
    capture_output=True, text=True
)
print(result.stdout, result.stderr)

# Add to path
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/Workspace/insurance-conformal/src")

# COMMAND ----------

# Run pytest on the tests directory
result = subprocess.run(
    [sys.executable, "-m", "pytest", "/Workspace/insurance-conformal/tests", "-x", "--tb=short", "-q"],
    capture_output=True, text=True,
    env={**__import__("os").environ, "PYTHONPATH": "/Workspace/insurance-conformal/src"}
)
print(result.stdout[-10000:])  # last 10k chars
print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
