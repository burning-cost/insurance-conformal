"""
Submit benchmark_gbm.py to Databricks as a one-shot job run.

Uses the same pattern as all other successful Burning Cost jobs:
- Notebook task (no environment_key, no new_cluster)
- %pip install in first cell
- Source uploaded to Workspace

Usage:
    python run_benchmark_databricks.py
"""

from __future__ import annotations

import base64
import os
import sys
import time

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as dbr_jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

# ---------------------------------------------------------------------------
# Build the notebook: pip install cell + benchmark code
# ---------------------------------------------------------------------------

SCRIPT_LOCAL = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_gbm.py")
with open(SCRIPT_LOCAL) as f:
    benchmark_src = f.read()

# Exact format that works in this workspace: %pip install, then COMMAND ----------, then code
notebook_src = f"""# Databricks notebook source
# MAGIC %pip install "insurance-conformal[catboost]>=0.4.1" polars>=1.0 --quiet

# COMMAND ----------

{benchmark_src}
"""

# ---------------------------------------------------------------------------
# Upload to Workspace
# ---------------------------------------------------------------------------

WS_DIR  = "/insurance-conformal-benchmarks"
WS_PATH = f"{WS_DIR}/benchmark_gbm"

try:
    w.workspace.mkdirs(WS_DIR)
except Exception:
    pass

w.workspace.import_(
    path=WS_PATH,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(notebook_src.encode()).decode(),
    overwrite=True,
)
print(f"Uploaded notebook to {WS_PATH}")

# ---------------------------------------------------------------------------
# Submit: no new_cluster, no environment_key (uses workspace default serverless)
# ---------------------------------------------------------------------------

run_response = w.jobs.submit(
    run_name="insurance-conformal-benchmark-gbm",
    tasks=[
        dbr_jobs.SubmitTask(
            task_key="benchmark",
            notebook_task=dbr_jobs.NotebookTask(
                notebook_path=WS_PATH,
                source=dbr_jobs.Source.WORKSPACE,
            ),
        )
    ],
)

run_id = run_response.run_id
print(f"Job submitted. Run ID: {run_id}")
print(f"Track at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")
print()
print("Waiting for completion (polling every 30s)...")

while True:
    time.sleep(30)
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    life_cycle = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    result_state = state.result_state.value if state and state.result_state else ""
    print(f"  [{time.strftime('%H:%M:%S')}] {life_cycle} {result_state}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print()
if result_state == "SUCCESS":
    print("Benchmark completed successfully.")
    print(f"View output at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")
    # Try to get notebook output
    try:
        run = w.jobs.get_run(run_id=run_id)
        for t in (run.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    print("\n=== NOTEBOOK OUTPUT ===")
                    print(out.notebook_output.result[:10000])
                elif out.logs:
                    print("\n=== LOGS ===")
                    print(out.logs[:10000])
            except Exception as e2:
                print(f"  (task output err): {e2}")
    except Exception as e:
        print(f"Could not retrieve output: {e}")
else:
    print(f"Run ended with state: {result_state}")
    try:
        run = w.jobs.get_run(run_id=run_id)
        for t in (run.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.error:
                    print(f"Error: {out.error}")
                if out.error_trace:
                    print(f"Trace: {out.error_trace[:3000]}")
            except Exception as e2:
                print(f"  (task output err): {e2}")
    except Exception as e:
        print(f"Could not retrieve error details: {e}")
    print(f"View details at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

sys.exit(0 if result_state == "SUCCESS" else 1)
