"""
Run MOPI (ShapeAdaptiveCP) tests on Databricks serverless compute.

Usage:
    python run_mopi_tests_databricks.py
"""

import os
import sys
import time
import base64

# Load credentials from ~/.config/burning-cost/databricks.env
_env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k, _v)

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import NotebookTask, Task, Source

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/insurance-conformal-mopi"

# ---------------------------------------------------------------------------
# Upload source tree
# ---------------------------------------------------------------------------

import pathlib

repo_root = pathlib.Path(__file__).parent

def upload_file(local_path: pathlib.Path, remote_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    try:
        w.workspace.import_(
            path=remote_path,
            content=encoded,
            format=ws_svc.ImportFormat.AUTO,
            overwrite=True,
        )
    except Exception as e:
        print(f"  WARNING upload {remote_path}: {e}")

def upload_dir(local_dir: pathlib.Path, remote_base: str, extensions=(".py", ".toml", ".cfg", ".txt")):
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and f.suffix in extensions and "__pycache__" not in str(f):
            rel = f.relative_to(local_dir)
            remote_path = f"{remote_base}/{rel}".replace("\\", "/")
            upload_file(f, remote_path)
            print(f"  uploaded {rel}")

print("Uploading source files...")
upload_dir(repo_root / "src", f"{WORKSPACE_DIR}/src")
upload_dir(repo_root / "tests", f"{WORKSPACE_DIR}/tests", extensions=(".py",))

# Upload pyproject.toml
upload_file(repo_root / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
print("  uploaded pyproject.toml")

# ---------------------------------------------------------------------------
# Build test notebook
# ---------------------------------------------------------------------------

NOTEBOOK_PATH = f"{WORKSPACE_DIR}/run_tests"

NOTEBOOK_SOURCE = r"""# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars>=1.0 scikit-learn>=1.6 scipy>=1.12 -q

# COMMAND ----------
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-conformal-mopi", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")

# COMMAND ----------
import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-conformal-mopi/tests/test_mopi.py",
        "-v", "--tb=short", "-x",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-conformal-mopi"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
print("Exit code:", result.returncode)
assert result.returncode == 0, f"Tests failed! Exit code {result.returncode}"
"""

encoded_nb = base64.b64encode(NOTEBOOK_SOURCE.encode()).decode()
try:
    w.workspace.import_(
        path=NOTEBOOK_PATH,
        content=encoded_nb,
        format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
        overwrite=True,
    )
    print(f"Notebook uploaded to {NOTEBOOK_PATH}")
except Exception as e:
    print(f"Notebook upload error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Submit as job run
# ---------------------------------------------------------------------------

print("Submitting test job...")
run_waiter = w.jobs.submit(
    run_name="insurance-conformal-mopi-tests",
    tasks=[
        Task(
            task_key="run_tests",
            notebook_task=NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                source=Source.WORKSPACE,
            ),
        )
    ],
)

run_id = run_waiter.run_id
print(f"Job submitted. Run ID: {run_id}")

# ---------------------------------------------------------------------------
# Poll for result
# ---------------------------------------------------------------------------

print("Waiting for job to complete...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state.value
    print(f"  State: {life_cycle}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(20)

result_state = run_state.state.result_state.value if run_state.state.result_state else "UNKNOWN"
print(f"Final state: {result_state}")

# Fetch output
for task in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n=== NOTEBOOK OUTPUT ===")
            print(output.notebook_output.result)
        if output.error:
            print("\n=== ERROR ===")
            print(output.error)
        if output.error_trace:
            print("\n=== ERROR TRACE ===")
            print(output.error_trace[:3000])
    except Exception as e:
        print(f"Could not fetch output: {e}")

if result_state != "SUCCESS":
    print("\nTests FAILED on Databricks")
    sys.exit(1)
else:
    print("\nAll tests PASSED on Databricks")
