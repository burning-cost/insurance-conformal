"""
Run ConditionalValidityIndex v1.3.0 tests on Databricks serverless.
Tests: n<800 warning, expected_calibration_error(), plot_conditional_validity().
"""
import os
import base64
import time
import sys

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import NotebookTask, Task, Source

w = WorkspaceClient()

import pathlib

repo_root = pathlib.Path(__file__).parent
workspace_base = "/Workspace/insurance-conformal-v130-test"

created_dirs: set = set()


def ensure_dir(ws_dir: str):
    if ws_dir in created_dirs or ws_dir == "/Workspace":
        return
    try:
        w.workspace.mkdirs(path=ws_dir)
    except Exception:
        pass
    created_dirs.add(ws_dir)


def upload_file(local_path: pathlib.Path, workspace_path: str):
    parent = str(pathlib.PurePosixPath(workspace_path).parent)
    ensure_dir(parent)
    content = local_path.read_bytes()
    b64 = base64.b64encode(content).decode()
    w.workspace.import_(
        path=workspace_path,
        content=b64,
        format=ws_svc.ImportFormat.AUTO,
        overwrite=True,
    )


ensure_dir(workspace_base)

SKIP_DIRS = {".venv", "__pycache__", ".git", "build", "dist", ".eggs", ".tox"}

for py_file in sorted(repo_root.rglob("*.py")):
    rel = py_file.relative_to(repo_root)
    parts_set = set(rel.parts[:-1])
    if parts_set & SKIP_DIRS:
        continue
    if any(p.startswith(".") for p in rel.parts[:-1]):
        continue
    if rel.name == "run_cvi_v130_tests_databricks.py":
        continue
    ws_path = f"{workspace_base}/{rel}"
    print(f"  Uploading {rel}")
    upload_file(py_file, ws_path)

upload_file(repo_root / "pyproject.toml", f"{workspace_base}/pyproject.toml")

notebook_content = '''# Databricks notebook source
import subprocess, sys, os, shutil

log = []

r = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "pytest", "pytest-cov", "scipy", "polars", "numpy", "pandas",
     "pyarrow", "scikit-learn", "lightgbm", "matplotlib"],
    capture_output=True, text=True
)
log.append(f"DEP_RC={r.returncode}")
if r.returncode != 0:
    log.append("DEP_STDERR:" + r.stderr[-500:])
    dbutils.notebook.exit("\\n".join(log))

WS = "/Workspace/insurance-conformal-v130-test"
TMP = "/tmp/insurance-conformal-v130-test"
if os.path.exists(TMP):
    shutil.rmtree(TMP)
shutil.copytree(WS, TMP)
log.append("copytree OK")

src_path = f"{TMP}/src"
env = {**os.environ, "PYTHONPATH": src_path, "PYTHONDONTWRITEBYTECODE": "1",
       "MPLBACKEND": "Agg"}

# Import check
r2 = subprocess.run(
    [sys.executable, "-c",
     f"import sys; sys.path.insert(0, \\"{src_path}\\"); "
     "from insurance_conformal.conditional_coverage import ConditionalValidityIndex; "
     "print(\\'ConditionalValidityIndex OK\\')"],
    capture_output=True, text=True, env=env
)
log.append(f"IMPORT_RC={r2.returncode}")
log.append("IMPORT_OUT:" + r2.stdout.strip())
if r2.returncode != 0:
    log.append("IMPORT_ERR:" + r2.stderr[-1000:])
    dbutils.notebook.exit("\\n".join(log))

# Run full test suite (all conditional coverage tests including new v1.3.0 tests)
r3 = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{TMP}/tests/test_conditional_coverage.py",
     "-v", "--tb=short",
     "-p", "no:cacheprovider",
     "--import-mode=importlib"],
    capture_output=True, text=True,
    cwd=TMP,
    env=env
)
log.append(f"PYTEST_RC={r3.returncode}")
out = r3.stdout
if r3.stderr:
    out += "\\nSTDERR:" + r3.stderr[-1000:]
log.append(out[-14000:] if len(out) > 14000 else out)

dbutils.notebook.exit("\\n".join(log))
'''

nb_b64 = base64.b64encode(notebook_content.encode()).decode()
nb_path = f"{workspace_base}/run_cvi_v130_tests"
w.workspace.import_(
    path=nb_path,
    content=nb_b64,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print(f"Notebook uploaded to {nb_path}")

run = w.jobs.submit(
    run_name="insurance-conformal-v130-cvi-tests",
    tasks=[
        Task(
            task_key="run_tests",
            notebook_task=NotebookTask(
                notebook_path=nb_path,
                source=Source.WORKSPACE,
            ),
        )
    ],
)
run_id = run.run_id
print(f"Submitted run_id={run_id}")

for _ in range(80):
    time.sleep(15)
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state.value
    print(f"  State: {life_cycle}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        result_state = run_state.state.result_state
        print(f"  Result: {result_state}")
        task_run = run_state.tasks[0] if run_state.tasks else None
        nb_result = ""
        if task_run:
            output = w.jobs.get_run_output(run_id=task_run.run_id)
            if output.notebook_output and output.notebook_output.result:
                nb_result = output.notebook_output.result
                print("\n=== NOTEBOOK OUTPUT ===")
                print(nb_result)
            if output.error:
                print("\n=== ERROR ===")
                print(output.error)
            if output.error_trace:
                print("\n=== TRACEBACK ===")
                print(output.error_trace[-2000:])
        if "PYTEST_RC=0" in nb_result:
            print("\nSUCCESS - all tests passed")
        else:
            print("\nFAILED")
            sys.exit(1)
        break
else:
    print("Timed out")
    sys.exit(1)
