"""
Run insurance-conformal selection_scores unit tests on Databricks.
Tests: tests/risk/test_selection_scores.py
Uses existing job: insurance-conformal-fraud-tests-v2 (792664693639819)
by uploading to /Workspace/insurance-conformal-fix-test/run_fix_tests
"""
from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path

env_file = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_file.read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc

w = WorkspaceClient()
HOST = os.environ["DATABRICKS_HOST"].rstrip("/")

WORKSPACE_ROOT = "/Workspace/insurance-conformal-fix-test"
NOTEBOOK_PATH = f"{WORKSPACE_ROOT}/run_fix_tests"
JOB_ID = 792664693639819

LOCAL_ROOT = Path(__file__).parent


def ws_mkdirs(path: str) -> None:
    try:
        w.workspace.mkdirs(path=path)
    except Exception:
        pass


def upload_file_raw(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ws_svc.ImportFormat.AUTO,
    )


def upload_directory(local_dir: Path, remote_dir: str) -> None:
    for local_path in sorted(local_dir.rglob("*")):
        if not local_path.is_file():
            continue
        if any(part.startswith(".") or part in ("__pycache__", "dist", "build") for part in local_path.parts):
            continue
        if local_path.suffix in {".pyc", ".pyo"}:
            continue
        rel = local_path.relative_to(local_dir)
        remote_path = f"{remote_dir}/{rel.as_posix()}"
        ws_mkdirs(remote_path.rsplit("/", 1)[0])
        print(f"  {rel}")
        upload_file_raw(local_path, remote_path)


print("Uploading insurance-conformal source + tests...")
ws_mkdirs(WORKSPACE_ROOT)
upload_file_raw(LOCAL_ROOT / "pyproject.toml", f"{WORKSPACE_ROOT}/pyproject.toml")
upload_directory(LOCAL_ROOT / "src", f"{WORKSPACE_ROOT}/src")
upload_directory(LOCAL_ROOT / "tests", f"{WORKSPACE_ROOT}/tests")

NOTEBOOK_CONTENT = r"""# Databricks notebook source
import subprocess, sys, os, shutil

log = []

r = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "pytest", "scipy", "polars", "numpy", "pandas",
     "pyarrow", "scikit-learn", "catboost", "lightgbm"],
    capture_output=True, text=True
)
log.append(f"DEP_RC={r.returncode}")
if r.returncode != 0:
    log.append("DEP_STDERR:" + r.stderr[-500:])
    dbutils.notebook.exit("\n".join(log))

WS = "/Workspace/insurance-conformal-fix-test"
TMP = "/tmp/insurance-conformal-fix-test"
if os.path.exists(TMP):
    shutil.rmtree(TMP)
shutil.copytree(WS, TMP)
log.append("copytree OK")

src_path = f"{TMP}/src"
env = {**os.environ, "PYTHONPATH": src_path, "PYTHONDONTWRITEBYTECODE": "1"}

r3 = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{TMP}/tests/risk/test_selection_scores.py",
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
    out += "\nSTDERR:" + r3.stderr[-1000:]
log.append(out[-12000:] if len(out) > 12000 else out)

dbutils.notebook.exit("\n".join(log))
"""

print(f"Uploading notebook to {NOTEBOOK_PATH}")
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=base64.b64encode(NOTEBOOK_CONTENT.encode()).decode(),
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print("Notebook uploaded.")

print(f"\nTriggering job {JOB_ID} (insurance-conformal-fraud-tests-v2)...")
run_response = w.jobs.run_now(job_id=JOB_ID)
run_id = run_response.run_id
print(f"Run submitted: run_id={run_id}")
print(f"Track at: {HOST}/#job/{JOB_ID}/run/{run_id}")

poll_interval = 20
max_wait = 900
elapsed = 0

while elapsed < max_wait:
    status = w.jobs.get_run(run_id=run_id)
    state = status.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state.result_state else ""
    print(f"  [{elapsed:3d}s] {lc} {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    import time
    time.sleep(poll_interval)
    elapsed += poll_interval

print("\n--- Run output ---")
try:
    output = w.jobs.get_run_output(run_id=run_id)
    if output.notebook_output and output.notebook_output.result:
        nb_result = output.notebook_output.result
        print(nb_result)
    elif output.error:
        print("ERROR:", output.error)
        if output.error_trace:
            print(output.error_trace[-3000:])
    else:
        print("(no notebook output captured)")
except Exception as exc:
    nb_result = ""
    print(f"Could not fetch output: {exc}")

final = w.jobs.get_run(run_id=run_id).state
result = final.result_state.value if final.result_state else "UNKNOWN"
print(f"\nFinal result: {result}")

if result != "SUCCESS":
    sys.exit(1)
