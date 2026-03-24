"""
Run CQR tests on Databricks.
All files already uploaded — this just submits the job.
Use this after Databricks job submission is re-enabled.
"""
import os
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
from databricks.sdk.service.jobs import NotebookTask, Task, Source

w = WorkspaceClient()

run = w.jobs.submit(
    run_name="insurance-conformal-v062-cqr-tests",
    tasks=[
        Task(
            task_key="run_tests",
            notebook_task=NotebookTask(
                notebook_path="/Workspace/insurance-conformal-test/run_tests",
                source=Source.WORKSPACE,
            ),
        )
    ],
)
run_id = run.run_id
print(f"Submitted run_id={run_id}")

for _ in range(60):
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
