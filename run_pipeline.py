# ==========================================
# RUN PIPELINE: Orchestrazione del workflow
# ==========================================
import subprocess
import sys
import time
from clearml import Task

PROJECT_NAME = 'Progetto_MLOps_Esame'
MARCO_TASK_NAME = 'Pipeline_Marco_RF_Baseline'
PAOLO_TASK_NAME = 'Pipeline_Paolo_XGBoost'

MARCO_SCRIPT = 'pipeline_marco.py'
PAOLO_SCRIPT = 'pipeline_paolo.py'
OPTIMIZER_SCRIPT = 'pipeline_optimizer.py'


def run_step(command, step_name):
    print(f"\n=== Avvio {step_name} ===")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Errore durante l'esecuzione di {step_name}")
    print(f"=== {step_name} completato ===")


def get_task_or_fail(project_name, task_name, wait_seconds=3):
    time.sleep(wait_seconds)
    task = Task.get_task(project_name=project_name, task_name=task_name)
    if task is None:
        raise RuntimeError(f"Task non trovato: {task_name}")
    return task


def main():
    run_step([sys.executable, MARCO_SCRIPT], "Pipeline Marco")

    marco_task = get_task_or_fail(PROJECT_NAME, MARCO_TASK_NAME)
    print(f"Task Marco trovato: {marco_task.id}")

    if 'taxi_data_cleaned' not in marco_task.artifacts:
        raise RuntimeError("Artifact 'taxi_data_cleaned' non trovato nel task di Marco")

    run_step(
        [sys.executable, PAOLO_SCRIPT, '--source_task_id', marco_task.id],
        "Pipeline Paolo"
    )

    paolo_task = get_task_or_fail(PROJECT_NAME, PAOLO_TASK_NAME)
    print(f"Task Paolo trovato: {paolo_task.id}")

    run_step(
        [sys.executable, OPTIMIZER_SCRIPT, '--base_task_id', paolo_task.id],
        "Pipeline Optimizer"
    )

    print("\nWorkflow completato con successo!")


if __name__ == '__main__':
    main()