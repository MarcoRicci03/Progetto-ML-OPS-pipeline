# -*- coding: utf-8 -*-
import argparse
from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna

PROJECT_NAME = 'Progetto_MLOps_Esame'
TASK_NAME = 'Pipeline_HPO_XGBoost_Tuning'


parser = argparse.ArgumentParser(description='ClearML HPO per il task XGBoost di Paolo')
parser.add_argument(
    '--base_task_id',
    required=True,
    help='ID del task ClearML da usare come template per l’ottimizzazione'
)
args = parser.parse_args()


# 1. INIZIALIZZAZIONE TASK DI OTTIMIZZAZIONE
task = Task.init(
    project_name=PROJECT_NAME,
    task_name=TASK_NAME,
    task_type=Task.TaskTypes.optimizer
)

# 2. VALIDAZIONE DEL TASK TEMPLATE
paolo_task = Task.get_task(task_id=args.base_task_id)

if paolo_task is None:
    raise RuntimeError('Task di Paolo non trovato su ClearML')

print(f"Inizializzazione Ottimizzatore basato sul Task di Paolo: {paolo_task.id}")

# 3. CONFIGURAZIONE DELL'OTTIMIZZATORE
optimizer = HyperParameterOptimizer(
    base_task_id=paolo_task.id,

    hyper_parameters=[
        DiscreteParameterRange('General/learning_rate', values=[0.01, 0.05, 0.1, 0.2]),
        UniformIntegerParameterRange('General/max_depth', min_value=3, max_value=8, step_size=1),
        UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
    ],

    objective_metric_title='Metrics',
    objective_metric_series='ROC_AUC',
    objective_metric_sign='max',

    max_number_of_concurrent_tasks=2,
    optimizer_class=OptimizerOptuna,
    execution_queue='default',
    total_max_jobs=10,
    max_iteration_per_job=30
)

# 4. ESECUZIONE
print("Avvio della ricerca automatica degli iperparametri (HPO)...")
optimizer.set_report_period(1)
optimizer.start_locally(job_complete_callback=None)

optimizer.wait()

# 5. RISULTATI
print("\nOttimizzazione completata!")

top_experiments = optimizer.get_top_experiments(top_k=1)
if top_experiments:
    print(f"L'esperimento migliore è stato il Task ID: {top_experiments[0].id}")
    print("Controlla la dashboard di ClearML nel tab 'Plots' di questo task per vedere il grafico a dispersione!")
else:
    print("Nessun esperimento completato correttamente.")

# 6. CHIUSURA
optimizer.stop()
task.close()