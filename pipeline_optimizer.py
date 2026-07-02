# ==========================================
# PIPELINE OPTIMIZER: Hyperparameter Tuning (HPO)
# ==========================================
from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna # Usa Optuna, il motore di ottimizzazione state-of-the-art

# ---------------------------------------------------------
# 1. INIZIALIZZAZIONE TASK DI OTTIMIZZAZIONE
# ---------------------------------------------------------
# Questo script è esso stesso un Task ClearML (il "Manager")
task = Task.init(
    project_name='Progetto_MLOps_Esame',
    task_name='Pipeline_HPO_XGBoost_Tuning_V2',
    task_type=Task.TaskTypes.optimizer
)
task.add_tags(['hpo', 'optuna', 'xgboost'])

# ---------------------------------------------------------
# 2. DEFINIZIONE DEL TEMPLATE (Il Task di Paolo)
# ---------------------------------------------------------
# ATTENZIONE: Inserisci qui l'ID del Task di Paolo che ha girato con successo!
# (Lo trovi nella UI di ClearML, è la stringa alfanumerica sotto il nome del Task)
PAOLO_TASK_ID = '953685d1d7f84023ae30fc1746723d03' 

print(f"Inizializzazione Ottimizzatore basato sul Task di Paolo: {PAOLO_TASK_ID}")

# ---------------------------------------------------------
# 3. CONFIGURAZIONE DELL'OTTIMIZZATORE
# ---------------------------------------------------------
optimizer = HyperParameterOptimizer(
    base_task_id=PAOLO_TASK_ID,
    
    hyper_parameters=[
        DiscreteParameterRange('General/learning_rate', values=[0.01, 0.05, 0.1, 0.2]),
        UniformIntegerParameterRange('General/max_depth', min_value=3, max_value=8, step_size=1),
        UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
    ],
    
    objective_metric_title='Metrics',
    objective_metric_series='Accuracy',
    objective_metric_sign='max',
    
    # Impostazioni di esecuzione dell'ottimizzatore
    max_number_of_concurrent_tasks=2, # Quanti cloni far girare contemporaneamente
    optimizer_class=OptimizerOptuna,  # Il motore matematico (Optuna è eccellente)
    execution_queue='default',        # La coda in cui verranno messi i cloni
    # Limiti di tempo e numero di prove (per non far girare il PC all'infinito)
    total_max_jobs=10,                # Fara al massimo 10 tentativi (puoi aumentarlo)
    max_iteration_per_job=30          # Tempo massimo per singolo job
)

# ---------------------------------------------------------
# 4. ESECUZIONE (La Magia)
# ---------------------------------------------------------
print("Avvio della ricerca automatica degli iperparametri (HPO)...")
# Impostiamo l'ottimizzatore per girare localmente sul tuo PC
optimizer.set_report_period(1) # Aggiorna la dashboard ogni minuto
optimizer.start_locally(job_complete_callback=None)

# Lo script attendera qui finché non ha finito tutte le prove
optimizer.wait()

# ---------------------------------------------------------
# 5. RISULTATI
# ---------------------------------------------------------
print("\nOttimizzazione Completata!")
top_experiment = optimizer.get_top_experiments(top_k=1)[0]
print(f"L'esperimento migliore è stato il Task ID: {top_experiment.id}")
print("Controlla la dashboard di ClearML nel tab 'Plots' di questo task per vedere il grafico a dispersione!")

# Alla fine chiudiamo l'ottimizzatore
optimizer.stop()