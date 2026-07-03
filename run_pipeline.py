from clearml import PipelineController

PROJECT_NAME = 'Progetto_MLOps_Esame'
MARCO_TASK_NAME = 'Pipeline_Marco_RF_Baseline'
PAOLO_TASK_NAME = 'Pipeline_Paolo_XGBoost'
OPTIMIZER_TASK_NAME = 'Pipeline_HPO_XGBoost_Tuning'

def main():
    print("=== Inizializzazione PipelineController ===")
    
    # 1. Creazione della Pipeline
    pipe = PipelineController(
        name='Taxi_NY_Orchestrator',
        project=PROJECT_NAME,
        version='1.0'
    )

    # Imposta la coda di default (utile se usi agent remoti, altrimenti in locale viene ignorata)
    pipe.set_default_execution_queue('default')

    # ==========================================
    # 2. Definizione degli Step (Il DAG)
    # ==========================================

    # Step 1: Pipeline Marco (Baseline & Data Prep)
    pipe.add_step(
        name='step_marco',
        base_task_project=PROJECT_NAME,
        base_task_name=MARCO_TASK_NAME,
        task_overrides={
            'script.repository': 'https://github.com/MarcoRicci03/Progetto-ML-OPS-pipeline.git',
            'script.branch': 'main',
            'script.version_num': ''
        }
    )

    # Step 2: Pipeline Paolo (XGBoost)
    # 'parents' crea la dipendenza: questo step attende la fine di Marco
    pipe.add_step(
        name='step_paolo',
        parents=['step_marco'],
        base_task_project=PROJECT_NAME,
        base_task_name=PAOLO_TASK_NAME,
        parameter_override={
            'Args/source_task_id': '${step_marco.id}'
        },
        task_overrides={
            'script.repository': 'https://github.com/MarcoRicci03/Progetto-ML-OPS-pipeline.git',
            'script.branch': 'main',
            'script.version_num': ''
        }
    )

    # Step 3: Pipeline Optimizer (HPO)
    pipe.add_step(
        name='step_optimizer',
        parents=['step_paolo'],
        base_task_project=PROJECT_NAME,
        base_task_name=OPTIMIZER_TASK_NAME,
        parameter_override={
            # Passiamo l'ID generato dallo step_paolo al parametro dell'Optimizer
            'Args/base_task_id': '${step_paolo.id}'
        },
        task_overrides={
            'script.repository': 'https://github.com/MarcoRicci03/Progetto-ML-OPS-pipeline.git',
            'script.branch': 'main',
            'script.version_num': ''
        }
    )

    # ==========================================
    # 3. Esecuzione
    # ==========================================
    print("=== Avvio della Pipeline su ClearML ===")
    
    # run_pipeline_steps_locally=True esegue tutto sul tuo computer
    # Se imposti a False, ClearML invierà i task alle code in attesa degli agent remoti!
    pipe.start_locally(run_pipeline_steps_locally=False)
    
    print("\nWorkflow completato con successo! Controlla la dashboard UI di ClearML per visualizzare il grafo.")

if __name__ == '__main__':
    main()