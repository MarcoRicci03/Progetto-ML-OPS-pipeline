from clearml.automation.controller import PipelineController

# 1. Inizializzazione dell'Orchestratore del DAG
pipe = PipelineController(
    name='Orchestratore_DAG_Produzione',
    project='Progetto_MLOps_Esame',
    version='1.0.0',
    add_pipeline_tags=False
)

# 2. Parametri Globali
pipe.add_parameter(
    name='dataset_url', 
    default='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
)

# 3. NODO A: Pipeline di Marco (Ingestion & Baseline)
pipe.add_step(
    name='step_marco_baseline',
    base_task_project='Progetto_MLOps_Esame',
    base_task_name='Pipeline_Marco_RF_Baseline',
    parameter_override={'General/data_url': '${pipeline.dataset_url}'}
)

# 4. NODO B: Pipeline di Paolo (Avanzato & Gatekeeper)
pipe.add_step(
    name='step_paolo_xgboost',
    parents=['step_marco_baseline'], # Attende Marco
    base_task_project='Progetto_MLOps_Esame',
    base_task_name='Pipeline_Paolo_XGBoost',
    parameter_override={
        'Args/source_task_id': '${step_marco_baseline.id}' 
    }
)

# 5. NODO C: Ottimizzazione Iperparametri (HPO)
pipe.add_step(
    name='step_hpo_tuning',
    parents=['step_paolo_xgboost'], # Attende che Paolo finisca l'addestramento base
    base_task_project='Progetto_MLOps_Esame',
    base_task_name='Pipeline_HPO_XGBoost_Tuning', # Il nome esatto del task nel tuo script
    parameter_override={
        # Inietta l'ID del task di Paolo come "base_task_id" per l'ottimizzatore
        'Args/base_task_id': '${step_paolo_xgboost.id}'
    }
)

if __name__ == '__main__':
    print("Avvio del PipelineController ClearML...")
    pipe.start_locally(run_pipeline_steps_locally=True)