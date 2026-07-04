from clearml.automation.controller import PipelineController

# 1. Inizializzazione dell'Orchestratore del DAG
pipe = PipelineController(
    name='Orchestratore_DAG_Produzione',
    project='Progetto_MLOps_Esame',
    version='1.0.0',
    add_pipeline_tags=False
)

# 2. Parametri Globali (Prevenzione Data Drift)
# In futuro, basterà cambiare questo URL per far girare tutto sui nuovi dati mensili
pipe.add_parameter(
    name='dataset_url', 
    default='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet'
)

# 3. Nodo A: Pipeline di Marco (Ingestion & Baseline)
pipe.add_step(
    name='step_marco_baseline',
    base_task_project='Progetto_MLOps_Esame',
    base_task_name='Pipeline_Marco_RF_Baseline',
    parameter_override={'General/data_url': '${pipeline.dataset_url}'}
)

# 4. Nodo B: Pipeline di Paolo (Avanzato & Gatekeeper)
pipe.add_step(
    name='step_paolo_xgboost',
    parents=['step_marco_baseline'], # Il DAG forza l'attesa: Paolo parte SOLO se Marco finisce
    base_task_project='Progetto_MLOps_Esame',
    base_task_name='Pipeline_Paolo_XGBoost',
    parameter_override={
        # Passaggio deterministico e nativo dell'ID tra i due nodi! Nessuna race condition.
        'Args/source_task_id': '${step_marco_baseline.id}' 
    }
)

if __name__ == '__main__':
    print("Avvio del PipelineController ClearML...")
    # L'esecuzione locale è perfetta per le istanze di GitHub Actions.
    # In azienda si userebbe pipe.start(queue='default') per delegare a cluster remoti con GPU.
    pipe.start_locally(run_pipeline_steps_locally=True)