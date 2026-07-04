from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from clearml import Model, Task
import joblib
import pandas as pd
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connessione a ClearML in corso...")
    try:
        print("Ricerca del task HPO...")
        hpo_task = Task.get_task(
            project_name='Progetto_MLOps_Esame', 
            task_name='Pipeline_HPO_XGBoost_Tuning'
        )
        
        if 'best_xgboost_model' not in hpo_task.artifacts:
            raise RuntimeError("Artifact 'best_xgboost_model' non trovato nel task HPO!")
            
        print("Artifact trovato! Scaricamento del modello migliore in corso...")
        
        model_path = hpo_task.artifacts['best_xgboost_model'].get_local_copy()
        
        ml_models["xgboost"] = joblib.load(model_path)
        print("Modello caricato in memoria con successo e pronto per le previsioni!")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
    
    yield
    
    ml_models.clear()
    print("Risorse liberate. Server spento.")


app = FastAPI(
    title="NY Taxi Tip Predictor",
    description="API MLOps per prevedere se una corsa in taxi riceverà una mancia alta",
    version="1.0",
    lifespan=lifespan
)

class TripData(BaseModel):
    duration_min: float
    fare_amount: float
    trip_distance: float
    tolls_amount: float
    Airport_fee: float
    is_weekend: int
    is_rush_hour: int
    is_airport: int

@app.post("/predict")
def predict_tip(trip: TripData):
    model = ml_models.get("xgboost")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Modello non inizializzato.")
    
    input_data = pd.DataFrame([trip.model_dump()])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "mancia_alta": bool(prediction == 1),
        "probabilita_mancia_alta": round(float(probability), 4),
        "messaggio": "Il passeggero lascerà una buona mancia!" if prediction == 1 else "Probabile mancia bassa."
    }