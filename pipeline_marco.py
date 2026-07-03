import pandas as pd
import pyarrow
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from clearml import Task, Logger, OutputModel

# 1. INIZIALIZZAZIONE CLEARML
task = Task.init(
    project_name='Progetto_MLOps_Esame', 
    task_name='Pipeline_Marco_RF_Baseline'
)

params = {
    'n_estimators': 50,
    'max_depth': 5,
    'test_size': 0.2,
    'random_state': 42,
    'data_url': "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
}
task.connect(params)

# 2. CARICAMENTO DATI
print("Scaricamento dei dati dei Taxi di NY...")
df_raw = pd.read_parquet(params['data_url'])

# 3. PULIZIA E FILTRI INIZIALI
print("Pulizia e filtraggio dei dati...")
df = df_raw[
    (df_raw['payment_type'] == 1) &
    (df_raw['RatecodeID'].isin([1, 2, 3, 4, 5, 6])) &
    (df_raw['trip_distance'] > 0) &
    (df_raw['passenger_count'] > 0)
].copy().dropna()

df = df.sample(n=100000, random_state=42)

# 4. FEATURE ENGINEERING
print("Creazione nuove feature (Feature Engineering)...")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['is_weekend'] = (df['tpep_pickup_datetime'].dt.dayofweek >= 5).astype(int)

hour = df['pickup_hour']
day = df['tpep_pickup_datetime'].dt.dayofweek
is_weekday = day < 5
morning_rush = (hour >= 7) & (hour <= 9)
evening_rush = (hour >= 16) & (hour <= 19)
df['is_rush_hour'] = ((morning_rush | evening_rush) & is_weekday).astype(int)
df['is_airport'] = df['RatecodeID'].isin([2, 3]).astype(int)

# Filtriamo viaggi anomali
df = df[(df['duration_min'] > 0) & (df['duration_min'] < 720)].copy()

# 5. CREAZIONE DEL TARGET (Percentuale Mancia > Mediana)
df_final = df[df['tip_amount'] >= 0].copy()
tip_pct = (df_final['tip_amount'] / df_final['fare_amount'].where(df_final['fare_amount'] > 0)).fillna(0) * 100
mediana = tip_pct.median()

df_final['target'] = (tip_pct > mediana).astype(int)

print("Ordinamento temporale dei dati per evitare Data Leakage...")
df_final = df_final.sort_values('tpep_pickup_datetime')

features_model = [
    'duration_min', 'fare_amount', 'trip_distance', 'tolls_amount', 
    'Airport_fee', 'is_weekend', 'is_rush_hour', 'is_airport'
]

df_to_share = df_final[features_model].copy()
df_to_share['target'] = df_final['target']

# 6. CONDIVISIONE ASSET
print("Upload dell'Artifact (Dataset Pulito) su ClearML...")
task.upload_artifact(name='taxi_data_cleaned', artifact_object=df_to_share)

# 7. SPLIT DATI
X = df_to_share.drop(columns=['target'])
y = df_to_share['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_size'], shuffle=False # shuffle=False per rispettare l'ordine temporale
)

# 8. ADDESTRAMENTO MODELLO BASELINE
print(f"Addestramento Random Forest Baseline (Depth: {params['max_depth']})...")
rf_base = RandomForestClassifier(
    n_estimators=params['n_estimators'], 
    max_depth=params['max_depth'], 
    random_state=params['random_state']
)
rf_base.fit(X_train, y_train)

# 9. VALUTAZIONE E LOGGING METRICHE
y_pred = rf_base.predict(X_test)
y_proba = rf_base.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nRISULTATI BASELINE:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

# Logghiamo i risultati finali su ClearML
Logger.current_logger().report_single_value(name='Accuracy_Test', value=acc)
Logger.current_logger().report_single_value(name='ROC_AUC_Test', value=auc)

# 10. NUOVA SEZIONE: PLOTS & MODEL REGISTRY
print("\nGenerazione grafici di valutazione...")

# Grafico 1: Matrice di Confusione
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_estimator(rf_base, X_test, y_test, ax=ax, cmap='Blues')
plt.title('Matrice di Confusione - Marco Baseline')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Matrice di Confusione", iteration=0, figure=fig
)
plt.close(fig)

# Grafico 2: Curva ROC
fig2, ax2 = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_estimator(rf_base, X_test, y_test, ax=ax2)
plt.title('Curva ROC - Marco Baseline')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Curva ROC", iteration=0, figure=fig2
)
plt.close(fig2)

# Grafico 3: Feature Importance
print("Generazione Feature Importance...")
importances = rf_base.feature_importances_
feature_names = X_train.columns
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
fig3, ax3 = plt.subplots(figsize=(8, 5))
forest_importances.plot.bar(ax=ax3, color='teal')
ax3.set_title("Feature Importance - Random Forest")
ax3.set_ylabel("Importanza Relativa")
fig3.tight_layout()
Logger.current_logger().report_matplotlib_figure(
    title="Explainability", series="Feature Importance", iteration=0, figure=fig3
)
plt.close(fig3)

print("Salvataggio del modello nel Model Registry di ClearML...")
joblib.dump(rf_base, 'marco_baseline_rf.pkl')

output_model = OutputModel(task=task, framework="Scikit-Learn")
output_model.update_weights(weights_filename='marco_baseline_rf.pkl', auto_delete_file=False)

print("\nPipeline completata con successo!")