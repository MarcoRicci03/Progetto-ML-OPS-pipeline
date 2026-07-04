import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from clearml import Task, Logger, OutputModel # IMPORTAZIONE CLEARML PORTATA SU

parser = argparse.ArgumentParser()
parser.add_argument('--source_task_id', required=True)
args = parser.parse_args()

PROJECT_NAME = 'Progetto_MLOps_Esame'

# 1. INIZIALIZZAZIONE CLEARML E RECUPERO ARTIFACT (PRIMA DI XGBOOST)
source_task = Task.get_task(task_id=args.source_task_id)
if source_task is None:
    raise RuntimeError('Task sorgente non trovato su ClearML')

if 'taxi_data_cleaned' not in source_task.artifacts:
    raise RuntimeError("Artifact 'taxi_data_cleaned' non trovato nel task sorgente")

df = source_task.artifacts['taxi_data_cleaned'].get()
print(f"Dataset ottenuto dal task {source_task.id}")

task = Task.init(
    project_name=PROJECT_NAME,
    task_name='Pipeline_Paolo_XGBoost'
)

# 2. IMPORT XGBOOST (DOPO L'INIZIALIZZAZIONE DEL TASK) E ALTRE LIBRERIE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

params = {
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 5,
    'test_size': 0.2,
    'val_size': 0.1,
    'early_stopping_rounds': 15,
    'random_state': 42
}
task.connect(params)

# 3. SPLIT DATI
X = df.drop(columns=['target'])
y = df['target']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=params['test_size'], shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=params['val_size'], shuffle=False
)

# 4. ADDESTRAMENTO MODELLO AVANZATO
print(f"Addestramento XGBoost Classifier (Estimators: {params['n_estimators']})...")
xgb_model = XGBClassifier(
    n_estimators=params['n_estimators'], 
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'], 
    random_state=params['random_state'],
    eval_metric='auc', # Utilizziamo l'AUC come metrica da monitorare per l'early stopping
    early_stopping_rounds=params['early_stopping_rounds']
)
xgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)], # TRACCIAMO SIA TRAIN CHE VAL PER LA CURVA
    verbose=False # Evitiamo di inondare la console con troppi log
)

best_iteration = xgb_model.best_iteration
print(f"L'addestramento si è fermato all'albero n. {best_iteration} (Early Stopping)")

# 5. VALUTAZIONE E LOGGING METRICHE
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nRISULTATI PAOLO (XGBOOST):")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

Logger.current_logger().report_scalar(title='Metrics', series='Accuracy', value=acc, iteration=0)
Logger.current_logger().report_scalar(title='Metrics', series='ROC_AUC', value=auc, iteration=0)
Logger.current_logger().report_scalar(title='Metrics', series='Best_Iteration', value=best_iteration, iteration=0)

# 5.5 LOGGING MANUALE DELLA CURVA DI APPRENDIMENTO (LOSS) NEI SCALAR
print("Estrazione della Loss e invio a ClearML Scalar...")
evals_result = xgb_model.evals_result()
logger = Logger.current_logger()

# Poiché abbiamo impostato eval_metric='auc', il dizionario conterrà i valori AUC
for i in range(len(evals_result['validation_0']['auc'])):
    logger.report_scalar(
        title="Learning Curve (AUC per Boosting Round)", 
        series="Train AUC", 
        value=evals_result['validation_0']['auc'][i], 
        iteration=i
    )
    logger.report_scalar(
        title="Learning Curve (AUC per Boosting Round)", 
        series="Validation AUC", 
        value=evals_result['validation_1']['auc'][i], 
        iteration=i
    )

# 6. PLOTS, EXPLAINABILITY & MODEL REGISTRY
print("\nGenerazione grafici di valutazione per XGBoost...")

# --- Matrice di Confusione ---
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, ax=ax, cmap='Oranges') # Uso l'arancione per distinguerlo da Marco
plt.title('Matrice di Confusione - Paolo XGBoost')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Matrice di Confusione", iteration=0, figure=fig
)
plt.close(fig)

# --- Curva ROC ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test, ax=ax2)
plt.title('Curva ROC - Paolo XGBoost')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Curva ROC", iteration=0, figure=fig2
)
plt.close(fig2)

# --- Feature Importance (XGBoost) ---
print("Generazione Feature Importance...")
importances = xgb_model.feature_importances_
feature_names = X_train.columns
xgb_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(8, 5))
xgb_importances.plot.bar(ax=ax3, color='darkorange')
ax3.set_title("Feature Importance - XGBoost")
ax3.set_ylabel("Importanza Relativa")
fig3.tight_layout()

Logger.current_logger().report_matplotlib_figure(
    title="Explainability", series="Feature Importance", iteration=0, figure=fig3
)
plt.close(fig3)

# --- Salvataggio Modello ---
print("Salvataggio del modello nel Model Registry...")
joblib.dump(xgb_model, 'paolo_xgboost_model.pkl')
output_model = OutputModel(task=task, framework="XGBoost")
output_model.update_weights(
    weights_filename='paolo_xgboost_model.pkl',
    auto_delete_file=False
)
print("\nSincronizzazione con il server in corso (10 secondi per garantire l'HPO)...")
task.flush(wait_for_uploads=True)

import time
time.sleep(10)
task.close()

print("\nPipeline di Paolo completata con successo!")