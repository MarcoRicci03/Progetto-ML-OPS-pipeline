import argparse
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Logger, Model, OutputModel, Task
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--source_task_id', required=True)
parser.add_argument('--baseline_model_id', required=True) # NUOVO PARAMETRO
args = parser.parse_args()

PROJECT_NAME = 'Progetto_MLOps_Esame'

source_task = Task.get_task(task_id=args.source_task_id)
if source_task is None:
    raise RuntimeError('Task sorgente non trovato su ClearML')

if 'taxi_data_cleaned' not in source_task.artifacts:
    raise RuntimeError("Artifact 'taxi_data_cleaned' non trovato nel task sorgente")

df = source_task.artifacts['taxi_data_cleaned'].get()
print(f"Dataset ottenuto dal task {source_task.id}")

# Fase 1: inizializzazione del run ClearML.
task = Task.init(
    project_name=PROJECT_NAME,
    task_name='Pipeline_Paolo_XGBoost'
)

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

# Fase 2: split temporale del dataset.
X = df.drop(columns=['target'])
y = df['target']

# Niente shuffle: il dataset è ordinato temporalmente e va trattato come sequenza.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=params['test_size'], shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=params['val_size'], shuffle=False
)

# Fase 3: training XGBoost con early stopping.
print(f"Addestramento XGBoost Classifier (Estimators: {params['n_estimators']})...")
xgb_model = XGBClassifier(
    n_estimators=params['n_estimators'], 
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'], 
    random_state=params['random_state'],
    eval_metric='auc',
    early_stopping_rounds=params['early_stopping_rounds']
)
xgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

best_iteration = xgb_model.best_iteration
print(f"L'addestramento si è fermato all'albero n. {best_iteration} (Early Stopping)")

# Fase 4: metriche finali sul test set.
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nRISULTATI PAOLO (XGBOOST):")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

Logger.current_logger().report_scalar(title='Metrics', series='Accuracy', value=float(acc), iteration=0)
Logger.current_logger().report_scalar(title='Metrics', series='ROC_AUC', value=float(auc), iteration=0)
Logger.current_logger().report_scalar(title='Metrics', series='Best_Iteration', value=best_iteration, iteration=0)

# La curva AUC per boosting round rende leggibile il comportamento dell'early stopping.
# Fase 5: curva di apprendimento del boosting.
print("Estrazione della Loss e invio a ClearML Scalar...")
evals_result = xgb_model.evals_result()
logger = Logger.current_logger()

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

# Fase 6: grafici di performance ed explainability.
print("\nGenerazione grafici di valutazione per XGBoost...")

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, ax=ax, cmap='Oranges')
plt.title('Matrice di Confusione - Paolo XGBoost')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Matrice di Confusione", iteration=0, figure=fig
)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test, ax=ax2)
plt.title('Curva ROC - Paolo XGBoost')
Logger.current_logger().report_matplotlib_figure(
    title="Performance", series="Curva ROC", iteration=0, figure=fig2
)
plt.close(fig2)

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

# Fase 7: registrazione e gate di promozione.
print("Salvataggio del nuovo modello nel Model Registry...")
joblib.dump(xgb_model, 'paolo_xgboost_model.pkl')

output_model = OutputModel(task=task, framework="XGBoost")
output_model.update_weights(weights_filename='paolo_xgboost_model.pkl', auto_delete_file=False)

print("\n--- Fase di Model Evaluation (Gatekeeper Deterministico) ---")

promote_to_production = True

try:
    # Recupero del modello di Marco tramite l'ID passato dall'orchestratore
    from clearml import Model
    current_prod_model = Model(model_id=args.baseline_model_id)
    print(f"Modello baseline recuperato via ID ({current_prod_model.id}). Inizio il confronto...")
    
    # 1. Download modello baseline
    prod_model_path = current_prod_model.get_local_copy()
    vecchio_modello = joblib.load(prod_model_path)
    
    # 2. Testiamo il modello baseline sul NOSTRO X_test per un confronto alla pari
    y_proba_vecchio = vecchio_modello.predict_proba(X_test)[:, 1]
    auc_vecchio = roc_auc_score(y_test, y_proba_vecchio)
    
    print(f"-> ROC-AUC Modello Baseline (Marco): {auc_vecchio:.4f}")
    print(f"-> ROC-AUC Modello XGBoost (Paolo):  {auc:.4f}")
    
    # 3. Valutazione effettiva
    if auc > auc_vecchio:
        print("L'XGBoost è migliore! Rimuovo eventuali tag di produzione dal vecchio e promuovo il nuovo.")
        
        # Gestione dei tag sul vecchio modello
        vecchi_tags = current_prod_model.tags
        if "production" in vecchi_tags:
            vecchi_tags.remove("production")
        if "archiviata" not in vecchi_tags:
            vecchi_tags.append("archiviata")
        current_prod_model.tags = vecchi_tags
    else:
        print("L'XGBoost NON migliora le prestazioni. Nessuna promozione in produzione.")
        promote_to_production = False
        
except Exception as e:
    print(f"Errore durante la valutazione del modello baseline: {e}. Annullamento promozione per sicurezza.")
    promote_to_production = False

# Assegnazione dei tag al nuovo modello
nuovi_tags = ["candidato"]
if promote_to_production:
    nuovi_tags.append("production")
    print("Nuovo modello taggato con successo come 'production'.")

output_model.tags = nuovi_tags

print("\nPipeline di Paolo completata con successo!")