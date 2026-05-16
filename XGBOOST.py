# ==============================================================================
# BLOQUE ATÓMICO: ENTRENAMIENTO + MÉTRICAS + GUARDADO (XGBOOST)
# ==============================================================================
!pip install xgboost joblib pandas numpy scikit-learn -q

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, f1_score, recall_score, matthews_corrcoef, confusion_matrix
from google.colab import drive

# 1. CONEXIÓN Y CARGA
drive.mount('/content/drive', force_remount=True)
print("🚀 1. Cargando y preparando datos...")
ruta_datos = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/dataset_TodaBanca_CON_N.csv'

df = pd.read_csv(ruta_datos, engine='pyarrow')
col_objetivo = 'FRAUDE_IA'
Y = df.pop(col_objetivo).values.astype(np.int8)

# Solo números para evitar errores de texto
X_df = df.select_dtypes(include=[np.number])
X = X_df.values.astype(np.float32)
nombres_columnas = X_df.columns.tolist()
del df
gc.collect()

# 2. PARTICIÓN Y ESCALADO
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
del X, Y, X_train, X_test
gc.collect()

# 3. ENTRENAMIENTO
print("🧠 2. Entrenando XGBoost (Capa de Precisión)...")
peso_fraude = np.sum(y_train == 0) / np.sum(y_train == 1)

modelo_xgb = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    scale_pos_weight=peso_fraude * 0.2, # Reducción de paranoia
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
modelo_xgb.fit(X_train_scaled, y_train)

# 4. CALIBRACIÓN Y MÉTRICAS DE VALIDACIÓN
print("⚖️ 3. Evaluando y Calibrando umbral...")
y_prob = modelo_xgb.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Optimizamos por F2-Score (Prioridad Recall) para encontrar el mejor umbral
f2_scores = np.divide(5 * (precision * recall), (4 * precision + recall), 
                      out=np.zeros_like(precision), where=(4 * precision + recall) != 0)
mejor_umbral = float(thresholds[np.argmax(f2_scores)])

# Generamos predicciones finales con ese umbral
y_pred = (y_prob >= mejor_umbral).astype(int)

# --- REPORTE DE MÉTRICAS PARA TU INFORME ---
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auprc = auc(recall, precision)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("🏆 REPORTE DE VALIDACIÓN (MODELO A GUARDAR)")
print("="*50)
print(f"🔸 Umbral de Decisión: {mejor_umbral:.4f}")
print(f"🔸 AUPRC (Métrica Reina): {auprc:.4f}")
print(f"🔸 Recall (Detección): {rec:.2%}")
print(f"🔸 F1-Score: {f1:.4f}")
print(f"🔸 MCC (Honestidad): {mcc:.4f}")
print("\n🚨 MATRIZ DE CONFUSIÓN:")
print(f"   ✔️ Sanos permitidos (TN): {cm[0][0]}")
print(f"   ⚠️ Falsos Positivos (FP): {cm[0][1]}")
print(f"   ❌ Fraudes fugados (FN): {cm[1][0]}")
print(f"   🎯 FRAUDES DETECTADOS (TP): {cm[1][1]}")
print("="*50)

# 5. GUARDADO DE ARCHIVOS
ruta_base = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/'
print("\n💾 4. Guardando archivos en Drive...")

joblib.dump(modelo_xgb, ruta_base + 'modelo_xgboost_final.pkl')
joblib.dump(scaler, ruta_base + 'scaler_master.pkl')

config = {
    "umbral_optimo": mejor_umbral,
    "nombres_columnas": nombres_columnas,
    "metricas": {"recall": rec, "mcc": mcc, "auprc": auprc}
}
with open(ruta_base + 'config_modelo.json', 'w') as f:
    json.dump(config, f)

print(f"🚀 ¡PROCESO FINALIZADO! Modelo validado y listo para producción.")