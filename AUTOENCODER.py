# ==============================================================================
# BLOQUE ATÓMICO: AUTOENCODER (VERSIÓN BLINDADA ANTI-NaN)
# ==============================================================================
!pip install tensorflow pandas numpy scikit-learn joblib -q

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef, confusion_matrix, precision_recall_curve, auc
import joblib
import json
import gc
from google.colab import drive

# 1. CONEXIÓN Y CARGA
drive.mount('/content/drive', force_remount=True)
print("🚀 1. Cargando datos...")
ruta_datos = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/dataset_TodaBanca_CON_N.csv'

df = pd.read_csv(ruta_datos, engine='pyarrow')
col_objetivo = 'FRAUDE_IA'
Y = df.pop(col_objetivo).values.astype(np.int8)

# Solo números para la Red Neuronal
X_df = df.select_dtypes(include=[np.number])
X = X_df.values.astype(np.float32)
nombres_columnas = X_df.columns.tolist()
del df
gc.collect()

# 2. PARTICIÓN Y ESCALADO BLINDADO
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
del X, Y
gc.collect()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 🔥 CAPA DE PROTECCIÓN: ELIMINACIÓN DE CORTOCIRCUITOS (NaN/Inf) ---
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

X_train_normal = X_train_scaled[y_train == 0]
print(f"   ✅ Datos desinfectados. Entrenando con {X_train_normal.shape[0]} registros sanos.")

# 3. ENTRENAMIENTO DEL AUTOENCODER
print("\n🧠 2. Entrenando Autoencoder...")
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamos con una muestra del 20% de los normales para evitar que la RAM colapse en el entrenamiento
autoencoder.fit(X_train_normal, X_train_normal, 
                epochs=15, 
                batch_size=1024, 
                validation_split=0.1, 
                callbacks=[early_stopping], 
                verbose=1)

# 4. CALIBRACIÓN DEL UMBRAL (99%)
print("\n⚖️ 3. Calibrando umbral...")
recon_train = autoencoder.predict(X_train_normal[:30000], verbose=0)
mse_train = np.mean(np.power(X_train_normal[:30000] - recon_train, 2), axis=1)
umbral_ae = float(np.percentile(mse_train, 99))

# 5. EVALUACIÓN FINAL
print("📊 4. Evaluando desempeño real...")
recon_test = autoencoder.predict(X_test_scaled, verbose=0)
mse_test = np.mean(np.power(X_test_scaled - recon_test, 2), axis=1)
# ¡OJO! Limpieza final de la salida para las métricas de sklearn
mse_test = np.nan_to_num(mse_test)
y_pred = (mse_test > umbral_ae).astype(int)

# --- MÉTRICAS ---
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision, recall_curve, _ = precision_recall_curve(y_test, mse_test)
auprc = auc(recall_curve, precision)

print("\n" + "="*50)
print("🏆 REPORTE FINAL: AUTOENCODER")
print("="*50)
print(f"🔸 AUPRC: {auprc:.4f}")
print(f"🔸 Recall: {rec:.2%}")
print(f"🔸 F1-Score: {f1:.4f} | MCC: {mcc:.4f}")
print("\n🚨 MATRIZ:")
print(f"   ✔️ Sanos (TN): {cm[0][0]} | ⚠️ Falsos Positivos (FP): {cm[0][1]}")
print(f"   ❌ Fugas (FN): {cm[1][0]} | 🎯 Atrapados (TP): {cm[1][1]}")
print("="*50)

# 6. GUARDAR
ruta_base = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/'
print("\n💾 5. Guardando archivos para la App...")
autoencoder.save(ruta_base + 'modelo_autoencoder_final.keras')
joblib.dump(scaler, ruta_base + 'scaler_autoencoder.pkl')
config_ae = {"umbral_anomalia": umbral_ae, "nombres_columnas": nombres_columnas}
with open(ruta_base + 'config_autoencoder.json', 'w') as f:
    json.dump(config_ae, f)
print("🚀 ¡PROCESO COMPLETADO!")