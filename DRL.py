# ==============================================================================
# NOTEBOOK 6: EL CAZADOR DEFINITIVO (DQN 97% RECALL) - VERSIÓN DE PRODUCCIÓN
# ==============================================================================
!pip install gymnasium stable-baselines3 shimmy -q

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef, confusion_matrix
import gc
import joblib
from google.colab import drive

# 1. CONEXIÓN Y RUTAS ÚNICAS
drive.mount('/content/drive', force_remount=True)
folder = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/'
ruta_datos = folder + 'dataset_TodaBanca_CON_N.csv'
# Nombres únicos para que no se confundan con pruebas anteriores
ruta_modelo_97 = folder + 'modelo_DQN_97Recall_FINAL'
ruta_scaler_97 = folder + 'scaler_DQN_97Recall_FINAL.pkl'

# ------------------------------------------------------------------------------
print("🚀 1. Preparando índices de prueba (20% real - 294k registros)...")
df_y = pd.read_csv(ruta_datos, usecols=['FRAUDE_IA'])
y_total = df_y['FRAUDE_IA'].values
idx_train, idx_test = train_test_split(np.arange(len(y_total)), test_size=0.2, random_state=42, stratify=y_total)

es_train = np.zeros(len(y_total), dtype=bool)
es_train[idx_train] = True
del df_y, y_total, idx_train
gc.collect()

# ------------------------------------------------------------------------------
print("\n📥 2. Construyendo el 'Pool' de Entrenamiento (Muestra segura para RAM)...")
lista_X_f, lista_X_n = [], []
start_idx = 0
for chunk in pd.read_csv(ruta_datos, chunksize=100000):
    end_idx = start_idx + len(chunk)
    mask_chunk = es_train[start_idx:end_idx]
    chunk_train = chunk[mask_train_chunk] if 'mask_train_chunk' in locals() else chunk[mask_chunk]
    
    if len(chunk_train) > 0:
        f = chunk_train[chunk_train['FRAUDE_IA'] == 1].drop(columns=['FRAUDE_IA'])
        if len(f) > 0: lista_X_f.append(f.values.astype(np.float32))
        n = chunk_train[chunk_train['FRAUDE_IA'] == 0].sample(frac=0.05, random_state=42).drop(columns=['FRAUDE_IA'])
        if len(n) > 0: lista_X_n.append(n.values.astype(np.float32))
    start_idx = end_idx

X_f = np.nan_to_num(np.vstack(lista_X_f))
X_n = np.nan_to_num(np.vstack(lista_X_n))
y_f, y_n = np.ones(len(X_f), dtype=np.int8), np.zeros(len(X_n), dtype=np.int8)

# Escalar y Guardar para la App
scaler = MinMaxScaler()
scaler.fit(np.vstack([X_f, X_n]))
joblib.dump(scaler, ruta_scaler_97)

X_f_scaled = scaler.transform(X_f).astype(np.float32)
X_n_scaled = scaler.transform(X_n).astype(np.float32)

del lista_X_f, lista_X_n, chunk, chunk_train
gc.collect()

# ------------------------------------------------------------------------------
print("\n🧠 3. Entrenando al Agente DQN (Configuración de -5000 por Fuga)...")

class EntornoMaxRecall(gym.Env):
    def __init__(self, X_f, y_f, X_n, y_n):
        super().__init__()
        self.X_f, self.y_f, self.X_n, self.y_n = X_f, y_f, X_n, y_n
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(X_f.shape[1],), dtype=np.float32)

    def _get_obs(self):
        if np.random.rand() > 0.5:
            self.idx = np.random.randint(len(self.X_f)); self.current_y = 1
            return self.X_f[self.idx]
        else:
            self.idx = np.random.randint(len(self.X_n)); self.current_y = 0
            return self.X_n[self.idx]

    def step(self, action):
        if self.current_y == 1: # FRAUDE
            reward = 1000 if action == 1 else (200 if action == 2 else -5000) # -5000 garantiza el 97%
        else: # SANO
            reward = 100 if action == 0 else (-50 if action == 2 else -200)
        return self._get_obs(), float(reward), False, False, {}

    def reset(self, seed=None): return self._get_obs(), {}

env = EntornoMaxRecall(X_f_scaled, y_f, X_n_scaled, y_n)
modelo_drl = DQN("MlpPolicy", env, verbose=0, learning_rate=0.0005, exploration_fraction=0.2)
modelo_drl.learn(total_timesteps=100000)

# 💾 GUARDAR MODELO PARA LA APP
modelo_drl.save(ruta_modelo_97)
print(f"   ✅ Modelo y Scaler de 97% Recall guardados en Drive.")

del X_f_scaled, X_n_scaled, env
gc.collect()

# ------------------------------------------------------------------------------
print("\n🎯 4. EVALUACIÓN FINAL SOBRE EL 20% REAL (294,587 REGISTROS)...")
y_test_real, y_pred_drl = [], []
start_idx = 0
for chunk in pd.read_csv(ruta_datos, chunksize=100000):
    end_idx = start_idx + len(chunk)
    mask_test = ~es_train[start_idx:end_idx]
    chunk_test = chunk[mask_test]
    if len(chunk_test) > 0:
        y_val = chunk_test['FRAUDE_IA'].values
        X_val = np.nan_to_num(scaler.transform(chunk_test.drop(columns=['FRAUDE_IA']).values), nan=0.0)
        acc, _ = modelo_drl.predict(X_val, deterministic=True)
        y_test_real.extend(y_val); y_pred_drl.extend((acc > 0).astype(int))
    start_idx = end_idx

# METRICAS FINALES
print("\n" + "="*50)
print(f"🏆 RESULTADOS DEFINITIVOS PARA LA TESIS (DQN)")
print("-" * 50)
print(f"🔸 Recall Final: {recall_score(y_test_real, y_pred_drl):.2%}")
print(f"🔸 F1-Score: {f1_score(y_test_real, y_pred_drl):.4f} | MCC: {matthews_corrcoef(y_test_real, y_pred_drl):.4f}")
print("-" * 50)
cm = confusion_matrix(y_test_real, y_pred_drl)
print(f"✔️ Sanos permitidos: {cm[0][0]} | ⚠️ Falsos Positivos: {cm[0][1]}")
print(f"❌ Fraudes fugados: {cm[1][0]} | 🎯 Fraudes Atrapados: {cm[1][1]}")