# ==============================================================================
# MOTOR DE INFERENCIA: TORNEO DE MODELOS (Veredictos Individuales)
# ==============================================================================
!pip install xgboost stable-baselines3 gymnasium shimmy tensorflow scikit-learn joblib -q

import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from IPython.display import display, HTML
from google.colab import drive

# 1. CONEXIÓN Y CARGA DE MODELOS
# ------------------------------------------------------------------------------
drive.mount('/content/drive', force_remount=True)
ruta_base = '/content/drive/MyDrive/Colab Notebooks/MAESTRIA IA/PROYECTO - DESARROLLO DE SOLUCIONES/'

print("🚀 Cargando Agentes expertos...")

# Agente 1: XGBoost (Patrones conocidos)
modelo_xgb = joblib.load(ruta_base + 'modelo_xgboost_final.pkl')
scaler_xgb = joblib.load(ruta_base + 'scaler_master.pkl')

# Agente 2: Autoencoder (Detección de lo raro/anomalía)
modelo_ae = load_model(ruta_base + 'modelo_autoencoder_final.keras')
scaler_ae = joblib.load(ruta_base + 'scaler_autoencoder.pkl')
with open(ruta_base + 'config_autoencoder.json') as f:
    umbral_ae = json.load(f)['umbral_anomalia']

# Agente 3: DRL DQN (Supervisor de políticas)
modelo_drl = DQN.load(ruta_base + 'modelo_DQN_97Recall_FINAL')
scaler_drl = joblib.load(ruta_base + 'scaler_DQN_97Recall_FINAL.pkl')

print("✅ Todos los agentes están en sus puestos. Listos para el torneo.")

# ------------------------------------------------------------------------------
# 2. MOTOR DEL TORNEO (INFERENCIA INDIVIDUAL)
# ------------------------------------------------------------------------------
def ejecutar_torneo(datos_crudos_fila, id_transaccion="TRX-NEW-001"):
    # Preparamos los datos para cada modelo (escalado individual)
    X_xgb = np.nan_to_num(scaler_xgb.transform(datos_crudos_fila).astype(np.float32))
    X_ae  = np.nan_to_num(scaler_ae.transform(datos_crudos_fila).astype(np.float32))
    X_drl = np.nan_to_num(scaler_drl.transform(datos_crudos_fila).astype(np.float32))

    # --- VERDICTO AGENTE 1: XGBOOST ---
    prob_f = modelo_xgb.predict_proba(X_xgb)[0][1]
    v_xgb = "🔴 FRAUDE" if prob_f > 0.50 else "🟢 SANO"
    p_xgb = f"{prob_f*100:.1f}%"

    # --- VERDICTO AGENTE 2: AUTOENCODER ---
    recon = modelo_ae.predict(X_ae, verbose=0)
    mse = np.mean(np.power(X_ae - recon, 2))
    v_ae = "🔴 ANÓMALO" if mse > umbral_ae else "🟢 HABITUAL"
    p_ae = f"Error: {mse:.5f}"

    # --- VERDICTO AGENTE 3: DRL DQN ---
    accion, _ = modelo_drl.predict(X_drl, deterministic=True)
    accion_num = accion[0] if isinstance(accion, np.ndarray) else accion
    map_acc = {0: "🟢 APROBAR", 1: "🔴 BLOQUEAR", 2: "🟡 PEDIR OTP"}
    v_drl = map_acc[accion_num]

    # RENDERIZACIÓN DEL TORNEO
    html = f"""
    <div style="font-family: Arial; border: 2px solid #333; padding: 20px; border-radius: 10px; width: 500px; background: #f4f4f4;">
        <h2 style="text-align: center; margin-top: 0; color: #1a237e;">🏆 TORNEO DE MODELOS (Votación Individual)</h2>
        <p style="text-align: center; color: #666;">ID Transacción: <b>{id_transaccion}</b></p>
        <hr>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #e0e0e0;">
                <th style="padding: 10px; text-align: left;">Agente de IA</th>
                <th style="padding: 10px; text-align: center;">Veredicto</th>
                <th style="padding: 10px; text-align: right;">Confianza/Métrica</th>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ccc;"><b>XGBoost</b><br><small>Patrones Históricos</small></td>
                <td style="padding: 10px; border-bottom: 1px solid #ccc; text-align: center;">{v_xgb}</td>
                <td style="padding: 10px; border-bottom: 1px solid #ccc; text-align: right;">{p_xgb}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ccc;"><b>Autoencoder</b><br><small>Detección Anomalías</small></td>
                <td style="padding: 10px; border-bottom: 1px solid #ccc; text-align: center;">{v_ae}</td>
                <td style="padding: 10px; border-bottom: 1px solid #ccc; text-align: right;">{p_ae}</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><b>DRL (PPO)</b><br><small>Políticas de Negocio</small></td>
                <td style="padding: 10px; text-align: center;">{v_drl}</td>
                <td style="padding: 10px; text-align: right;">Acción: {accion_num}</td>
            </tr>
        </table>
    </div>
    """
    display(HTML(html))

# ------------------------------------------------------------------------------
# 3. PRUEBA CON UN DATO NUEVO (EJEMPLO)
# ------------------------------------------------------------------------------
# Para la demo, cargamos el archivo de prueba y tomamos una fila al azar
# (En la aplicación web, esto vendría del formulario del usuario)
df_prueba = pd.read_csv(ruta_base + 'dataset_TodaBanca_CON_N.csv', nrows=100)
transaccion_nueva = df_prueba.drop(columns=['FRAUDE_IA']).iloc[[5]] # Tomamos la fila 5 como ejemplo

print("\n🚀 PROCESANDO NUEVA TRANSACCIÓN...")
ejecutar_torneo(transaccion_nueva.values, id_transaccion="TRX-DEMO-2026")