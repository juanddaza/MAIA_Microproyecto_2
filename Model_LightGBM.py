# ==============================================================================
# MODELO RETADOR: LightGBM - Detección de Fraude Transaccional
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

print("🚀 Iniciando pipeline de LightGBM para Detección de Fraude...")

# ---------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE COLUMNAS
# ---------------------------------------------------------
file_path = 'universidad.xlsx'
df = pd.read_excel(file_path)
print(f"✅ Archivo '{file_path}' cargado exitosamente. Total registros: {len(df)}")

def limpiar_columnas(col):
    col = str(col).replace('ÃƒÂ­', 'i').replace('ÃƒÂ³', 'o').replace('ÃƒÂ±', 'n')
    col = col.replace('ÃƒÂº', 'u').replace('ÃƒÂ©', 'e').replace('ÃƒÂ', 'a')
    col = col.replace('"', '').replace(' ', '_').replace('-', '').replace('__', '_')
    return col.strip().lower()

df.columns = [limpiar_columnas(c) for c in df.columns]

# ---------------------------------------------------------
# 2. HUELLA DIGITAL (RESCATE DE NULOS) Y FECHAS
# ---------------------------------------------------------
# Fecha y Hora unificada
df['fecha_hora'] = pd.to_datetime(df['fecha_inicial'].astype(str) + ' ' + df['hora_inicial'].astype(str), errors='coerce')

# Rescatamos registros sin cliente creando un ID Sintético basado en IP + OS
df['huella_digital'] = df['afd__direccion_ip'].astype(str) + '_' + df['rt_afd_sistema_operativo'].astype(str)
df['id_analisis'] = np.where(
    df['afd__id_cliente/rfc'].isna() | (df['afd__id_cliente/rfc'].astype(str).str.strip() == ''),
    df['huella_digital'],
    df['afd__id_cliente/rfc']
)

# ---------------------------------------------------------
# 3. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ---------------------------------------------------------
print("Calculando variables de velocidad, ratios y contexto...")

# Orden vital para cálculos temporales
df = df.sort_values(by=['id_analisis', 'fecha_hora'])

# A. VELOCITY RULES (Usando id_analisis para incluir a todos)
df_temp = df.set_index('fecha_hora')
grouper = df_temp.groupby('id_analisis')

# Contamos transacciones en ventanas de tiempo
df['trx_ult_1h'] = grouper['afd__código_de_transacción'].rolling('1h').count().values
df['trx_ult_6h'] = grouper['afd__código_de_transacción'].rolling('6h').count().values
df['trx_ult_24h'] = grouper['afd__código_de_transacción'].rolling('24h').count().values

# Intentos fallidos
df_temp['es_fallida'] = np.where(df_temp['mfi_resultado_transacción'] != 1, 1, 0)
df['intentos_fallidos_ult_1h'] = df_temp.groupby('id_analisis')['es_fallida'].rolling('1h').sum().values

# B. RATIOS MONETARIOS
df['monto_actual'] = pd.to_numeric(df['afd__monto_total_trx'], errors='coerce').fillna(0)
df['monto_promedio_6m'] = pd.to_numeric(df['mfi_promedio_ultimos_6_meses_producto_origen'], errors='coerce').fillna(0)
df['ratio_monto_vs_promedio'] = np.where(df['monto_promedio_6m'] > 0, df['monto_actual'] / df['monto_promedio_6m'], 0)
df['alerta_monto_excesivo'] = (df['ratio_monto_vs_promedio'] > 10).astype(int)

# C. COMPORTAMIENTO DE DISPOSITIVO
df['ip_actual'] = df['afd__direccion_ip'].astype(str).str.strip()
df['ip_inscrita'] = df['afd__direccion_ip_inscrita'].astype(str).str.strip()
df['es_ip_nueva'] = np.where(df['ip_actual'] != df['ip_inscrita'], 1, 0)

cols_cambio = ['rt_afd__isp_internet_service_provider', 'rt_afd_sistema_operativo', 'afd_codigo_de_browser']
for col in cols_cambio:
    valor_anterior = df.groupby('id_analisis')[col].shift(1)
    df[f'cambio_subito_{col}'] = (df[col].notna() & valor_anterior.notna() & (df[col] != valor_anterior)).astype(int)

# D. HORARIOS Y CALENDARIO
df['hora_num'] = df['fecha_hora'].dt.hour
bins = [-1, 5, 11, 17, 24]
labels = ['Madrugada', 'Manana', 'Tarde', 'Noche']
df['franja_horaria'] = pd.cut(df['hora_num'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['franja_horaria'], prefix='franja', dtype=int)
df['es_fin_de_semana'] = (df['fecha_hora'].dt.dayofweek >= 5).astype(int)

# ---------------------------------------------------------
# 4. PREPARACIÓN PARA MODELADO (SPLIT TEMPORAL)
# ---------------------------------------------------------
print("📅 Realizando partición temporal de los datos...")

# Consolidamos variables
features_creadas = [
    'trx_ult_1h', 'trx_ult_6h', 'trx_ult_24h', 'intentos_fallidos_ult_1h',
    'ratio_monto_vs_promedio', 'alerta_monto_excesivo', 'es_ip_nueva',
    'cambio_subito_rt_afd__isp_internet_service_provider',
    'cambio_subito_rt_afd_sistema_operativo', 'cambio_subito_afd_codigo_de_browser',
    'es_fin_de_semana'
]
franjas = [col for col in df.columns if col.startswith('franja_')]
features_originales = ['afd__monto_total_trx', 'mfi_promedio_ultimos_6_meses_producto_origen', 'afd_score_dinamico']

X_cols = features_creadas + franjas + features_originales
y_col = 'fraude'

# Ordenamos por fecha general antes de cortar
df = df.sort_values('fecha_hora')

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[X_cols].astype(float)
y_train = train_df[y_col].astype(int)
X_test = test_df[X_cols].astype(float)
y_test = test_df[y_col].astype(int)

# ---------------------------------------------------------
# 5. ENTRENAMIENTO MODELO LIGHTGBM
# ---------------------------------------------------------
print("⚡ Entrenando el algoritmo LightGBM...")

conteo_clases = y_train.value_counts()
ratio_desbalance = conteo_clases[0] / max(conteo_clases[1], 1)

# Configuración optimizada contra overfitting
modelo_lgb = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,                 
    min_child_samples=20,          
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio_desbalance,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

modelo_lgb.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. EVALUACIÓN Y VALIDACIÓN DE OVERFITTING
# ---------------------------------------------------------
print("📊 Evaluando resultados...")

# Predicciones Test
y_pred_proba_test = modelo_lgb.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

# Predicciones Train (para comparar)
y_pred_proba_train = modelo_lgb.predict_proba(X_train)[:, 1]
y_pred_train = (y_pred_proba_train >= 0.5).astype(int)

# Métricas
precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)
pr_auc_test = auc(recall_test, precision_test)

precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_proba_train)
pr_auc_train = auc(recall_train, precision_train)

print("\n=== RENDIMIENTO EN ENTRENAMIENTO (TRAIN) ===")
print(classification_report(y_train, y_pred_train))

print("\n=== RENDIMIENTO EN PRUEBA (TEST - Out of Time) ===")
print(classification_report(y_test, y_pred_test))

print("\n--- MATRIZ DE CONFUSIÓN (TEST) ---")
print(confusion_matrix(y_test, y_pred_test))

print(f"\n🏆 PR-AUC Train: {pr_auc_train:.4f} vs PR-AUC Test: {pr_auc_test:.4f}")
print("✅ Fin del pipeline LightGBM.")