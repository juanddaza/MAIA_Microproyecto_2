import pandas as pd
import os
file_path = 'universidad.xlsx'
df = pd.read_excel(file_path)
print(f"✅ Archivo '{file_path}' cargado exitosamente.")

def limpiar_columnas(col):
  # Convertimos a string por si hay nombres numéricos
  col = str(col)
  # Reemplaza errores de encoding y caracteres especiales
  col = col.replace('ÃƒÂ­', 'i').replace('ÃƒÂ³', 'o').replace('ÃƒÂ±', 'n')
  col = col.replace('ÃƒÂº', 'u').replace('ÃƒÂ©', 'e').replace('ÃƒÂ', 'a')
  # Limpieza de formato: minúsculas, sin espacios ni guiones
  col = col.replace('"', '').replace(' ', '_').replace('-', '').replace('__', '_')
  return col.strip().lower()

df.columns = [limpiar_columnas(c) for c in df.columns]
# 3. Mostrar variables encontradas
print("\nVariables en el dataset:")
print(df.columns.tolist())

# 1. Diccionario de mapeo
mapeo_transacciones = {
    'ACH': {'canal_nombre': 'BV', 'tipo_trx': 'Administrativas'},
    'ATQD': {'canal_nombre': 'BV-BM', 'tipo_trx': 'Monetarias'},
    'ATTD': {'canal_nombre': 'BM', 'tipo_trx': 'Monetarias'},
    'ATWI': {'canal_nombre': 'BM', 'tipo_trx': 'Monetarias'},
    'AVAL': {'canal_nombre': 'BV', 'tipo_trx': 'Administrativas'},
    'AVAV': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVIT': {'canal_nombre': 'BV-BM', 'tipo_trx': 'Administrativas'},
    'AVOA': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVOB': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVPE': {'canal_nombre': 'PSE', 'tipo_trx': 'Monetarias'},
    'AVPF': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVPT': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVTL': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVTN': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVTO': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVTR': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'AVUA': {'canal_nombre': 'BV', 'tipo_trx': 'Monetarias'},
    'BDB': {'canal_nombre': 'BV', 'tipo_trx': 'Administrativas'},
    'FDA-AUBV': {'canal_nombre': 'BV', 'tipo_trx': 'Administrativas'},
    'FDA-AUTH': {'canal_nombre': 'BV', 'tipo_trx': 'Administrativas'},
    'MOTD': {'canal_nombre': 'BV-BM', 'tipo_trx': 'Administrativas'},
    'BMRS': {'canal_nombre': 'DIGITAL', 'tipo_trx': 'Administrativas'},
    'FDA-RXP': {'canal_nombre': 'DIGITAL', 'tipo_trx': 'Administrativas'},
    'CRLD': {'canal_nombre': 'DIGITAL', 'tipo_trx': 'Administrativas'},
    'CRN1': {'canal_nombre': 'DIGITAL', 'tipo_trx': 'Administrativas'}
}
def obtener_canal(cod):
    # Convertimos a string, quitamos espacios y pasamos a MAYÚSCULAS para asegurar la coincidencia
    codigo_normalizado = str(cod).strip().upper()
    return mapeo_transacciones.get(codigo_normalizado, {}).get('canal_nombre', 'OTRO')
def obtener_tipo(cod):
    # Convertimos a string, quitamos espacios y pasamos a MAYÚSCULAS
    codigo_normalizado = str(cod).strip().upper()
    return mapeo_transacciones.get(codigo_normalizado, {}).get('tipo_trx', 'OTRO')

# Nota: Asegúrate de que 'afd__código_de_transacción' sea el nombre exacto de la columna
df['canal_agrupado'] = df['afd__código_de_transacción'].apply(obtener_canal)
df['tipo_trx_agrupado'] = df['afd__código_de_transacción'].apply(obtener_tipo)

# 4. Filtrar solo por canal BV o BV-BM
#df_BV = df[df['canal_agrupado'].isin(['BV', 'BV-BM'])].copy()

# 5. Verificación y resultados
#print(f"Total de registros originales: {len(df)}")
#print(f"Registros filtrados (Canal BV o BV-BM): {len(df_BV)}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.countplot(x='fraude', data=df, palette='viridis')
plt.title('Distribución de la variable FRAUDE')
plt.xlabel('Fraude')
plt.ylabel('Cantidad de Transacciones')
plt.show()

# 1. Crear una columna de Fecha y Hora unificada
# Nota: Ajusta el formato (format=...) si tus datos tienen un formato de fecha particular (ej. '%Y-%m-%d %H:%M:%S')
df['fecha_hora'] = pd.to_datetime(df['fecha_inicial'].astype(str) + ' ' + df['hora_inicial'].astype(str), errors='coerce')

# 2. Filtrar el DataFrame descartando nulos y vacíos en el ID del cliente
df_validos = df.dropna(subset=['afd__id_cliente/rfc']).copy()
df_validos = df_validos[df_validos['afd__id_cliente/rfc'].astype(str).str.strip() != '']

# 3. Ordenar cronológicamente por cliente para asegurar que el orden de los eventos sea correcto
df_validos = df_validos.sort_values(by=['afd__id_cliente/rfc', 'fecha_hora'])

# 4. Identificar Logins Exitosos según tus reglas
cond_login = (
    (df_validos['afd__código_de_transacción'] == 'FDA-AUBV') &
    (df_validos['mfi_resultado_transacción'] == 1) &
    (df_validos['mfi_descripción/razón_respuesta_trx'].isin(['Cliente sin token.', 'RSA ingreso sin token.', 'Token valido.']))
)

# Marcar el tiempo exacto del login
df_validos['tiempo_ultimo_login'] = pd.NaT
df_validos.loc[cond_login, 'tiempo_ultimo_login'] = df_validos.loc[cond_login, 'fecha_hora']

# Propagar (forward fill) el tiempo del último login hacia adelante dentro de las transacciones de cada cliente
df_validos['tiempo_ultimo_login'] = df_validos.groupby('afd__id_cliente/rfc')['tiempo_ultimo_login'].ffill()

# Calcular el tiempo transcurrido en segundos desde el último login
df_validos['tiempo_desde_login_segundos'] = (df_validos['fecha_hora'] - df_validos['tiempo_ultimo_login']).dt.total_seconds()

# Evitar tiempos negativos (por errores de cruce de sesiones o datos mal registrados)
df_validos = df_validos[df_validos['tiempo_desde_login_segundos'] >= 0]

# 5. Filtrar el resto de transacciones válidas (donde el resultado es 0)
cond_resto_trx = (
    (df_validos['afd__código_de_transacción'] != 'FDA-AUBV') &
    (df_validos['mfi_resultado_transacción'] == 0)
)

# Quedarnos solo con transacciones que tuvieron un login previo en la misma sesión/historial
df_otras_trx = df_validos[cond_resto_trx & df_validos['tiempo_desde_login_segundos'].notnull()]

# Calcular el tiempo promedio por cliente y por tipo de transacción agrupado (Administrativas vs Monetarias)
tiempo_promedio = df_otras_trx.groupby(['afd__id_cliente/rfc', 'tipo_trx_agrupado'])['tiempo_desde_login_segundos'].mean().reset_index()

# Pivotear la tabla para tener las columnas: ID, Promedio_Administrativas, Promedio_Monetarias
df_features_tiempo = tiempo_promedio.pivot(index='afd__id_cliente/rfc', columns='tipo_trx_agrupado', values='tiempo_desde_login_segundos').reset_index()
# Limpiar nombres de columnas
df_features_tiempo.columns.name = None
df_features_tiempo.rename(columns={
    'Administrativas': 'prom_segundos_login_a_administrativa',
    'Monetarias': 'prom_segundos_login_a_monetaria'
}, inplace=True)

# 7. Unir estas nuevas variables al dataframe original
df = df.merge(df_features_tiempo, on='afd__id_cliente/rfc', how='left')

print("✅ Variables de tiempo promedio desde login creadas exitosamente.")

# Asegurémonos de que la fecha sea el índice temporal para cálculos de velocidad,
# pero guardemos el índice original si es necesario.
df = df.sort_values(by=['afd__id_cliente/rfc', 'fecha_hora'])

# ---------------------------------------------------------
# 1. VELOCITY RULES (Velocidad de Transacciones)
# ---------------------------------------------------------
import numpy as np
print("Calculando variables de velocidad...")

# 1. Guardamos el índice original para poder mapear los datos de vuelta a las 180k filas
df['idx_original'] = df.index

# 2. Filtramos el subconjunto de los ~50k registros que sí tienen un ID válido
mascara_validos = df['afd__id_cliente/rfc'].notna() & (df['afd__id_cliente/rfc'].astype(str).str.strip() != '')
df_validos = df[mascara_validos].copy()

# 3. Ordenamos por cliente y luego cronológicamente (vital para calcular el tiempo)
df_validos = df_validos.sort_values(by=['afd__id_cliente/rfc', 'fecha_hora'])

# 4. Ponemos la fecha_hora como índice temporal para que el comando 'rolling' funcione
df_validos = df_validos.set_index('fecha_hora')

# 5. Agrupamos y calculamos las ventanas de tiempo
grouper = df_validos.groupby('afd__id_cliente/rfc')['afd__código_de_transacción']

# Como lo aplicamos sobre df_validos, ambos tienen 51,308 filas, por lo que '.values' ahora funciona perfecto
df_validos['trx_ult_1h'] = grouper.rolling('1h').count().values
df_validos['trx_ult_6h'] = grouper.rolling('6h').count().values
df_validos['trx_ult_24h'] = grouper.rolling('24h').count().values

# --- Cantidad de intentos fallidos recientes ---
# Asumimos que un resultado != 1 es un fallo (ajusta esto si el código de éxito de tu banco es diferente)
df_validos['es_fallida'] = np.where(df_validos['mfi_resultado_transacción'] != 1, 1, 0)
df_validos['intentos_fallidos_ult_1h'] = df_validos.groupby('afd__id_cliente/rfc')['es_fallida'].rolling('1h').sum().values

# 6. Mapeamos los resultados calculados de vuelta al DataFrame original de 180k registros
# Usamos el índice original que guardamos en el paso 1 como llave
df_validos = df_validos.set_index('idx_original')

df['trx_ult_1h'] = df['idx_original'].map(df_validos['trx_ult_1h'])
df['trx_ult_6h'] = df['idx_original'].map(df_validos['trx_ult_6h'])
df['trx_ult_24h'] = df['idx_original'].map(df_validos['trx_ult_24h'])
df['intentos_fallidos_ult_1h'] = df['idx_original'].map(df_validos['intentos_fallidos_ult_1h'])

# Limpiamos las columnas auxiliares que ya no necesitamos
df.drop(columns=['idx_original'], inplace=True)

print("✅ Variables de velocidad calculadas correctamente.")

# ---------------------------------------------------------
# 1. VARIABLES MONETARIAS (Ratios)
# ---------------------------------------------------------
print("1. Calculando ratios monetarios...")

# Convertir columnas a numéricas, forzando errores a NaN y luego llenando con 0
df['monto_actual'] = pd.to_numeric(df['afd__monto_total_trx'], errors='coerce').fillna(0)
df['monto_promedio_6m'] = pd.to_numeric(df['mfi_promedio_ultimos_6_meses_producto_origen'], errors='coerce').fillna(0)

# Calcular el ratio de forma segura (evitando división por cero)
df['ratio_monto_vs_promedio'] = np.where(
    df['monto_promedio_6m'] > 0,
    df['monto_actual'] / df['monto_promedio_6m'],
    0  # Si el promedio es 0, dejamos el ratio en 0
)

# Flag de alerta fuerte: 1 si la transacción es > 10 veces su promedio
df['alerta_monto_excesivo'] = (df['ratio_monto_vs_promedio'] > 10).astype(int)

# ---------------------------------------------------------
# 2. COMPORTAMIENTO DE DISPOSITIVO / CONEXIÓN
# ---------------------------------------------------------
print("2. Analizando dispositivo y conexión...")

# --- Device Hopping (IP Nueva o No Registrada) ---
# Limpiamos espacios y convertimos a string para comparar con seguridad
ip_actual = df['afd__direccion_ip'].astype(str).str.strip()
ip_inscrita = df['afd__direccion_ip_inscrita'].astype(str).str.strip()

# Si son diferentes, marcamos 1 (es nueva/sospechosa), sino 0
df['es_ip_nueva'] = np.where(ip_actual != ip_inscrita, 1, 0)

# --- Cambios Súbitos (ISP, OS, Browser) ---
# Analizamos columnas específicas buscando saltos entre la transacción anterior y la actual
cols_cambio = [
    'rt_afd__isp_internet_service_provider',
    'rt_afd_sistema_operativo',
    'afd_codigo_de_browser'
]

for col in cols_cambio:
    # Obtenemos el valor de la transacción inmediatamente anterior para el mismo cliente
    valor_anterior = df.groupby('afd__id_cliente/rfc')[col].shift(1)

    # Marcamos como cambio (1) si el valor actual es diferente al anterior
    # y nos aseguramos de que no sean valores nulos comparándose entre sí
    es_cambio = (df[col].notna() & valor_anterior.notna() & (df[col] != valor_anterior))
    df[f'cambio_subito_{col}'] = es_cambio.astype(int)

# ---------------------------------------------------------
# 3. HORARIOS Y DÍAS INUSUALES
# ---------------------------------------------------------
print("3. Generando variables de tiempo y calendario...")

# Extraer la hora (0-23)
df['hora_num'] = df['fecha_hora'].dt.hour

# Crear franjas horarias
# Usamos 'Manana' sin la 'ñ' para evitar problemas de codificación más adelante en el modelo
bins = [-1, 5, 11, 17, 24]
labels = ['Madrugada', 'Manana', 'Tarde', 'Noche']
df['franja_horaria'] = pd.cut(df['hora_num'], bins=bins, labels=labels)

# Convertir la franja a variables One-Hot (Dummies)
# Esto creará columnas como 'franja_Madrugada', 'franja_Manana', etc. (con valores 0 o 1)
df = pd.get_dummies(df, columns=['franja_horaria'], prefix='franja', dtype=int)

# Fin de semana: dt.dayofweek devuelve 0=Lunes, 6=Domingo.
# Sábado (5) y Domingo (6) serán marcados con 1
df['es_fin_de_semana'] = (df['fecha_hora'].dt.dayofweek >= 5).astype(int)

# Limpieza opcional de columnas temporales usadas para cálculos
df.drop(columns=['monto_actual', 'monto_promedio_6m', 'hora_num'], inplace=True, errors='ignore')

print("✅ Todas las variables de contexto generadas exitosamente.")

print("--- DISTRIBUCIÓN POR FRANJAS HORARIAS ---")
# Sumamos cuántas transacciones cayeron en cada franja
franjas = ['franja_Madrugada', 'franja_Manana', 'franja_Tarde', 'franja_Noche']
# Filtramos solo las columnas que realmente existen (por si alguna franja quedó vacía)
franjas_existentes = [col for col in franjas if col in df.columns]
print(df[franjas_existentes].sum().to_frame(name='Total Transacciones'))

print("\n--- TRANSCACCIONES FIN DE SEMANA VS ENTRE SEMANA ---")
# normalize=True nos da el porcentaje (multiplicado por 100)
print(df['es_fin_de_semana'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\n--- ALERTAS DE MONTO Y CAMBIOS DE IP ---")
print("Transacciones con monto > 10x su promedio (Alerta Fuerte):")
print(df['alerta_monto_excesivo'].value_counts())

print("\nTransacciones realizadas desde una IP nueva/no inscrita:")
print(df['es_ip_nueva'].value_counts())

# Si ya tienes la variable objetivo 'fraude' (1 o 0), este cruce es oro puro:
if 'fraude' in df.columns:
    print("\n--- CRUCE: FRANJA DE MADRUGADA VS FRAUDE ---")
    if 'franja_Madrugada' in df.columns:
        print(pd.crosstab(df['franja_Madrugada'], df['fraude'], normalize='index').mul(100).round(2))

    print("\n--- CRUCE: IP NUEVA VS FRAUDE ---")
    print(pd.crosstab(df['es_ip_nueva'], df['fraude'], normalize='index').mul(100).round(2))

#!pip install xgboost scikit-learn

# Creamos una "Huella Digital" para agrupar transacciones sin cliente
df['huella_digital'] = df['afd__direccion_ip'].astype(str) + '_' + df['rt_afd_sistema_operativo'].astype(str)

# Si el ID del cliente es nulo o vacío, usamos la huella digital
df['id_analisis'] = np.where(
    df['afd__id_cliente/rfc'].isna() | (df['afd__id_cliente/rfc'].astype(str).str.strip() == ''),
    df['huella_digital'],
    df['afd__id_cliente/rfc']
)

# ---------------------------------------------------------
# PASO 2: SELECCIÓN DE VARIABLES (Features) Y TARGET
# ---------------------------------------------------------
# Variables que creamos en los pasos anteriores
features_creadas = [
    'trx_ult_1h', 'trx_ult_6h', 'trx_ult_24h', 'intentos_fallidos_ult_1h',
    'ratio_monto_vs_promedio', 'alerta_monto_excesivo', 'es_ip_nueva',
    'cambio_subito_rt_afd__isp_internet_service_provider',
    'cambio_subito_rt_afd_sistema_operativo', 'cambio_subito_afd_codigo_de_browser',
    'es_fin_de_semana'
]

# Añadir las columnas de franjas horarias (One-Hot Encoding)
franjas = [col for col in df.columns if col.startswith('franja_')]
features_creadas.extend(franjas)

# Otras variables originales numéricas que aportan valor directo
features_originales = [
    'afd__monto_total_trx',
    'mfi_promedio_ultimos_6_meses_producto_origen',
    'afd_score_dinamico' # Si ya existe un score previo, el modelo puede aprender de sus errores
]

# Consolidamos las variables predictoras (X) y la variable objetivo (y)
X_cols = features_creadas + features_originales
# Aseguramos que la columna objetivo sea entera (0 = Legítimo, 1 = Fraude)
y_col = 'fraude'

# ---------------------------------------------------------
# PASO 3: SPLIT TEMPORAL (Train / Test)
# ---------------------------------------------------------
print("📅 Realizando partición temporal de los datos...")

# Ordenamos estrictamente por fecha y hora
df = df.sort_values('fecha_hora')

# Tomamos el 80% inicial de los datos (cronológicamente) para entrenar
# y el 20% más reciente para validar (simulando el paso a producción)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[X_cols].astype(float) # Forzamos float para XGBoost
y_train = train_df[y_col].astype(int)

X_test = test_df[X_cols].astype(float)
y_test = test_df[y_col].astype(int)

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# PASO 4: ENTRENAMIENTO DEL MODELO XGBOOST
# ---------------------------------------------------------
print("🧠 Entrenando el algoritmo Gradient Boosting...")

# Calculamos el ratio de desbalanceo para ayudar al modelo a prestar atención al fraude
conteo_clases = y_train.value_counts()
ratio_desbalance = conteo_clases[0] / max(conteo_clases[1], 1) # Legítimos / Fraudes

# Configuración del modelo
modelo_xgb = xgb.XGBClassifier(
    n_estimators=300,              # Número de árboles
    learning_rate=0.05,            # Tasa de aprendizaje
    max_depth=6,                   # Profundidad de cada árbol (evita sobreajuste)
    scale_pos_weight=ratio_desbalance, # ¡Clave para datos desbalanceados!
    eval_metric='aucpr',           # Evaluamos usando el Área bajo la curva Precision-Recall
    random_state=42,
    n_jobs=-1                      # Usa todos los núcleos de tu procesador
)

# Entrenamiento
modelo_xgb.fit(X_train, y_train)

# ---------------------------------------------------------
# PASO 5: PREDICCIÓN Y EVALUACIÓN BANCARIA
# ---------------------------------------------------------
print("📊 Evaluando resultados en el conjunto de prueba (Out-of-Time)...")

# Predecimos la probabilidad de fraude (no solo el 0 o 1)
y_pred_proba = modelo_xgb.predict_proba(X_test)[:, 1]

# Definimos un umbral de decisión (por defecto 0.5, pero en bancos suele bajarse a 0.7 o 0.8 para evitar molestar clientes)
umbral = 0.5
y_pred = (y_pred_proba >= umbral).astype(int)

print("\n--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))

# Calcular PR-AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"\n🏆 Área bajo la curva Precisión-Recall (PR-AUC): {pr_auc:.4f}")

# 1. Predecimos sobre los datos de entrenamiento (X_train)
y_train_pred_proba = modelo_xgb.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_pred_proba >= umbral).astype(int)

# 2. Imprimimos el reporte comparativo
print("=== RENDIMIENTO EN ENTRENAMIENTO (TRAIN) ===")
print(classification_report(y_train, y_train_pred))

print("\n=== RENDIMIENTO EN PRUEBA (TEST - Out of Time) ===")
print(classification_report(y_test, y_pred))

# 3. Comparación de PR-AUC
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)
pr_auc_train = auc(recall_train, precision_train)
print(f"\n🏆 PR-AUC Train: {pr_auc_train:.4f} vs PR-AUC Test: {pr_auc:.4f}")

import shap
import matplotlib.pyplot as plt

print("Calculando SHAP Values (esto puede tomar unos minutos dependiendo de tu procesador)...")

# 1. Inicializar el explicador de árboles para XGBoost
explainer = shap.TreeExplainer(modelo_xgb)

# Nota: Si X_test es muy grande (>30,000 filas), SHAP puede tardar bastante.
# Es común tomar una muestra representativa para el Summary Plot si hay prisa.
# X_test_sample = X_test.sample(10000, random_state=42)
shap_values = explainer.shap_values(X_test)

# 2. Gráfico de Importancia Global (Summary Plot)
# Muestra qué variables impactan más y en qué dirección (aumentan o bajan el riesgo de fraude)
plt.figure(figsize=(10, 8))
plt.title("Impacto Global de Variables en la Predicción de Fraude")
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.tight_layout()
plt.show()

# 3. Explicabilidad Local (Waterfall Plot para una sola transacción)
# Vamos a buscar un Falso Positivo (cliente bueno bloqueado) para ver qué confundió al modelo
indices_falsos_positivos = np.where((y_test.values == 0) & (y_pred == 1))[0]

if len(indices_falsos_positivos) > 0:
    idx_fp = indices_falsos_positivos[0] # Tomamos el primer falso positivo
    print(f"\nAnalizando el Falso Positivo en el índice {idx_fp}:")

    # Preparamos el Waterfall Plot
    shap_explainer_obj = explainer(X_test)
    plt.figure()
    shap.plots.waterfall(shap_explainer_obj[idx_fp], max_display=10, show=False)
    plt.title(f"¿Por qué se bloqueó esta transacción legítima?")
    plt.tight_layout()
    plt.show()