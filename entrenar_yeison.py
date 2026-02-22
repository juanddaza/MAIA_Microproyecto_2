import pandas as pd
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score

# --- CONFIGURACION DE CARGA ---
# Usamos latin-1 para soportar tildes y eñes del sector bancario
# Usamos sep=None para que el motor de python detecte si es tabulacion o coma


# --- CONFIGURACION DE CARGA ---
# --- CONFIGURACION DE CARGA CORREGIDA ---
try:
    print("Iniciando carga con separador de punto y coma (;)...")
    df = pd.read_csv(
        'data/datos_fraude_anon.csv', 
        encoding='latin-1', 
        sep=';',              # Forzamos el punto y coma detectado en el error
        on_bad_lines='skip', 
        low_memory=False
    )
    # Limpiar espacios y posibles comillas de los nombres de columnas
    df.columns = df.columns.str.strip().str.replace('"', '')
    print("Carga exitosa. Registros: " + str(len(df)) + " Columnas: " + str(len(df.columns)))
except Exception as e:
    print("Error al cargar el archivo: " + str(e))
    exit()

# --- BUSQUEDA DE LA VARIABLE OBJETIVO ---
# Buscamos cualquier columna que se llame exactamente 'Fraude' o termine en ';Fraude'
if 'Fraude' in df.columns:
    target = 'Fraude'
else:
    # Intento de encontrar la columna si quedó pegada a otra
    posibles = [c for c in df.columns if 'Fraude' in c]
    if posibles:
        target = posibles[0]
    else:
        print("Error: No se encontro la columna 'Fraude'.")
        print("Columnas disponibles (primeras 10):")
        print(df.columns[:10].tolist())
        exit()

print("Variable objetivo para entrenamiento: " + target)

# --- LIMPIEZA DE DATOS ---
# Convertir la columna objetivo a numerico (por si acaso)
df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)

# Seleccionamos solo columnas numericas para el modelo
X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]
X = X.fillna(0)

print("Entrenando con " + str(len(X.columns)) + " variables numericas.")


# --- PREPROCESAMIENTO ---
# Limpieza de espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()



# Identificacion de la variable objetivo (Target)
# El codigo buscara la columna 'Fraude' que se agrego al final
if 'Fraude' in df.columns:
    target = 'Fraude'
elif 'AFD -  Indicador de Fraude' in df.columns:
    target = 'AFD -  Indicador de Fraude'
else:
    print("Error: No se encontro la columna objetivo. Columnas disponibles: " + str(df.columns[:5]))
    exit()

print("Variable objetivo identificada: " + target)

# Seleccion de variables numericas y manejo de valores nulos
# XGBoost maneja bien los datos pero necesitamos limpiar columnas no numericas para esta prueba
X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]

# Rellenar valores nulos con 0 para evitar errores en el entrenamiento
X = X.fillna(0)

# Division del dataset: 80% entrenamiento y 20% validacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CONFIGURACION MLFLOW ---
# Se define el experimento para el seguimiento en el servidor AWS
mlflow.set_experiment("Optimizacion_XGBoost_YeisonToro")

def ejecutar_entrenamiento(estimadores, learning_rate, profundidad):
    run_name = "XGB_LR_" + str(learning_rate) + "_D_" + str(profundidad)
    with mlflow.start_run(run_name=run_name):
        # Configuracion del modelo XGBoost
        model = XGBClassifier(
            n_estimators=estimadores,
            learning_rate=learning_rate,
            max_depth=profundidad,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Entrenamiento del modelo
        model.fit(X_train, y_train)
        
        # Calculo de predicciones y metricas
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        
        # Obtener probabilidades para el calculo de AUC
        probabilidades = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilidades)
        
        # Registro de parametros y metricas en MLflow
        mlflow.log_param("n_estimators", estimadores)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", profundidad)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Guardado del modelo en el servidor
        mlflow.xgboost.log_model(model, "modelo_final_fraude")
        
        print("Experimento registrado: LR=" + str(learning_rate) + " F1=" + str(round(f1, 4)))

# --- EJECUCION DE PRUEBAS ---
if __name__ == "__main__":
    # Prueba 1: Configuracion conservadora
    ejecutar_entrenamiento(100, 0.01, 3)
    
    # Prueba 2: Configuracion estandar
    ejecutar_entrenamiento(200, 0.1, 6)
    
    # Prueba 3: Configuracion agresiva
    ejecutar_entrenamiento(300, 0.2, 9)
