import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# -------------------------------------------------------------------------
# 1. CARGA DEL DATASET PROCESADO
# -------------------------------------------------------------------------
# Utilizamos el archivo generado por el pipeline de limpieza previo
PATH_DATOS = 'data/dataset_final_ia.csv'

try:
    print(f"Cargando dataset procesado: {PATH_DATOS}")
    df = pd.read_csv(PATH_DATOS, sep=';')
    print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
except Exception as e:
    print(f"Error al cargar el archivo procesado: {e}")
    exit()


# -------------------------------------------------------------------------
# 2. PREPARACIÓN DE MATRICES PARA EL MODELO (CORREGIDO)
# -------------------------------------------------------------------------
TARGET = 'Fraude'

if TARGET not in df.columns:
    print(f"Error: La columna {TARGET} no se encuentra en el dataset.")
    exit()

# Seleccionamos SOLO las columnas numéricas y eliminamos el Target
# Esto quita automáticamente las columnas "object" que causan el error
X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore')
y = df[TARGET]

print(f"Variables finales para entrenamiento: {X.shape[1]}")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------------------------------
# 3. CONFIGURACIÓN DE EXPERIMENTOS MLFLOW
# -------------------------------------------------------------------------
# Definimos el nombre del experimento para el servidor de seguimiento
mlflow.set_experiment("Optimizacion_Dataset_Procesado_Yeison")

def ejecutar_corrida(nombre_prueba, lr, profundidad, estimadores):
    """
    Función para entrenar el modelo XGBoost y registrar los resultados
    en el servidor de MLflow para comparativa técnica.
    """
    with mlflow.start_run(run_name=nombre_prueba):
        # Configuración del algoritmo XGBoost
        model = XGBClassifier(
            n_estimators=estimadores,
            learning_rate=lr,
            max_depth=profundidad,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Proceso de ajuste del modelo
        model.fit(X_train, y_train)
        
        # Obtención de métricas de desempeño
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        
        # Registro de parámetros e indicadores de rendimiento
        mlflow.log_params({
            "learning_rate": lr,
            "max_depth": profundidad,
            "n_estimators": estimadores
        })
        
        mlflow.log_metrics({
            "f1_score": f1,
            "precision": prec,
            "recall": rec
        })
        
        # Persistencia del modelo para despliegue futuro
        mlflow.xgboost.log_model(model, "modelo_final_fraude")
        
        print(f"Finalizado: {nombre_prueba} -> F1-Score: {round(f1, 4)}")

# -------------------------------------------------------------------------
# 4. EJECUCIÓN DE PRUEBAS COMPARATIVAS
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando ciclo de experimentos de optimización...")
    
    # Prueba A: Configuración estándar de aprendizaje rápido
    ejecutar_corrida("XGB_Estandar", lr=0.1, profundidad=6, estimadores=100)
    
    # Prueba B: Configuración profunda para capturar patrones complejos
    ejecutar_corrida("XGB_Profundo", lr=0.05, profundidad=12, estimadores=200)
    
    # Prueba C: Configuración balanceada (Candidata a producción)
    ejecutar_corrida("XGB_Balanceado", lr=0.2, profundidad=9, estimadores=150)
