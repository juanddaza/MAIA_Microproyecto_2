import pandas as pd
import numpy as np
import csv
import os

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustado al nuevo nombre del archivo
INPUT_FILE = 'data/universidad.csv' 
OUTPUT_FILE = 'data/dataset_final_ia.csv'

try:
    print("Iniciando Pipeline de Limpieza sobre universidad.csv...")
    
    # --- CARGA ROBUSTA ---
    # Usamos el formato de punto y coma que detectamos antes
    df = pd.read_csv(
        INPUT_FILE, 
        sep=';', 
        encoding='latin-1', 
        quoting=csv.QUOTE_NONE, 
        on_bad_lines='skip', 
        low_memory=False
    )
    
    # Limpieza de cabeceras (espacios y comillas)
    df.columns = [c.strip().replace('"', '').replace(';', '') for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # --- INGENIERÍA DE CARACTERÍSTICAS ---
    print("Ejecutando rescate de variables comportamentales...")
    
    # Rescate de Hora (detectando la columna automáticamente)
    col_hora = [c for c in df.columns if 'Hora Trx' in c][0]
    df['Hora_Num'] = pd.to_datetime(df[col_hora], errors='coerce').dt.hour
    df['Es_Madrugada'] = df['Hora_Num'].apply(lambda x: 1 if x <= 5 or x >= 23 else 0)

    # Rescate de Frecuencia de Cuenta Origen
    col_cta = [c for c in df.columns if 'Cuenta Origen' in c][0]
    df['Frecuencia_Cta'] = df.groupby(col_cta)[col_cta].transform('count')

    # --- DEPURACIÓN ---
    print("Eliminando ruido y variables sensibles...")
    cols_a_borrar = [col_hora, col_cta, 'Usuarios']
    df = df.drop(columns=[c for c in cols_a_borrar if c in df.columns])

    # --- TRANSFORMACIÓN (ONE-HOT ENCODING) ---
    print("Convirtiendo variables categóricas...")
    # Solo variables con cardinalidad razonable para no saturar memoria
    cols_ohe = [c for c in df.select_dtypes(include=['object']).columns if df[c].nunique() < 50]
    df_final = pd.get_dummies(df, columns=cols_ohe, drop_first=True)

    # --- GUARDAR RESULTADO ---
    df_final.to_csv(OUTPUT_FILE, index=False, sep=';')
    print(f"Pipeline completado con éxito.")
    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"Dimensiones finales: {df_final.shape}")

except Exception as e:
    print(f"Error en la ejecución del pipeline: {e}")
