import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM_soft import do_ICM_soft

def convert_to_soft_predictions(df, model_columns):
    """
    Convierte las salidas numéricas de un modelo en tuplas de probabilidades (sí/no)
    
    Parámetros:
    - df: DataFrame que contiene la columna del modelo
    - model_columns: Lista con el nombre de la columna del modelo (en este caso solo una)
    
    Retorna:
    - Lista de tuplas donde cada tupla es (prob_si, prob_no)
    """
    # Aseguramos que estamos trabajando con un único modelo
    if len(model_columns) != 1:
        raise ValueError("Esta función está diseñada para un único modelo. model_columns debe contener exactamente un elemento.")
    
    # Obtenemos el nombre de la columna del modelo
    model_column = model_columns[0]
    
    # Obtener los valores del modelo
    values = df[model_column].values
    
    # Normalizar al rango [0,1] si los valores están fuera de ese rango
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val < 0 or max_val > 1:
        # Normalización Min-Max
        normalized_values = (values - min_val) / (max_val - min_val)
    else:
        normalized_values = values
    
    # Crear las tuplas (prob_si, prob_no)
    probability_tuples = [(float(prob_si), float(1.0 - prob_si)) for prob_si in normalized_values]
    
    return probability_tuples

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el CSV
    df = pd.read_csv('../../../entrenamiento/audio/results/resultados_train.csv')
    
    # Columnas de los modelos
    model_columns = [
        #'facebook/wav2vec2-large-xlsr-53',
        #'google/vivit-b-16x2-kinetics400',
        #'FacebookAI/xlm-roberta-large_transcripcion',
        #'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
    
    # Convertir a predicciones soft
    resultado = convert_to_soft_predictions(df, model_columns)
       
    icm,icm_norm,cross_entropy = do_ICM_soft(resultado, df["id_exist"].values, "soft_individual.json")
    
    print(f"MODELO: {model_columns[0]}")
    print(f"ICM_SOFT: {icm}")
    print(f"ICM_SOFT_NORM: {icm_norm}")
    print(f"Cross Entropy: {cross_entropy}")
    
    