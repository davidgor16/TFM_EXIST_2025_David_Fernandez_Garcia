from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.metrics.metricfactory import MetricFactory
import json
import pandas as pd

from do_eval import do_eval

def do_ICM_soft(prediction_tuples, ids, json_filename):
    """
    Crea un archivo JSON con predicciones soft (probabilidades) y evalúa el modelo
    
    Parámetros:
    - prediction_tuples: Lista de tuplas donde cada tupla es (prob_si, prob_no)
    - ids: Lista de identificadores
    - json_filename: Nombre del archivo JSON a generar
    
    Retorna:
    - Resultado de la evaluación del modelo
    """
    # Lista para almacenar todos los registros en el formato deseado
    json_data = []
    
    for id_, prediction_tuple in zip(ids, prediction_tuples):
        # Extraer las probabilidades de la tupla
        prob_si, prob_no = prediction_tuple
            
        # Crear diccionario para cada ID, incluyendo las probabilidades
        record = {
            "test_case": "EXIST2025",
            "id": str(id_),  # Asegurar que es string
            "value": {
                "YES": float(prob_si),  # Probabilidad de "YES"
                "NO": float(prob_no)    # Probabilidad de "NO"
            }
        }
        json_data.append(record)
    
    # Guardar en un archivo JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    
    # Evaluamos el modelo
    return do_eval(json_filename,"SOFT")
    
    
    
