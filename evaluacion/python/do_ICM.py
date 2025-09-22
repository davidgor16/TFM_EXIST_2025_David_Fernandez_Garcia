from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.metrics.metricfactory import MetricFactory
import json
import pandas as pd

from do_eval import do_eval

def do_ICM(predictions, ids, json_filename):
    
    # Lista para almacenar todos los registros en el formato deseado
    json_data = []
    
    for id_, prediction in zip(ids, predictions):
        
        value = "NO" if int(prediction) == 0 else "YES"
        
        # Crear diccionario para cada ID
        record = {
            "test_case": "EXIST2025",
            "id": str(id_),  # Asegurar que es string (si es necesario)
            "value": value  # Convertir a string y mayúsculas (ej. "NO")
        }
        json_data.append(record)
    
    # Guardar en un archivo JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
    # Evaluamos el modelo
    return do_eval(json_filename)
    
    
    
