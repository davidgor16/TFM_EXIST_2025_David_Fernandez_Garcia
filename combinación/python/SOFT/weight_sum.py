import pandas as pd
import numpy as np
import sys
import os
import json
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM_soft import do_ICM_soft

def convert_to_soft_predictions(df, model_columns, target_column='ground_truth', val_split=0.2, random_state=42, temp_file='temp_optimization.json'):
    """
    Combina las salidas de múltiples modelos usando promedio ponderado con pesos optimizados
    para minimizar Cross Entropy y devuelve tuplas de probabilidades.
    
    Parámetros:
    - df: DataFrame con las columnas de los modelos
    - model_columns: Lista con los nombres de las columnas de los modelos
    - target_column: Nombre de la columna objetivo
    - val_split: Proporción de datos a usar como validación para optimizar pesos
    - random_state: Semilla aleatoria para reproducibilidad
    - temp_file: Archivo temporal para optimización
    
    Retorna:
    - Lista de tuplas donde cada tupla es (prob_si, prob_no)
    """
    # Si solo hay un modelo, procesarlo directamente
    if len(model_columns) == 1:
        # Obtener los valores del modelo
        values = df[model_columns[0]].values
        
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
        
        print(f"Se usó un solo modelo: {model_columns[0]}")
        
        return probability_tuples
    
    # Para múltiples modelos, implementar promedio ponderado con optimización de pesos
    print(f"Combinando {len(model_columns)} modelos usando promedio ponderado optimizado para Cross Entropy")
    
    # Obtener y normalizar los valores de cada modelo
    normalized_model_values = []
    
    for column in model_columns:
        values = df[column].values
        min_val = np.min(values)
        max_val = np.max(values)
        
        if min_val < 0 or max_val > 1:
            # Normalización Min-Max
            normalized_values = (values - min_val) / (max_val - min_val)
        else:
            normalized_values = values
            
        normalized_model_values.append(normalized_values)
    
    # Crear un array con forma (n_muestras, n_modelos)
    X = np.column_stack(normalized_model_values)
    
    # Obtener las etiquetas
    y = df[target_column].values
    
    # Dividir en conjuntos de entrenamiento y validación
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    val_size = int(val_split * len(df))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_val = X[val_indices]
    y_val = y[val_indices]
    id_val = df['id_exist'].values[val_indices]
    
    # Definir la función objetivo para optimizar Cross Entropy directamente
    def objective_function(weights):
        # Normalizar los pesos para que sumen 1
        weights = weights / np.sum(weights)
        
        # Calcular predicciones ponderadas
        weighted_preds = np.dot(X_val, weights)
        
        # Convertir a tuplas de probabilidades
        probability_tuples = [(float(prob_si), float(1.0 - prob_si)) for prob_si in weighted_preds]
        
        # Calcular Cross Entropy usando do_ICM_soft
        _, _, cross_entropy = do_ICM_soft(probability_tuples, id_val, temp_file)
        
        return cross_entropy  # Queremos minimizar esta métrica
    
    # Optimizar los pesos
    # Inicializar con pesos iguales
    initial_weights = np.ones(len(model_columns)) / len(model_columns)
    
    # Restricciones: pesos no negativos, suma = 1
    bounds = [(0, 1) for _ in range(len(model_columns))]
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    print("Optimizando pesos para minimizar Cross Entropy... (esto puede tomar un tiempo)")
    
    # Ejecutar la optimización
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
        options={'disp': False, 'maxiter': 100, 'ftol': 1e-3}
    )
    
    # Obtener los pesos optimizados
    optimized_weights = result.x
    
    # Normalizar los pesos para asegurar que sumen 1
    optimized_weights = optimized_weights / np.sum(optimized_weights)
    
    # Mostrar los pesos optimizados
    print("\nPesos optimizados para cada modelo:")
    for column, weight in zip(model_columns, optimized_weights):
        print(f"  {column}: {weight:.4f}")
    
    # Calcular las predicciones finales usando los pesos optimizados
    weighted_predictions = np.dot(X, optimized_weights)
    
    # Crear las tuplas (prob_si, prob_no)
    probability_tuples = [(float(prob_si), float(1.0 - prob_si)) for prob_si in weighted_predictions]
    
    # Evaluar el rendimiento en todo el conjunto de datos
    binary_preds = (weighted_predictions > 0.5).astype(int)
    
    # Calcular exactitud
    accuracy = np.mean(binary_preds == y)
    
    # Calcular F1-score
    tp = np.sum((y == 1) & (binary_preds == 1))
    fp = np.sum((y == 0) & (binary_preds == 1))
    fn = np.sum((y == 1) & (binary_preds == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nRendimiento con pesos optimizados (conjunto completo):")
    print(f"  Exactitud: {accuracy:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Precisión: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Intentar eliminar el archivo temporal si existe
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except:
        pass
    
    return probability_tuples

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el CSV
    df = pd.read_csv('../../../entrenamiento/audio/results/resultados_train.csv')
    
    # Columnas de los modelos
    model_columns = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        #'FacebookAI/xlm-roberta-large_original'
    ]
    
    # Archivo temporal para optimización
    temp_file = 'temp_optimization.json'
    
    # Convertir a predicciones soft usando promedio ponderado optimizado para Cross Entropy
    resultado = convert_to_soft_predictions(df, model_columns, temp_file=temp_file)
       
    # Nombre del archivo de salida
    output_file = "soft_weighted_optimized_cross_entropy.json"
    
    # Evaluar con do_ICM_soft
    icm, icm_norm, cross_entropy = do_ICM_soft(resultado, df["id_exist"].values, output_file)
    
    print(f"\nRESULTADOS FINALES (Promedio ponderado optimizado para Cross Entropy):")
    print(f"ICM_SOFT: {icm}")
    print(f"ICM_SOFT_NORM: {icm_norm}")
    print(f"Cross Entropy: {cross_entropy}")