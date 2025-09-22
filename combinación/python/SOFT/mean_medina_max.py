import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM_soft import do_ICM_soft

def convert_to_soft_predictions(df, model_columns, method='median', target_column='ground_truth'):
    """
    Combina las salidas de múltiples modelos usando la regla del máximo, mínimo o mediana
    y devuelve tuplas de probabilidades.
    
    Parámetros:
    - df: DataFrame con las columnas de los modelos
    - model_columns: Lista con los nombres de las columnas de los modelos
    - method: Método de combinación ('max', 'min', 'median', 'mean')
    - target_column: Nombre de la columna objetivo (para mostrar métricas)
    
    Retorna:
    - Lista de tuplas donde cada tupla es (prob_si, prob_no)
    """
    # Verificar que el método sea válido
    valid_methods = ['max', 'min', 'median', 'mean']
    if method not in valid_methods:
        raise ValueError(f"El método '{method}' no es válido. Opciones: {valid_methods}")
    
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
    
    # Si hay múltiples modelos, combinarlos según el método elegido
    print(f"Combinando {len(model_columns)} modelos usando el método '{method}'")
    
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
    model_predictions = np.column_stack(normalized_model_values)
    
    # Aplicar la regla seleccionada a lo largo del eje 1 (modelos)
    if method == 'max':
        combined_predictions = np.max(model_predictions, axis=1)
        print("Usando el valor MÁXIMO de las predicciones de los modelos")
    elif method == 'min':
        combined_predictions = np.min(model_predictions, axis=1)
        print("Usando el valor MÍNIMO de las predicciones de los modelos")
    elif method == 'median':
        combined_predictions = np.median(model_predictions, axis=1)
        print("Usando la MEDIANA de las predicciones de los modelos")
    else:  # method == 'mean'
        combined_predictions = np.mean(model_predictions, axis=1)
        print("Usando la MEDIA de las predicciones de los modelos")
    
    # Analizar la contribución de cada modelo (correlación con la predicción final)
    print("\nAnálisis de contribución de cada modelo:")
    for i, column in enumerate(model_columns):
        correlation = np.corrcoef(normalized_model_values[i], combined_predictions)[0, 1]
        print(f"  {column}: correlación con predicción final = {correlation:.4f}")
    
    # Crear las tuplas (prob_si, prob_no)
    probability_tuples = [(float(prob_si), float(1.0 - prob_si)) for prob_si in combined_predictions]
    
    # Evaluar el rendimiento si tenemos las etiquetas reales
    if target_column in df.columns:
        y_true = df[target_column].values
        y_pred = [1 if p[0] > 0.5 else 0 for p in probability_tuples]
        
        # Calcular exactitud
        accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
        
        # Calcular F1-score manualmente (para evitar dependencias adicionales)
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nRendimiento con método '{method}':")
        print(f"  Exactitud: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")
    
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
        'FacebookAI/xlm-roberta-large_original'
    ]
    
    for metodo in ["max", "min", "median", "mean"]:
        
        # Convertir a predicciones soft usando la regla seleccionada
        resultado = convert_to_soft_predictions(df, model_columns, method=metodo)
        
        # Nombre del archivo de salida basado en el método
        output_file = f"soft_{metodo}.json"
        
        # Evaluar con do_ICM_soft
        icm, icm_norm, cross_entropy = do_ICM_soft(resultado, df["id_exist"].values, output_file)
        
        print(f"\nRESULTADOS FINALES ({metodo.upper()}):")
        print(f"ICM_SOFT: {icm}")
        print(f"ICM_SOFT_NORM: {icm_norm}")
        print(f"Cross Entropy: {cross_entropy}")