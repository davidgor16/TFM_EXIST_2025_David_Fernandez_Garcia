import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM_soft import do_ICM_soft

def convert_to_soft_predictions(df, model_columns, target_column='ground_truth', n_folds=5, random_state=42):
    """
    Implementa stacking con validación cruzada para combinar múltiples señales
    y obtener tuplas de probabilidades.
    
    Parámetros:
    - df: DataFrame con las columnas de los modelos y la columna objetivo
    - model_columns: Lista de nombres de columnas que contienen las salidas de los modelos
    - target_column: Nombre de la columna objetivo (etiquetas reales)
    - n_folds: Número de folds para la validación cruzada
    - random_state: Semilla aleatoria para reproducibilidad
    
    Retorna:
    - Lista de tuplas donde cada tupla es (prob_si, prob_no)
    """
    # Extraer características (X) y etiquetas (y)
    X = df[model_columns].values
    y = df[target_column].values
    
    # Crear objeto KFold para validación cruzada
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Crear arreglo para almacenar las predicciones out-of-fold
    meta_features = np.zeros((X.shape[0],))  # Una columna para la prob de clase positiva
    
    # Realizar la validación cruzada
    for train_idx, val_idx in kf.split(X):
        # Dividir los datos
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Entrenar el meta-modelo (regresión logística)
        meta_model = LogisticRegression(solver='liblinear', random_state=random_state)
        meta_model.fit(X_train, y_train)
        
        # Generar predicciones out-of-fold
        meta_features[val_idx] = meta_model.predict_proba(X_val)[:, 1]
    
    # Entrenar el meta-modelo final usando todos los datos
    final_meta_model = LogisticRegression(solver='liblinear', random_state=random_state)
    final_meta_model.fit(X, y)
    
    # Calibrar el modelo final para obtener mejores probabilidades
    calibrated_model = CalibratedClassifierCV(final_meta_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X, y)
    
    # Imprimir información sobre el modelo
    print(f"Modelo de stacking entrenado con {len(model_columns)} modelos base")
    print(f"Coeficientes del meta-modelo (importancia de cada modelo):")
    for model_name, coef in zip(model_columns, final_meta_model.coef_[0]):
        print(f"  {model_name}: {coef:.4f}")
    
    # Si solo se proporcionó un modelo, usamos las predicciones out-of-fold
    # En caso contrario, usamos el modelo calibrado final
    if len(model_columns) == 1:
        probability_values = meta_features
    else:
        # Obtener las probabilidades finales con el modelo calibrado
        probability_values = calibrated_model.predict_proba(X)[:, 1]
        
    # Convertir a lista de tuplas (prob_si, prob_no)
    probability_tuples = [(float(p), float(1.0 - p)) for p in probability_values]
    
    # Evaluar el rendimiento
    binary_predictions = [1 if p > 0.5 else 0 for p in probability_values]
    accuracy = accuracy_score(y, binary_predictions)
    f1 = f1_score(y, binary_predictions)
    
    print(f"\nRendimiento del modelo:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return probability_tuples

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el CSV
    df = pd.read_csv('../../../entrenamiento/audio/results/resultados_train.csv')
    
    # Columnas de los modelos - ahora podemos usar uno o más modelos
    model_columns = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        #'FacebookAI/xlm-roberta-large_transcripcion',
        #'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
    
    # Convertir a predicciones soft usando stacking
    resultado = convert_to_soft_predictions(df, model_columns)
       
    icm, icm_norm, cross_entropy = do_ICM_soft(resultado, df["id_exist"].values, "soft_stacking.json")
    
    print(f"\nRESULTADOS FINALES:")
    print(f"ICM_SOFT: {icm}")
    print(f"ICM_SOFT_NORM: {icm_norm}")
    print(f"Cross Entropy: {cross_entropy}")