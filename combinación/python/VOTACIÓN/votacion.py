import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM import do_ICM

tipo = sys.argv[1]

# Leer el archivo CSV
df = pd.read_csv('../../../entrenamiento/audio/results/resultados_train.csv')

if tipo == "mio":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        #'FacebookAI/xlm-roberta-large_original'
    ]
    min_votos = 3
    
if tipo == "original":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        #'FacebookAI/xlm-roberta-large_transcripcion',
        #'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
    min_votos = 2

if tipo == "todos":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
    min_votos = 3

# Ground truth
ground_truth = df['ground_truth'].values

def predecir_votacion(df, umbrales, min_votos=3):
    """
    Predice 1 si al menos 'min_votos' modelos superan su umbral
    
    Parámetros:
    - df: DataFrame con los scores de los modelos
    - umbrales: lista/array con los umbrales para cada modelo
    - min_votos: número mínimo de modelos que deben votar 'sí' (por defecto 3 de 5)
    """
    n_samples = len(df)
    predicciones = np.zeros(n_samples)
    
    for i in range(n_samples):
        votos = 0
        
        # Contar votos de cada modelo
        for j, modelo in enumerate(modelos):
            if df[modelo].iloc[i] >= umbrales[j]:
                votos += 1
        
        # Decidir por mayoría
        if votos >= min_votos:
            predicciones[i] = 1
    
    return predicciones

# Función para calcular métricas
def calcular_metricas(y_pred):
    ids = df["id_exist"].values
    icm, icm_norm, f1 = do_ICM(y_pred, ids, "votacion_"+tipo+".json")
    
    return {
        'icm': icm,
        'icm_norm': icm_norm,
        'f1': f1
    }

# Búsqueda exhaustiva
def busqueda_exhaustiva(n_puntos=2):
    """
    Prueba todas las combinaciones posibles de umbrales
    """
    print(f"Probando todas las combinaciones con {n_puntos} puntos por modelo...")
    print(f"Total de combinaciones: {n_puntos**len(modelos)}")
    
    # Crear grilla de umbrales posibles
    umbrales_posibles = np.linspace(0, 1, n_puntos)
    
    mejores_resultados = {
        'icm': {'umbrales': None, 'valor': -np.inf, 'metricas': None},
        'icm_norm': {'umbrales': None, 'valor': -np.inf, 'metricas': None},
        'f1': {'umbrales': None, 'valor': -np.inf, 'metricas': None},
    }
    
    total_combinaciones = n_puntos ** len(modelos)
    contador = 0
    inicio = time.time()
    
    # Probar todas las combinaciones
    for combinacion in product(umbrales_posibles, repeat=len(modelos)):
        contador += 1
        
        # Mostrar progreso
        if contador % 1000 == 0:
            tiempo_transcurrido = time.time() - inicio
            velocidad = contador / tiempo_transcurrido
            tiempo_restante = (total_combinaciones - contador) / velocidad
            print(f"Progreso: {contador}/{total_combinaciones} ({contador/total_combinaciones*100:.1f}%) - Tiempo restante: {tiempo_restante:.0f}s")
        
        # Hacer predicciones
        predicciones = predecir_votacion(df, combinacion, min_votos=min_votos)
        
        # Calcular métricas
        metricas = calcular_metricas(predicciones)
        
        # Actualizar mejores resultados
        for metrica in ['icm', 'icm_norm', 'f1']:
            if metricas[metrica] > mejores_resultados[metrica]['valor']:
                mejores_resultados[metrica]['valor'] = metricas[metrica]
                mejores_resultados[metrica]['umbrales'] = combinacion
                mejores_resultados[metrica]['metricas'] = metricas
    
    tiempo_total = time.time() - inicio
    print(f"\nBúsqueda completada en {tiempo_total:.1f} segundos")
    
    return mejores_resultados

# Ejecutar búsqueda
print("=== BÚSQUEDA EXHAUSTIVA DE UMBRALES ÓPTIMOS ===\n")

# Puedes ajustar n_puntos para más o menos precisión (más puntos = más tiempo)
resultados = busqueda_exhaustiva(n_puntos=5)

# Mostrar resultados
print("\n=== MEJORES RESULTADOS ===\n")

for metrica_objetivo in ['icm', 'icm_norm', 'f1']:
    print(f"\nOptimizando para {metrica_objetivo.upper()}:")
    resultado = resultados[metrica_objetivo]
    
    print(f"Valor óptimo: {resultado['valor']:.4f}")
    print("Umbrales:")
    for i, modelo in enumerate(modelos):
        print(f"  {modelo}: {resultado['umbrales'][i]:.3f}")
    
    print("Todas las métricas:")
    for metrica, valor in resultado['metricas'].items():
        print(f"  {metrica}: {valor:.4f}")

# Mejor combinación general (optimizando F1)
print("\n=== MEJOR COMBINACIÓN (ICM) ===")
mejor_f1 = resultados['icm']
print(f"F1: {mejor_f1['valor']:.4f}")
print("\nUmbrales óptimos:")
for i, modelo in enumerate(modelos):
    print(f"{modelo}: {mejor_f1['umbrales'][i]:.3f}")

print("\nMétricas completas:")
for metrica, valor in mejor_f1['metricas'].items():
    print(f"{metrica}: {valor:.4f}")

# Aplicar los mejores umbrales
predicciones_optimas = predecir_votacion(df, mejor_f1['umbrales'], min_votos=min_votos)
n_positivos = np.sum(predicciones_optimas)
print(f"\nPredicciones positivas: {n_positivos} de {len(predicciones_optimas)} ({n_positivos/len(predicciones_optimas)*100:.1f}%)")
