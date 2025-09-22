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

# Definir las columnas de los modelos

if tipo == "mio":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        #'FacebookAI/xlm-roberta-large_original'
    ]
    
if tipo == "original":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        #'FacebookAI/xlm-roberta-large_transcripcion',
        #'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]

if tipo == "todos":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]

# Ground truth
ground_truth = df['ground_truth'].values

# Función para hacer predicciones con estrategia OR
def predecir_or(df, umbrales):
    """
    Predice 1 si al menos un modelo supera su umbral
    """
    n_samples = len(df)
    predicciones = np.zeros(n_samples)
    
    for i in range(n_samples):
        for j, modelo in enumerate(modelos):
            if df[modelo].iloc[i] >= umbrales[j]:
                predicciones[i] = 1
                break
    
    return predicciones

# Función para calcular métricas
def calcular_metricas(y_pred):
    ids = df["id_exist"].values
    icm, icm_norm, f1 = do_ICM(y_pred, ids, "con_uno_vale_"+tipo+".json")
    
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
        predicciones = predecir_or(df, combinacion)
        
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
#resultados = busqueda_exhaustiva(n_puntos=1)

resultados = {
        'icm': {'umbrales': [0.778,0.889,0.889], 'valor': 0.2059, 'metricas':  {'icm': 0.2059, 'icm_norm': 0.6031, 'f1': 0.7351}},
        'icm_norm': {'umbrales': [0.778,0.889,0.889], 'valor': 0.6031, 'metricas': {'icm': 0.2059, 'icm_norm': 0.6031, 'f1': 0.7351}},
        'f1': {'umbrales': [0.778,0.889,0.889], 'valor': 0.7351, 'metricas': {'icm': 0.2059, 'icm_norm': 0.6031, 'f1': 0.7351}},
    }


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

# Mejor combinación general (optimizando ICM)
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
predicciones_optimas = predecir_or(df, mejor_f1['umbrales'])
n_positivos = np.sum(predicciones_optimas)
print(f"\nPredicciones positivas: {n_positivos} de {len(predicciones_optimas)} ({n_positivos/len(predicciones_optimas)*100:.1f}%)")

# Función para análisis de solapamiento
def analizar_solapamiento(df, umbrales):
    """
    Analiza el solapamiento entre modelos con los umbrales dados
    """
    # Crear predicciones binarias para cada modelo
    predicciones_por_modelo = {}
    for j, modelo in enumerate(modelos):
        predicciones_por_modelo[modelo] = (df[modelo] >= umbrales[j]).astype(int)
    
    # Calcular cuántos modelos predicen 1 para cada muestra
    num_modelos_positivos = np.zeros(len(df))
    for modelo in modelos:
        num_modelos_positivos += predicciones_por_modelo[modelo]
    
    # Distribución del solapamiento
    distribucion = {}
    total_muestras = len(df)
    
    for n in range(len(modelos) + 1):
        count = np.sum(num_modelos_positivos == n)
        percentage = (count / total_muestras) * 100
        distribucion[n] = {
            'count': count,
            'percentage': percentage
        }
    
    # Calcular Jaccard Generalizado
    arrays_pred = [predicciones_por_modelo[modelo] for modelo in modelos]
    
    # Intersección (todos predicen 1)
    interseccion = np.ones(len(df), dtype=bool)
    for pred in arrays_pred:
        interseccion &= (pred == 1)
    
    # Unión (al menos uno predice 1)
    union = np.zeros(len(df), dtype=bool)
    for pred in arrays_pred:
        union |= (pred == 1)
    
    num_interseccion = np.sum(interseccion)
    num_union = np.sum(union)
    jaccard_generalizado = num_interseccion / num_union if num_union > 0 else 0
    
    # Matrices de solapamiento entre pares
    matriz_jaccard = np.zeros((len(modelos), len(modelos)))
    matriz_correlacion = np.zeros((len(modelos), len(modelos)))
    
    for i, modelo1 in enumerate(modelos):
        for j, modelo2 in enumerate(modelos):
            pred1 = predicciones_por_modelo[modelo1]
            pred2 = predicciones_por_modelo[modelo2]
            
            if i == j:
                matriz_jaccard[i, j] = 1.0
                matriz_correlacion[i, j] = 1.0
            else:
                # Jaccard
                interseccion_par = np.sum((pred1 == 1) & (pred2 == 1))
                union_par = np.sum((pred1 == 1) | (pred2 == 1))
                matriz_jaccard[i, j] = interseccion_par / union_par if union_par > 0 else 0
                
                # Correlación
                matriz_correlacion[i, j] = np.corrcoef(pred1, pred2)[0, 1]
    
    # Análisis de contribuciones
    contribuciones = np.zeros(len(modelos))
    muestras_positivas = np.where(num_modelos_positivos > 0)[0]
    
    for idx in muestras_positivas:
        for j, modelo in enumerate(modelos):
            if df[modelo].iloc[idx] >= umbrales[j]:
                contribuciones[j] += 1
                break  # Solo cuenta el primero que dice sí
    
    return {
        'distribucion': distribucion,
        'jaccard_generalizado': jaccard_generalizado,
        'matriz_jaccard': matriz_jaccard,
        'matriz_correlacion': matriz_correlacion,
        'contribuciones': contribuciones,
        'num_modelos_positivos': num_modelos_positivos
    }
    
    
# ANÁLISIS DE SOLAPAMIENTO CON UMBRALES ÓPTIMOS
print("\n=== ANÁLISIS DE SOLAPAMIENTO CON UMBRALES ÓPTIMOS ===")
solapamiento = analizar_solapamiento(df, mejor_f1['umbrales'])

# Mostrar distribución del solapamiento
print("\nDistribución del solapamiento:")
for n in range(len(modelos) + 1):
    stats = solapamiento['distribucion'][n]
    print(f"{n} modelos predicen 1: {stats['count']} casos ({stats['percentage']:.2f}%)")

print(f"\nJaccard Generalizado: {solapamiento['jaccard_generalizado']:.4f}")

# Contribuciones por modelo
print("\nContribuciones por modelo (primer modelo en decir sí):")
total_positivos = np.sum(solapamiento['contribuciones'])
for i, modelo in enumerate(modelos):
    contrib = solapamiento['contribuciones'][i]
    print(f"{modelo}: {int(contrib)} casos ({contrib/total_positivos*100:.1f}%)")

# Calcular porcentaje de clasificaciones positivas por modelo
print("\nPorcentaje de casos clasificados como 1 por cada modelo:")
total_casos = len(df)
for i, modelo in enumerate(modelos):
    umbral = mejor_f1['umbrales'][i]
    predicciones = (df[modelo] >= umbral).astype(int)
    positivos = np.sum(predicciones)
    porcentaje = (positivos / total_casos) * 100
    print(f"{modelo}: {positivos} casos ({porcentaje:.1f}%)")

# NUEVA SECCIÓN: Análisis de contribución individual al total de predicciones positivas
print("\n=== CONTRIBUCIÓN INDIVIDUAL AL TOTAL DE PREDICCIONES POSITIVAS ===")

# Calcular el total de predicciones positivas de todos los modelos
total_predicciones_positivas = 0
predicciones_por_modelo = []

for i, modelo in enumerate(modelos):
    umbral = mejor_f1['umbrales'][i]
    predicciones = (df[modelo] >= umbral).astype(int)
    positivos = np.sum(predicciones)
    predicciones_por_modelo.append(positivos)
    total_predicciones_positivas += positivos

print(f"Total de predicciones positivas (sumando todos los modelos): {total_predicciones_positivas}")
print("\nContribución de cada modelo al total de predicciones positivas:")

contribuciones_individuales = []
for i, modelo in enumerate(modelos):
    contribucion_pct = (predicciones_por_modelo[i] / total_predicciones_positivas) * 100
    contribuciones_individuales.append(contribucion_pct)
    print(f"{modelo}: {predicciones_por_modelo[i]} predicciones ({contribucion_pct:.1f}% del total)")

# Visualización - Solo gráfico de contribución individual
modelos_cortos = ["AUDIO", "IMAGE", "TEXT"]

# Crear el gráfico de contribución individual
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

bars = ax.bar(modelos_cortos, contribuciones_individuales, color='lightcoral', edgecolor='darkred', alpha=0.8)
ax.set_ylabel('Contribution to total positive predictions (%)')
ax.set_title('Individual contribution to total positive predictions')
ax.tick_params(axis='x', rotation=45, labelsize=18)
ax.grid(axis='y', alpha=0.3)

for bar, pct in zip(bars, contribuciones_individuales):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{pct:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("contribucion_individual_"+tipo+".png", dpi=300, bbox_inches='tight')
plt.show()